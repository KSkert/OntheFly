# src/onthefly/control.py
from __future__ import annotations

import atexit
import json
import math
import select
import sys
import threading
import time
import traceback
import queue
from typing import Any, Dict, Optional, Callable
from collections.abc import Mapping, Sequence

JsonDict = Dict[str, Any]
Handler = Callable[[JsonDict], Any]  # receives payload, returns reply data (dict/primitive)


# ---------------------- JSON helpers ----------------------

def _safe_default(o):
    """
    Convert numpy / torch objects into JSON-serializable Python types.
    Fallback: stringify unknown types.
    """
    try:
        import numpy as np
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass

    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.item() if o.ndim == 0 else o.detach().cpu().tolist()
    except Exception:
        pass

    return str(o)


def _coerce_nonfinite(x):
    # Replace NaN/Inf/-Inf with None so JSON is valid and Node can parse it.
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _sanitize(obj):
    """
    Walk an object (after _safe_default has done its best) and replace
    any non-finite floats with None. Handles nested mappings and sequences.
    Strings/bytes are left as-is.
    """
    if isinstance(obj, Mapping):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, bytes, bytearray)):
        return obj
    return _coerce_nonfinite(obj)


# ---------------------- Control bus (stdin -> command queue) ----------------------

class ControlBus:
    """
    Duplex channel (JSON Lines over stdio).

      UI -> trainer:  {"id":"<uuid>","cmd":"pause","payload":{...}}
      trainer -> UI:
        - reply: {"id":"<uuid>","ok":true,"data":{...}}
        - error: {"id":"<uuid>","ok":false,"error":"..."}
        - event: {"type":"metric"|"event"|"log", ...}

    Notes:
      * Spawn Python with -u so stdout is unbuffered.
      * We avoid blocking reads using select() + nonblocking queue.
    """
    def __init__(self, max_queue: int = 1000):
        self._q: "queue.Queue[JsonDict]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1)
        except Exception:
            pass

    def _reader(self) -> None:
        while not self._stop.is_set():
            # Non-blocking poll of stdin for commands from the UI.
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue

            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue

            try:
                msg = json.loads(line.strip())
                # Only enqueue proper command messages with a "cmd".
                if isinstance(msg, dict) and "cmd" in msg:
                    try:
                        self._q.put_nowait(msg)
                    except queue.Full:
                        # Backpressure: drop oldest to keep trainer responsive.
                        try:
                            _ = self._q.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._q.put_nowait(msg)
                        except Exception:
                            pass
            except Exception:
                # Ignore garbage lines
                pass

    def poll_cmd(self, timeout: float = 0.0) -> Optional[JsonDict]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------- Command router ----------------------

class CommandRouter:
    def __init__(self):
        self._handlers: Dict[str, Handler] = {}

    def on(self, cmd: str):
        def deco(fn: Handler):
            self._handlers[cmd] = fn
            return fn
        return deco

    def dispatch(self, cmd: str, payload: JsonDict) -> Any:
        if cmd not in self._handlers:
            raise RuntimeError(f"Unknown cmd: {cmd}")
        return self._handlers[cmd](payload)


def serve_commands(bus: ControlBus, router: CommandRouter, poll_sec: float = 0.05) -> None:
    """
    Call this from your trainer/main loop (or a dedicated thread) to
    process incoming commands and send replies.
    """
    while True:
        msg = bus.poll_cmd(timeout=poll_sec)
        if not msg:
            return  # return to caller so the trainer loop can continue

        req_id = msg.get("id") or ""
        cmd = msg.get("cmd")
        payload = msg.get("payload") or {}

        try:
            data = router.dispatch(cmd, payload)
            if req_id:
                send_reply(req_id, data)
        except Exception as e:
            tb = traceback.format_exc()
            if req_id:
                send_error(req_id, str(e), detail=tb)
            else:
                send_event({"type": "error", "text": str(e)})


# ---------------------- Async stdout writer ----------------------

class _AsyncStdoutWriter:
    """
    Two-queue async writer with small coalescing window.
    High-priority queue for replies/errors; low-priority for metrics/events.
    Robust against serialization errors (will not crash).
    """
    def __init__(self, max_latency_ms=50, max_batch_bytes=256 * 1024):
        self._hi = queue.SimpleQueue()   # replies/errors
        self._lo = queue.SimpleQueue()   # metrics/events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._max_latency = max_latency_ms / 1000.0
        self._max_bytes = max_batch_bytes
        self._lo_soft_cap = 20000

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=1)
            except Exception:
                pass

    def enqueue(self, obj, *, priority: bool = False):
        # Soft bound for low-priority growth: drop oldest if necessary.
        if not priority and getattr(self._lo, "qsize", None):
            try:
                if self._lo.qsize() > self._lo_soft_cap:
                    _ = self._lo.get_nowait()
            except Exception:
                pass
        (self._hi if priority else self._lo).put(obj)

    def _run(self):
        batch: list[str] = []
        size = 0
        last = time.time()

        def flush():
            nonlocal batch, size, last
            if not batch:
                return
            try:
                sys.stdout.write("".join(batch))
                sys.stdout.flush()
            except Exception as e:
                # stderr is not parsed by the extension as JSON, so it’s safe to log here
                try:
                    sys.stderr.write(f"[writer] stdout write error: {e}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            batch.clear()
            size = 0
            last = time.time()

        while not self._stop.is_set():
            now = time.time()
            got = None
            try:
                got = self._hi.get_nowait()
            except Exception:
                try:
                    got = self._lo.get(timeout=self._max_latency)
                except Exception:
                    pass

            if got is not None:
                try:
                    s = json.dumps(_sanitize(got), default=_safe_default, allow_nan=False) + "\n"
                except Exception as e:
                    # Never let a bad payload kill the writer thread.
                    # Emit a structured error to stderr so it’s visible.
                    try:
                        sys.stderr.write(f"[writer] serialization error: {e}\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    # Fallback: emit a minimal log line describing the issue.
                    try:
                        s = json.dumps(
                            {"type": "log", "text": f"serialization error for payload: {repr(got)[:200]}"},
                            allow_nan=True
                        ) + "\n"
                    except Exception:
                        # Absolute last resort: skip this message.
                        s = ""
                if s:
                    batch.append(s)
                    size += len(s)

            if size >= self._max_bytes or (batch and (now - last) >= self._max_latency):
                flush()

        flush()


_writer = _AsyncStdoutWriter()
atexit.register(lambda: _writer.stop())

def _ensure_writer_started():
    _writer.start()


# ---------------------- Public emitters ----------------------

def send_event(obj: dict) -> None:
    """
    Default event path. For per-step 'trainStep' events, prefer the fast path
    (direct write) so updates are minimally latent and not batched.
    Everything remains JSON-valid via _sanitize with allow_nan=False.
    """
    if isinstance(obj, dict) and obj.get("type") == "trainStep":
        try:
            s = json.dumps(_sanitize(obj), default=_safe_default, allow_nan=False) + "\n"
            sys.stdout.write(s)
            sys.stdout.flush()
            return
        except Exception:
            # Fall back to async writer if direct I/O or serialization failed
            _ensure_writer_started()
            _writer.enqueue(obj, priority=True)
            return

    _ensure_writer_started()
    _writer.enqueue(obj, priority=False)


def send_reply(req_id: str, data) -> None:
    _ensure_writer_started()
    _writer.enqueue({"id": req_id, "ok": True, "data": data}, priority=True)


def send_error(req_id: str, message: str, *, detail: str | None = None) -> None:
    _ensure_writer_started()
    payload = {"id": req_id, "ok": False, "error": message}
    if detail:
        payload["detail"] = detail
    _writer.enqueue(payload, priority=True)

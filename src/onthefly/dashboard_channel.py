from __future__ import annotations

import os
import queue
import socket
import sys
import threading
import time
from collections import deque
from typing import Optional


def _default_port() -> int:
    value = os.environ.get("ONTHEFLY_DASHBOARD_PORT", "") or ""
    try:
        port = int(value)
        if port > 0:
            return port
    except Exception:
        pass
    return 47621


class SocketChannel:
    """
    Duplex JSON-line channel that connects to the VS Code dashboard server.
    It tolerates the dashboard being closed: events are buffered until a
    connection is available, and commands are simply absent when nobody is
    connected.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        *,
        connect_timeout: float = 1.0,
        reconnect_interval: float = 2.0,
        backlog_limit: int = 20000,
        auto_connect: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port or _default_port())
        self._connect_timeout = max(0.1, float(connect_timeout))
        self._reconnect_interval = max(0.5, float(reconnect_interval))
        self._stop = threading.Event()
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._conn_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._backlog: "deque[str]" = deque(maxlen=max(1, int(backlog_limit)))
        self._cmd_queue: "queue.Queue[str]" = queue.Queue()
        if auto_connect:
            self.start()

    # ------------------------------------------------------------------ public API

    def start(self) -> None:
        if self._conn_thread and self._conn_thread.is_alive():
            return
        self._conn_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._conn_thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    # Writers used by control.send_event via the global channel -------------------

    def write_text(self, data: str) -> None:
        if not data:
            return
        with self._lock:
            sock = self._sock
        if not sock:
            self._backlog.append(data)
            return
        try:
            sock.sendall(data.encode("utf-8"))
        except Exception:
            self._backlog.append(data)
            self._handle_disconnect(sock)

    def flush(self) -> None:
        # sockets don't require flushing; method exists for interface parity
        return

    # Command consumption for ControlBus -----------------------------------------

    def read_line(self, timeout: float = 0.0) -> Optional[str]:
        try:
            return self._cmd_queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ internals

    def _connect_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                alive = self._sock is not None
            if alive:
                time.sleep(0.25)
                continue

            try:
                sock = socket.create_connection(
                    (self.host, self.port),
                    timeout=self._connect_timeout,
                )
            except Exception:
                time.sleep(self._reconnect_interval)
                continue

            try:
                sock.settimeout(0.5)
            except Exception:
                pass

            with self._lock:
                self._sock = sock

            self._reader = threading.Thread(target=self._reader_loop, args=(sock,), daemon=True)
            self._reader.start()
            self._flush_backlog()
            self._debug_log(f"[onthefly] dashboard connected on tcp://{self.host}:{self.port}")

        # When stop is set, reader loop will notice due to socket shutdown.

    def _flush_backlog(self) -> None:
        while self._backlog:
            data = self._backlog.popleft()
            self.write_text(data)

    def _reader_loop(self, sock: socket.socket) -> None:
        buffer = ""
        try:
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while True:
                    idx = buffer.find("\n")
                    if idx < 0:
                        break
                    line = buffer[:idx]
                    buffer = buffer[idx + 1 :]
                    if line:
                        self._cmd_queue.put(line)
        except Exception:
            pass
        finally:
            self._handle_disconnect(sock)

    def _handle_disconnect(self, sock: Optional[socket.socket]) -> None:
        with self._lock:
            if sock is not None and sock is self._sock:
                self._sock = None
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        self._debug_log("[onthefly] dashboard connection closed; buffering events until it reopens.")
        time.sleep(self._reconnect_interval)

    @staticmethod
    def _debug_log(msg: str) -> None:
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass


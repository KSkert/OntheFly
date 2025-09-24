from __future__ import annotations
import time
from typing import Dict, Any, Optional, List
from ..control import send_event

class EventsMixin:
    """
    Uniform, ordered emission for *all* UI/telemetry events.
    Also includes helpers that snapshot runtime config for the UI.
    """
    _event_seq: int
    _run_gen: int
    session_id: str

    def _event(self, obj: Dict[str, Any]) -> None:
        self._event_seq += 1
        obj.setdefault("run_id", self.cfg.run_name)
        obj.setdefault("run_gen", self._run_gen)
        obj.setdefault("ts", time.time())
        obj.setdefault("session_id", getattr(self, "session_id", None))
        obj["event_seq"] = self._event_seq
        send_event(obj)

    # Keep _emit as an alias for backwards compatibility
    def _emit(self, obj: Dict[str, Any]) -> None:
        self._event(obj)

    def _emit_new_run(self, run_id: str, parents: List[str], meta: Optional[Dict[str, Any]] = None):
        evt = {"type": "newRun", "run_id": run_id, "parents": parents}
        if meta: evt["meta"] = meta
        self._event(evt)

    def _snapshot_autofork_config(self):
        r = self._autofork_engine.rules
        return {"rules": dict(r.__dict__), "sampling": dict(self._af_cfg), "runtime": dict(self._af_rt)}

    def _ping_merge_gating(self, reason: str, extra: dict | None = None):
        payload = {"type": "merge_gating", "reason": reason}
        if extra: payload.update(extra)
        (getattr(self, "_emit", self._event))(payload)

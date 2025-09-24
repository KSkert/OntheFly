from __future__ import annotations
import torch

class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __call__(self): return self

def _sync_device_by_name(device: str | None):
    try:
        dev = str(device or "")
        if "cuda" in dev and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif "mps" in dev and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass

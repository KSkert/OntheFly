from __future__ import annotations
from typing import Optional, Union
import torch

DeviceLike = Union[str, torch.device, None]


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


def _as_device(device: DeviceLike, *, default: DeviceLike = "cpu") -> torch.device:
    """
    Best-effort conversion to torch.device with a configurable fallback.
    """
    def _normalize(value: DeviceLike) -> Optional[torch.device]:
        if isinstance(value, torch.device):
            return value
        if value is None:
            return None
        name = str(value).strip()
        if not name:
            return None
        try:
            return torch.device(name)
        except Exception:
            return None

    target = _normalize(device)
    if target is not None:
        return target
    fallback_dev = _normalize(default)
    if fallback_dev is not None:
        return fallback_dev
    return torch.device("cpu")


def _resolve_device(
    model: torch.nn.Module | None,
    *,
    prefer: DeviceLike = None,
    fallback: DeviceLike = "cpu",
) -> torch.device:
    """
    Resolve the device that a model should be associated with.

    Priority:
        1) Concrete parameters of `model`
        2) Optional `prefer` hint (e.g. user provided device or strategy root)
        3) Fallback (defaults to CPU)
    """
    if model is not None:
        try:
            param = next(model.parameters())
            dev = getattr(param, "device", None)
            if dev is not None:
                return _as_device(dev, default=fallback)
        except Exception:
            pass
        attr_dev = getattr(model, "device", None)
        if attr_dev is not None:
            return _as_device(attr_dev, default=fallback)
    if prefer is not None:
        return _as_device(prefer, default=fallback)
    return _as_device(fallback, default="cpu")

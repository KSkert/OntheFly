from .trainer import Trainer
from .wrappers.lightning import attach_lightning

try:  # pragma: no cover - importlib metadata shim
    from importlib import metadata as importlib_metadata
except Exception:  # Python <3.8 compatibility
    import importlib_metadata  # type: ignore

try:
    __version__ = importlib_metadata.version("onthefly-ai")
except Exception:
    __version__ = "0.0.0"

__all__ = ["Trainer", "attach_lightning", "__version__"]

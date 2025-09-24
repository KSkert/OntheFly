# src/seamless_modeldev/__init__.py
from __future__ import annotations

# Public API: main entry points
from .session import SeamlessSession, quickstart
from .config import SessionConfig

# Auto-fork rules — support both module names for backward compatibility
try:
    from .autofork import AutoForkRules  # singular file name
except Exception:  # pragma: no cover
    from .autoforks import AutoForkRules  # fallback if project uses plural

# Convenience re-exports
from .data_explorer import compute_per_sample_losses as per_sample_loss
from .metrics_utils import _to_scalar_loss as to_scalar_loss

__all__ = [
    "SeamlessSession",
    "quickstart",
    "SessionConfig",
    "AutoForkRules",
    "per_sample_loss",
    "to_scalar_loss",
]

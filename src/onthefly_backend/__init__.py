# src/onthefly_backend/__init__.py
from __future__ import annotations

# Public API: main entry points
from .session import OnTheFlySession, quickstart
from .config import SessionConfig
from .autofork import AutoForkRules  # singular file name

# Convenience re-exports
from .data_explorer import compute_per_sample_losses as per_sample_loss
from .metrics_utils import _to_scalar_loss as to_scalar_loss

__all__ = [
    "OnTheFlySession",
    "quickstart",
    "SessionConfig",
    "AutoForkRules",
    "per_sample_loss",
    "to_scalar_loss",
]

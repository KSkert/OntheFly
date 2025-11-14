from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SessionConfig:
    """
    Holds minimal, serializable settings for a training run.
    Keep this small and stable; other runtime state lives on the session.
    """
    project: str = "default"
    run_name: str = "run"
    device: Optional[str] = None
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    save_dir: str = "./checkpoints"
    ckpt_keep: int = 10
    ckpt_every_steps: int = 200

    # determinism control plane (non-invasive by default)
    # - "user": never touch sampler order
    # - "epoch_reseed": reproducible fresh shuffle per epoch
    # - "fixed_order": one reproducible shuffle reused every epoch
    data_order_policy: str = "user"      # "user" | "epoch_reseed" | "fixed_order"
    enforce_sampler_state: bool = True   # if True and policy != "user", wrap samplers so we can resume mid-epoch

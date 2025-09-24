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

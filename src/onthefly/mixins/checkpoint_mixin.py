from __future__ import annotations
import os, json
from typing import Optional, Dict, Any
from ..ckpt_utils import _parse_step

class CheckpointMixin:
    """
    Manages ring-checkpoints and latest-ckpt queries.
    """
    _ckpts: list


    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": (self.optimizer.state_dict() if self.optimizer is not None else None),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            "scaler": (
                getattr(self, "scaler", None).state_dict()
                if getattr(self, "scaler", None) else None
            ),
            "step": int(self.step),
            "epoch": int(self.epoch),
            "last_val_loss": (
                float(self._last_val_loss)
                if self._last_val_loss is not None else None
            ),
        }


    def _write_checkpoint_to(self, path: str) -> str:
        payload = self._checkpoint_payload()
        import torch, os as _os
        directory = _os.path.dirname(path)
        if directory:
            _os.makedirs(directory, exist_ok=True)
        torch.save(payload, path)
        return path


    def _save_ring_checkpoint(self, *, force: bool = False) -> Optional[str]:
        # Respect your "only when paused" rule *unless* someone explicitly forces it
        if not force and not getattr(self, "_paused", False):
            return None

        path = os.path.join(
            self.cfg.save_dir,
            f"{self.cfg.project}__{self.cfg.run_name}__step{self.step}.pt"
        )

        self._write_checkpoint_to(path)
        self._ckpts.append(path)
        if len(self._ckpts) > self.cfg.ckpt_keep:
            old = self._ckpts.pop(0)
            for p in (old, old + ".meta.json"):
                try:
                    os.remove(p)
                except Exception:
                    pass

        return path


    def _latest_ckpt_for_run(self, run_name: str) -> Optional[str]:
        patt = f"{self.cfg.project}__{run_name}__step"
        try:
            paths = [os.path.join(self.cfg.save_dir, p)
                     for p in os.listdir(self.cfg.save_dir)
                     if p.startswith(patt) and p.endswith(".pt")]
        except FileNotFoundError:
            return None
        if not paths: return None
        paths.sort(key=_parse_step)
        return paths[-1]

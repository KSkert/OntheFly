from __future__ import annotations
import os, json
from typing import Optional
from ..ckpt_utils import _parse_step
from ..checkpoints import save_checkpoint, load_checkpoint
from ..snapshots import SnapshotManager
from ..ids import _safe_component, _unique_component

class CheckpointMixin:
    """
    Manages ring-checkpoints, latest-ckpt queries, and snapshot publication.
    """
    _ckpts: list
    _snapshots: SnapshotManager


    def _save_ring_checkpoint(self, *, force: bool = False) -> Optional[str]:
        # Respect your "only when paused" rule *unless* someone explicitly forces it
        if not force and not getattr(self, "_paused", False):
            return None

        path = os.path.join(
            self.cfg.save_dir,
            f"{self.cfg.project}__{self.cfg.run_name}__step{self.step}.pt"
        )

        payload = {
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

        import torch, os as _os
        _os.makedirs(self.cfg.save_dir, exist_ok=True)
        torch.save(payload, path)

        self._ckpts.append(path)
        if len(self._ckpts) > self.cfg.ckpt_keep:
            old = self._ckpts.pop(0)
            for p in (old, old + ".meta.json"):
                try:
                    _os.remove(p)
                except Exception:
                    pass

        # publish snapshot for worker on cadence, only while paused (unchanged)
        try:
            every = int(self._feature_sampling_cfg.get("psl_every", 0) or 0)
        except Exception:
            every = 0
        if self._paused and every and (self.step % every) == 0:
            token, ref = self._snapshots.push_from_model(
                owner=self.cfg.run_name, version=self.step, model=self.model
            )
            self._expected_token_by_owner[self.cfg.run_name] = token
            try:
                self._feature_worker.submit_snapshot(
                    owner=self.cfg.run_name,
                    version=self.step,
                    token=token,
                    snapshot=ref,
                    ckpt_path=path,
                )
            except Exception:
                pass

        return path


    def _find_ckpt_for_rewind(self, steps_back: int) -> Optional[str]:
        target = max(0, self.step - steps_back)
        before = [p for p in self._ckpts if _parse_step(p) <= target]
        if not before: return self._ckpts[0] if self._ckpts else None
        return before[-1]

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

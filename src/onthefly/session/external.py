from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from ..config import SessionConfig
from ..control import ControlDelegate
from ..factory import _build_model_factory
from ..ids import _short_hash
from ..device_utils import _resolve_device
from ..runtime_metrics import ActivationZeroTracker, DeviceStatsMonitor, runtime_snapshot
from .base import OnTheFlySessionBase
from ..wrappers.base import FrameworkDelegate


class _LightningControlDelegate(ControlDelegate):
    def __init__(self, session: "OnTheFlyExternalSession", gate=None):
        self._session = session
        self._gate = gate

    def on_pause(self, req: dict[str, Any]) -> None:
        self._session._paused = True
        delegate = self._session._framework_delegate
        if delegate:
            try:
                delegate.request_pause(reason=str(req.get("reason", "manual")))
            except Exception:
                pass

    def on_resume(self) -> None:
        self._session._paused = False
        delegate = self._session._framework_delegate
        if delegate:
            try:
                delegate.request_resume()
            except Exception:
                pass

    def on_fork(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._session._do_fork(payload or {})

    def on_merge(
        self,
        *,
        parents: list[str] | None,
        strategy: str | None,
        paths: list[str] | None,
        new_name: str | None,
    ) -> dict[str, Any]:
        parents_list = [str(p) for p in (parents or []) if p]
        strategy_val = str(strategy).lower() if strategy else "swa"
        paths_list = [str(p) for p in (paths or []) if p]
        if not parents_list and not paths_list:
            raise RuntimeError("merge requires either 'parents' or explicit 'paths'")
        if parents_list and not paths_list:
            ckpts: list[str] = []
            for run_id in parents_list:
                ckpt = self._session._latest_ckpt_for_run(run_id)
                if not ckpt:
                    raise RuntimeError(f"no checkpoint found for parent run: {run_id}")
                ckpts.append(ckpt)
            paths_list = ckpts

        merged_name = self._session._merge_from_checkpoints(
            paths_list,
            strategy=strategy_val,
            parents=parents_list,
            new_name=new_name,
        )

        self._session._rebind_train_loader_to_subset(None)
        self._session._active_subset_indices = None
        self._session._save_ring_checkpoint()

        return {
            "new_run": merged_name,
            "parents": parents_list or None,
            "strategy": strategy_val,
            "paths": paths_list,
        }

    def on_test(self, label: str, source: str) -> dict[str, Any]:
        return self._session._run_labeled_test_and_ckpt(label=label, source=source)


class OnTheFlyExternalSession(OnTheFlySessionBase):
    """
    Control plane used by framework adapters (Lightning, Accelerate, ...).

    The wrapped framework owns the training loop; this session handles VS Code
    wiring, pause/resume, and telemetry so the dashboard behaves exactly like a
    native OnTheFly run.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        save_dir: str = "./checkpoints",
    ) -> None:
        cfg = SessionConfig(project=project, run_name=run_name, save_dir=save_dir)
        # Data/module slots must exist before the base installs determinism guards.
        self.datamodule = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self._train_root_ds = None
        self._active_subset_indices: Optional[List[int]] = None
        self._initial_run_name = str(run_name)
        super().__init__(
            cfg=cfg,
            delegate_factory=lambda gate: _LightningControlDelegate(self, gate=gate),
            start_running=False,
        )
        self._init_determinism(
            seed=42,
            data_order_policy="user",
            enforce_sampler_state=True,
            deterministic_pauses=True,
        )
        self._framework_delegate: FrameworkDelegate | None = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self._activation_tracker: ActivationZeroTracker | None = None
        self._device_monitor: DeviceStatsMonitor | None = None
        self._model_factory = None
        self._embedding_hook_fn = None

        self.raw_loss_fn = None
        self.loss_fn = None
        self._ckpts: List[str] = []
        self._test_inflight = False

        self._run_name_counts: Dict[str, int] = {}

        @self._router.on("stop")
        def _stop(_payload):
            self._running = False
            delegate = getattr(self, "_framework_delegate", None)
            if delegate:
                with contextlib.suppress(Exception):
                    delegate.request_stop()
            self._event({"type": "stopping"})
            return {"status": "stopping"}

        self.start_command_loop()

    # ------------------------------------------------------------------ wiring

    def bind_runtime(
        self,
        *,
        model=None,
        optimizer=None,
        scheduler=None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        if model is not None and model is not self.model:
            self.model = model
            self._activation_tracker = ActivationZeroTracker(model)

        target_model = self.model if self.model is not None else model
        resolved = _resolve_device(
            target_model,
            prefer=device,
            fallback=self.device or "cpu",
        )
        resolved_name = str(resolved)
        if resolved_name and self.device != resolved_name:
            self.device = resolved_name
            self._device_monitor = DeviceStatsMonitor(self.device)

        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
        if self.model is not None:
            self._refresh_model_factory()

    # ------------------------------------------------------------------ configuration helpers

    def _refresh_model_factory(self) -> None:
        if self.model is None:
            return
        self._model_factory = _build_model_factory(self.model, getattr(self, "_model_factory_override", None))

    def configure_model_factory(self, factory) -> None:
        self._model_factory_override = factory
        self._refresh_model_factory()

    def configure_embedding_hook(self, fn) -> None:
        self._embedding_hook_fn = fn

    def configure_datamodule(self, datamodule) -> None:
        self.datamodule = datamodule

    def _setup_datamodule(self, stage: str) -> None:
        dm = self.datamodule
        if dm is None or not hasattr(dm, "setup"):
            return
        attr = f"_onthefly_setup_{stage}"
        if getattr(dm, attr, False):
            return
        try:
            dm.setup(stage)
        except Exception:
            pass
        setattr(dm, attr, True)

    def configure_loss_fn(self, loss_fn) -> None:
        if loss_fn is None:
            return
        if not isinstance(loss_fn, torch.nn.Module):
            class _WrappedCriterion(torch.nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self._fn = fn
                def forward(self, *args, **kwargs):
                    return self._fn(*args, **kwargs)
            loss_fn = _WrappedCriterion(loss_fn)
            loss_fn._otf_uses_batch = True
            loss_fn._otf_batch_call_cfg = ("loss_fn(model, batch)", True, False)
        self.raw_loss_fn = loss_fn
        self.loss_fn = loss_fn

    def _set_train_loader(self, loader: DataLoader | None) -> None:
        if loader is None:
            return
        self.train_loader = loader
        self._train_root_ds = getattr(loader, "dataset", None)
        self._reapply_determinism()

    def _set_val_loader(self, loader: DataLoader | None) -> None:
        if loader is None:
            return
        self.val_loader = loader
        self._reapply_determinism()

    def _set_test_loader(self, loader: DataLoader | None) -> None:
        if loader is None:
            return
        self.test_loader = loader
        self._reapply_determinism()

    def configure_dataloaders(
        self,
        *,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
    ) -> None:
        self._set_train_loader(train_loader)
        self._set_val_loader(val_loader)
        self._set_test_loader(test_loader)

    def _ensure_train_loader(self) -> DataLoader | None:
        if self.train_loader is not None:
            return self.train_loader
        loader = None
        if self.datamodule and hasattr(self.datamodule, "train_dataloader"):
            self._setup_datamodule("fit")
            try:
                loader = self.datamodule.train_dataloader()
            except Exception:
                loader = None
        if loader is None and hasattr(self.model, "train_dataloader"):
            try:
                loader = self.model.train_dataloader()
            except Exception:
                loader = None
        if loader is not None:
            self._set_train_loader(loader)
        return self.train_loader

    def _ensure_val_loader(self) -> DataLoader | None:
        if self.val_loader is not None:
            return self.val_loader
        loader = None
        if self.datamodule and hasattr(self.datamodule, "val_dataloader"):
            self._setup_datamodule("fit")
            try:
                loader = self.datamodule.val_dataloader()
            except Exception:
                loader = None
        if loader is None and hasattr(self.model, "val_dataloader"):
            try:
                loader = self.model.val_dataloader()
            except Exception:
                loader = None
        if loader is not None:
            self._set_val_loader(loader)
        return self.val_loader

    def _ensure_test_loader(self) -> DataLoader | None:
        if self.test_loader is not None:
            return self.test_loader
        loader = None
        if self.datamodule and hasattr(self.datamodule, "test_dataloader"):
            self._setup_datamodule("test")
            try:
                loader = self.datamodule.test_dataloader()
            except Exception:
                loader = None
        if loader is None and hasattr(self.model, "test_dataloader"):
            try:
                loader = self.model.test_dataloader()
            except Exception:
                loader = None
        if loader is not None:
            self._set_test_loader(loader)
        return self.test_loader

    def _load_checkpoint_into_state(self, path: str) -> int:
        try:
            return super()._load_checkpoint_into_state(path)
        except Exception as exc:
            delegate = getattr(self, "_framework_delegate", None)
            if delegate is None:
                raise
            self._event(
                {
                    "type": "log",
                    "level": "info",
                    "text": f"[ckpt] falling back to Lightning loader for {os.path.basename(path)} ({exc})",
                }
            )
            step = delegate.load_checkpoint(path, None)
            self.step = int(step)
            return self.step

    def _notify_train_loader_changed(self) -> None:
        delegate = self._framework_delegate
        if delegate and hasattr(delegate, "refresh_train_loader"):
            try:
                delegate.refresh_train_loader()
            except Exception:
                pass

    def _rebind_train_loader_to_subset(self, indices: Optional[List[int]]):
        if self._train_root_ds is None:
            self._active_subset_indices = None
            return

        base_loader = self._ensure_train_loader()
        bs = getattr(base_loader, "batch_size", 256)
        cf = getattr(base_loader, "collate_fn", None)
        shuffle = getattr(base_loader, "shuffle", True)
        drop_last = getattr(base_loader, "drop_last", False)

        if indices and len(indices) > 0:
            sub = Subset(self._train_root_ds, list(indices))
            loader = DataLoader(sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self.train_loader = loader
            self._active_subset_indices = list(indices)
        else:
            loader = DataLoader(
                self._train_root_ds,
                batch_size=bs,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=cf,
            )
            self.train_loader = loader
            self._active_subset_indices = None
        self._notify_train_loader_changed()

    def _run_dir_exists(self, run_name: str) -> bool:
        return os.path.exists(os.path.join(self.cfg.save_dir, run_name))

    def attach_framework(self, delegate: FrameworkDelegate) -> None:
        self._framework_delegate = delegate
        delegate.install_batch_boundary_hook(self.console_gate)

    # ------------------------------------------------------------------ lifecycle helpers

    def start_streaming(self) -> None:
        self.before_training()

    def close(self, status: str = "stopped") -> None:
        self.after_training(status=status)
        if self._activation_tracker is not None:
            try:
                self._activation_tracker.close()
            except Exception:
                pass
        if self._device_monitor is not None:
            try:
                self._device_monitor.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ event emission

    def record_train_step(
        self,
        *,
        loss: float,
        batch: Any,
        metrics: Dict[str, Any],
        step_duration: float,
        step: int,
        epoch: int,
    ) -> None:
        raw_step = int(step)
        self.step = self._normalize_backend_step(raw_step)
        self.epoch = int(epoch)

        snapshot = runtime_snapshot(
            metrics,
            batch,
            step_duration,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            activation_tracker=self._activation_tracker,
            device_monitor=self._device_monitor,
            prev_lr=self._last_runtime_lr,
        )
        if snapshot.get("lr") is not None:
            self._last_runtime_lr = float(snapshot["lr"])

        payload: Dict[str, Any] = {
            "type": "trainStep",
            "step": self.step,
            "loss": float(loss),
            "val_loss": (float(self._last_val_loss) if self._last_val_loss is not None else None),
        }

        if self._last_emitted_epoch is None or epoch != self._last_emitted_epoch:
            payload["epoch"] = epoch
            self._last_emitted_epoch = epoch
            self._event({
                "type": "log",
                "level": "info",
                "phase": "train",
                "text": f"[epoch] now at epoch {epoch} (step {self.step})",
            })

        payload.update(snapshot)
        self._event(payload)

        last_v = f"{self._last_val_loss:.6f}" if self._last_val_loss is not None else "None"
        self._event({
            "type": "log",
            "level": "info",
            "text": f"step {self.step}: train_loss = {loss:.6f}, val_loss = {last_v}",
        })

    def record_epoch_end(self, epoch: int, *, val_loss: Optional[float]) -> None:
        self.epoch = int(epoch)
        self._event({"type": "epoch_end", "epoch": epoch, "val_loss": val_loss})

    def set_last_val_loss(self, value: Optional[float]) -> None:
        self._last_val_loss = float(value) if value is not None else None

    # ------------------------------------------------------------------ reset helpers

    def _dedupe_run_name(self, base: str) -> str:
        base_name = (base or "").strip() or "baseline"
        counts = self._run_name_counts
        current = counts.get(base_name, 0)
        counts[base_name] = current + 1
        return base_name if current == 0 else f"{base_name}#{current + 1}"

    def _restore_baseline_state(self) -> bool:
        delegate = self._framework_delegate
        if not delegate:
            return False
        try:
            restored = delegate.restore_initial_state()
        except Exception as exc:
            self._event({
                "type": "log",
                "level": "warn",
                "text": f"[reset] delegate restore failed: {exc}",
            })
            return False
        if not restored:
            self._event({
                "type": "log",
                "level": "warn",
                "text": "[reset] no baseline snapshot available; weights unchanged.",
            })
        return bool(restored)

    def _reset_session_state(self, *, run_hint: Optional[str], dedupe: bool) -> Dict[str, Any]:
        base = (
            run_hint
            or getattr(self, "_initial_run_name", None)
            or self.cfg.run_name
            or "baseline"
        )
        new_name = self._dedupe_run_name(base) if dedupe else base
        self._mark_backend_step_reset(raw_step=self.step)
        restored = self._restore_baseline_state()
        self.step = 0
        self.epoch = 0
        self._last_val_loss = None
        self._last_emitted_epoch = None
        self._event_seq = 0
        self._run_gen = 0
        self._active_subset_indices = None
        self.session_id = f"sess-{_short_hash(f'{self.cfg.project}|{new_name}|{time.time()}', n=12)}"
        self.cfg.run_name = new_name
        self._pause_ckpt_path = None
        if restored:
            self._event({
                "type": "log",
                "level": "info",
                "text": f"[reset] session cleared; '{new_name}' ready for resume.",
            })
        return {
            "run_id": new_name,
            "display_name": new_name,
            "project": self.cfg.project,
            "session_id": self.session_id,
        }

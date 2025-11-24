from __future__ import annotations
from typing import Optional, Callable, Any, Dict, List
import time, os, tempfile, shutil, contextlib, threading
import torch
from torch.utils.data import DataLoader


from ..config import SessionConfig
from ..factory import _build_model_factory
from ..device_utils import _noop_ctx
from ..scale import _SafeScaler
from ..ids import _short_hash
from ..control import OnTheFlyTrainerDelegate
from ..mixins.train_mixin import TrainMixin
from ..runtime_metrics import ActivationZeroTracker, DeviceStatsMonitor
from .base import OnTheFlySessionBase


class OnTheFlySession(OnTheFlySessionBase, TrainMixin):
    """
    Orchestrates a *single* training run while delegating functionality to focused mixins.
    Public API and method names match the original.
    """
    def __init__(
        self,
        project: str,
        run_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: Optional[str] = None,
        scheduler: Optional[Any] = None,
        amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        save_dir: str = "./checkpoints",
        seed: int = 42,
        embedding_hook: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        model_factory: Optional[Callable[[], torch.nn.Module]] = None,
        data_order_policy: str = "user",    # "user" | "epoch_reseed" | "fixed_order"
        deterministic_pauses: bool = True,
        enforce_sampler_state: bool = True, # keep True; just means "wrap if policy != user"
        val_every_n_epochs: Optional[int] = 1,
    ):
        self.cfg = SessionConfig(project, run_name, device, amp, grad_clip_norm, save_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self._val_every_n_epochs = self._normalize_val_schedule(val_every_n_epochs)
        self._model_factory = _build_model_factory(self.model, model_factory)
        self._embedding_hook_fn = embedding_hook

        super().__init__(
            cfg=self.cfg,
            delegate_factory=lambda gate: OnTheFlyTrainerDelegate(session=self, gate=gate),
            start_running=True,
        )
        self.__init__identity_and_device(project, run_name)
        self.__init__loss_and_scaler(loss_fn)
        self.__init__train_context()
        self._init_determinism(
            seed=seed,
            data_order_policy=data_order_policy,
            enforce_sampler_state=enforce_sampler_state,
            deterministic_pauses=deterministic_pauses,
        )
        self._initial_run_name = str(run_name)
        self._reset_snapshot_dir: Optional[str] = None
        self._reset_snapshot_path: Optional[str] = None
        self._reset_snapshot_evt = threading.Event()
        self._reset_snapshot_thread = threading.Thread(
            target=self._capture_reset_snapshot,
            name="otf-reset-snapshot",
            daemon=True,
        )
        self._reset_snapshot_thread.start()

    # small helpers to keep names intact
    def _run_dir_exists(self, run_name: str) -> bool:
        import os
        return os.path.exists(os.path.join(self.cfg.save_dir, run_name))

    def _dedupe_run_name(self, base: str) -> str:
        if not self._run_dir_exists(base) and base != self.cfg.run_name:
            return base
        i = 2
        candidate = f"{base}#{i}"
        while self._run_dir_exists(candidate) or candidate == self.cfg.run_name:
            i += 1; candidate = f"{base}#{i}"
        return candidate
    
    @staticmethod
    def _normalize_val_schedule(freq: Optional[int]) -> int:
        if freq is None:
            return 0
        try:
            value = int(freq)
        except Exception:
            return 0
        return max(0, value)

    def _capture_reset_snapshot(self) -> None:
        try:
            self._cleanup_reset_snapshot()
            tmpdir = tempfile.mkdtemp(prefix="onthefly-reset-")
            fname = f"{self.cfg.project}__{self.cfg.run_name}__baseline.pt"
            path = os.path.join(tmpdir, fname)
            self._write_checkpoint_to(path)
            self._reset_snapshot_dir = tmpdir
            self._reset_snapshot_path = path
        except Exception as e:
            self._reset_snapshot_dir = None
            self._reset_snapshot_path = None
            with contextlib.suppress(Exception):
                self._event({
                    "type": "log",
                    "level": "warn",
                    "text": f"[reset] could not cache baseline snapshot: {e}",
                })
        finally:
            evt = getattr(self, "_reset_snapshot_evt", None)
            if evt:
                evt.set()

    def _cleanup_reset_snapshot(self) -> None:
        directory = getattr(self, "_reset_snapshot_dir", None)
        if directory and os.path.isdir(directory):
            with contextlib.suppress(Exception):
                shutil.rmtree(directory, ignore_errors=True)
        self._reset_snapshot_dir = None
        self._reset_snapshot_path = None

    def _wait_for_reset_snapshot(self, timeout: Optional[float] = None) -> None:
        evt = getattr(self, "_reset_snapshot_evt", None)
        if evt is not None:
            evt.wait(timeout=timeout)

    def _restore_from_reset_snapshot(self) -> bool:
        self._wait_for_reset_snapshot()
        path = getattr(self, "_reset_snapshot_path", None)
        if not path or not os.path.exists(path):
            return False
        try:
            blob = torch.load(path, map_location="cpu")
        except Exception as e:
            with contextlib.suppress(Exception):
                self._event({
                    "type": "log",
                    "level": "warn",
                    "text": f"[reset] failed to load baseline snapshot: {e}",
                })
            return False

        model_state = blob.get("model")
        if model_state:
            self.model.load_state_dict(model_state, strict=True)

        opt_state = blob.get("optimizer")
        if self.optimizer is not None:
            if opt_state:
                with contextlib.suppress(Exception):
                    self.optimizer.load_state_dict(opt_state)
            with contextlib.suppress(Exception):
                self.optimizer.zero_grad(set_to_none=True)

        sched_state = blob.get("scheduler")
        if self.scheduler is not None and sched_state:
            with contextlib.suppress(Exception):
                self.scheduler.load_state_dict(sched_state)

        scaler_state = blob.get("scaler")
        if getattr(self, "scaler", None):
            if scaler_state:
                with contextlib.suppress(Exception):
                    self.scaler.load_state_dict(scaler_state)
            else:
                self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))
        else:
            self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))

        self.step = int(blob.get("step") or 0)
        self.epoch = int(blob.get("epoch") or 0)
        last_val = blob.get("last_val_loss")
        self._last_val_loss = float(last_val) if last_val is not None else None
        self._epoch_batch_idx = 0
        self._last_emitted_epoch = None
        return True

    def _reset_session_state(self, *, run_hint: Optional[str], dedupe: bool = True) -> Dict[str, Any]:
        base_name = str(
            run_hint
            or getattr(self, "_initial_run_name", None)
            or self.cfg.run_name
            or "baseline"
        ).strip() or "baseline"

        new_name = base_name
        if dedupe:
            try:
                new_name = self._dedupe_run_name(base_name)
            except Exception:
                new_name = base_name

        restored = self._restore_from_reset_snapshot()
        if not restored:
            with contextlib.suppress(Exception):
                self._event({
                    "type": "log",
                    "level": "warn",
                    "text": "[reset] baseline snapshot missing; weights left unchanged.",
                })

        self._active_subset_indices = None
        with contextlib.suppress(Exception):
            self._rebind_train_loader_to_subset(None)

        self._ckpts.clear()
        self._pause_ckpt_path = None
        self._run_gen = 0
        self._event_seq = 0
        self._last_val_loss = None
        self._last_emitted_epoch = None
        self._epoch_batch_idx = 0
        self.step = 0
        self.epoch = 0

        if getattr(self, "optimizer", None):
            with contextlib.suppress(Exception):
                self.optimizer.zero_grad(set_to_none=True)
        if getattr(self, "scaler", None) is None:
            self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))

        self.cfg.run_name = new_name
        self.session_id = f"sess-{_short_hash(f'{self.cfg.project}|{new_name}|{time.time()}', n=12)}"
        return {
            "run_id": new_name,
            "display_name": new_name,
            "project": self.cfg.project,
            "session_id": self.session_id,
        }

    def __init__identity_and_device(self, project: str, run_name: str) -> None:
        if self.cfg.device:
            self.device = self.cfg.device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self._activation_tracker = ActivationZeroTracker(self.model)
        self._device_monitor = DeviceStatsMonitor(self.device)

    def __init__loss_and_scaler(self, loss_fn: Callable) -> None:
        self.raw_loss_fn = loss_fn

        def _wrapped_loss_fn(*args, **kwargs):
            out = self.raw_loss_fn(*args, **kwargs)
            from ..metrics_utils import _to_scalar_loss
            return _to_scalar_loss(out, device=self.device)

        self.loss_fn = _wrapped_loss_fn
        self.autocast = torch.cuda.amp.autocast if (self.cfg.amp and "cuda" in self.device) else _noop_ctx
        self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))

    def __init__train_context(self) -> None:
        self._training_step_fn = self._default_training_step
        self._validation_step_fn = self._default_validation_step
        self._train_root_ds = getattr(self.train_loader, "dataset", None)
        self._active_subset_indices: Optional[List[int]] = None

    # def _skip_train_batches_if_needed(self) -> None:
    #     """
    #     If resuming mid-epoch from a fresh iterator, advance by the last seen cursor.
    #     Safe no-op if you resume with a live iterator.
    #     """
    #     k = int(getattr(self, "_epoch_batch_idx", 0) or 0)
    #     if k <= 0:
    #         return
    #     import itertools as _it
    #     try:
    #         _ = next(_it.islice(iter(self.train_loader), k, None), None)
    #     except Exception:
    #         pass

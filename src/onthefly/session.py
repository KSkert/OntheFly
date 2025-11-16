from __future__ import annotations
from typing import Optional, Callable, Any, Dict, List
import time, threading
import torch
from torch.utils.data import DataLoader
from functools import partial


from .config import SessionConfig
from .factory import _build_model_factory
from .device_utils import _noop_ctx
from .utils import _seed_worker
from .scale import _SafeScaler
from .ids import _short_hash
from .control import ControlBus, CommandRouter
from .mixins.events_mixin import EventsMixin
from .mixins.checkpoint_mixin import CheckpointMixin
from .mixins.feature_mixin import FeatureMixin
from .mixins.run_management_mixin import RunManagementMixin
from .mixins.commands_mixin import CommandsMixin
from .mixins.train_mixin import TrainMixin
from .sampler_utils import EpochSeededRandomSampler
from .runtime_metrics import ActivationZeroTracker, DeviceStatsMonitor

class OnTheFlySession(EventsMixin, CheckpointMixin, FeatureMixin, RunManagementMixin, CommandsMixin, TrainMixin):
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

        self.__init__identity_and_device(project, run_name)
        self.__init__loss_and_scaler(loss_fn)
        self.__init__runtime_state()
        self.__init__control_plane()
        self.__init__train_context()
        self.__init__determinism(
            seed=seed,
            data_order_policy=data_order_policy,
            enforce_sampler_state=enforce_sampler_state,
            deterministic_pauses=deterministic_pauses,
        )

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

    # --- rebuild a DataLoader with a specific sampler (no other behavior changes) ---
    def _rebuild_loader_with_sampler(self, loader: DataLoader, sampler):
        if not isinstance(loader, DataLoader):
            return loader
        kwargs = dict(
            dataset=loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            shuffle=False,  # sampler controls order
            num_workers=getattr(loader, "num_workers", 0),
            collate_fn=getattr(loader, "collate_fn", None),
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
            timeout=getattr(loader, "timeout", 0),
            persistent_workers=getattr(loader, "persistent_workers", False),
        )
        # only pass if present/non-None
        wif = getattr(loader, "worker_init_fn", None)
        if wif is not None:
            kwargs["worker_init_fn"] = wif
        gen = getattr(loader, "generator", None)
        if gen is not None:
            kwargs["generator"] = gen
        pf = getattr(loader, "prefetch_factor", None)
        if pf is not None:
            kwargs["prefetch_factor"] = pf

        return DataLoader(**kwargs)

    def __init__identity_and_device(self, project: str, run_name: str) -> None:
        self.session_id = f"sess-{_short_hash(f'{project}|{run_name}|{time.time()}', n=12)}"
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
            from .metrics_utils import _to_scalar_loss
            return _to_scalar_loss(out, device=self.device)

        self.loss_fn = _wrapped_loss_fn
        self.autocast = torch.cuda.amp.autocast if (self.cfg.amp and "cuda" in self.device) else _noop_ctx
        self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))

    def __init__runtime_state(self) -> None:
        self.step = 0
        self.epoch = 0
        self._running = True
        self._paused = False
        self._event_seq = 0
        self._run_gen = 0
        self._ckpts: List[str] = []
        self._last_val_loss: Optional[float] = None
        self._halt_evt = threading.Event()
        self._pause_gen = 0
        self._pause_ckpt_path: Optional[str] = None
        self._feature_sampling_cfg: Dict[str, Any] = dict(
            psl_every=0,
            psl_budget=4000,
            mirror_train=True,
            amp_for_psl=True,
            compute_margins=True,
            compute_embeddings=False,
            embed_max_dim=256,
        )

    def __init__control_plane(self) -> None:
        self._bus = ControlBus()
        self._router = CommandRouter()
        self._register_command_handlers()
        try:
            self._bus.start()
        except Exception:
            pass

    def __init__train_context(self) -> None:
        self._training_step_fn = self._default_training_step
        self._validation_step_fn = self._default_validation_step
        self._train_root_ds = getattr(self.train_loader, "dataset", None)
        self._active_subset_indices: Optional[List[int]] = None

    def __init__determinism(
        self,
        *,
        seed: int,
        data_order_policy: Optional[str],
        enforce_sampler_state: bool,
        deterministic_pauses: bool,
    ) -> None:
        self._data_order_policy = (data_order_policy or "user").lower()
        self._enforce_sampler_state = bool(enforce_sampler_state)
        self._deterministic_pauses = bool(deterministic_pauses)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._apply_determinism_policy(seed)
        if self._deterministic_pauses:
            self._install_determinism_guards(seed)
        else:
            self._dl_determinism_managed = False
            self._det_seed = None
            self._epoch_batch_idx = 0

    

    def _install_determinism_guards(self, base_seed: int) -> None:
        import numpy as _np
        from torch.utils.data import DataLoader as _DL
        from torch.utils.data import RandomSampler as _RandSampler
        try:
            from torch.utils.data.distributed import DistributedSampler as _DistSampler
        except Exception:
            _DistSampler = tuple()

        if base_seed is None or int(base_seed) < 0:
            self._dl_determinism_managed = False
            self._det_seed = None
            self._epoch_batch_idx = 0
            return

        def _clone_loader(dl: _DL, seed: int) -> _DL:
            if not isinstance(dl, _DL):
                return dl
            sampler = getattr(dl, "sampler", None)
            has_native_state = bool(callable(getattr(sampler, "state_dict", None)))
            if has_native_state:
                return dl  # native sampler restart is sufficient

            gen = getattr(dl, "generator", None) or torch.Generator(device="cpu")
            gen.manual_seed(int(seed))

            want_shuffle = isinstance(sampler, _RandSampler) or bool(getattr(dl, "shuffle", False))
            keep_sampler = isinstance(sampler, _DistSampler)

            kwargs = dict(
                dataset=dl.dataset,
                batch_size=dl.batch_size,
                shuffle=(want_shuffle and not keep_sampler),
                sampler=(sampler if keep_sampler else None),
                num_workers=dl.num_workers,
                collate_fn=dl.collate_fn,
                pin_memory=dl.pin_memory,
                drop_last=dl.drop_last,
                timeout=dl.timeout,
                persistent_workers=getattr(dl, "persistent_workers", False),
                generator=gen,
            )

            # Preserve any existing worker_init_fn but wrap it with our seeding
            orig_wif = getattr(dl, "worker_init_fn", None)
            if dl.num_workers and dl.num_workers > 0:
                kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=seed, orig=orig_wif)
            elif orig_wif is not None:
                kwargs["worker_init_fn"] = orig_wif

            pf = getattr(dl, "prefetch_factor", None)
            if pf is not None:
                kwargs["prefetch_factor"] = pf

            return _DL(**kwargs)

    
        self._det_seed = int(base_seed)
        managed_any = False

        if isinstance(self.train_loader, torch.utils.data.DataLoader):
            before = self.train_loader
            self.train_loader = _clone_loader(self.train_loader, base_seed)
            managed_any |= (self.train_loader is not before)

        if isinstance(self.val_loader, torch.utils.data.DataLoader):
            before = self.val_loader
            self.val_loader = _clone_loader(self.val_loader, base_seed + 17)
            managed_any |= (self.val_loader is not before)

        if isinstance(self.test_loader, torch.utils.data.DataLoader):
            before = self.test_loader
            self.test_loader = _clone_loader(self.test_loader, base_seed + 33)
            managed_any |= (self.test_loader is not before)

        self._dl_determinism_managed = bool(managed_any)
        self._epoch_batch_idx = 0


    # --- apply the chosen determinism policy (no-op when policy="user") ---
    def _apply_determinism_policy(self, base_seed: int):
        policy = getattr(self, "_data_order_policy", "user")
        if policy == "user" or not getattr(self, "_enforce_sampler_state", True):
            self._sampler_stateful = False
            return

        # In DDP we let DistributedSampler handle per-epoch determinism (via set_epoch)
        def _is_ddp():
            try:
                import torch.distributed as dist
                return dist.is_available() and dist.is_initialized()
            except Exception:
                return False

        def _wrap_loader(loader, seed_offset: int):
            if not isinstance(loader, DataLoader) or loader.dataset is None:
                return loader, False

            # Keep user’s DistributedSampler in DDP
            try:
                from torch.utils.data.distributed import DistributedSampler as _DistSampler
                if isinstance(getattr(loader, "sampler", None), _DistSampler):
                    return loader, False
            except Exception:
                pass

            # Need dataset length to build a random permutation
            try:
                n = len(loader.dataset)
                if not isinstance(n, int) or n <= 0:
                    return loader, False
            except Exception:
                return loader, False

            fixed = (policy == "fixed_order")
            epoch_fn = lambda: getattr(self, "epoch", 0)
            sampler = EpochSeededRandomSampler(
                data_len=n,
                base_seed=int(base_seed + seed_offset),
                epoch_fn=epoch_fn,
                fixed_order=fixed,
            )
            return self._rebuild_loader_with_sampler(loader, sampler), True

        # Wrap train/val (test usually not shuffled; leave it)
        any_wrapped = False
        if not _is_ddp():
            self.train_loader, a = _wrap_loader(self.train_loader, 0)
            self.val_loader,   b = _wrap_loader(self.val_loader, 17)
            any_wrapped = a or b

        self._sampler_stateful = any_wrapped

        # Keep DDP happy if user is using DistributedSampler
        self._sampler_set_epoch(self.epoch)

    def _sampler_set_epoch(self, epoch: int) -> None:
        """Call this at the start of each epoch to keep DistributedSampler deterministic."""
        try:
            from torch.utils.data.distributed import DistributedSampler as _DistSampler
            s = getattr(self.train_loader, "sampler", None)
            if isinstance(s, _DistSampler):
                try: s.set_epoch(int(epoch))
                except Exception: pass
        except Exception:
            pass

    def _skip_train_batches_if_needed(self) -> None:
        """
        If resuming mid-epoch from a fresh iterator, advance by the last seen cursor.
        Safe no-op if you resume with a live iterator.
        """
        k = int(getattr(self, "_epoch_batch_idx", 0) or 0)
        if k <= 0:
            return
        import itertools as _it
        try:
            _ = next(_it.islice(iter(self.train_loader), k, None), None)
        except Exception:
            pass


def quickstart(*, project, run_name, model, optimizer, loss_fn,
               train_loader, val_loader, test_loader=None,
               max_epochs=None, max_steps=None, do_test_after=False,
               val_every_n_epochs: Optional[int] = None,
               model_factory=None, **kwargs):
    s = OnTheFlySession(
        project=project, run_name=run_name,
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        model_factory=model_factory,
        val_every_n_epochs=val_every_n_epochs,
        **kwargs
    )
    s.serve(max_steps=max_steps, max_epochs=max_epochs, do_test_after=do_test_after)

from __future__ import annotations
from typing import Optional, Callable, Any, Dict, List
import os, time, threading, random
import torch
from torch.utils.data import DataLoader
from queue import Queue

from .config import SessionConfig
from .factory import _build_model_factory
from .device_utils import _noop_ctx
from .scale import _SafeScaler
from .ids import _short_hash
from .snapshots import SnapshotManager
from .feature_worker import FeatureWorker
from .control import ControlBus, CommandRouter
from .mixins.events_mixin import EventsMixin
from .mixins.checkpoint_mixin import CheckpointMixin
from .mixins.feature_mixin import FeatureMixin
from .mixins.autofork_mixin import AutoForkMixin
from .mixins.commands_mixin import CommandsMixin
from .mixins.train_mixin import TrainMixin

class OnTheFlySession(EventsMixin, CheckpointMixin, FeatureMixin, AutoForkMixin, CommandsMixin, TrainMixin):
    """
    Orchestrates a *single* training run and an AutoForkEngine, delegating most
    behavior to focused mixins. Public API and method names match the original.
    """
    def __init__(
        self,
        project: str,
        run_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[str] = None,
        scheduler: Optional[Any] = None,
        amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        save_dir: str = "./checkpoints",
        seed: int = 42,
        embedding_hook: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        model_factory: Optional[Callable[[], torch.nn.Module]] = None,
    ):
        self.cfg = SessionConfig(project, run_name, device, amp, grad_clip_norm, save_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self._model_factory = _build_model_factory(self.model, model_factory)
        self._embedding_hook_fn = embedding_hook

        # session identity & device
        self.session_id = f"sess-{_short_hash(f'{project}|{run_name}|{time.time()}', n=12)}"
        if self.cfg.device: self.device = self.cfg.device
        elif torch.cuda.is_available(): self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"
        self.model.to(self.device)

        # loss / amp / scaler
        self.raw_loss_fn = loss_fn
        def _wrapped_loss_fn(*args, **kwargs):
            out = self.raw_loss_fn(*args, **kwargs)
            from .metrics_utils import _to_scalar_loss
            return _to_scalar_loss(out, device=self.device)
        self.loss_fn = _wrapped_loss_fn
        self.autocast = torch.cuda.amp.autocast if (self.cfg.amp and "cuda" in self.device) else _noop_ctx
        self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))

        # state
        self.step = 0; self.epoch = 0
        self._running = True; self._paused = False
        self._auto_exec_before_pause: Optional[bool] = None
        self._auto_merge_exec_before_pause: Optional[bool] = None
        self._event_seq = 0; self._run_gen = 0
        self._ckpts: List[str] = []
        self._expected_token_by_owner: Dict[str, str] = {}
        self._children_registered: set[str] = set()
        self._child_parent: dict[str, str] = {}
        self._last_val_loss: Optional[float] = None
        self._halt_evt = threading.Event()
        self._pause_gen = 0
        self._pause_ckpt_path: Optional[str] = None

        # snapshots + feature worker infra
        self._snapshots = SnapshotManager(keep_per_owner=3, default_spill_dir=self.cfg.save_dir)
        self._snapshots.attach_session(self)
        self._feature_queue = Queue(maxsize=4)
        self._FeatureWorkerCtor = FeatureWorker

        # --- pending feature latch (needed by FeatureMixin/AutoForkMixin) ---
        self._pending_feature_feed = None
        self._pending_feature_owner = None
        self._pending_feature_step = None
        self._pending_feature_token = None

        # autofork
        from .autofork import AutoForkRules, AutoForkEngine
        self._autofork_engine = AutoForkEngine(AutoForkRules())
        self._af_cfg: Dict[str, Any] = dict(psl_every=200, psl_budget=4000, mirror_train=True, amp_for_psl=True,
                                            compute_margins=True, compute_embeddings=False, embed_max_dim=256)
        self._af_rt: Dict[str, Any] = dict(auto_execute=False, auto_merge_execute=True, variant_policy="first",
                                           variant_index=0, name_template="{parent}-auto@{step}",
                                           min_train_steps_between_autoforks=100, max_branch_depth=1)
        self._af_rr = 0
        self._last_autoexec_ckpt_step_by_owner: Dict[str, int] = {}

        # bus + router
        self._bus = ControlBus(); self._router = CommandRouter()
        self._register_command_handlers()
        try: self._bus.start()
        except Exception: pass

        # train hooks & dataset roots
        self._training_step_fn = self._default_training_step
        self._validation_step_fn = self._default_validation_step
        self._train_root_ds = getattr(self.train_loader, "dataset", None)
        self._active_subset_indices: Optional[List[int]] = None

        # start background feature worker immediately
        self._spawn_feature_worker()

        # seeding
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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

def quickstart(*, project, run_name, model, optimizer, loss_fn,
               train_loader, val_loader, test_loader=None,
               max_epochs=None, max_steps=None, do_test_after=False,
               model_factory=None, **kwargs):
    s = OnTheFlySession(
        project=project, run_name=run_name,
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        model_factory=model_factory,
        **kwargs
    )
    s.serve(max_steps=max_steps, max_epochs=max_epochs, do_test_after=do_test_after)

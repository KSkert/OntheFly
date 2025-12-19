from __future__ import annotations

import contextlib
import os
import threading
import time
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import SessionConfig
from ..control import CommandRouter, ConsoleAction, ControlBus, ControlDelegate, PauseGate, serve_commands
from ..ids import _short_hash
from ..metrics_utils import _to_scalar_loss
from ..mixins.checkpoint_mixin import CheckpointMixin
from ..mixins.commands_mixin import CommandsMixin
from ..mixins.events_mixin import EventsMixin
from ..mixins.feature_mixin import FeatureMixin
from ..mixins.run_management_mixin import RunManagementMixin
from ..device_utils import _resolve_device
from ..sampler_utils import EpochSeededRandomSampler
from ..utils import _seed_worker
from ..data_explorer import (
    model_input_candidates,
    ensure_tensor_output,
    should_retry_model_input,
    _call_batch_loss,
    _apply_loss_with_warning_guard,
    _extract_loss_value,
)




class OnTheFlySessionBase(  # pylint: disable=too-many-instance-attributes
    EventsMixin,
    CheckpointMixin,
    FeatureMixin,
    RunManagementMixin,
    CommandsMixin,
):
    """
    Shared runtime/control plane used by both native and external sessions.
    Provides ordered event emission, checkpoint helpers, and uniform command routing.
    """

    def __init__(
        self,
        *,
        cfg: SessionConfig,
        delegate_factory: Callable[[PauseGate], ControlDelegate],
        start_running: bool = True,
    ) -> None:
        self.cfg = cfg
        self.session_id = f"sess-{_short_hash(f'{cfg.project}|{cfg.run_name}|{time.time()}', n=12)}"

        # runtime state
        self.step = 0
        self.epoch = 0
        self._running = bool(start_running)
        self._paused = False
        self._event_seq = 0
        self._run_gen = 0
        self._pause_gen = 0
        self._pause_ckpt_path: Optional[str] = None
        self._last_val_loss: Optional[float] = None
        self._last_emitted_epoch: Optional[int] = None
        self._last_runtime_lr: Optional[float] = None
        self._feature_sampling_cfg: Dict[str, Any] = dict(
            psl_every=0,
            psl_budget=4000,
            mirror_train=True,
            amp_for_psl=True,
            compute_margins=True,
            compute_embeddings=False,
            embed_max_dim=256,
        )
        self._ckpts: list[str] = []
        self._test_inflight = False
        self._test_counter = 0
        self._halt_evt = threading.Event()
        self._tick_pause_notified = False
        self._fork_counter_seed = 0
        self._merge_counter_seed = 0
        self._counter_seed_gen = 0
        self._lifecycle_started = False
        self._lifecycle_finished = False
        self._dl_determinism_managed = False
        self._det_seed: int | None = None
        self._epoch_batch_idx = 0
        self._sampler_stateful = False
        self._deterministic_pauses = True
        self._data_order_policy = "user"
        self._enforce_sampler_state = True
        self._reset_backend_step_tracking()
        self._fatal_error: Optional[str] = None

        # IO/control plumbing
        self._bus = ControlBus()
        self._router = CommandRouter()
        self.console_gate = PauseGate()
        delegate = delegate_factory(self.console_gate)
        self.console_action = ConsoleAction(session=self, delegate=delegate, gate=self.console_gate)
        self._register_command_handlers()
        self._bus.start()

        self._stop_commands = threading.Event()
        self._command_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle hooks

    def before_training(self) -> None:
        """Emit the standard session header once before training/driving begins."""
        if self._lifecycle_started:
            return
        self._lifecycle_started = True
        self._running = True
        self._event(
            {"type": "session_started", "project": self.cfg.project, "run_name": self.cfg.run_name}
        )
        self._event({"type": "log", "level": "info", "text": f"model session_id={self.session_id}"})
        self._event({"type": "log", "level": "info", "text": "training"})
        self._event({"type": "log", "level": "info", "text": self.cfg.run_name or "baseline"})
        self._emit_new_run(self.cfg.run_name, parents=[], meta={"display_name": self.cfg.run_name})

    def after_training(self, *, status: str = "completed") -> None:
        """Emit a terminal event and stop background loops exactly once."""
        if self._lifecycle_finished:
            return
        self._lifecycle_finished = True
        self._running = False
        self._event(
            {
                "type": "trainingFinished",
                "status": status,
                "run_id": self.cfg.run_name,
                "step": int(self.step),
                "session_id": self.session_id,
            }
        )
        self.stop_command_loop()
        with contextlib.suppress(Exception):
            self._bus.stop()

    # ------------------------------------------------------------------ command helpers

    def start_command_loop(self, poll_sec: float = 0.1) -> None:
        """Spin a background thread that continuously drains the command queue."""
        if self._command_thread and self._command_thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop_commands.is_set():
                serve_commands(self._bus, self._router, poll_sec=poll_sec)

        self._command_thread = threading.Thread(target=_loop, daemon=True)
        self._command_thread.start()

    def stop_command_loop(self) -> None:
        if not self._command_thread:
            return
        self._stop_commands.set()
        try:
            self._command_thread.join(timeout=1)
        except Exception:
            pass
        self._command_thread = None
        self._stop_commands.clear()

    def _maybe_handle_commands(self) -> None:
        serve_commands(self._bus, self._router, poll_sec=0.0)

    def tick(self, *, idle_sleep: float = 0.05) -> None:
        """
        Run a lightweight servicing pass used by external loops.
        Mirrors what TrainMixin does around each batch so pause/resume semantics match.
        """
        self._maybe_handle_commands()
        if not self._running:
            self._tick_pause_notified = False
            return
        if not self._paused:
            self._tick_pause_notified = False
            return
        if not self._tick_pause_notified:
            try:
                self._event({"type": "paused", "run_id": self.cfg.run_name, "step": int(self.step)})
            except Exception:
                pass
            self._tick_pause_notified = True
        while self._paused and self._running:
            self._maybe_handle_commands()
            time.sleep(idle_sleep)

    # ------------------------------------------------------------------ checkpoint helpers

    def _safe_load_state(self, blob: Dict[str, Any]) -> None:
        if "model" in blob and getattr(self, "model", None) is not None:
            self.model.load_state_dict(blob["model"])
        if "optimizer" in blob and getattr(self, "optimizer", None) is not None:
            with contextlib.suppress(Exception):
                self.optimizer.load_state_dict(blob["optimizer"])
        if "scheduler" in blob and getattr(self, "scheduler", None) is not None:
            with contextlib.suppress(Exception):
                self.scheduler.load_state_dict(blob["scheduler"])
        if "scaler" in blob and getattr(self, "scaler", None) is not None:
            with contextlib.suppress(Exception):
                self.scaler.load_state_dict(blob["scaler"])
        if "epoch" in blob:
            with contextlib.suppress(Exception):
                self.epoch = int(blob["epoch"])
        if "last_val_loss" in blob:
            with contextlib.suppress(Exception):
                self._last_val_loss = float(blob["last_val_loss"])

    def _load_checkpoint_into_state(self, path: str) -> int:
        """
        Load a Seamless checkpoint and return the resume step.
        Supports both safe weights-only loads and trusted fallbacks for older formats.
        """
        import inspect

        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        blob: Dict[str, Any] | None = None

        try:
            if supports_weights_only:
                blob = torch.load(path, map_location=getattr(self, "device", "cpu"), weights_only=True)
            else:
                blob = torch.load(path, map_location=getattr(self, "device", "cpu"))
        except Exception:
            blob = torch.load(path, map_location=getattr(self, "device", "cpu"), weights_only=False)

        state = blob.get("state", blob) if isinstance(blob, dict) else blob
        self._safe_load_state(state)
        return int(state.get("step", state.get("global_step", 0)))

    # ------------------------------------------------------------------ dataloader helpers

    def _ensure_train_loader(self) -> DataLoader | None:
        return getattr(self, "train_loader", None)

    def _ensure_val_loader(self) -> DataLoader | None:
        return getattr(self, "val_loader", None)

    def _ensure_test_loader(self) -> DataLoader | None:
        return getattr(self, "test_loader", None)

    # ------------------------------------------------------------------ determinism helpers

    def _rebuild_loader_with_sampler(self, loader: DataLoader, sampler):
        if not isinstance(loader, DataLoader):
            return loader
        kwargs = dict(
            dataset=loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=getattr(loader, "num_workers", 0),
            collate_fn=getattr(loader, "collate_fn", None),
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
            timeout=getattr(loader, "timeout", 0),
            persistent_workers=getattr(loader, "persistent_workers", False),
        )
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

    def _install_determinism_guards(self, base_seed: int) -> None:
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
                return dl

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

        if isinstance(self.train_loader, DataLoader):
            before = self.train_loader
            self.train_loader = _clone_loader(self.train_loader, base_seed)
            managed_any |= (self.train_loader is not before)

        if isinstance(self.val_loader, DataLoader):
            before = self.val_loader
            self.val_loader = _clone_loader(self.val_loader, base_seed + 17)
            managed_any |= (self.val_loader is not before)

        if isinstance(self.test_loader, DataLoader):
            before = self.test_loader
            self.test_loader = _clone_loader(self.test_loader, base_seed + 33)
            managed_any |= (self.test_loader is not before)

        self._dl_determinism_managed = bool(managed_any)
        self._epoch_batch_idx = 0

    def _apply_determinism_policy(self, base_seed: int) -> None:
        policy = getattr(self, "_data_order_policy", "user")
        if policy == "user" or not getattr(self, "_enforce_sampler_state", True):
            self._sampler_stateful = False
            return

        def _is_ddp():
            try:
                import torch.distributed as dist
                return dist.is_available() and dist.is_initialized()
            except Exception:
                return False

        def _wrap_loader(loader, seed_offset: int):
            if not isinstance(loader, DataLoader) or loader.dataset is None:
                return loader, False
            try:
                from torch.utils.data.distributed import DistributedSampler as _DistSampler
                if isinstance(getattr(loader, "sampler", None), _DistSampler):
                    return loader, False
            except Exception:
                pass
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

        any_wrapped = False
        if not _is_ddp():
            self.train_loader, a = _wrap_loader(self.train_loader, 0)
            self.val_loader, b = _wrap_loader(self.val_loader, 17)
            any_wrapped = a or b

        self._sampler_stateful = any_wrapped
        self._sampler_set_epoch(self.epoch)

    def _sampler_set_epoch(self, epoch: int) -> None:
        try:
            from torch.utils.data.distributed import DistributedSampler as _DistSampler
            sampler = getattr(self.train_loader, "sampler", None)
            if isinstance(sampler, _DistSampler):
                with contextlib.suppress(Exception):
                    sampler.set_epoch(int(epoch))
        except Exception:
            pass

    def _reapply_determinism(self) -> None:
        seed = getattr(self, "_det_seed", None)
        if seed is None:
            return
        self._apply_determinism_policy(seed)
        if getattr(self, "_deterministic_pauses", False):
            self._install_determinism_guards(seed)
        else:
            self._epoch_batch_idx = 0

    def _init_determinism(
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

    # ------------------------------------------------------------------ evaluation helpers

    def _run_test(self) -> float:
        loader = self._ensure_test_loader()
        if loader is None:
            raise RuntimeError("test_loader is not configured; cannot run test.")
        if getattr(self, "loss_fn", None) is None:
            raise RuntimeError("loss_fn/model not configured; cannot run test.")
        if getattr(self, "model", None) is None:
            raise RuntimeError("model is not bound; cannot run test.")

        model = self.model
        torch_device = _resolve_device(
            model,
            prefer=getattr(self, "device", None),
            fallback="cpu",
        )
        device_hint = str(torch_device)

        loss_fn = self.loss_fn

        # Static hint from attribute + type (same as compute_per_sample_losses)
        is_batch_aware_flag = bool(getattr(loss_fn, "_otf_uses_batch", False))

        was_training = bool(getattr(model, "training", True))

        model.eval()

        try:
            losses: list[float] = []
            self._event(
                {"type": "log", "level": "info", "phase": "test", "text": "[test] starting evaluation pass"}
            )

            with torch.no_grad():
                for tstep, batch in enumerate(loader, start=1):
                    batch = self._move_to_device(batch, torch_device)

                    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                        raise RuntimeError("Unexpected batch format; expected (inputs, targets, *...).")

                    first = batch[0]
                    y = batch[1]
                    rest = list(batch[2:])

                    uses_batch = is_batch_aware_flag
                    batch_loss_probe = None

                    if callable(loss_fn) and not uses_batch:
                        try:
                            batch_loss_probe = _apply_loss_with_warning_guard(
                                lambda: _call_batch_loss(loss_fn, model, batch)
                            )
                            uses_batch = True
                            if not is_batch_aware_flag:
                                try:
                                    setattr(loss_fn, "_otf_uses_batch", True)
                                except Exception:
                                    pass
                                is_batch_aware_flag = True
                        except TypeError:
                            batch_loss_probe = None
                        except Exception:
                            batch_loss_probe = None
                            uses_batch = False

                    if uses_batch:
                        if batch_loss_probe is not None:
                            raw_loss = batch_loss_probe
                        else:
                            raw_loss = _apply_loss_with_warning_guard(
                                lambda: _call_batch_loss(loss_fn, model, batch)
                            )
                    else:
                        candidates = model_input_candidates(model, first, rest)
                        logits = None
                        last_exc: Exception | None = None

                        for cand in candidates:
                            try:
                                out = model(cand)
                                logits = ensure_tensor_output(out)
                                break
                            except Exception as exc:
                                if should_retry_model_input(exc):
                                    last_exc = exc
                                    continue
                                raise

                        if logits is None:
                            if last_exc is not None:
                                raise last_exc
                            raise RuntimeError("Could not prepare inputs for model in _run_test.")

                        raw_loss = loss_fn(logits, y)

                    try:
                        loss_tensor_candidate = _extract_loss_value(raw_loss)
                        scalar = _to_scalar_loss(loss_tensor_candidate, device=device_hint)
                        if torch.is_tensor(scalar):
                            value = float(scalar.detach().cpu().item())
                        else:
                            value = float(scalar)
                    except Exception:
                        value = float("nan")

                    losses.append(value)
                    self._event({"type": "testStep", "step": tstep, "loss": value})
                    self._event(
                        {
                            "type": "log",
                            "level": "info",
                            "phase": "test",
                            "text": f"step {tstep}: test_loss = {value:.6f}",
                        }
                    )

            avg = sum(losses) / max(1, len(losses))
            self._event({"type": "log", "level": "info", "phase": "test", "text": f"test_avg_loss = {avg:.6f}"})
            self._event({"type": "log", "level": "info", "phase": "test", "text": "[test] evaluation pass complete"})
            return float(avg)
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

    def _move_to_device(self, obj: Any, device: torch.device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v, device) for v in obj)
        if isinstance(obj, list):
            return [self._move_to_device(v, device) for v in obj]
        return obj

    def _run_labeled_test_and_ckpt(self, *, label: str = "final", source: str = "manual") -> Dict[str, Any]:
        loader = self._ensure_test_loader()
        if loader is None:
            raise RuntimeError("test_loader is not configured; cannot run test.")

        if getattr(self, "_test_inflight", False):
            return {
                "status": "busy",
                "run_id": self.cfg.run_name,
                "session_id": getattr(self, "session_id", None),
                "step": int(self.step),
                "label": label,
            }

        self._test_inflight = True
        test_tag = self._prepare_test_env_label()
        try:
            self._event(
                {
                    "type": "log",
                    "level": "info",
                    "phase": "test",
                    "text": f"[test] requested evaluation for label '{label}' (source={source})",
                }
            )
            avg = float(self._run_test())
            ckpt_path = self._save_ring_checkpoint(force=True)

            if ckpt_path is not None:
                import os

                self._event(
                    {
                        "type": "log",
                        "level": "info",
                        "phase": "test",
                        "text": f"[test] checkpoint saved for label '{label}': {os.path.basename(ckpt_path)}",
                    }
                )
            else:
                self._event(
                    {
                        "type": "log",
                        "level": "warn",
                        "phase": "test",
                        "text": f"[test] requested checkpoint for label '{label}' but save returned None",
                    }
                )

            result: Dict[str, Any] = {
                "status": "ok",
                "avg_loss": avg,
                "run_id": self.cfg.run_name,
                "session_id": getattr(self, "session_id", None),
                "step": int(self.step),
                "label": label,
                "ckpt_path": ckpt_path,
                "env_test_label": test_tag,
            }

            event = dict(result)
            event["type"] = "test_complete"
            event["source"] = source
            self._event(event)

            if source == "auto":
                auto_evt = dict(result)
                auto_evt["type"] = "auto_test_complete"
                auto_evt["source"] = "auto"
                self._event(auto_evt)
            return result
        finally:
            self._test_inflight = False

    def _prepare_test_env_label(self) -> str:
        self._test_counter += 1
        tag = f"test_{self._test_counter}"
        try:
            os.environ["ONTHEFLY_TEST_LABEL"] = tag
        except Exception:
            pass
        return tag

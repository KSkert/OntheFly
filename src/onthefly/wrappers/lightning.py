from __future__ import annotations

import copy
import contextlib
import importlib
import os
import sys
import time
from typing import Any, Callable, Dict, Optional

import torch
import types
from torch.utils.data import DataLoader

from ..control import PauseGate, set_channel
from ..dashboard_channel import SocketChannel
from ..device_utils import _resolve_device
from ..metrics_utils import _to_scalar_loss
from ..runtime_metrics import canonicalize_metrics
from ..session.external import OnTheFlyExternalSession
from .base import FrameworkDelegate

_GLOBAL_BASELINE_DEVICE: Optional[torch.device] = None

def _resolve_lightning_callback():
    try:
        module = importlib.import_module("lightning.pytorch.callbacks")
        return module.Callback
    except Exception:
        module = importlib.import_module("pytorch_lightning.callbacks")
        return module.Callback


class _LightningFitDriver:
    """
    Wrap Lightning's `.fit(...)` so auto-test pauses behave like the native
    trainer: after the Lightning segment finishes and auto-test pauses, we
    block until the dashboard resumes/stops. On resume, we extend the trainer's
    epoch/step limits and call the original `.fit(...)` again to continue
    training, all under a single user-level `trainer.fit(...)` call.
    """

    def __init__(self, delegate: "LightningFrameworkDelegate"):
        self.delegate = delegate
        self.session = delegate.session
        self.trainer = delegate.trainer
        self._orig_fit = getattr(self.trainer, "fit")
        self._active = False  # guard against re-entrant calls
        self._resume_epoch_window = self._infer_epoch_window()
        self._resume_step_window = self._infer_step_window()

    # ------------------------------------------------------------------ config inference

    def _infer_epoch_window(self) -> int:
        candidates: list[int] = []
        try:
            val = getattr(self.trainer, "max_epochs", None)
            if isinstance(val, int) and val > 0:
                candidates.append(val)
        except Exception:
            pass
        try:
            loop = getattr(self.trainer, "fit_loop", None)
            if loop is not None:
                val = getattr(loop, "max_epochs", None)
                if isinstance(val, int) and val > 0:
                    candidates.append(val)
        except Exception:
            pass
        # Fallback to 1 epoch if nothing obvious is configured
        return max(1, max(candidates or [1]))

    def _infer_step_window(self) -> Optional[int]:
        candidates: list[int] = []
        try:
            val = getattr(self.trainer, "max_steps", None)
            if isinstance(val, int) and val > 0:
                candidates.append(val)
        except Exception:
            pass
        try:
            loop = getattr(self.trainer, "fit_loop", None)
            if loop is not None:
                val = getattr(loop, "max_steps", None)
                if isinstance(val, int) and val > 0:
                    candidates.append(val)
        except Exception:
            pass
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------ binding

    def bind(self) -> None:
        """
        Monkey-patch trainer.fit so user code can keep calling
            trainer.fit(model, datamodule=...)
        but we transparently re-enter the original `.fit` as needed after
        auto-test pauses.
        """

        def _wrapped(trainer_self, *args, **kwargs):
            return self(trainer_self, *args, **kwargs)

        self.trainer._onthefly_fit_wrapped = True
        self.trainer._onthefly_orig_fit = self._orig_fit
        self.trainer.fit = types.MethodType(_wrapped, self.trainer)


    # ------------------------------------------------------------------ main driver

    def __call__(self, _trainer_self, *args, **kwargs):
        # Avoid recursion if someone somehow calls trainer.fit() from inside our own logic.
        if self._active:
            return self._orig_fit(*args, **kwargs)

        status = "completed"
        last_result = None
        self._active = True
        try:
            while True:
                # Reset per-segment flags
                self.delegate._auto_test_triggered = False
                self.delegate._fit_end_status = None

                
                # Run one Lightning fit segment
                last_result = self._orig_fit(*args, **kwargs)

                # Status from the callback's on_fit_end hook, or fallback
                status = (
                    self.delegate._fit_end_status
                    or ("stopped" if getattr(self.trainer, "should_stop", False) else "completed")
                )
                self.session._event({
                    "type": "log",
                    "level": "info",
                    "text": (
                        f"[fit_driver] segment finished; "
                        f"status={status}, "
                        f"auto_test_triggered={self.delegate._auto_test_triggered}, "
                        f"post_hold={self.delegate._post_auto_test_hold}, "
                        f"paused={getattr(self.session, '_paused', None)}, "
                        f"running={getattr(self.session, '_running', None)}"
                    ),
                })

                if self._should_hold_after_auto_test():
                    if not self._wait_for_resume():
                        # User stopped from the dashboard
                        status = "stopped"
                        break

                    # User resumed after auto-test:
                    #  - clear per-segment hold flags so we don't re-block
                    #  - extend epoch/step window and re-enter fit
                    #
                    # One-shot semantics for auto-test are now handled by
                    # self.delegate._auto_test_count in _run_auto_test_after_training.
                    self.delegate._auto_test_triggered = False
                    self.delegate._post_auto_test_hold = False
                    self._prepare_for_resume()
                    continue

                # No auto-test hold; respect session liveness.
                if not getattr(self.session, "_running", False):
                    status = "stopped"
                break
        finally:
            try:
                # Single place where we close the external session after
                # all segments (and any auto-test resumes) are done.
                self.session.close(status=status)
            finally:
                self._active = False

        return last_result

    # ------------------------------------------------------------------ helpers

    def _should_hold_after_auto_test(self) -> bool:
        return (
            (self.delegate._auto_test_triggered or self.delegate._post_auto_test_hold)
            and getattr(self.session, "_paused", False)
        )

    def _wait_for_resume(self) -> bool:
        """
        Block in Python (not in Lightning) until:
          * user resumes (session._paused -> False) -> return True
          * user stops (session._running -> False) -> return False
        """
        self.session._event({
            "type": "log",
            "level": "info",
            "text": "[fit_driver] entering wait_for_resume after auto-test pause",
        })
        while True:
            self.session._maybe_handle_commands()
            if not getattr(self.session, "_running", False):
                self.session._event({"type": "log", "level": "info",
                                    "text": "[fit_driver] wait_for_resume: _running is False -> stop"})
                return False
            if not getattr(self.session, "_paused", False):
                self.session._event({"type": "log", "level": "info",
                                    "text": "[fit_driver] wait_for_resume: _paused is False -> resume"})
                return True
            time.sleep(0.05)

    def _prepare_for_resume(self) -> None:
        """
        After the user resumes from an auto-test pause, adjust Lightning's
        stopping criteria so training can continue.

        Semantics:
          * Before any auto-test, we would extend by a local epoch/step window
            (kept here for completeness).
          * AFTER the first auto-test has run (self.delegate._auto_test_count >= 1),
            we effectively "ignore" the original max_epochs/max_steps by setting
            them to very large values, so training only stops when the user
            sends a Stop command (or Lightning stops for some other reason).
        """
        trainer = self.trainer

        # Clear any generic should_stop flag Lightning set.
        try:
            trainer.should_stop = False
        except Exception:
            pass

        loop = getattr(trainer, "fit_loop", None)

        # Are we in "post auto-test" mode?
        post_auto = getattr(self.delegate, "_auto_test_count", 0) >= 1

        # Current progress
        cur_epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        cur_step = int(getattr(trainer, "global_step", 0) or 0)

        # Decide target limits
        target_max_epochs: Optional[int] = None
        target_max_steps: Optional[int] = None

        if post_auto:
            # After the first auto-test, effectively ignore the original
            # max_epochs/max_steps by pushing them way out.
            #
            # These values just need to be "large enough" that user-driven
            # Stop is the only practical termination.
            target_max_epochs = max(cur_epoch + 1, 10**9)

            if self._resume_step_window is not None:
                target_max_steps = max(cur_step + 1, 10**12)
        else:
            # (For completeness) pre-auto-test behavior: extend by the
            # inferred window relative to the current position.
            target_max_epochs = cur_epoch + self._resume_epoch_window
            if self._resume_step_window is not None:
                target_max_steps = cur_step + max(1, self._resume_step_window)

        # Apply to fit loop
        if loop is not None:
            try:
                if hasattr(loop, "max_epochs") and target_max_epochs is not None:
                    loop.max_epochs = target_max_epochs
            except Exception:
                pass
            try:
                if hasattr(loop, "max_steps") and target_max_steps is not None:
                    loop.max_steps = target_max_steps
            except Exception:
                pass

        # Mirror onto trainer attributes if present
        try:
            if hasattr(trainer, "max_epochs") and target_max_epochs is not None:
                trainer.max_epochs = target_max_epochs
        except Exception:
            pass
        try:
            if hasattr(trainer, "max_steps") and target_max_steps is not None:
                trainer.max_steps = target_max_steps
        except Exception:
            pass

        # Refresh dataloaders so any OnTheFly subset rebinding propagates
        with contextlib.suppress(Exception):
            self.delegate.refresh_train_loader()


class LightningFrameworkDelegate(FrameworkDelegate):
    """
    Framework delegate that wires a Lightning Trainer into an external session.
    """

    def __init__(
        self,
        *,
        session: OnTheFlyExternalSession,
        trainer: Any,
        model: Any,
        datamodule: Any = None,
        log_every_n_steps: int = 1,
        do_test_after: bool = False,
    ) -> None:
        self.session = session
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._patched_loaders = False
        self._do_test_after = bool(do_test_after)
        self._auto_test_triggered = False
        self._fit_end_status: Optional[str] = None
        self._post_auto_test_hold = False
        self._streaming_started = False  # ensure before_training runs only once
        self._auto_test_count: int = 0
        self._baseline_device = None  # device Lightning was using at baseline capture
        self._lightning_test_inflight = False

        Callback = _resolve_lightning_callback()

        class _LightningConsoleCallback(Callback):  # type: ignore
            def __init__(self, outer: "LightningFrameworkDelegate"):
                super().__init__()
                self._outer = outer
                self._gate: PauseGate | None = None
                self._step_start: Optional[float] = None

            def set_gate(self, gate: PauseGate) -> None:
                self._gate = gate

            def _wait_for_gate(self) -> None:
                gate = self._gate or self._outer.session.console_gate
                if gate and gate.should_block():
                    gate.wait_until_resumed()

            def _infer_device(self, trainer, pl_module):
                try:
                    strategy = getattr(trainer, "strategy", None)
                    prefer = getattr(strategy, "root_device", None)
                except Exception:
                    prefer = None
                session_dev = getattr(self._outer.session, "device", None)
                return _resolve_device(
                    pl_module,
                    prefer=prefer,
                    fallback=session_dev or "cpu",
                )

            def _ensure_session_device(self, trainer, pl_module) -> None:
                dev = self._infer_device(trainer, pl_module)
                if dev is None:
                    return
                current = self._outer.session.device
                current = str(current) if current is not None else None
                dev_str = str(dev)
                if current == dev_str:
                    return
                self._outer.session.bind_runtime(model=pl_module, device=dev)

            def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
                self._outer.model = pl_module

            def on_fit_start(self, trainer, pl_module) -> None:
                # Handles may change if the strategy/optimizers are reconfigured.
                self._outer._refresh_runtime_handles(trainer)
                dev_hint = self._infer_device(trainer, pl_module)
                self._outer.session.bind_runtime(
                    model=pl_module,
                    optimizer=self._outer.optimizer,
                    scheduler=self._outer.scheduler,
                    device=dev_hint,
                )
                self._ensure_session_device(trainer, pl_module)
                # Capture baseline + emit run header only once, on the first segment.
                if not self._outer._streaming_started:
                    self._outer._capture_initial_state()
                    self._outer.session.start_streaming()
                    self._outer._streaming_started = True
                self._outer.session.tick()

            def on_fit_end(self, trainer, pl_module) -> None:
                # Sync device one more time in case Lightning relocated the module.
                self._ensure_session_device(trainer, pl_module)
                # Run a single auto-test + pause if configured. Do NOT close
                # the session here; the fit driver owns that lifecycle.
                self._outer._run_auto_test_after_training()
                self._outer._fit_end_status = (
                    "stopped" if getattr(trainer, "should_stop", False) else "completed"
                )

            def on_train_epoch_start(self, trainer, pl_module) -> None:
                epoch = int(getattr(trainer, "current_epoch", 0) or 0)
                self._outer.session._sampler_set_epoch(epoch)
                self._outer.session.tick()

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
                self._ensure_session_device(trainer, pl_module)
                self._outer.session.tick()
                self._wait_for_gate()
                self._step_start = time.perf_counter()

            def _extract_loss(self, outputs, trainer) -> Optional[torch.Tensor]:
                if isinstance(outputs, dict):
                    if "loss" in outputs:
                        return outputs["loss"]
                    for key, value in outputs.items():
                        if "loss" in key.lower():
                            return value
                if torch.is_tensor(outputs):
                    return outputs
                metrics = getattr(trainer, "callback_metrics", None) or {}
                for key in ("loss", "train_loss", "training_loss"):
                    if key in metrics:
                        return metrics[key]
                for key, value in metrics.items():
                    if "loss" in key.lower():
                        return value
                return None

            def on_train_batch_end(
                self,
                trainer,
                pl_module,
                outputs,
                batch,
                batch_idx,
            ) -> None:
                self._ensure_session_device(trainer, pl_module)
                if self._outer.log_every_n_steps > 1:
                    step = int(getattr(trainer, "global_step", 0) or 0)
                    if step % self._outer.log_every_n_steps != 0:
                        return

                loss_tensor = self._extract_loss(outputs, trainer)
                if loss_tensor is None:
                    loss_val = float("nan")
                else:
                    try:
                        scalar = _to_scalar_loss(loss_tensor, device=str(pl_module.device))
                        loss_val = float(scalar.detach().cpu().item())
                    except Exception:
                        loss_val = float("nan")

                step_duration = 0.0
                if self._step_start is not None:
                    step_duration = max(time.perf_counter() - self._step_start, 1e-9)
                self._step_start = None

                metrics = self._outer._collect_stream_metrics(trainer, outputs)
                self._outer.session.record_train_step(
                    loss=loss_val,
                    batch=batch,
                    metrics=metrics,
                    step_duration=step_duration,
                    step=int(getattr(trainer, "global_step", 0) or 0),
                    epoch=int(getattr(trainer, "current_epoch", 0) or 0),
                )
                self._outer.session.tick()

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                metrics = getattr(trainer, "callback_metrics", None) or {}
                val = None
                for key in ("val_loss", "val/loss", "validation_loss"):
                    if key in metrics:
                        val = metrics[key]
                        break
                if val is None:
                    for key, value in metrics.items():
                        if "val" in key.lower() and "loss" in key.lower():
                            val = value
                            break
                if val is not None:
                    try:
                        scalar = float(
                            _to_scalar_loss(val, device=str(pl_module.device)).detach().cpu().item()
                        )
                    except Exception:
                        scalar = None
                    self._outer.session.set_last_val_loss(scalar)

            def on_train_epoch_end(self, trainer, pl_module) -> None:
                self._outer.session.record_epoch_end(
                    int(getattr(trainer, "current_epoch", 0) or 0),
                    val_loss=self._outer.session._last_val_loss,
                )

        self._callback = _LightningConsoleCallback(self)
        self.optimizer = None
        self.scheduler = None
        self._initial_state: Optional[Dict[str, Any]] = None
        self._fit_driver = _LightningFitDriver(self)

    # ------------------------------------------------------------------ framework wiring

    def attach(self, model, trainer, datamodule) -> None:
        self.model = model
        self.trainer = trainer
        self.datamodule = datamodule
        self.session.bind_runtime(model=model)
        self.session.configure_datamodule(datamodule)
        self._patch_dataloader_sources()


        if not any(cb is self._callback for cb in getattr(trainer, "callbacks", [])):
            trainer.callbacks.append(self._callback)

        # Wrap trainer.fit once we are attached
        self._fit_driver.bind()


    def install_batch_boundary_hook(self, gate: PauseGate) -> None:
        self._callback.set_gate(gate)

    # ----- baseline capture/restore -------------------------------------------

    @staticmethod
    def _deep_copy(obj: Any):
        if torch.is_tensor(obj):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: LightningFrameworkDelegate._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [LightningFrameworkDelegate._deep_copy(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(LightningFrameworkDelegate._deep_copy(v) for v in obj)
        return copy.deepcopy(obj)


    def _capture_initial_state(self) -> None:
        if self.model is None:
            return

        # Ask Lightning what device it is actually using
        device = None
        try:
            strategy = getattr(self.trainer, "strategy", None)
            device = getattr(strategy, "root_device", None)
        except Exception:
            device = None

        # Fallback: infer from the model parameters
        if device is None and self.model is not None:
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = torch.device("cpu")

        # Remember per-delegate + globally for all future forks/runs
        self._baseline_device = device

        global _GLOBAL_BASELINE_DEVICE
        # Only set once; first successful baseline run defines the "canonical" device
        if _GLOBAL_BASELINE_DEVICE is None and device is not None:
            _GLOBAL_BASELINE_DEVICE = device

        state: Dict[str, Any] = {
            "model": self._deep_copy(self.model.state_dict()),
        }
        if self.optimizer is not None:
            try:
                state["optimizer"] = self._deep_copy(self.optimizer.state_dict())
            except Exception:
                state["optimizer"] = None
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            try:
                state["scheduler"] = self._deep_copy(self.scheduler.state_dict())
            except Exception:
                state["scheduler"] = None
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            state["global_step"] = int(getattr(trainer, "global_step", 0) or 0)
            state["current_epoch"] = int(getattr(trainer, "current_epoch", 0) or 0)
        self._initial_state = state

    # ------------------------------------------------------------------ dataloaders

    def _patch_dataloader_sources(self) -> None:
        if self._patched_loaders:
            return
        owner = self.datamodule or self.model
        if owner is None:
            return

        def _wrap(attr: str, getter):
            fn = getattr(owner, attr, None)
            if not callable(fn):
                return
            orig_name = f"_onthefly_orig_{attr}"
            if hasattr(owner, orig_name):
                return
            original = fn

            def wrapped(inst, *args, **kwargs):
                loader = getter()
                if loader is not None:
                    return loader
                return original(*args, **kwargs)

            setattr(owner, orig_name, original)
            setattr(owner, attr, types.MethodType(wrapped, owner))

        _wrap("train_dataloader", lambda: self.session.train_loader)
        _wrap("val_dataloader", lambda: self.session.val_loader)
        _wrap("test_dataloader", lambda: self.session.test_loader)
        self._patched_loaders = True

    def refresh_train_loader(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        if hasattr(trainer, "reset_train_dataloader"):
            with contextlib.suppress(Exception):
                trainer.reset_train_dataloader(self.model)
                return
        connector = getattr(trainer, "_data_connector", None)
        if connector and hasattr(connector, "attach_data"):
            with contextlib.suppress(Exception):
                connector.attach_data(
                    model=self.model,
                    train_dataloader=self.session.train_loader,
                    val_dataloaders=self.session.val_loader,
                    datamodule=self.datamodule,
                )

    def run_lightning_test(self, *, test_loader, label: str, source: str) -> None:
        if self._lightning_test_inflight:
            return
        trainer = getattr(self, "trainer", None)
        model = getattr(self, "model", None)
        if trainer is None or model is None:
            return
        loader = test_loader
        if loader is None:
            loader = self.session._ensure_test_loader()
        if loader is None:
            return
        self._lightning_test_inflight = True
        self.session._event(
            {
                "type": "log",
                "level": "info",
                "phase": "test",
                "text": f"[lightning_test] invoking trainer.test for label='{label}' (source={source})",
            }
        )
        try:
            trainer.test(model=model, dataloaders=loader)
        except Exception as exc:
            self.session._event(
                {
                    "type": "log",
                    "level": "warn",
                    "phase": "test",
                    "text": f"[lightning_test] trainer.test failed: {exc}",
                }
            )
        finally:
            self._lightning_test_inflight = False

    # ------------------------------------------------------------------ baseline restore

    def restore_initial_state(self) -> bool:
        state = getattr(self, "_initial_state", None)
        if not state or self.model is None:
            return False
        try:
            model_state = state.get("model")
            if model_state:
                self.model.load_state_dict(copy.deepcopy(model_state), strict=True)

            opt_state = state.get("optimizer")
            if opt_state and self.optimizer is not None:
                with torch.no_grad():
                    self.optimizer.load_state_dict(copy.deepcopy(opt_state))

            sched_state = state.get("scheduler")
            if sched_state and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
                self.scheduler.load_state_dict(copy.deepcopy(sched_state))

            trainer = getattr(self, "trainer", None)
            if trainer is not None:
                trainer.global_step = int(state.get("global_step", 0) or 0)
                trainer.current_epoch = int(state.get("current_epoch", 0) or 0)
                try:
                    trainer.fit_loop.epoch_progress.current.completed = 0
                except Exception:
                    pass
                try:
                    trainer.fit_loop.epoch_loop._batches_that_stepped = 0  # type: ignore[attr-defined]
                except Exception:
                    pass

            # --- move restored state onto the chosen device ---
            device = self._target_device()
            if device is not None:
                self.model.to(device)
                if self.optimizer is not None:
                    for state in self.optimizer.state.values():
                        for k, v in list(state.items()):
                            if torch.is_tensor(v):
                                state[k] = v.to(device)

        except Exception:
            return False
        return True
    
    def _target_device(self) -> Optional[torch.device]:
        """
        Decide which device restored state should live on.

        Priority:
          1) Lightning's strategy.root_device (what batches are moved to)
          2) This delegate's baseline device (captured at baseline)
          3) Global baseline device (first successful baseline in this process)
          4) Current model parameter device (fallback)
        """
        # 1) Source of truth: Lightning's strategy root device
        try:
            strategy = getattr(self.trainer, "strategy", None)
            dev = getattr(strategy, "root_device", None)
            if isinstance(dev, torch.device):
                return dev
        except Exception:
            pass

        # 2) Delegate baseline
        if self._baseline_device is not None:
            return self._baseline_device

        # 3) Global baseline
        global _GLOBAL_BASELINE_DEVICE
        if _GLOBAL_BASELINE_DEVICE is not None:
            return _GLOBAL_BASELINE_DEVICE

        # 4) Fallback: whatever the model is currently on
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    # ----- delegate hooks used by the session -----------------------------------

    def request_pause(self, reason: str) -> None:
        _ = reason  # Lightning pauses at batch boundaries via the gate.

    def request_resume(self) -> None:
        return

    def request_stop(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            try:
                trainer.should_stop = True
            except Exception:
                pass

    def save_checkpoint(self, reason: str | None = None) -> Optional[str]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        save_dir = os.path.abspath(self.session.cfg.save_dir or os.getcwd())
        os.makedirs(save_dir, exist_ok=True)
        stamp = int(time.time())
        fname = f"{self.session.cfg.run_name}__step{int(self.session.step):08d}__{stamp}.ckpt"
        path = os.path.join(save_dir, fname)
        try:
            trainer.save_checkpoint(path)
            return path
        except Exception:
            return None

    def load_checkpoint(self, path: str, step_hint: Optional[int] = None) -> int:
        if not path:
            raise RuntimeError("load_checkpoint requires a path")

        # Load checkpoint on CPU for safety/portability
        blob = torch.load(path, map_location="cpu")

        state_dict = blob.get("state_dict", {})
        if state_dict and self.model is not None:
            # This will stuff CPU tensors into the model params
            self.model.load_state_dict(state_dict, strict=True)

        opt_states = blob.get("optimizer_states") or []
        if opt_states and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_states[0])
            except Exception:
                pass

        sched_states = blob.get("lr_schedulers") or []
        if sched_states and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(sched_states[0])
            except Exception:
                pass

        # ---- move restored state onto the chosen baseline device ----
        device = self._target_device()
        if device is not None:
            # Move model
            self.model.to(device)

            # Move optimizer state tensors as well
            if self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in list(state.items()):
                        if torch.is_tensor(v):
                            state[k] = v.to(device)

        step = int(blob.get("global_step") or blob.get("step") or step_hint or 0)
        return step

    # ------------------------------------------------------------------ internals

    def _refresh_runtime_handles(self, trainer) -> None:
        self.optimizer = self._first_optimizer(trainer)
        self.scheduler = self._first_scheduler(trainer)

    def _collect_stream_metrics(self, trainer, outputs) -> Dict[str, Any]:
        """
        Normalize whatever metrics Lightning exposed this step so record_train_step
        can stream canonical keys (accuracy, lr, ...).
        """
        sources = []
        if isinstance(outputs, dict):
            sources.append(outputs)
        for attr in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            payload = getattr(trainer, attr, None)
            if payload:
                sources.append(payload)
        return canonicalize_metrics(*sources)

    @staticmethod
    def _first_optimizer(trainer) -> Any:
        for attr in ("strategy",):
            handle = getattr(trainer, attr, None)
            optimizers = getattr(handle, "optimizers", None) if handle is not None else None
            if optimizers:
                return optimizers[0]
        optimizers = getattr(trainer, "optimizers", None)
        if optimizers:
            return optimizers[0]
        return None

    @staticmethod
    def _first_scheduler(trainer) -> Any:
        configs = getattr(trainer, "lr_scheduler_configs", None)
        if configs:
            cfg = configs[0]
            scheduler = getattr(cfg, "scheduler", None)
            if scheduler is not None:
                return scheduler
        strategy = getattr(trainer, "strategy", None)
        configs = getattr(strategy, "lr_scheduler_configs", None) if strategy is not None else None
        if configs:
            cfg = configs[0]
            scheduler = getattr(cfg, "scheduler", None)
            if scheduler is not None:
                return scheduler
        return None

    @property
    def callback(self):
        return self._callback

    # ------------------------------------------------------------------ auto-test

    def _run_auto_test_after_training(self) -> None:
        """
        Auto-test hook called from on_fit_end.

        Semantics:
          * Honors user config: only runs when do_test_after=True.
          * Runs at most once per session (auto_test_count == 0).
          * Emits a 'paused' event with action='auto_test_pause'.
          * The fit driver is responsible for actually waiting on resume/stop.
        """

        # 1) Config says "no auto-test"
        if not self._do_test_after:
            return

        # 2) Already ran once for this session → one-shot behavior
        if self._auto_test_count >= 1:
            return

        # 3) Usual safety guards
        if not getattr(self.session, "test_loader", None):
            return
        if not getattr(self.session, "_running", False):
            return
        if int(getattr(self.session, "step", 0) or 0) <= 0:
            return

        try:
            result = self.session._run_labeled_test_and_ckpt(label="final", source="auto")
        except Exception as exc:
            self.session._event(
                {
                    "type": "log",
                    "level": "warn",
                    "text": f"[auto_test] failed: {exc}",
                }
            )
            return

        # Mark that an auto-test ran and we should now be in a hold
        self._auto_test_triggered = True
        self._post_auto_test_hold = True
        self._auto_test_count += 1

        ckpt_path = result.get("ckpt_path")
        self.session._paused = True
        self.session._pause_ckpt_path = ckpt_path
        self.session._pause_gen = int(getattr(self.session, "_pause_gen", 0) or 0) + 1
        pause_action = {"action": "auto_test_pause", "reason": "auto_test_complete"}
        pause_evt: Dict[str, Any] = {
            "type": "paused",
            "run_id": self.session.cfg.run_name,
            "step": int(getattr(self.session, "step", 0)),
            "request": pause_action,
        }
        if ckpt_path:
            pause_evt["ckpt"] = ckpt_path
        self.session._event(
            {
                "type": "log",
                "level": "info",
                "text": (
                    f"[auto_test] triggered; step={int(getattr(self.session, 'step', 0))}, "
                    f"paused={getattr(self.session, '_paused', None)}, "
                    f"running={getattr(self.session, '_running', None)}"
                ),
            }
        )
        self.session._event(pause_evt)



def attach_lightning(
    *,
    trainer: Any,
    model: Any,
    datamodule: Any = None,
    project: str,
    run_name: str,
    save_dir: str = "./checkpoints",
    auto_connect: bool = True,
    channel: Optional[SocketChannel] = None,
    channel_kwargs: Optional[Dict[str, Any]] = None,
    log_every_n_steps: int = 1,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    loss_fn: Optional[Any] = None,
    model_factory: Optional[Callable[[], Any]] = None,
    embedding_hook: Optional[Any] = None,
    do_test_after: bool = False,
) -> LightningFrameworkDelegate:
    """
    High level helper: attach OnTheFly to a Lightning Trainer.

    Usage:
        model = LitModel()
        trainer = L.Trainer(max_epochs=10, callbacks=[])
        attach_lightning(trainer=trainer, model=model, project="demo", run_name="baseline")
        trainer.fit(model, datamodule=...)
    """

    if auto_connect:
        if channel is None:
            channel = SocketChannel(**(channel_kwargs or {}))
        set_channel(channel)

    session = OnTheFlyExternalSession(project=project, run_name=run_name, save_dir=save_dir)
    session.configure_datamodule(datamodule)
    session.configure_dataloaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    if embedding_hook is not None:
        session.configure_embedding_hook(embedding_hook)
    if model_factory is not None:
        session.configure_model_factory(model_factory)
    resolved_loss = loss_fn or getattr(model, "loss_fn", None) or getattr(model, "criterion", None)
    if resolved_loss is None:
        raise ValueError("attach_lightning requires a loss_fn (pass via loss_fn=...).")
    session.configure_loss_fn(resolved_loss)

    delegate = LightningFrameworkDelegate(
        session=session,
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        log_every_n_steps=log_every_n_steps,
        do_test_after=do_test_after,
    )
    delegate.attach(model=model, trainer=trainer, datamodule=datamodule)
    session.attach_framework(delegate)
    return delegate

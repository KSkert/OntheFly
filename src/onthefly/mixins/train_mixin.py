from __future__ import annotations
import os, time, math
from typing import Dict, Any, Optional
import torch

from ..device_utils import _sync_device_by_name, _noop_ctx
from ..scale import _SafeScaler
from ..metrics_utils import _to_scalar_loss, _grad_norm
from ..runtime_metrics import (
    batch_accuracy as _batch_accuracy,
    estimate_batch_size as _estimate_batch_size,
    weight_norm as _weight_norm,
)
from ..control import serve_commands


def _metric_float(val):
    if val is None:
        return None
    try:
        if torch.is_tensor(val):
            if val.numel() == 0:
                return None
            data = val.detach()
            if data.ndim > 0:
                data = data.reshape(-1)
            out = float(data.mean().item())
        else:
            out = float(val)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def lr_from_optimizer(opt):
    if opt is None:
        return None
    vals = []
    for group in getattr(opt, "param_groups", []) or []:
        try:
            v = float(group.get("lr", float("nan")))
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


class TrainMixin:
    """
    The training/validation/test loops + step defaults and state exposure.
    Keeps the outer 'OnTheFlySession' thin while preserving method names.
    """


    def _safe_load_state(self, blob: Dict[str, Any]):
        if "model" in blob:
            self.model.load_state_dict(blob["model"])
        if "optimizer" in blob and self.optimizer is not None:
            try: self.optimizer.load_state_dict(blob["optimizer"])
            except Exception: pass
        if "scheduler" in blob and self.scheduler is not None:
            try: self.scheduler.load_state_dict(blob["scheduler"])
            except Exception: pass
        if "scaler" in blob and hasattr(self, "scaler") and self.scaler is not None:
            try: self.scaler.load_state_dict(blob["scaler"])
            except Exception: pass
        if "epoch" in blob:  # optional
            try: self.epoch = int(blob["epoch"])
            except Exception: pass
        if "last_val_loss" in blob:
            try: self._last_val_loss = float(blob["last_val_loss"])
            except Exception: pass

    def _load_checkpoint_into_state(self, path: str) -> int:
        """
        Load a Seamless checkpoint. Prefer the safe weights-only path first
        (PyTorch >=2.6 default). If that fails or isn't supported, fall back
        to a full, trusted load. Returns the resume step (default 0).
        """
        import inspect
        blob = None

        # Does this torch.load accept weights_only?
        _supports_weights_only = 'weights_only' in inspect.signature(torch.load).parameters

        try:
            if _supports_weights_only:
                # Try safe path first (may raise WeightsOnly errors on older objects)
                blob = torch.load(path, map_location=self.device, weights_only=True)
            else:
                # Older torch: no weights_only kw → just try normal load
                blob = torch.load(path, map_location=self.device)
        except Exception as e_safe:
            # Fall back to a full unpickle. Only do this for your own, trusted files.
            if _supports_weights_only:
                blob = torch.load(path, map_location=self.device, weights_only=False)
            else:
                blob = torch.load(path, map_location=self.device)

        # Accept both raw dicts and SnapshotManager payloads
        state = blob.get("state", blob) if isinstance(blob, dict) else blob
        self._safe_load_state(state)

        # Step may be stored under different keys; normalize
        return int(state.get("step", state.get("global_step", 0)))


    def _sync_device(self):
        _sync_device_by_name(self.device)

    def _state(self, train=True) -> Dict[str, Any]:
        return {
            "model": self.model.train() if train else self.model.eval(),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "device": self.device,
            "scaler": self.scaler,
            "loss_fn": self.loss_fn,
            "step": self.step,
            "grad_clip_norm": self.cfg.grad_clip_norm,
            "autocast": self.autocast() if train else _noop_ctx(),
            "train_loader": self.train_loader,
        }

    def _default_training_step(self, batch, state):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        loss = _to_scalar_loss(loss, device=self.device)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        grad_norm = _grad_norm(self.model)
        acc = _batch_accuracy(logits, y)
        lr = lr_from_optimizer(self.optimizer)
        payload = {"loss": loss.detach(), "grad_norm": grad_norm, "lr": lr}
        
        if acc is not None:
            payload["accuracy"] = acc
        return payload

    def _default_validation_step(self, batch, state):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        return {"val_loss": _to_scalar_loss(loss, device=self.device).detach()}

    def _run_validation(self) -> float:
        if self.val_loader is None:
            raise RuntimeError("val_loader is not configured; cannot run validation.")
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                out = self._validation_step_fn(batch, self._state(train=False))
                losses.append(float(out["val_loss"]))
        self.model.train()
        avg = sum(losses)/max(1, len(losses))

        return avg

    def _run_test(self) -> float:
        if self.test_loader is None:
            return float("nan")
        self.model.eval()
        losses = []
        with torch.no_grad():
            self._event({"type":"log","level":"info","phase":"test","text":"[test] starting evaluation pass"})
            tstep = 0
            for batch in self.test_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                loss_t = _to_scalar_loss(self.loss_fn(logits, y), device=self.device).detach()
                l = float(loss_t)
                tstep += 1
                self._emit({"type":"testStep","step":tstep,"loss":l})
                self._event({"type":"log","level":"info","phase":"test","text":f"step {tstep}: test_loss = {l:.6f}"})
                losses.append(l)
        avg = (sum(losses) / max(1, len(losses)))
        self._event({"type":"log","level":"info","phase":"test","text":f"test_avg_loss = {avg:.6f}"})
        self._event({"type":"log","level":"info","phase":"test","text":"[test] evaluation pass complete"})
        self.model.train()
        return avg

    def _runtime_metric_snapshot(self, metrics: Dict[str, Any], batch, step_duration: float, scheduler=None) -> Dict[str, Optional[float]]:
        snapshot: Dict[str, Optional[float]] = {}

        accuracy = _metric_float(metrics.get("accuracy"))
        snapshot["accuracy"] = accuracy

        grad_norm = _metric_float(metrics.get("grad_norm"))
        if grad_norm is None:
            try:
                grad_norm = float(_grad_norm(self.model))
            except Exception:
                grad_norm = None
        snapshot["grad_norm"] = grad_norm

        lr = _metric_float(metrics.get("lr"))
        if lr is None:
            lr = lr_from_optimizer(self.optimizer)
        if lr is None and scheduler is not None:
            try:
                if hasattr(scheduler, "get_last_lr"):
                    last_lr_seq = scheduler.get_last_lr()
                    if isinstance(last_lr_seq, (list, tuple)):
                        last_vals = [float(v) for v in last_lr_seq if math.isfinite(float(v))]
                        if last_vals:
                            lr = float(sum(last_vals) / len(last_vals))
                    elif last_lr_seq is not None:
                        val = float(last_lr_seq)
                        if math.isfinite(val):
                            lr = val
            except Exception:
                pass
        prev_lr = getattr(self, "_last_runtime_lr", None)
        if lr is None and prev_lr is not None:
            lr = float(prev_lr)
        if lr is not None:
            self._last_runtime_lr = float(lr)
        snapshot["lr"] = lr

        weight = _metric_float(metrics.get("weight_norm"))
        if weight is None:
            weight = _weight_norm(self.model)
        snapshot["weight_norm"] = weight

        tracker = getattr(self, "_activation_tracker", None)
        snapshot["activation_zero_frac"] = tracker.pop_recent() if tracker is not None else None

        throughput = _metric_float(metrics.get("throughput"))
        if throughput is None:
            batch_size = _estimate_batch_size(batch)
            if batch_size and step_duration > 0:
                throughput = float(batch_size) / max(step_duration, 1e-9)
        snapshot["throughput"] = throughput

        monitor = getattr(self, "_device_monitor", None)
        mem_vram = None
        gpu_util = None
        if monitor is not None:
            try:
                mem_vram, gpu_util = monitor.snapshot()
            except Exception:
                mem_vram, gpu_util = None, None
        snapshot["mem_vram"] = mem_vram
        snapshot["gpu_util"] = gpu_util

        return snapshot

    def _maybe_handle_commands(self):
        serve_commands(self._bus, self._router, poll_sec=0.0)

    def _validation_frequency(self) -> int:
        freq = getattr(self, "_val_every_n_epochs", None)
        try:
            freq = int(freq) if freq is not None else 0
        except Exception:
            freq = 0
        return max(0, freq)

    def _validation_enabled(self) -> bool:
        return self.val_loader is not None and self._validation_frequency() > 0

    def _validation_disabled_reason(self) -> Optional[str]:
        freq = self._validation_frequency()
        if freq <= 0:
            return "val_every_n_epochs <= 0"
        if self.val_loader is None:
            return "no val_loader provided"
        return None

    def _should_run_validation_epoch(self) -> bool:
        if not self._validation_enabled():
            return False
        freq = self._validation_frequency()
        epoch_idx = int(getattr(self, "epoch", 0) or 0)
        return ((epoch_idx + 1) % freq) == 0

    # ---------------- public API ----------------
    def training_step(self, fn):
        self._training_step_fn = fn
        return fn

    def validation_step(self, fn):
        self._validation_step_fn = fn
        return fn

    def embedding_hook(self, fn):
        self._embedding_hook_fn = fn
        return fn
    

    def serve(self, max_steps: Optional[int] = None, max_epochs: Optional[int] = None, do_test_after: bool = False):
        hit_criterion = False

        # --- 1) Extension-assisted resume (imported bundles only) ---
        resume_run    = os.getenv("ONTHEFLY_RESUME_RUN_ID") or None
        init_ckpt     = os.getenv("ONTHEFLY_INIT_CKPT") or None
        init_step_env = os.getenv("ONTHEFLY_INIT_STEP") or None
        is_resume = bool(resume_run or init_ckpt)

        if resume_run:
            prev = getattr(self.cfg, "run_name", None)
            self.cfg.run_name = str(resume_run)

        if init_ckpt:
            if os.path.exists(init_ckpt):
                try:
                    ckpt_step = int(self._load_checkpoint_into_state(init_ckpt))
                    if init_step_env is not None:
                        try:
                            self.step = int(init_step_env)
                        except Exception:
                            self.step = ckpt_step
                    else:
                        self.step = ckpt_step

                    self._event({"type":"log","level":"info",
                                "text": f"[resume] restored from {os.path.basename(init_ckpt)}  step={self.step}  epoch={getattr(self,'epoch',0)}"})
                    self._event({"type":"checkpoint_loaded","path": init_ckpt, "step": int(self.step)})

                except Exception as e:
                    self._event({"type":"log","level":"error", "text": f"[resume] failed: {e}"})

        start_step = int(getattr(self, "step", 0) or 0)
        start_epoch = int(getattr(self, "epoch", 0) or 0)

        steps_before = int(start_step)
        epochs_before = int(start_epoch)

        target_step  = None if max_steps  is None else start_step  + max_steps
        target_epoch = None if max_epochs is None else start_epoch + max_epochs

        self._last_emitted_epoch = None

        # --- 2) Session header & run identity (unchanged) ---
        self._event({"type":"session_started","project": self.cfg.project, "run_name": self.cfg.run_name})

        self._event({"type":"log","level":"info","text": f"model session_id={self.session_id}"})
        self._event({"type":"log","level":"info","text": "training"})
        self._event({"type":"log","level":"info","text": self.cfg.run_name or "baseline"})

        self._maybe_handle_commands()

        self._emit_new_run(self.cfg.run_name, parents=[], meta={"display_name": self.cfg.run_name})
        val_reason = self._validation_disabled_reason()
        if val_reason:
            freq = self._validation_frequency()
            level = "warn" if (val_reason == "no val_loader provided" and freq > 0) else "info"
            self._event({"type": "log", "level": level, "text": f"validation disabled ({val_reason})"})
        try:
            while self._running and (target_step is None or self.step < target_step) and (target_epoch is None or self.epoch < target_epoch):
                self._maybe_handle_commands()

                if self._paused:
                    time.sleep(0.05)
                    continue

                self._sampler_set_epoch(self.epoch)

                self._event({"type": "log", "level": "info", "text": f"epoch {self.epoch}"})

                k = int(getattr(self, "_epoch_batch_idx", 0) or 0)
                it = iter(self.train_loader)
                if k > 0:
                    for _ in range(k):
                        try:
                            next(it)
                        except StopIteration:
                            break

                for batch in it:
                    self._maybe_handle_commands()
                    if not self._running:
                        break

                    if self._paused:
                        try:
                            self._event({"type": "paused", "run_id": self.cfg.run_name, "step": self.step})
                        except Exception:
                            pass
                    while self._paused and self._running:
                        self._maybe_handle_commands()
                        time.sleep(0.05)
                    if not self._running:
                        break

                    step_start = time.perf_counter()
                    metrics = self._training_step_fn(batch, self._state())
                    step_elapsed = max(time.perf_counter() - step_start, 1e-9)
                    self._sync_device()

                    loss = float(metrics.get("loss", float("inf")))
                    if not math.isfinite(loss):
                        self._event({"type": "log", "level": "error",
                                    "text": f"Non-finite loss at step {self.step}. "
                                            f"Restoring last checkpoint if available and skipping step."})
                        if self._ckpts:
                            try:
                                self.step = self._load_checkpoint_into_state(self._ckpts[-1])
                            except Exception:
                                pass
                        self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))
                        continue

                    runtime_metrics = self._runtime_metric_snapshot(metrics, batch, step_elapsed, self.scheduler)

                    self.step += 1
                    self._epoch_batch_idx = int(getattr(self, "_epoch_batch_idx", 0) or 0) + 1
                    current_epoch = int(getattr(self, "epoch", 0) or 0)

                    payload = {
                        "type":     "trainStep",
                        "step":     self.step,
                        "loss":     loss,
                        "val_loss": (float(self._last_val_loss) if self._last_val_loss is not None else None),
                    }

                    # Only include epoch when it changes (edge-triggered)
                    last_emitted = getattr(self, "_last_emitted_epoch", None)
                    if last_emitted is None or current_epoch != last_emitted:
                        payload["epoch"] = current_epoch
                        self._last_emitted_epoch = current_epoch
                        self._event({
                            "type":  "log",
                            "level": "info",
                            "phase": "train",
                            "text":  f"[epoch] now at epoch {current_epoch} (step {self.step})",
                        })

                    payload.update(runtime_metrics)
                    self._emit(payload)

                    last_v = (f"{self._last_val_loss:.6f}" if self._last_val_loss is not None else "None")
                    self._event({"type": "log", "level": "info",
                                "text": f"step {self.step}: train_loss = {loss:.6f}, val_loss = {last_v}"})

                    if self.scheduler:
                        self.scheduler.step()

                    if max_steps and self.step >= max_steps:
                        break

                if not self._running:
                    break

                while self._paused and self._running:
                    self._maybe_handle_commands()
                    time.sleep(0.05)
                if not self._running:
                    break

                epoch_val_loss = None
                if self._should_run_validation_epoch():
                    vloss = self._run_validation()
                    epoch_val_loss = float(vloss)
                    self._last_val_loss = epoch_val_loss
                    self._event({"type": "log", "level": "info", "text": f"epoch {self.epoch} val_loss = {vloss:.6f}"})

                self._event({"type": "epoch_end", "epoch": self.epoch, "val_loss": epoch_val_loss})

                self._epoch_batch_idx = 0
                self.epoch += 1
            
            progressed = (int(self.step) > steps_before) or (int(self.epoch) > epochs_before)

            if progressed:
                if target_epoch is not None and self.epoch >= target_epoch:
                    hit_criterion = True
                if target_step is not None and self.step >= target_step:
                    hit_criterion = True

            if (
                progressed
                and not is_resume
                and hit_criterion
                and do_test_after
                and self.test_loader is not None
                and self._running
            ):
                _ = self._run_labeled_test_and_ckpt(label="final", source="auto")

        finally:
            self._bus.stop()
            tracker = getattr(self, "_activation_tracker", None)
            if tracker is not None:
                try:
                    tracker.close()
                except Exception:
                    pass
            monitor = getattr(self, "_device_monitor", None)
            if monitor is not None:
                try:
                    monitor.close()
                except Exception:
                    pass


    def _run_labeled_test_and_ckpt(self, label: str = "final", source: str = "manual") -> Dict[str, Any]:
        if self.test_loader is None:
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
        try:
            self._event({
                "type": "log",
                "level": "info",
                "phase": "test",
                "text": f"[test] requested evaluation for label '{label}' (source={source})",
            })

            avg = float(self._run_test())

            # force a checkpoint even though we're not paused
            ckpt_path = self._save_ring_checkpoint(force=True)

            if ckpt_path is not None:
                import os
                self._event({
                    "type": "log",
                    "level": "info",
                    "phase": "test",
                    "text": f"[test] checkpoint saved for label '{label}': {os.path.basename(ckpt_path)}",
                })
            else:
                # Should not happen with force=True
                self._event({
                    "type": "log",
                    "level": "warn",
                    "phase": "test",
                    "text": f"[test] requested checkpoint for label '{label}' but save returned None",
                })

            result: Dict[str, Any] = {
                "status": "ok",
                "avg_loss": avg,
                "run_id": self.cfg.run_name,
                "session_id": getattr(self, "session_id", None),
                "step": int(self.step),
                "label": label,
                "ckpt_path": ckpt_path,
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

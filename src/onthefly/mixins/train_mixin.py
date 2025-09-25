from __future__ import annotations
import os, time, math, threading, random
from typing import Dict, Any, Optional, Callable, List
import torch
from torch.utils.data import DataLoader

from ..device_utils import _sync_device_by_name, _noop_ctx
from ..scale import _SafeScaler
from ..metrics_utils import _to_scalar_loss, _grad_norm
from ..control import ControlBus, CommandRouter, serve_commands
from ..snapshots import SnapshotManager
from ..autofork import AutoForkRules, AutoForkEngine

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
            # Let the UI know we’re retrying with the unsafe path (trusted checkpoints only)
            try:
                self._event({"type":"log","level":"warn",
                            "text": f"[resume.debug] weights_only/normal load failed, retrying full unpickle for trusted ckpt: {e_safe}"})
            except Exception:
                pass
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
        return {"loss": loss.detach(), "grad_norm": grad_norm}

    def _default_validation_step(self, batch, state):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        return {"val_loss": _to_scalar_loss(loss, device=self.device).detach()}

    def _run_validation(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                out = self._validation_step_fn(batch, self._state(train=False))
                losses.append(float(out["val_loss"]))
        self.model.train()
        avg = sum(losses)/max(1, len(losses))

        try:
            if self.cfg.run_name in self._children_registered:
                self._autofork_engine.report_child_result(self.cfg.run_name, {"loss": float(avg)})
            else:
                self._autofork_engine.observe_eval("val", {"loss": float(avg)})
        except Exception:
            pass
        return avg

    def _run_test(self) -> float:
        if self.test_loader is None:
            return float("nan")
        self.model.eval()
        losses = []
        with torch.no_grad():
            tstep = 0
            for batch in self.test_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                loss_t = _to_scalar_loss(self.loss_fn(logits, y), device=self.device).detach()
                l = float(loss_t)
                tstep += 1
                self._emit({"type":"testStep","step":tstep,"loss":l})
                self._event({"type":"log","level":"info","text":f"step {tstep}: test_loss = {l:.6f}"})
                losses.append(l)
        avg = (sum(losses) / max(1, len(losses)))
        self._event({"type":"log","level":"info","text":f"test_avg_loss = {avg:.6f}"})
        try:
            self._autofork_engine.observe_eval("test", {"loss": float(avg)})
        except Exception:
            pass
        self.model.train()
        return avg

    def _maybe_handle_commands(self):
        serve_commands(self._bus, self._router, poll_sec=0.0)

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
        # --- 1) Extension-assisted resume (imported bundles only) ---
        resume_run    = os.getenv("ONTHEFLY_RESUME_RUN_ID") or None
        init_ckpt     = os.getenv("ONTHEFLY_INIT_CKPT") or None
        init_step_env = os.getenv("ONTHEFLY_INIT_STEP") or None

        # DEBUG: echo what we saw from the environment
        self._event({"type":"log","level":"info",
                    "text": f"[resume.debug] env: RESUME_RUN_ID={resume_run!r} INIT_CKPT={init_ckpt!r} INIT_STEP={init_step_env!r}"})

        # If the UI asked us to resume under a specific run id, adopt it *before* emitting anything.
        if resume_run:
            prev = getattr(self.cfg, "run_name", None)
            self.cfg.run_name = str(resume_run)
            self._event({"type":"log","level":"info",
                        "text": f"[resume.debug] adopting run_name from env: {prev!r} → {self.cfg.run_name!r}"})

        # Try to restore from the provided checkpoint (if any).
        if init_ckpt:
            if os.path.exists(init_ckpt):
                try:
                    ckpt_step = int(self._load_checkpoint_into_state(init_ckpt))
                    # Prefer explicit ONTHEFLY_INIT_STEP when provided; otherwise trust ckpt contents.
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

                    # Extra DEBUG: confirm effective values
                    self._event({"type":"log","level":"info",
                                "text": f"[resume.debug] post-load: step={self.step} epoch={getattr(self,'epoch',0)} last_val={getattr(self,'_last_val_loss',None)}"})
                except Exception as e:
                    self._event({"type":"log","level":"error", "text": f"[resume] failed: {e}"})
            else:
                # explicit debug if we were told a path that doesn’t exist
                self._event({"type":"log","level":"warn",
                            "text": f"[resume.debug] INIT_CKPT provided but missing on disk: {init_ckpt}"})
        else:
            self._event({"type":"log","level":"info",
                        "text": "[resume.debug] no INIT_CKPT provided → cold start"})

        # --- 2) Session header & run identity (unchanged) ---
        self._event({"type":"session_started","project": self.cfg.project, "run_name": self.cfg.run_name})

        _main_wall_start = time.perf_counter()
        _total_compute_s = 0.0

        self._event({"type":"log","level":"info","text": f"model session_id={self.session_id}"})
        self._event({"type":"log","level":"info","text": "training"})
        self._event({"type":"log","level":"info","text": self.cfg.run_name or "baseline"})

        # open command loop
        self._maybe_handle_commands()

        # finally emit the designated or default run
        self._emit_new_run(self.cfg.run_name, parents=[], meta={"display_name": self.cfg.run_name})
        try:
            while self._running and (max_steps is None or self.step < max_steps) and (max_epochs is None or self.epoch < max_epochs):
                self._maybe_handle_commands()

                # Outer-loop idle while paused (no epoch finalize reached yet)
                if self._paused:
                    time.sleep(0.05)
                    continue

                self._event({"type": "log", "level": "info", "text": f"epoch {self.epoch}"})
                self._drain_feature_queue()

                _steps_per_epoch = len(self.train_loader)
                _slice_size = 50
                _local_step = 0
                _slice_accum_compute = 0.0

                for batch in self.train_loader:
                    self._maybe_handle_commands()
                    if not self._running:
                        break

                    # ---------------- GATE A: pause BEFORE consuming the next batch ----------------
                    if self._paused:
                        # notify UI of the pause point (step is last-completed global step)
                        try:
                            self._event({"type": "paused", "run_id": self.cfg.run_name, "step": self.step})
                        except Exception:
                            pass
                    while self._paused and self._running:
                        self._maybe_handle_commands()
                        time.sleep(0.05)
                    if not self._running:
                        break
                    # -------------------------------------------------------------------------------

                    _local_step += 1
                    _t0 = time.perf_counter()
                    metrics = self._training_step_fn(batch, self._state())  # compute
                    self._sync_device()
                    _dt = time.perf_counter() - _t0

                    _total_compute_s += _dt
                    _slice_accum_compute += _dt

                    loss = float(metrics.get("loss", float("inf")))
                    if not math.isfinite(loss):
                        self._event({"type": "log", "level": "error",
                                    "text": f"Non-finite loss at step {self.step}. Restoring last checkpoint if available and skipping step."})
                        if self._ckpts:
                            try:
                                self.step = self._load_checkpoint_into_state(self._ckpts[-1])
                            except Exception:
                                pass
                        self.scaler = _SafeScaler(torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and "cuda" in self.device)))
                        continue

                    grad_norm = float(metrics.get("grad_norm", 0.0))
                    self.step += 1

                    if (_local_step % _slice_size) == 0:
                        if _local_step == _slice_size:
                            _slice_accum_compute = 0.0
                        else:
                            start_b = _local_step - _slice_size
                            end_b   = _local_step
                            avg = _slice_accum_compute / _slice_size
                            print(f"[slice] epoch {self.epoch:02d} steps {start_b}-{end_b}  compute_wall={_slice_accum_compute:.3f}s  avg={avg:.4f}s/step")
                            _slice_accum_compute = 0.0

                    self._emit({
                        "type":     "trainStep",
                        "step":     self.step,
                        "loss":     loss,
                        "val_loss": (float(self._last_val_loss) if self._last_val_loss is not None else None),
                    })

                    last_v = (f"{self._last_val_loss:.6f}" if self._last_val_loss is not None else "None")
                    self._event({"type": "log", "level": "info",
                                "text": f"step {self.step}: train_loss = {loss:.6f}, val_loss = {last_v}"})

                    if self.scheduler: self.scheduler.step()
                    if self.step % self.cfg.ckpt_every_steps == 0:
                        self._save_ring_checkpoint()

                    self._maybe_feed_autofork(loss, grad_norm)

                    if max_steps and self.step >= max_steps:
                        break

                if not self._running:
                    break

                # ---------------- GATE B: pause at epoch boundary BEFORE finalize ----------------
                while self._paused and self._running:
                    # wait here; do not run validation or increment epoch while paused
                    self._maybe_handle_commands()
                    time.sleep(0.05)
                if not self._running:
                    break
                # -------------------------------------------------------------------------------

                vloss = self._run_validation()
                self._last_val_loss = float(vloss)
                self._event({"type": "log", "level": "info", "text": f"epoch {self.epoch} val_loss = {vloss:.6f}"})

                self._event({"type": "epoch_end", "epoch": self.epoch, "val_loss": vloss})
                self._maybe_suggest_or_do_merge()
                self.epoch += 1

            _train_wall_s = time.perf_counter() - _main_wall_start
            print(f"[timing] total_train_compute={_total_compute_s:.2f}s  total_train_wall={_train_wall_s:.2f}s  "
                f"(~{_total_compute_s/max(1,self.epoch):.2f}s compute/epoch)")

            if do_test_after and self.test_loader is not None:
                self._event({"type": "log", "level": "info", "text": "testing"})
                self._run_test()

            self._event({"type": "training_finished", "code": 0})
        finally:
            self._bus.stop()
            try:
                self._feature_worker.stop()
            except Exception:
                pass

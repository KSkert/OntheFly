# Session Lifecycle & Training Flow

OnTheFly exposes two concrete session types inside `src/onthefly/session/`:

- `native.py::OnTheFlySession` drives the entire training loop inside Python. It is what the public `Trainer.fit(...)` instantiates.
- `external.py::OnTheFlyExternalSession` keeps the same control-plane surface but lets another framework (Lightning, Accelerate, etc.) own the batch loop via a `FrameworkDelegate`.

Both classes inherit from `session/base.py::OnTheFlySessionBase`, which supplies lifecycle hooks (`before_training`, `after_training`, `serve`), event emission, command handling, checkpoint helpers, and determinism guards. The base class also wires the `ControlBus`, `CommandRouter`, `PauseGate`, and `ConsoleAction` so command handling is uniform no matter who drives the batches.

## Bootstrapping a Native Session

`OnTheFlySession.__init__` receives everything the training run needs: PyTorch model/optimizer/scheduler objects, dataloaders, loss function, gradient clipping/AMP options, and determinism flags such as `data_order_policy` and `enforce_sampler_state`. Key steps during construction:

1. `SessionConfig` is materialized so immutable run metadata (project/run name, checkpoint directory, device preference, AMP flag, checkpoint retention) is centralized (`src/onthefly/config.py`).
2. A reusable model factory is derived via `factory._build_model_factory`, allowing mixins to spawn clean models for forks, merges, and analysis jobs.
3. `OnTheFlySessionBase.__init__` is called with a delegate factory that produces an `control.OnTheFlyTrainerDelegate` bound to this session's `PauseGate`. That delegate services pause/resume/fork/merge/test requests from the dashboard.
4. Training context is prepared:
   - `_init__identity_and_device` resolves device placement via `device_utils._resolve_device`.
   - `_init__loss_and_scaler` wraps the user loss with `_SafeScaler` from `scale.py` for AMP-safe backward calls.
   - `_init__train_context` installs default training/validation/test step callables from `mixins/train_mixin.py`.
   - `_init_determinism` seeds RNGs, optionally wraps samplers with `sampler_utils.EpochSeededRandomSampler`, and remembers whether sampler state needs to be maintained across pauses.
5. A background thread captures a "baseline" checkpoint (`_capture_reset_snapshot`) so the `resetSession` command can roll the session back even before any user checkpoints exist.

## Mixins and Responsibilities

`OnTheFlySessionBase` composes several mixins that keep responsibilities isolated:

- `mixins.events_mixin.EventsMixin` orders outbound metrics/log/events and provides `_event(...)`.
- `mixins.checkpoint_mixin.CheckpointMixin` materializes payloads, writes ring checkpoints, and exposes `_latest_ckpt_for_run`.
- `mixins.commands_mixin.CommandsMixin` registers `CommandRouter` handlers for pause/resume, forks, merges, subset exports, auto-tests, and report generation commands.
- `mixins.feature_mixin.FeatureMixin` powers the per-sample/embedding scans used by subset selection and the explorer UI.
- `mixins.run_management_mixin.RunManagementMixin` keeps run naming, fork/merge bookkeeping, loader rebinding, and subset activation logic together.
- `mixins.train_mixin.TrainMixin` (mixed into `OnTheFlySession`) implements the actual training/validation/test loops, gradient scaling, metric snapshots, and validation scheduling.

Because mixins live directly on the session instance, they can reuse attributes such as `self.model`, `self.optimizer`, `self.train_loader`, and `self.cfg` without creating fragile circular dependencies.

## Lifecycle Hooks & Command Loop

- `before_training` emits a `session_started` event, logs the session identifier, and calls `_emit_new_run` so the dashboard has an initial run node.
- `start_command_loop` begins a background thread that continuously invokes `control.serve_commands(...)`, draining the JSON command queue while training runs.
- `tick(idle_sleep=0.05)` gives external runtimes a way to drive the same pause/resume semantics by periodically servicing commands and idling while `_paused` is `True`.
- `after_training(status="completed")` makes sure `trainingFinished` is emitted exactly once, then stops the command loop and `ControlBus`.

Native sessions call `serve(...)` (defined on `OnTheFlySessionBase`) to wrap the lifecycle: `before_training`, command loop, training loop via `TrainMixin`, optional auto-test, and `after_training`. External sessions instead rely on delegates calling `before_training` then `tick(...)` inside their framework callbacks.

## Training/Validation/Test Loops

`mixins/train_mixin.py` contains composable pieces that keep the native loop readable:

- `_state(train=True)` returns a dictionary of the live training objects (model/optimizer/scheduler/device/grad scaler/loader) so overridable hooks receive a uniform view.
- `_default_training_step` zeroes gradients, runs the model under `torch.cuda.amp.autocast` when enabled, clips gradients, steps the scaler/optimizer, and returns metric payloads (`loss`, `grad_norm`, `accuracy`, `lr`).
- `_default_validation_step` mirrors the training step without gradient updates.
- `_run_validation` executes `val_loader` iterators and averages the resulting losses. `_validation_frequency`/`_validation_enabled` consult `val_every_n_epochs` to decide when `TrainMixin` should trigger validation inside the epoch loop.
- `_run_test` (in `OnTheFlySessionBase`) is shared by native and external sessions. It resolves the device with `device_utils._resolve_device`, marshals batches to that device, supports batch-aware losses via `_call_batch_loss`, and emits `testStep` + aggregate log events. When labeled tests are requested, `_run_labeled_test_and_ckpt` forces a checkpoint save so results are reproducible.
- `_runtime_metric_snapshot` normalizes in-flight metrics (`accuracy`, `grad_norm`, `lr`, `weight_norm`, `activation_zero_frac`, `throughput`, GPU mem/utilization) into the canonical form consumed by the dashboard. It fuses user-reported metrics with automatic measurements computed from batches and the device monitor.

Training itself happens in `_training_loop` (in `TrainMixin`), which repeatedly:

1. Calls `_maybe_handle_commands` before and after each batch so pause/fork/test commands take effect quickly.
2. Measures step duration for throughput estimates.
3. Emits `trainStep` events with the metric snapshots above plus step/epoch counters.
4. Runs validation on the configured cadence, logging results and updating `self._last_val_loss`.

## Deterministic Dataloader Policies

`OnTheFlySessionBase._init_determinism` governs reproducibility:

- Seeds Python, NumPy, PyTorch CPU, and CUDA RNGs up front.
- Records whether sampler state should be enforced (`enforce_sampler_state=True` wraps samplers so `state_dict()`/`load_state_dict(...)` survive pauses).
- When `data_order_policy` is `epoch_reseed` or `fixed_order`, installs an `EpochSeededRandomSampler` that reseeds itself every epoch but remembers cursor position so mid-epoch pauses resume at the right spot.
- When `deterministic_pauses=True`, `_install_determinism_guards` clones the loaders so batch cursors, worker seeds, and sampler order remain stable whenever the dashboard pauses training. Otherwise, `_epoch_batch_idx` simply resets between epochs.
- `_sampler_set_epoch` forwards epoch counters into PyTorch `DistributedSampler` when present so DDP jobs stay in sync.

These controls ensure that forks, merges, tests, and per-sample scans always replay the same mini-batch ordering unless the user explicitly opts out.

## Reset & Resume

Native sessions offer first-class reset support:

- `_capture_reset_snapshot` runs in a daemon thread that writes a checkpoint with model/optimizer/scheduler/scaler state plus counters to a temporary directory as soon as training begins.
- `_restore_from_reset_snapshot` reloads that blob when the dashboard issues `resetSession`, including RNG states and grad scaler configuration.
- `_reset_session_state` coordinates run-name selection, checkpoint cleanup, and run metadata updates by delegating to `RunManagementMixin`.

Because these operations piggy-back on the same checkpoint serialization as regular ring checkpoints, resets are deterministic and amenable to the same disk/storage policies users already configure via `SessionConfig`.

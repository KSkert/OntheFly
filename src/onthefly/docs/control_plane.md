# Control Plane & Dashboard Wiring

All dashboard interactions flow through `src/onthefly/control.py` and `src/onthefly/dashboard_channel.py`. The control plane is responsible for transporting metrics/commands, gating pause/fork/test requests, and bridging into external trainers when Lightning or other frameworks drive the batch loop.

## Transport & Serialization

- **`ControlBus`** opens a background reader thread that consumes JSON lines from the active channel (stdio by default or `SocketChannel` when a dashboard is available). Messages with a `"cmd"` field are enqueued in a bounded queue so training threads never block on I/O.
- **`CommandRouter`** lets mixins register handlers via `router.on("pause")(handler)`. `serve_commands(...)` repeatedly drains `ControlBus`, invokes the appropriate handler, and replies via `send_reply`/`send_error`.
- **`_AsyncStdoutWriter`** batches outbound replies, errors, metrics, and events. High-priority replies bypass the coalescing delay; `trainStep` events take an even faster path that writes directly to the channel before falling back to the writer on serialization issues.
- **`dashboard_channel.SocketChannel`** replaces stdio with a TCP connection to the VS Code dashboard. It reconnects automatically, buffers events while the dashboard is closed, and owns a separate reader thread that feeds command lines into the same `ControlBus` queue. The channel is installed globally through `control.set_channel(...)` so every emitter writes to the same transport.

## Pause Gate & Console Actions

The pause lifecycle is coordinated by the trio of `PauseGate`, `ConsoleAction`, and `ControlDelegate`:

- **`PauseGate`** exposes `request(...)`, `should_block(...)`, and `wait_until_resumed()` so batch loops (native or external) can stop fetching data whenever a dashboard action needs an exclusive window.
- **`ConsoleAction`** wraps the delegate with `_enter_pause_window(...)`/`_exit_pause_window(...)`. It emits `paused`/`resumed` events, triggers ring checkpoints when a pause starts, and runs sensitive actions (`fork`, `merge`, `test`) inside `_with_paused_window(...)` to guarantee that no optimizer steps happen concurrently.
- **`ControlDelegate`** is the minimal interface the pause gate needs. `OnTheFlyTrainerDelegate` (used by native sessions) mutates session flags and forwards fork/merge/test calls into `RunManagementMixin`. External sessions supply their own delegate that understands how to pause the host framework.

Dashboard commands ultimately call into `ConsoleAction` through handlers registered in `CommandsMixin`. For example, the `"pause"` command calls `ConsoleAction.pause(...)`, and the reply contains the checkpoint that the dashboard can offer for download. `"fork"`, `"merge"`, and `"test"` all follow the same pattern, ensuring reproducible state transitions.

## Events & Metrics

`control.send_event(...)` normalizes every outbound payload through `_sanitize` so non-finite floats become `None` and so Torch/Numpy objects are JSON-serializable. Mixins call `_event(...)` (from `EventsMixin`) to stamp monotonically increasing sequence numbers, attach timestamps, and push metrics/logs to the writer.

Runtime telemetry (GPU utilization, VRAM, throughput, activation sparsity, grad norms, etc.) is collected in `mixins/train_mixin.TrainMixin._runtime_metric_snapshot` by combining:

- Live metrics returned by `_default_training_step`.
- Helpers in `runtime_metrics.py` (`canonicalize_metrics`, `weight_norm`, `ActivationZeroTracker`, `DeviceStatsMonitor`, `move_batch_like`, etc.).
- Loss/gradient utilities in `metrics_utils.py`.

Metric snapshots are emitted as `{"type":"trainStep", ...}` events, which the dashboard renders in charts without needing any external logging hooks.

## Dashboard Connectivity & Autostart

The public `Trainer` (`src/onthefly/trainer.py`) optionally owns a `SocketChannel`. When `auto_connect=True`, `Trainer.__init__` immediately connects to the dashboard server, calls `set_channel(...)`, and keeps the socket alive until `Trainer.close()` is invoked. When no dashboard is available, the backend falls back to stdio so scripts can still log locally.

`dashboard_channel.SocketChannel` is resilient:

- Commands are only present when the dashboard is open; otherwise the trainer simply never sees `ControlBus` messages.
- Outbound events are buffered in `_backlog` and flushed when a connection returns.
- `_connect_loop` keeps trying to reconnect with exponential wait while logging to `stderr` so users can diagnose networking issues.

## Framework Delegates & External Loops

External integrations live under `src/onthefly/wrappers/` and implement `wrappers.base.FrameworkDelegate`:

- `FrameworkDelegate.attach(...)` installs itself into the target framework (e.g., Lightning callbacks) but does not start training.
- `install_batch_boundary_hook(...)` registers pause-gate checks so the dashboard can pause/resume at safe points.
- Optional hooks like `request_pause`, `save_checkpoint`, or `restore_initial_state` allow the dashboard to control frameworks that own their own persistence layers.

`wrappers/lightning.py` demonstrates the pattern:

- `LightningFrameworkDelegate` spins up a `SocketChannel` (if needed), instantiates `OnTheFlyExternalSession`, and forwards Lightning callbacks into session methods that emit metrics/events.
- `_LightningFitDriver` monkey-patches `trainer.fit(...)` so Lightning jobs can go through dashboard-driven auto-test pauses without the user having to restructure their scripts.
- Metrics reported by Lightning are canonicalized via `runtime_metrics.canonicalize_metrics` before being sent to the dashboard, keeping parity with the native loop.

The same control-plane building blocks therefore serve native PyTorch scripts and third-party frameworks; the only difference is whether `TrainMixin` drives batches or the delegate calls `session.tick()` between the host framework’s steps.

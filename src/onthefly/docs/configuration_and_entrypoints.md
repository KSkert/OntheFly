# Configuration & Entry Points

This guide explains how user code boots the backend, which objects carry configuration, and how integrations outside the native trainer wire themselves into the session layer.

## SessionConfig

`src/onthefly/config.py::SessionConfig` is a compact `@dataclass` that travels with every session. Fields include:

- `project` and `run_name` — identifiers echoed back to the dashboard and used in checkpoint filenames.
- `device` — preferred device string (e.g., `"cuda:0"`). When unset, `device_utils._resolve_device` infers the device from model parameters.
- `amp`, `grad_clip_norm` — enable autocast/mixed precision and gradient clipping thresholds.
- `save_dir`, `ckpt_keep`, `ckpt_every_steps` — checkpoint directory, ring retention size, and optional cadence for auto checkpoints.
- `data_order_policy`, `enforce_sampler_state`, `grad_clip_norm` — determinism knobs consumed by `OnTheFlySessionBase._init_determinism`.

The config stays intentionally small so it can be serialized and shipped alongside events without leaking large runtime objects.

## Trainer API

`src/onthefly/trainer.py::Trainer` is the ergonomic entry point modeled after Lightning's trainer:

- `__init__` arguments include `project`, default `run_name`, `max_epochs`, `max_steps`, validation cadence, and dashboard socket parameters (host/port, connect timeouts, backlog size). When `auto_connect=True`, the trainer instantiates a `dashboard_channel.SocketChannel` and calls `control.set_channel(...)` so events immediately flow to the dashboard if it is open.
- `session_defaults` lets callers pre-fill settings (e.g., `save_dir`, `ckpt_keep`, determinism flags) that will be merged into every session launched from this trainer.
- `fit(...)` requires `model`, `optimizer`, `loss_fn`, and `train_loader`. Optional parameters mirror those exposed by `OnTheFlySession` (scheduler, device override, AMP flag, gradient clipping norm, manual seed, embedding hook, `model_factory`, determinism toggles, validation cadence). The method:
  1. Resolves the actual run name (explicit `run_name` argument overrides the default).
  2. Merges `session_defaults` and per-call overrides into a single kwargs dict.
  3. Constructs `session/native.py::OnTheFlySession` and immediately calls `session.serve(...)`, supplying `max_steps`, `max_epochs`, and `do_test_after`.
  4. Returns the session so advanced users can inspect it after training completes.
- `close()` tears down the dashboard socket when the trainer leaves scope.

Because `Trainer.fit(...)` re-uses `OnTheFlySession`, everything documented in `session_lifecycle.md` applies whether users instantiate sessions directly or go through the trainer facade.

## Model Factories & Rehydration

`src/onthefly/factory.py` builds factories that can recreate models without needing the original constructor arguments:

- `_normalize_user_factory` accepts a callable, `(callable, args, kwargs)` tuple, or `{"factory": fn, "args": ..., "kwargs": ...}` mapping and wraps it so later calls return a fully constructed `torch.nn.Module`.
- `_build_model_factory(model, user_factory=None)` tries, in order: a user-supplied factory, a `model.factory/build/make/new/spawn` method, a zero-argument constructor, and finally `copy.deepcopy`. Before deep-copying, `_strip_lock_prone_attrs` removes fields that tend to hold non-pickleable locks or trainer handles (loggers, callbacks, TensorBoard writers) and `_patched_deepcopy_for_locks` temporarily teaches `copy.deepcopy` how to handle threading locks.
- `_ensure_module(...)` enforces that the factory ultimately returns a `torch.nn.Module`, giving clear error messages if not.

These factories are used whenever the backend needs fresh model instances: forks, merges, reports, and subset scans never mutate the live training model.

## Device & Precision Utilities

- `src/onthefly/device_utils.py` houses `_resolve_device` (prefers user-specified device, otherwise inspects model parameters/buffers), `_sync_device_by_name` (initializes CUDA/MPS contexts), and `_noop_ctx` (used when AMP/autocast is disabled).
- `src/onthefly/scale.py::_SafeScaler` wraps `torch.cuda.amp.GradScaler` so callers can always call `.scale(...)`, `.unscale_(...)`, `.step(...)`, and `.update(...)` even when AMP is disabled. It also ensures `.backward()` behaves on non-scalar losses by reducing them to `mean()` when required.

These utilities let the trainer present a uniform API even when users toggle AMP or move models between CPU/GPU/MPS devices.

## External Sessions & Delegates

`src/onthefly/session/external.py::OnTheFlyExternalSession` mirrors the native session but expects an external framework to call `tick()`:

- `bind_runtime(...)` records references to the framework-owned model/optimizer/scheduler/device, infers device placement, and attaches `DeviceStatsMonitor`/`ActivationZeroTracker` so telemetry stays uniform.
- `attach_framework(delegate: FrameworkDelegate)` installs the adapter and registers the pause gate with the framework's batch boundaries. Delegates live under `src/onthefly/wrappers/`—for example, `wrappers/lightning.py` installs callbacks into Lightning's trainer, monkey-patches `trainer.fit`, and forwards Lightning metrics via `runtime_metrics.canonicalize_metrics`.
- `configure_loss_fn(...)` wraps callables into lightweight `nn.Module` instances and sets metadata such as `_otf_uses_batch` or `_otf_batch_call_cfg` so per-sample analyses can treat custom loss functions consistently.

External sessions therefore share the exact same mixin stack and control plane as native sessions; the only differences are how batches are advanced and how pause/resume requests reach the framework.

## Identifiers & Helpers

- `src/onthefly/ids.py::_short_hash` generates stable yet human-readable IDs for sessions and runs (`sess-<hash>`).
- `src/onthefly/utils.py` contains utility helpers (`_seed_worker`, JSON-safe math helpers, etc.) used by loader determinism and feature scans.
- Logging/diagnostics reuse Python's stdlib plus the event system described in `control_plane.md`, so no external logging libraries are required.

Taken together, these components make it straightforward to embed the backend in any training script: instantiate `Trainer`, call `fit(...)`, and the rest of the infrastructure (sessions, control plane, docs above) automatically handles dashboard connectivity, checkpoints, determinism, and instrumentation.

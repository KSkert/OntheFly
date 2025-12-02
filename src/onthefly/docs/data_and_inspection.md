# Data Determinism & Inspection Tooling

Understanding what the backend does with your dataloaders, samplers, and analysis passes is critical when debugging pause/resume behavior or building custom selection policies. This document explains the pieces that live under `src/onthefly/sampler_utils.py`, `src/onthefly/data_explorer.py`, and the mixins that use them.

## Deterministic Data Ordering

- **Policies (`SessionConfig.data_order_policy`)**: `OnTheFlySessionBase._init_determinism` reads `data_order_policy` (`"user"`, `"epoch_reseed"`, `"fixed_order"`) and `enforce_sampler_state`. When the policy is not `"user"`, it calls `_apply_determinism_policy` to wrap the train loader with an `EpochSeededRandomSampler`.
- **`EpochSeededRandomSampler` (`sampler_utils.py`)**: Generates reproducible shuffles keyed by `base_seed + epoch`. It tracks `_cursor` so a pause can resume mid-epoch, and exposes `state_dict`/`load_state_dict` for checkpoint persistence. `_sampler_set_epoch` detects `DistributedSampler` and calls `set_epoch(...)` so DDP replicas stay consistent.
- **Pause Guarding**: When `deterministic_pauses` is true, `_install_determinism_guards` (in `OnTheFlySessionBase`) snapshots loader state, worker seeds, and sampler cursors every time the session pauses. `_reapply_determinism` reapplies the policy after any loader swap and reinstalls the guards.

## Data-Explorer Utilities

`src/onthefly/data_explorer.py` is the backbone for all inspection workflows (per-sample losses, embeddings, subset exports, etc.). Highlights include:

- **Model Input Preparation**: `model_input_candidates(model, first, rest)` builds a priority-ordered list of tensors/structures to try when calling the model. It inspects `model.embedding_model` hooks if present, supports tuples/lists/dicts, and filters incompatible shapes. `should_retry_model_input(exc)` tells callers when to try the next candidate.
- **Output Normalization**: `ensure_tensor_output(output)` unwraps dictionaries, tuples, and nested lists until a tensor is found. `_extract_loss_value` turns heterogeneous loss returns (scalar tensor, tuple, dict) into a tensor suitable for `_to_scalar_loss`.
- **Batch-Aware Loss Invocation**: `_call_batch_loss(loss_fn, model, batch)` executes losses that accept the full batch object. `_apply_loss_with_warning_guard(fn)` converts PyTorch warnings about shape/broadcast mismatches into hard errors so we never hide misconfigured losses during analysis.
- **Per-Sample Loss Pipeline**: `compute_per_sample_losses(...)` takes either a dataset or a prebuilt `DataLoader`, optional index subsets, AMP flags, deterministic seeds, and `materialize=False` toggles. It iterates batches, reuses `model_input_candidates` and `_call_batch_loss`, and writes both per-sample losses and sample indices into either Python lists or a `ChunkedArraySpool`.
- **Embedding + Clustering Helpers**: `compute_embeddings(...)` runs forward passes to collect embedding vectors (optionally using a user-provided hook). `cluster_embeddings(...)` falls back gracefully if scikit-learn is unavailable, and `select_hard_clusters(...)` ranks clusters by average loss so fork policies can target "hard" regions.
- **Persistence Helpers**: `ChunkedArraySpool` streams floats/ints to a temporary file so large scans never blow past RAM. `export_subset_table(...)` writes compact CSV/Parquet/Feather summaries, relying on `_jsonable_list` to convert tensors and numpy arrays into serializable lists.

## Feature & Selection Mixins

`mixins/feature_mixin.py` orchestrates the data-explorer primitives for commands such as `fork`, `report`, and `exportSubset`:

- `_loss_module_for_features` produces an `nn.Module` wrapper for callable loss functions so TorchScript/AMP hooks continue to work during scans.
- `_rng_guard(seed)` temporarily overrides RNG seeds (Python/NumPy/Torch/CUDA) to keep deterministic behavior during analysis without perturbing the main training loop.
- `_compute_subset_losses(...)` routes to `compute_per_sample_losses` with knobs for AMP, deterministic seeds, streaming spools, or prebuilt loaders.
- `_compute_margins_and_embeddings(...)` collects per-sample softmax margins via `metrics_utils._top2_margin`, runs optional embedding hooks, and stores values in chunked spools. It handles dataset slicing via `torch.utils.data.Subset` when a subset of indices is requested.
- `_model_from_checkpoint(...)` respawns fresh model copies using the `_model_factory` built during session initialization so reports never mutate the live training model.

`mixins.commands_mixin.CommandsMixin` uses these helpers to implement dashboard commands:

- `reportRunHealth`, `reportActivationStats`, and `reportGradients` all rely on `_health_header_for(...)`, `_jsonable_list(...)`, and runtime metrics to build textual diagnostics.
- `export_subset_table` supports both manual index lists and sampler-derived lists (`_coerce_indices`) with safety limits (10M items) so requests remain bounded.
- Fork commands call into `_build_features_for_selection` (defined in `RunManagementMixin`) which in turn uses feature-mixin APIs to calculate loss quartiles, KMeans clusters (`kmeans_utils.py`), and embeddings before deciding which samples to keep.

## Metrics & Batch Utilities

Supporting modules fill in the gaps:

- `runtime_metrics.py` offers `move_batch_like`, accuracy helpers, `_metric_tokens` for alias detection, GPU telemetry monitors (`DeviceStatsMonitor`), and activation sparsity tracking (`ActivationZeroTracker`). The training loop uses these during every batch to produce enriched `trainStep` events.
- `metrics_utils.py` houses `_to_scalar_loss`, `_grad_norm`, `_top2_margin`, and helper routines for extracting scalar values from nested tensors.
- `sampler_utils.py` and `sampler_utils.EpochSeededRandomSampler` (covered above) provide the reproducible data-order foundation that makes all analysis reproducible.

Together these components let the backend pause, fork, merge, and analyze massive datasets deterministically without asking user code to change how it feeds data into PyTorch.

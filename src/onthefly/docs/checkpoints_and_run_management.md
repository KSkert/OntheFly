# Checkpoints, Forks, & Run Management

OnTheFly treats checkpoints as first-class artifacts because every dashboard command—pauses, forks, merges, resets, subset exports—relies on deterministic snapshots of model state. This guide covers the modules responsible for persistence and run orchestration.

## Ring Checkpoints & Serialization

- **`mixins/checkpoint_mixin.CheckpointMixin`** defines `_checkpoint_payload`, `_write_checkpoint_to`, and `_save_ring_checkpoint`. The payload always contains the model, optimizer, scheduler, AMP scaler, current `step`/`epoch`, and last validation loss. Ring checkpoints live in `SessionConfig.save_dir` and respect `SessionConfig.ckpt_keep` for retention.
- **`src/onthefly/checkpoints.py`** provides standalone `save_checkpoint`/`load_checkpoint` helpers used by mixins and commands. Saving captures:
  - RNG state for Python, NumPy, PyTorch CPU, and CUDA so resumes replay deterministically.
  - DataLoader sampler state via `_sampler_state(...)`, enabling mid-epoch resume/pause fidelity.
  - Optional grad scaler state and scheduler state if those objects expose `state_dict()`.
  - A lightweight `.meta.json` sidecar (path + step) to speed up UI queries.
- Loading injects the serialized state back into the live objects and returns the restored `step`/`epoch` counters. `load_checkpoint` deliberately tolerates missing scaler/scheduler states and only loads what the callee supplies.
- **`ckpt_utils._parse_step`** (used throughout the mixins) extracts numeric step IDs from filenames so `_latest_ckpt_for_run` can return the newest snapshot without reading metadata.

Pauses always trigger `_save_ring_checkpoint` (unless explicitly suppressed), so the dashboard can surface “download latest checkpoint” links after every manual pause, fork, or test.

## Reset Snapshots & Resume

`session/native.py` spins up `_capture_reset_snapshot` as soon as a session starts. That thread writes a checkpoint to a temporary directory so the `resetSession` command can atomically restore model/optimizer/scheduler/scaler state and step counters without requiring the user to configure extra hooks. `_reset_session_state` coordinates run-name deduplication, checkpoint cleanup, and new-run metadata so the UI sees the reset as a new run generation.

## RunManagementMixin

`mixins/run_management_mixin.py` centralizes everything related to run transitions:

- **Run identity**: `_next_run_name`, `_switch_to_new_run`, and `_emit_new_run` keep run IDs unique and emit the right dashboard events (`runTransition`, `finalizeRun`). Environment variables can seed fork/merge counters so reconnecting sessions pick up where they left off.
- **Subset activation**: `_rebind_train_loader_to_subset(indices)` swaps the train loader to point at a `torch.utils.data.Subset` while preserving `batch_size`, `collate_fn`, `drop_last`, and `shuffle`. `_active_subset_indices` tracks the current subset for the explorer UI. When merges complete, the mixin resets back to the full dataset.
- **Fork orchestration**: `_do_fork(payload)` stitches together selection inputs (manual indices, quantiles, clustering, “hard sample” selectors), kicks off per-sample loss/embedding scans via `FeatureMixin`, writes JSON manifests describing the selection, and spawns new run names.
- **Merge orchestration**: `_merge_from_checkpoints(paths, strategy, parents, new_name)` loads checkpoints, respawns cloned models using `_model_factory`, calls the requested merge strategy (see below), and writes metadata so the UI can show merge provenance. It also emits the usual run transition events and checkpoints the merged model for immediate use.

## Merge Strategies

`src/onthefly/merging.py` implements the supported strategies:

- **Stochastic Weight Averaging (`stochastic_weight_averaging`)**: Equal-weight average over all checkpoint state dicts.
- **Fisher Soup (`fisher_soup_merge`)**: Uses per-parameter Fisher information (when provided) to weight contributions. Falls back to SWA if Fisher matrices are missing.
- **Adapter Fusion (`adapter_fuse_merge`)**: Copies base parameters from the first model and averages only parameters whose names contain `"adapter"`, ideal for LoRA/adapter fine-tuning workflows.
- **Weighted Average (`weighted_average_merge`)**: Internal helper used by the strategies above.

Because merge outputs are plain state dictionaries, `RunManagementMixin` can load them back into the active session and continue training immediately.

## Commands & Console Integration

`mixins.commands_mixin` exposes these features to the dashboard:

- `loadCheckpoint` and `saveCheckpoint` wrap the helpers in `checkpoints.py` to operate on arbitrary paths.
- `resetSession`, `forkRun`, `mergeRuns`, and `testRun` all go through `ConsoleAction` so they pause cleanly before touching model state.
- Health/report commands emit textual summaries that include checkpoint details (hashes via `_metadata_hash`, latest steps via `_parse_step`).

Together, these layers provide a reproducible history of every fork/merge/reset while keeping actual checkpoint I/O centralized and auditable.

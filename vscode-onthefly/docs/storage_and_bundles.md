# Storage & Bundle Pipeline

`src/storage.ts` gives the extension a zero-dependency persistence layer that mirrors what the Python backend writes. It relies on `better-sqlite3` for synchronous access so event handlers can persist metrics/logs without async juggling.

## Database Initialization

- `initStorage(context)` opens a workspace-specific SQLite file (under the extension global storage path) and applies `applyPragmas` (`WAL`, `NORMAL` synchronous mode, cache tweaks). The module guards initialization with `initPromise` so multiple callers can await the same work.
- `ensureSchema` creates/updates every table the UI relies on:
  - `runs` & `run_edges` – canonical list of run IDs plus explicit parent edges used by the DAG view.
  - `run_subsets` – stores manually selected sample indices for each run.
  - `checkpoints` – tracks checkpoint paths, steps, and creation times.
  - `metrics` – per-step training metrics (loss, val loss, accuracy, runtime stats) with indexes over `(run_id, step)`.
  - `summaries` – final loss/val_loss snapshots.
  - `kv` – small key/value store for feature toggles.
  - `logs` – structured textual logs tagged by phase (train/test/info).
  - `test_metrics` – per-step test losses.
  - `session_final_models` – ties a session ID to the checkpoint path recorded during “Test Now” or auto-tests.
- Schema migrations enforce a consistent `metrics` column set and prune legacy columns (`theta`, `constraintSatisfied`, etc.) when older installs upgrade.

The module exposes `isReady()`, `whenReady()`, and `closeStorage({ retainFile })` to coordinate lifecycle with the activation layer.

## Write Helpers

- `insertRun(run_id, project, name, parents)` seeds a row in `runs`, updates parent edges, and stamps `created_at`.
- `insertMetric(run_id, StepRow)` writes metrics and automatically keeps prepared statements cached for streaming inserts.
- `insertCheckpoint(ckpt_id, run_id, step, path)` / `latestCheckpointForRun(run_id)` maintain a queue used by pause/merge/fork commands.
- `insertLog`, `insertTestMetric`, `upsertSummary`, `upsertFinalModel`, `setRunSubset`, and `insertReportLoss` (via `upsertReportLossDist`) give the activation layer simple CRUD operations while keeping SQL localized.
- `checkpointNow()` exposes `PRAGMA wal_checkpoint` so heavy exports can finalize WAL files before copying.

## Query Helpers

- `listRuns`, `getRunRows(runId)`, `getLogs(runId, phase?)`, `getLogsBySession(sessionId, phase?)`, `getTestRows(runId)`, `getReportLossDist(runId)`, and `getRunSubset(runId)` back every “request…” message the webview sends.
- Each helper returns plain objects/arrays so `extension.ts` can post them straight to the UI. `stripUiOnlyFields` (in `state.ts`) massages logs before they hit the dashboard to keep step tags tidy.

## Bundle Export & Import

`exportBundle(dir)` and `loadBundle(dir, context)` implement a simple portable format:

1. Copy the SQLite file into `dir/runs.sqlite` via `_copyDbToSingleFile`.
2. Copy every checkpoint referenced in `checkpoints` into `dir/checkpoints/<ckpt_id>.pt` and record the mapping in `bundle.json`.
3. Copy final tested models into `dir/final_models/<session>-final.pt` and add their paths to the manifest.
4. Write `bundle.json` with `{ schema_version, sqlite, checkpoints, final_models }`.

`loadBundle` reverses the process: it loads the bundled SQLite via `loadSessionFrom`, rewrites checkpoint paths to the new absolute locations, and updates `session_final_models` so “load session” buttons can immediately stream metrics/logs from the imported data.

These helpers power the “Export Session” / “Load Session” commands wired up in `extension.ts`. Because all paths are rewired inside the DB, users can zip/unzip bundles on other machines without breaking checkpoint references.

## Integration Points

- `extension.ts` calls `initStorage`, `closeStorage`, `listRuns`, `getLogs`, `getRunRows`, etc., whenever the panel needs to refresh or when the webview requests more data.
- `extensionHost/ipc.ts` persists streaming events through `insertMetric`, `insertLog`, and `insertCheckpoint` so nothing is lost if VS Code reloads.
- Bundle export/import functions are exposed via re-exports (`exportBundle as storageExportBundle`) so the activation layer can keep UI code free of filesystem details.

By keeping schema changes and SQL localized to this module, the rest of the extension can treat persistence as a simple typed API and stay focused on IPC/webview concerns.

# IPC, Control Server & Extension State

`src/extensionHost/` contains the runtime that keeps the dashboard in sync with Python trainers. `ipc.ts` owns the TCP server and message protocol, while `state.ts` exposes the UI-facing state bag and helper emitters.

## Dashboard Server & Connection Handling

- `ensureTrainerServer()` spins up a `net.Server` bound to `localhost:${ONTHEFLY_DASHBOARD_PORT||47621}`. Incoming sockets are accepted one at a time; if a second trainer tries to attach while another is still connected the new socket is dropped with a warning.
- On the first connection the IPC layer optionally triggers `hardResetSession` (via `registerHardResetHandler`) so stale state does not leak across runs. Every socket installs `'data'`, `'error'`, and `'close'` handlers that funnel into `handleTrainerData` and `disconnectTrainer`.
- `startRun()` backs commands such as “Resume after reset.” It shows a VS Code notification (`withProgress`) while waiting up to 60 seconds for a trainer to connect and updates the status banner once the socket is alive.

## Request/Response Plumbing

- `sendReq(cmd, payload, timeoutMs)` is the JSON-RPC style helper every command uses. It assigns a UUID, writes `{"id", "cmd", "payload"}` to the socket, and stores `{resolve,reject,timer}` in a `pending` map. Replies (objects containing the same `id` plus `ok`/`error`) are matched inside `handleLine`.
- `pending` is cleared automatically when sockets close to avoid promise leaks. On timeouts, `sendReq` rejects with `timeout waiting for <cmd>`.

## Fast Control Commands

- `sendCtl({ cmd: 'pause' | 'resume' | 'save_ckpt' | 'merge' | 'fork' })` writes control messages without expecting a reply. It optimistically emits events (`paused`, `resumed`) and toggles the activity state so the UI feels responsive while the backend catches up.
- `requestBackendPause` wraps `sendReq('pause')`, marks the session as paused, and returns whatever payload the backend emitted (typically the checkpoint path/step).
- `resumeTrainerOn(runId)` and `resumeAfterReset()` guard against concurrent resumes, call `ensureTrainerOnRun`, and then emit `sendCtl({ cmd: 'resume' })` to release the backend gate.

## Run Targeting & Context Seeding

- `computeForkMergeCounters()` scans `listRuns()` (via `storage.ts`) to determine the highest fork/merge suffix currently in use so backend counters restart at the appropriate values.
- `latestResumeHints` consults `latestCheckpointForRun` and returns `{ init_ckpt, init_step }` if a ring checkpoint has been persisted.
- `seedTrainerForRun(targetRunId)` tells the backend which run is currently in focus. If the trainer is already running the requested run it just updates counter seeds via `sendReq('set_counter_seeds', …)`. Otherwise it calls `sendReq('attach_context', …)` with the resume hints and counter seeds.
- `ensureTrainerOnRun(targetRunId)` is the public helper that waits for a trainer connection, compares the run ID with `extensionState.currentRunId`, and issues `sendReq('switch_run', …)` when a swap is needed. It then updates `extensionState` and echoes `modelNav.select` to the webview.

## Health Checks & Region Helpers

- `runHealth(cmd, payload, eventType, timeoutMs, forRunId?)` pauses the backend, issues the requested RPC (e.g., `dist_health`, `numerics_health`), and posts progress/completion/error messages back to the webview. Health buttons are wired to this helper in `extension.ts`.
- `getSelectedRegionIndices(runId, minLoss, maxLoss)` proxies UI selections to the backend for deterministic subset creation.

## Trainer Event Processing

- `handleTrainerData` accumulates socket chunks until a newline is encountered, then forwards each line to `handleLine`.
- Inside `handleLine`:
  - Reply objects (with `id`/`ok`) resolve pending `sendReq` promises.
  - `testStep`, `trainStep`, `log`, `newRun`, and other stream events are normalized and written to SQLite (`insertMetric`, `insertLog`, `insertTestMetric`, `insertRun`, `insertCheckpoint`, `upsertSummary`, `upsertFinalModel`).
  - Activity flags (`pauseInFlight`, `resumeInFlight`, `runActivityState`) are updated so `postStatus` can show `running`/`paused`.
  - Special cases such as `session_started` trigger one-off behavior (e.g., `clean_disk` when `needDiskCleanOnNextTrainer` is true).
- After persistence the event is re-posted to the webview via `post(obj)` so the dashboard charts/logs refresh in near real-time.

## Extension State Utilities

`extensionHost/state.ts` exposes the mutable runtime bag:

- `extensionState` holds the panel reference, run IDs, session IDs, `nativeRunsThisSession` set, pause/resume flags, and last export directory metadata.
- `post(msg)` is a thin guard around `panel.webview.postMessage`.
- `postStatus(connected)` derives running/paused booleans from `extensionState.runActivityState` and informs the UI so it can enable/disable controls.
- `postCurrentSession()` emits `fs.session.current` events so the webview keeps a breadcrumb of the active backend session.
- `stripUiOnlyFields(rows)` removes redundant `step` labels from log entries when the text already encodes the step number. This keeps the log viewer tidy across mixed telemetry sources.

`extensionHost/types.ts` keeps the discriminated unions for both IPC directions (`WebMsg`, `Ctl`) and shares small structs (`RunActivityState`, `StepRow`, `TrainerResetSeed`). Updating types there ensures the activation layer, webview, and IPC helpers line up at compile time.

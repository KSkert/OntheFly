# VS Code Extension Architecture

The VS Code package under `vscode-onthefly/` is the desktop front-end for the Python runtime. It is split into a thin activation layer (`src/extension.ts`), a cohesive extension host (`src/extensionHost/…`), a local SQLite adapter (`src/storage.ts`), and the dashboard webview bundle (`src/webview/`). Each piece can evolve independently: the host focuses on IPC + state, the storage layer persists run metadata, and the webview renders charts/logs/health widgets.

## Top-Level Layout

- **`src/extension.ts`** – VS Code entry point. Registers the dashboard command, manages the webview panel lifecycle, forwards webview messages into the host, and coordinates session resets.
- **`src/storage.ts`** – Typed wrapper around the SQLite database shared with the Python backend. Provides helpers for inserting runs, metrics, checkpoints, logs, reports, and exporting/importing bundles.
- **`src/extensionHost/`**
  - `ipc.ts` encapsulates the TCP server that accepts trainer connections, manages JSON-RPC–style calls to the Python process, handles streaming events, and exposes helpers such as `ensureTrainerOnRun`, `runHealth`, `requestBackendPause`, and `resumeTrainerOn`.
  - `state.ts` owns every mutable bit of runtime state (current run, session ID, pause/resume flags) plus the `post`/`postStatus` utilities for messaging the webview.
  - `types.ts` centralizes the discriminated unions for webview commands (`WebMsg`), outgoing control messages (`Ctl`), and shared data structures such as `StepRow` or `TrainerResetSeed`.
- **`src/webview/`** – Static JS/HTML bundle that VS Code injects into the dashboard panel. It consumes extension-originated `post(...)` messages to render charts, logs, DAGs, health panels, and reporting widgets.

## Activation Flow

1. **Activation** – VS Code calls `activate` in `src/extension.ts`. The module registers the `onthefly.showDashboard` command, rehydrates serialized panels, wires `registerWebviewPanelSerializer`, and calls `initStorage(...)` so SQLite is ready before events arrive.
2. **Panel creation** – `openPanel` (or `revivePanel`) instantiates the dashboard webview, injects resource URIs for the JS bundle, and wires `onDidReceiveMessage` so UI events are routed to the `onMessage` dispatcher.
3. **Trainer lifecycle** – The first panel/configuration call invokes `ensureTrainerServer()` inside `extensionHost/ipc.ts`, which starts a TCP server on `localhost`. When a Python trainer connects, `ipc.ts`:
   - Clears pending RPC promises.
   - Emits `status`/`log` updates to the webview via `post`.
   - Seeds the trainer with the current run context (fork/merge counters, checkpoint hints) using `seedTrainerForRun`.
   - Streams back `trainStep`, `log`, and other events, persisting them through `storage.ts` as needed.
4. **Webview commands** – UI actions (pause/resume, fork/merge, exports, health checks, report generation, DAG merges, etc.) arrive as `WebMsg` unions defined in `extensionHost/types.ts`. `extension.ts` handles each command by combining host helpers and storage APIs. Examples:
   - `pause` → `requestBackendPause` + `sendReq('save_ckpt', …)` + `insertCheckpoint`.
   - `resume` → `resumeTrainerOn` or `resumeAfterReset`.
   - `exportSubset` → gather indices from UI/storage, then call `sendReq('export_subset', …)` to make the backend write parquet/CSV/feather files.
   - `generateReport` → uses stored subsets, executes the RPC, ingests returned histograms, and updates `ChartReport`.
5. **Storage + bundles** – Export/import commands call `storageExportBundle` / `storageLoadBundle` to move the SQLite DB and referenced checkpoints on/off disk. During exports, the host requests the Python process to spill the latest checkpoints so the bundle is self-contained.

## Communication Channels

- **Extension ↔ Webview** – `post(...)` and `postStatus(...)` in `extensionHost/state.ts` send messages to the UI. The panel registers a single `onDidReceiveMessage` handler that forwards everything to `onMessage`.
- **Extension ↔ Python Trainer** – `extensionHost/ipc.ts` manages a newline-delimited JSON protocol identical to the backend’s stdio interface. Outgoing requests include an `id`; responses resolve/reject pending promises stored in a local `Map`. Streaming events (`trainStep`, `log`, `testStep`, `newRun`, etc.) are emitted to the webview and persisted through `storage.ts`.
- **Extension ↔ SQLite** – `storage.ts` offers synchronous helpers (using `better-sqlite3`) so the activation thread can insert metrics/logs without additional IPC. The module also exposes analytical queries (run list, logs by phase, subsets, report distributions) that the webview can request on demand.

## State Management

The host maintains all mutable state in `extensionHost/state.ts`. Key fields:

- `panel` – active webview panel reference (or `null` when closed/revived later).
- `currentRunId` / `modelNavSelectedRunId` – drive status updates and inform which run future pause/resume/fork commands should target.
- `currentSessionId` – tags logs, drives grouped log fetches, and helps detect whether a run was created in the current trainer session (`nativeRunsThisSession` Set).
- `pauseInFlight` / `resumeInFlight` / `pendingResumeAwaitingFirstRun` – guard repeated UI actions so the extension does not issue overlapping RPCs.
- `needDiskCleanOnNextTrainer` – ensures the next trainer connection triggers a `clean_disk` RPC.

`state.ts` also exposes `postStatus` (UI indicator), `postErr`, `postCurrentSession`, and `stripUiOnlyFields` (prunes redundant `step` labels when logs already mention the step number).

## Event Handling Pipeline

1. The Python trainer sends an event (e.g., `{"type":"trainStep",…}`) over the socket.
2. `ipc.ts` parses the line, updates `extensionState.currentRunId` if necessary, persists the data through `storage.ts`, and emits user-facing events via `post(...)`.
3. Certain event types trigger additional logic:
   - `newRun` → ensures the run exists in SQLite, updates session IDs, and emits `runs` so the UI refreshes the run list.
   - `log` → heuristically infers the phase (train/test/info), extracts steps/epochs, and filters out redundant `step` badges before posting to the UI.
   - `trainStep` → writes metrics via `insertMetric`, echoes a friendly log line, and transitions the activity state to `running`.
   - `session_started` → optionally calls the Python `clean_disk` RPC on first connect to remove stale checkpoints.
   - `auto_test_complete` / `checkpoint_saved` → record fresh checkpoints so they appear in bundle exports immediately.

## Adding Features

- **New dashboard command** – Add a union member to `WebMsg` (`extensionHost/types.ts`), handle it inside `extension.ts:onMessage`, and (if needed) add IPC/storage helpers.
- **New trainer RPC** – Define the request/response logic in `ipc.ts` (using `sendReq`) and expose a typed helper so `extension.ts` can invoke it.
- **New persistence** – Extend `storage.ts` with the required SQL, import the helper into `extension.ts`, and call it from the appropriate event handler or stream hook.
- **New UI bundle assets** – Drop files into `src/webview/`, register them in `getHtml`, and ensure `dashboard.html` references the generated URIs.

Because the host/state/IPC modules are isolated, modifying trainer messaging or storage logic rarely requires touching the large activation file. The extension remains responsive by keeping heavy work (RPC waits, exports) inside `withProgress` blocks while the UI and trainer stay in sync via the shared state.

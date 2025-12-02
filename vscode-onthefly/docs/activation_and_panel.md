# Activation & Panel Lifecycle

The VS Code integration lives primarily in `src/extension.ts`. This document explains how the extension boots, how the dashboard panel stays in sync with trainer processes, and how UI messages flow back into the control plane.

## Activation & Shutdown

- `activate(context: vscode.ExtensionContext)` registers the `onthefly.showDashboard` command and a webview panel serializer so VS Code can restore panels after reloads. It immediately calls `initStorage(context)` to create/open the SQLite database before any trainer events arrive.
- `deactivate()` shuts down the trainer TCP server (`shutdownTrainerServer`) and closes SQLite via `closeStorage({ retainFile: true })` so the on-disk DB sticks around for later sessions.
- During activation, the file also registers a hard reset handler (`registerHardResetHandler(hardResetSession)`) so the IPC layer can force a clean slate whenever a new trainer connects.

## Hard Reset Semantics

`hardResetSession` encapsulates the “start fresh” workflow:

1. Optionally ask the backend to `reset_session` and `clean_disk` so checkpoints are pruned on the Python side.
2. Disconnect from the trainer socket, clear `extensionState` (current run/session IDs, pause/resume flags, model-nav selection), and tear down the in-memory DB by calling `closeStorage({ retainFile: false })`.
3. Re-initialize SQLite (`initStorage(context)`), set `needDiskCleanOnNextTrainer = true`, and notify the webview via `post({ type: 'resetOk' })`.
4. If the backend returned a `TrainerResetSeed`, seed a placeholder run in SQLite and select it in the UI so the dashboard shows immediate feedback.

This same handler powers both the “Reset All” button in the webview and automatic resets triggered when a second trainer connects while another is still paused.

## Panel Creation & Wiring

- `openPanel` instantiates a `WebviewPanel` with `retainContextWhenHidden` so charts/logs keep their state when the user hides the tab. `revivePanel` rebinds event handlers when VS Code restores a serialized panel.
- `configurePanel`:
  - Calls `ensureTrainerServer()` so the TCP listener is live before UI interactions begin.
  - Stores the panel instance on `extensionState.panel` and registers lifecycle hooks (`onDidChangeViewState`, `onDidDispose`) to refresh run lists when the tab becomes visible and to drop stale references when closed.
  - Hooks `onDidReceiveMessage` so every webview post funnels into `onMessage`.
  - Generates the HTML by calling `getHtml(context, panel.webview, nonce)` and injects local resource roots so bundled JS/CSS can load.

`getHtml(...)` looks for `dashboard.html`/`dashboard.js` (falling back to the `src/webview/` copies when running unbundled) and replaces placeholders (`__CHART_REPORT_JS__`, `__RUN_STATE_JS__`, etc.) with `webview.asWebviewUri(...)` results. It also writes a strict CSP meta tag that pins every script to the generated nonce.

## Webview Message Handling

`onMessage(context, m: WebMsg)` is the central dispatcher:

- **Exports**: `exportChart` saves PNGs from `dataUrl` payloads; `exportSubset` collects indices from the report selection overlay (or stored subsets, or backend region queries) and calls `sendReq('export_subset', …)`.
- **Navigation**: `modelNav.select` updates `extensionState.modelNavSelectedRunId` and immediately posts `logs` for the selected run via `getLogs`.
- **Data Fetching**: `requestLogs`, `requestTestRows`, `requestRuns`, and `requestRows` proxy to SQLite helpers (`getLogs`, `getLogsBySession`, `getTestRows`, `getRunRows`) and echo the results to the webview.
- **Lifecycle Commands**: `pause`, `resume`, `testNow`, `fork`, `merge`, and the various health checks call into `extensionHost/ipc.ts` helpers (`requestBackendPause`, `resumeTrainerOn`, `ensureTrainerOnRun`, `runHealth`, etc.). They persist checkpoints or subsets in SQLite as needed so imports/exports stay consistent.
- **Bundle IO**: `exportSession` and `loadSession` call `storageExportBundle` / `storageLoadBundle`, prompting the user for a directory via `showOpenDialog` + `showSaveDialog`.
- **Reset**: `resetAll` simply invokes `hardResetSession(context, { fromUser: true })`.

All user-facing errors are routed through `postErr`, which surfaces the message both in the dashboard and VS Code’s UI.

## Tracking Export Paths

The extension remembers the last directory used for bundle/subset/chart exports via the `LAST_EXPORT_DIR_KEY` key in `context.globalState`. `getInitialExportDir` (inline helper in `extension.ts`) reads that value and falls back to `workspaceState`/`os.homedir()` so save dialogs start in a convenient location.

## Relationship to Extension State

Almost every branch in `onMessage` reads or mutates `extensionState`:

- `pause/resume` guard against concurrent requests with `pauseInFlight` / `resumeInFlight`.
- `fork/merge/testNow` update `currentRunId`, `modelNavSelectedRunId`, and `nativeRunsThisSession` so future commands target the right run.
- `exportSubset` and `generateReport` consult `getRunSubset` / `setRunSubset` to reuse previously selected regions.
- `postStatus(trainerActive())` runs whenever the panel becomes visible to keep the status banner current.

Keeping these updates centralized inside `extension.ts` avoids tight coupling between webview logic and the IPC layer while ensuring that the UI can always reconstruct state after a reload.

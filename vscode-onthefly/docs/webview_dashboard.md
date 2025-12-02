# Dashboard Webview

Everything under `vscode-onthefly/src/webview/` is bundled into the HTML/JS that runs inside VS Code’s webview. It talks to the extension through `window.vscode.postMessage` and consumes events via `window.addEventListener('message', ...)`.

## Entry Point (`dashboard.js`)

- Bootstraps the VS Code bridge via `acquireVsCodeApi()` and exposes it on `window.vscode` so other scripts can post messages.
- Imports helpers from `run_state.js` (aliased as `RunState`) to manage the list of runs, DAG edges, and follow-live behavior.
- Wires DOM references for buttons (`btnPause`, `btnResume`, `btnTestNow`, DAG toggles, report controls, etc.) and keeps UI state in sync with the latest `status`/`trainStep` events.
- Coordinates chart updates by delegating to `ChartCreation`, `ChartStream`, `ChartReport`, and `ChartReportSelection`.
- Maintains run navigation state (follow vs. manual mode), gating of health buttons (`gateHealthButtons`), and overlays such as the DAG panel (`dag_layout.js` + `dag_render.js`).
- Houses the `window.addEventListener('message', ...)` handler that routes every extension event to specialized updaters (metrics, logs, health panels, reports, exports).

## IPC Controls (`ipc_controls.js`)

Centralizes button wiring so the dashboard can stay declarative:

- Registers click handlers for pause/resume/test, session export/import, report generation, health checks, DAG open/close, and DAG merge actions.
- Tracks outstanding report-generation requests (`reportState`) so histogram updates can be matched to the originating run.
- Exposes an `init(...)` method that `dashboard.js` calls with callbacks (`send`, `currentPageRunId`, `setRunningFor`, etc.), keeping UI glue in one place.

## Run State (`run_state.js`)

A self-contained store for lineage and navigation:

- Maintains `parentsOf`, `childrenOf`, `edgeSet`, and `edges` so the DAG visualization can rebuild quickly.
- Tracks `NAV_LIST`, `PAGE_INDEX`, and `CURRENT_LIVE_RUN`, allowing the UI to follow the newest run until the user opts out.
- Stores per-run activity markers (`RUN_STATE`) and last paused steps (`LAST_PAUSED_STEP`) for visual cues next to the loss charts.
- Exposes helpers such as `rebuildNavListFromRows`, `gotoPageByRunId`, `setLiveRun`, `streamTargetRunId`, and `updateModelNav`.

## Chart & Report Modules

- `chart_creation.js`, `chart_bootstrap.js`, and `chart_plugins.js` wrap Chart.js setup for loss/accuracy charts and the histogram view. They keep the configuration logic (scales, legend, tooltips) decoupled from streaming.
- `chart_stream.js` holds a ring buffer (`STATE`/`PEND`) of streaming metrics. It batches updates, enforces point limits, and calls `Chart.update('none')` on the next animation frame for smooth rendering.
- `chart_report.js` caches per-run histogram data returned by the backend (`ChartReport.updateReportForRun`). It renders the distribution, tracks metadata (note, step, epoch), and exposes helpers like `selectedIndicesForRun` for manual fork/export flows.
- `report_selection.js` manages the draggable overlay that lets users select a loss range, preview sample counts, trigger manual forks (`send('fork', …)`), or export subsets (`send('exportSubset', …)`).
- `chart_utils.js` contains shared helpers (DOM lookups, histogram binning) used by the modules above.

## DAG & Health Widgets

- `dag_layout.js` and `dag_render.js` transform the run graph into SVG nodes/edges whenever the user opens the overlay. They rely on the topology maintained by `run_state.js`.
- `health_monitor.js` provides a responsive dock where health panels (distribution, throughput, activations, numerics, determinism) can be opened/closed and resized. The extension emits `dist_health`/`throughput_health` events that populate these panels.

## Logging & Buffers

- `log_buffer.js` keeps a 2,000-line ring buffer for the textual log textarea. It batches DOM writes so rapid log bursts don’t lock the UI.
- `run_state.js` exposes `LAST_PAUSED_STEP` and `AF_MARKERS`; `dashboard.js` uses those to annotate logs and charts when forks/auto-tests happen.

## Assets & Layout

- `dashboard.html` defines the basic layout (toolbar, charts, report side panel, DAG overlay, health panel container) and holds `__PLACEHOLDER__` slots that `getHtml` in `extension.ts` fills with resource URIs and nonce-aware script tags.
- `images/` houses icons (e.g., the fly logo). `getHtml` injects them via `webview.asWebviewUri(...)`.

## Message Flow Recap

1. Extension posts events via `panel.webview.postMessage`.
2. `dashboard.js` listens for them and forwards to modules (`ChartStream.append`, `RunState.addRun`, `LogBuffer.log`, `ChartReport.updateReportForRun`, `HealthMonitor.update`, etc.).
3. User actions call `window.vscode.postMessage({ command, ...payload })`, which `extension.ts` receives and routes through the IPC/storage helpers described elsewhere.

Keeping each concern in its own file allows the dashboard to be extended (new charts, new panels, new controls) without modifying monolithic scripts.

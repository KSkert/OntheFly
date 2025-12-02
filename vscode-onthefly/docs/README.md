# VS Code Dashboard Docs

This folder collects in-depth documentation for the `vscode-onthefly` extension so frontend workflows stay as transparent as the Python backend.

## Document Map

- `extension_architecture.md` – High-level overview of how the activation layer, extension host, storage adapter, and dashboard webview fit together.
- `activation_and_panel.md` – Details the VS Code entry point (`src/extension.ts`), panel lifecycle, reset semantics, and webview message handling.
- `ipc_and_state.md` – Explains the TCP dashboard server, request/response plumbing, `extensionHost/state.ts`, and how trainer events are normalized.
- `storage_and_bundles.md` – Describes the SQLite schema, persistence helpers, and the session import/export bundle format.
- `webview_dashboard.md` – Breaks down the browser-side modules (`dashboard.js`, chart helpers, DAG view, log buffer, health panels) and how they consume events.

Add more topic-specific docs here as new subsystems (e.g., new widgets or import paths) land.

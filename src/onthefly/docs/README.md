# Backend Documentation Hub

This folder groups together the in-depth documentation for the Python backend that powers OnTheFly. Each markdown file focuses on a subsystem so you can jump directly to the component you are touching without reading the entire source tree.

## Document Map

- `runtime_architecture.md` — high-level view of the session orchestration layer and how the primary runtime pieces interact.
- `session_lifecycle.md` — zooms in on `OnTheFlySession`/`OnTheFlySessionBase`, the mixin stack, and the training/evaluation loops.
- `control_plane.md` — details the JSON-over-stdio transport, command router, pause gate, and dashboard channel integrations.
- `data_and_inspection.md` — covers determinism policies, sampler helpers, data-explorer utilities, and the per-sample/embedding analysis stack.
- `checkpoints_and_run_management.md` — explains checkpoint capture/resume, ring checkpoint retention, fork/merge helpers, and merge strategies.
- `configuration_and_entrypoints.md` — documents `Trainer`, `SessionConfig`, model factories, framework delegates, and how external integrations bootstrap the backend.

Each file references the concrete modules/classes so you can trace behavior directly in `src/onthefly`. Feel free to expand this directory with additional topic-specific guides as the backend grows.

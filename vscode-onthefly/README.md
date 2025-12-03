# OnTheFly · VS Code Extension

Human-in-the-loop ML training directly inside VS Code. This extension streams live metrics, logs, and actions from your `onthefly-ai` trainers into a dashboard panel so you can pause, inspect, fork, and merge runs without leaving the editor.

---

## Requirements

- VS Code `1.102.0+`
- Python `3.9+` with `onthefly-ai ≥ 0.1.1`
- PyTorch `2.2+` (Lightning optional)
- Local training scripts that instantiate `OnTheFlyTrainer` or call `attach_lightning(...)`

The extension enforces the `onthefly-ai` minimum: if the connected trainer is older or cannot report its version, the connection is rejected with an upgrade prompt.

---

## Installation

1. Install **OnTheFly** from the VS Code Marketplace (or `vsce package` → “Install from VSIX…”).
2. Ensure the Python environment you use for training has `onthefly-ai`:
   ```bash
   pip install -U onthefly-ai
   ```
3. In VS Code: press `Ctrl/Cmd+Shift+P` → `OnTheFly: Show Dashboard`.

The dashboard listens on `localhost:47621`. Whenever your training script runs and instantiates an OnTheFly trainer, the dashboard connects automatically as long as the panel is open.

---

## Quickstart

```python
from onthefly import Trainer

trainer = Trainer(
    project="mnist-demo",
    run_name="baseline",
    max_epochs=1,
    do_test_after=True,
    val_every_n_epochs=1,
)

trainer.fit(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)
```

Lightning users can wrap an existing `L.Trainer`:

```python
from onthefly import attach_lightning

attach_lightning(
    trainer=trainer,
    model=model,
    project="demo",
    run_name="lightning-baseline",
    loss_fn=model.loss,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)
trainer.fit(model, datamodule=data_module)
```

Run your script normally (`python train.py`). Once the dashboard reports “Trainer connected”, live metrics, logs, checkpoints, and controls stream immediately.

---

## Dashboard Workflow

1. **Open the panel** (`OnTheFly: Show Dashboard`).
2. **Run training** in a terminal/notebook with `onthefly-ai` imported.
3. **Monitor** loss curves, runtime metrics, and logs in real time.
4. **Act**: pause/resume, trigger tests, generate reports, fork specialists, or merge checkpoints from the dashboard.
5. **Export sessions** or load previous bundles directly from the panel to revisit runs.

If the dashboard disconnects, simply reopen it—active trainers reconnect automatically.

---

## Features

- **Live metrics & logs**: per-step loss, val loss, throughput, GPU stats, and structured log streaming.
- **Mid-training controls**: pause/resume, save checkpoints, run health checks, launch reports without touching the script.
- **Fork & merge**: branch specialists from loss tails, compare experts, and merge via SWA, distillation, Fisher soup, or adapter fusion.
- **Session management**: export/import SQLite-backed bundles with checkpoints and final models for reproducible workflows.
- **Offline-first**: everything runs locally; no external accounts or telemetry.

---

## Troubleshooting

- **Version warning**: upgrade `onthefly-ai` (`pip install -U onthefly-ai`) until the dashboard accepts the trainer.
- **Port busy**: ensure nothing else uses `localhost:47621`, or set `ONTHEFLY_DASHBOARD_PORT` before launching VS Code.
- **No data**: confirm the training script instantiates `Trainer`/`attach_lightning` and that the dashboard tab stays open.

Need help? File an issue at [github.com/KSkert/OnTheFly](https://github.com/KSkert/OnTheFly/issues).

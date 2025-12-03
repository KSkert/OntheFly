# OnTheFly — interactive PyTorch training in VS Code

OnTheFly adds a live training dashboard to VS Code for **local/offline** PyTorch runs. Run your training script the way you already do; while it trains, you can watch per-sample behavior, pause safely, trigger tests/health checks, and export/import sessions for reproducible continuation.

> [!IMPORTANT]
> **Status: Beta.** APIs, UI flows, and file formats may change before v1.0. Expect rough edges—please report issues.

![OnTheFly dashboard](https://github.com/KSkert/onthefly/raw/main/docs/images/onthefly_dashboard.png)

---

## What you get

- **Live visibility**: per-sample loss + metrics + logs + runtime stats
- **Mid-run control**: pause/resume, trigger tests, run health checks
- **Reproducible continuation**: export/import sessions (includes optimizer state)
- **Optional specialization**: fork short-budget specialists on hard regions and merge improvements
- **Fully local**: no accounts, no external services, no cloud storage

---

## Quickstart (VS Code + Python)

1) **Install the extension**: *OnTheFly* (VS Code Marketplace)  
2) Install the Python package in the **same Python environment** you run training with:

```bash
pip install onthefly-ai
```

In VS Code, open the Command Palette (Cmd/Ctrl+Shift+P) and run **OnTheFly: Show Dashboard**.

Run your training script normally:

```bash
python train.py
```

When your run reaches `Trainer.fit(...)` (native) or you call `attach_lightning(...)` (Lightning), the dashboard attaches and begins streaming. You can open the dashboard before or after starting training; the session backfills and keeps streaming.

---

## Supported workflows

### Native PyTorch (`onthefly.Trainer`)
Use OnTheFly’s trainer for any `torch.nn.Module` + standard `DataLoader`s.

```python
from onthefly import Trainer

trainer = Trainer(project="demo", run_name="baseline", max_epochs=3)
trainer.fit(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
)
```

### Lightning (`attach_lightning(...)`)
Keep using `lightning.Trainer`. Call `attach_lightning(...)` to wire your run into the dashboard, then call `trainer.fit(...)` as usual.

```python
from onthefly import attach_lightning

attach_lightning(
    trainer=trainer,
    model=model,
    project="demo",
    run_name="lightning-baseline",
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=model.loss,
)
trainer.fit(model, train_loader, val_loader)
```

---

## Requirements
- VS Code 1.102+
- Python 3.9+
- PyTorch 2.2+
- OS: Linux / macOS / Windows

Optional extras:

```bash
pip install "onthefly-ai[explorer]"   # data explorer / slice export helpers
pip install "onthefly-ai[metrics]"    # GPU metrics (pynvml)
```

---

## How to think about it (workflow)
**Train → Observe → Pause → Focus → Compare → Merge → Export/Resume**

You can use only the “observe + pause + export” pieces, or go deeper with forking/merging when you need it. Forking is optional.

---

## Storage & privacy
- Everything runs locally (offline).
- Sessions are ephemeral until you export them.
- Exporting is how you save a run for resuming later (includes optimizer state).

---

## Troubleshooting

### Dashboard didn’t attach
- Confirm `onthefly-ai` is installed in the same Python environment VS Code uses to run `python train.py`.
- Make sure your script reaches `Trainer.fit(...)` (native) or calls `attach_lightning(...)` before `trainer.fit(...)` (Lightning).
- If you use multiple terminals/interpreters, verify the interpreter shown in VS Code matches the environment you installed into.

### Port issues
- OnTheFly attaches over localhost (default port `47621`).
- If something else is using the port, stop the other run or configure a different port if exposed.

### Lightning gotchas
- Call `attach_lightning(...)` before `trainer.fit(...)`.
- Provide the dataloaders you want visible in the dashboard, plus a callable loss function.

---

## Links
- Repo + full docs + examples: https://github.com/KSkert/onthefly
- Python package (PyPI): https://pypi.org/project/onthefly-ai/
- Issues: https://github.com/KSkert/onthefly/issues
- License: https://github.com/KSkert/onthefly/blob/main/LICENSE.txt

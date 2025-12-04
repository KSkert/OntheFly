# OnTheFly — interactive PyTorch training in VS Code

OnTheFly adds a live training dashboard to VS Code for **local/offline** PyTorch runs. Run your training script the way you already do; while it trains, you can watch per-sample behavior, pause safely, trigger tests/health checks, and export/import sessions for reproducible continuation.

> **Status: Beta.** APIs, UI flows, and file formats may change before v1.0. Expect rough edges and please report issues.

![OnTheFly dashboard](https://github.com/KSkert/onthefly/raw/main/docs/images/onthefly_dashboard.png)

---

## Feature Overview

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
Works with
Native PyTorch
from onthefly import Trainer
Trainer(project="demo", run_name="baseline", max_epochs=3).fit(...)

Lightning
from onthefly import attach_lightning
attach_lightning(trainer=trainer, model=model, project="demo", run_name="baseline", ...)
trainer.fit(...)


➡️ Integrations & advanced workflows (experiment tracking, export/import, specialists):
See the full docs on GitHub: https://github.com/KSkert/OnTheFly.

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

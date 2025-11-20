# OnTheFly

**OnTheFly** lets you steer PyTorch training runs live from VS Code — pause, inspect, fork, merge, and export without ever leaving your IDE or sending data to the cloud.

With OnTheFly, now you can make incremental, interactive progress on model development. When you start a long run, you can now see and do more during training with OnTheFly. Detect subtle faults in your model before training goes haywire. 

Being able to test, save a model, and continue training seamlessly is another big time saver. Many engineers quit training after seeing validation error increase when the model fits the training data, but double descent is a common theme in some setups.

![On-the-Fly overview](./docs/images/onthefly_dashboard.png)

OnTheFly turns that into an interactive loop:

- watch per-sample loss and slices **while you train**
- pause to inspect hard examples or drift
- fork short-budget specialists on rough regions
- merge improvements back into a single exportable model

All of this runs **fully offline** in a local VS Code extension — no accounts, no tokens, no external services.

This shifts training from a fixed, single-pass run into an incremental process that can be revisited and extended as new data arrives. Any previous session can be resumed with its full optimizer state, enabling controlled continuation rather than full retrains. Real-time visibility into failure regions turns continuous improvement into a measurable, iterative workflow rather than a periodic batch job.


> [!IMPORTANT]
> **Project status: Beta.** APIs, UI flows, and file formats may change before v1.0. Expect rough edges and please report issues. Currently, using another trainer at the same time (such as Lightning AI) is not possible, but OnTheFly's endgoal is to support wrapping around other trainers to support a univeral training dashboard.
---

## When should you use OnTheFly?

OnTheFly is aimed at people who:

- train **PyTorch models** (classification, regression, etc.) and want more actionability than TensorBoard/print logs
- are not currently using another trainer, such as Lightning, in your setup. 
- care about **bad slices / drift / outliers** and don't want to wait until the run is over to investigate
- prefer a **local, offline** workflow inside VS Code rather than wiring up cloud dashboards

---

## Getting Started

### Install

```bash
pip install onthefly-ai
```

**Requirements**

* Python ≥ 3.9
* PyTorch ≥ 2.2 (CUDA 12.x optional)
* OS: Linux, macOS, or Windows
* Visual Studio Code

### Open the VS Code dashboard

1. Open VS Code → Command Palette (`Ctrl/Cmd + Shift + P`).
2. Select the **“OnTheFly: Show Dashboard”** command.
3. Run your training script in a terminal (or notebook) so it instantiates `Trainer.fit(...)`.
4. The dashboard status badge turns green once a Trainer connects; metrics/controls stream automatically while the tab is open. An idle Trainer does not mean the dashboard lost connection, it just means your model is not currently training.


### Quickstart

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from onthefly import Trainer

# toy dataset
X = torch.randn(4096, 28 * 28)
y = (X[:, :50].sum(dim=1) > 0).long()
ds = TensorDataset(X, y)
train = DataLoader(ds, batch_size=128, shuffle=True)
val = DataLoader(ds, batch_size=256)
test = DataLoader(ds, batch_size=256)

# tiny model
model = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

trainer = Trainer(
    project="mnist-demo",
    run_name="baseline",
    max_epochs=1,
    do_test_after=True,
    val_every_n_epochs=1,
)

trainer.fit(
    model=model,
    optimizer=opt,
    loss_fn=loss,
    train_loader=train,
    val_loader=val,
    test_loader=test,
)
```

 -Run your script exactly as you normally would (`python train.py` or `python -m training`); if the script instantiates a `Trainer`, the VS Code dashboard will attach automatically (it listens on `localhost:47621`) whenever a dashboard tab is open.
 - Once you run your script, you will see this in your terminal: [onthefly] dashboard connected on tcp://127.0.0.1:47621. This means that even if your dashboard wasn't open yet, you can still open the dashboard from the Command Palette (Cmd+Shift+P) and you will see the live training.
 -Instantiating this Trainer will wrap around your training, just like tools like Lightning and Accelerate do. Now you can perform actions on your model from the dashboard.
 -To exit the Trainer, use Ctrl+C from the terminal to close the dashboard connection.

`Trainer` skips validation unless you pass `val_every_n_epochs`. Set it to the cadence you need (e.g., `1` for every epoch); omit or set `0` to disable validation entirely.


> **Sessions & storage**
>
> Every session is **ephemeral** in storage: when a new session begins, the previous session’s storage is cleaned up.
> Exporting a session is equivalent to saving a session.

---


## Features

**Mid-training control & visibility**
- Start and control training from the VS Code dashboard.
- Stream per-sample loss (optionally grad-norm, margin) with robust quantiles to surface loss tails early.
- Run mid-run health check-ups to detect instability and configuration issues before they cascade.

**Fork, specialize, and merge**
- Fork short-budget specialists from high-loss tails or residual clusters, then route with a lightweight gate.
- Compare experts side-by-side on target slices.
- Merge via SWA, distillation, Fisher Soup, or adapter fusion; view model lineage (parent/children) before committing.

**Data & sessions**
- One-click export of indices/rows for any slice to CSV / Parquet / JSON.
- Ephemeral sessions: storage is cleared when a new session begins; exporting is how you save.
- Portable sessions: exported sessions include the final model and can be re-imported to run tests, reports, or further training.

**Training backend**
- Mirrors training in OnTheFly’s backend for any `torch.nn.Module` (including custom ones) and standard `DataLoader`s.
- Deterministic distributed runs, with a surfaced determinism “health check” for monitoring.
- Seamless continuation of training after tests, whether they were triggered automatically or manually.

---

## Manual human-in-the-loop

Instead of trusting a long run and hoping for the best, you keep tight control over when to pause, inspect, fork, and merge — with deterministic actions and evidence in front of you.

**What you can do**

* **Pause/Resume** at any time to take a clean snapshot.
* **Inspect before acting**: View per-sample loss distributions, export subsets for offline analysis.
* **Approve or edit plan cards** prior to execution.
* **Compare experts** on target slices.
* **Merge on your terms** via SWA / Distill / Fisher Soup / adapter fusion.
* **Run health check-ups mid-run** to validate determinism, gradients, and metrics before committing to longer budgets.
* **Export & import sessions** knowing that exported sessions include the final model and can later be imported, tested, and trained further.

**Typical manual loop**

1. Pause when drift or a weak slice appears.
2. Inspect loss tails, export a subset for a quick notebook check.
3. Fork a short-budget specialist for chosen samples, with desired parameters.
4. Evaluate on target slices; iterate if needed.
5. Merge improvements and resume training.
6. Export the session for traceability, or import a prior session to continue training or generate reports.


## Method (at a glance)

> Train a generalist, detect hard cases, focus on those specialists, learn a gating network, and export a unified MoE for inference. Or, don't use forking at all; simply manage your model development from VS Code without connecting to any externals or cloud.

1. Train a compact **generalist** on all data.
2. **Hard-sample mining** flags high-loss examples online.
3. **Clustering** groups hard samples into candidate regimes.
4. Boost rough areas of the loss curve by forking specialists.
5. Choose a **gating network** to unify experts.
6. **Benchmark fairly** against a monolithic baseline with matched compute.


## License

This project is licensed under the MIT License – see the LICENSE.txt file for details.

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{onthefly2025,
  title        = {OnTheFly: Human-in-the-Loop ML Orchestrator},
  author       = {Luke Skertich},
  year         = {2025},
  url          = {https://github.com/KSkert/onthefly}
}
```

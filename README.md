# OnTheFly

OnTheFly is a **VS Code extension + Python package** for interactive PyTorch training. Run your training script exactly as you do today; while it trains, a VS Code dashboard can:

- stream per-sample loss, metrics, logs, and runtime stats
- pause/resume training and trigger tests
- export/import sessions (with optimizer state) for reproducible resumes
- fork short specialists on rough regions and merge improvements

Everything is local/offline with no accounts or external services. Sessions are ephemeral until you export them, so saving or exporting is how you keep a run around.

> [!IMPORTANT]
> **Project status: Beta.** APIs, UI flows, and file formats may change before v1.0. Expect rough edges and please report issues. Currently, the console only supports PyTorch modules and Lightning trainers, in addition to our native trainer.

![On-the-Fly overview](./docs/images/onthefly_dashboard.png)

OnTheFly turns training into a tight, iterative loop:

- watch per-sample loss and slices **while you train**
- pause to inspect hard examples or drift
- fork short-budget specialists on rough regions
- merge improvements back into a single exportable model

This shifts training from a fixed, single-pass run into an incremental process that can be revisited and extended as new data arrives. Any previous session can be resumed with its full optimizer state, enabling controlled continuation rather than full retrains. Real-time visibility into failure regions turns continuous improvement into a measurable, iterative workflow rather than a periodic batch job.
---

## When should you use OnTheFly?

OnTheFly is aimed at people who:

- train **PyTorch models** (classification, regression, etc.) and want more actionability than TensorBoard/print logs
- are currently using no trainer or a lightning trainer
- prefer a **local, offline** workflow inside VS Code rather than cloud dashboards

---

## Getting Started

### Quickstart (Python + VS Code)

1. Launch **OnTheFly: Show Dashboard** from the Command Palette (`Cmd/Ctrl+Shift+P`).
2. `pip install onthefly-ai` inside the same Python environment as your training script.
3. Run your script exactly as you do today; as soon as it calls `Trainer.fit(...)` or `attach_lightning(...)`, the VS Code dashboard listens on `localhost:47621` and attaches automatically.

The Python backend prints `[onthefly] dashboard connected on tcp://127.0.0.1:47621` when the dashboard is available. You can open the dashboard before or after launching the script—the session backfills metrics and keeps streaming so you can pause, resume, and trigger tests at any time.

### Install

#### 1) VS Code extension
- Install “OnTheFly” from the VS Code Marketplace (or use the `.vsix` while it’s unpublished).

#### 2) Python package

```bash
pip install onthefly-ai
```

Optional extras:
- Data Explorer downloads: `pip install "onthefly-ai[explorer]"`
- GPU metrics (pynvml): `pip install "onthefly-ai[metrics]"`

**Requirements**

* Visual Studio Code 1.102+
* Python ≥ 3.9
* PyTorch ≥ 2.2 (CUDA 12.x optional)
* OS: Linux, macOS, or Windows


### Minimal PyTorch script

<details>
<summary>Show PyTorch example</summary>

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onthefly import Trainer


def build_loaders():
    x = torch.randn(4096, 32)
    y = (x[:, :6].sum(dim=1) > 0).long()
    dataset = TensorDataset(x, y)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [3072, 768, 256], generator=torch.Generator().manual_seed(0)
    )
    return (
        DataLoader(train_ds, batch_size=128, shuffle=True),
        DataLoader(val_ds, batch_size=256),
        DataLoader(test_ds, batch_size=256),
    )


def main():
    train_loader, val_loader, test_loader = build_loaders()

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        project="demo",
        run_name="baseline",
        max_epochs=3,
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


if __name__ == "__main__":
    main()
```

</details>

### Minimal Lightning script

<details>
<summary>Show Lightning example</summary>

```python
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onthefly import attach_lightning


class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


def make_loaders():
    x = torch.randn(4096, 32)
    y = (x[:, :6].sum(dim=1) > 0).long()
    ds = TensorDataset(x, y)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [3072, 1024], generator=torch.Generator().manual_seed(0)
    )
    return (
        DataLoader(train_ds, batch_size=128, shuffle=True),
        DataLoader(val_ds, batch_size=256),
    )


def main():
    train_loader, val_loader = make_loaders()

    model = LitClassifier()
    trainer = L.Trainer(max_epochs=3, log_every_n_steps=1)

    attach_lightning(
        trainer=trainer,
        model=model,
        project="demo",
        run_name="lightning-baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=model.loss,
        do_test_after=True,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
```

</details>

Open the dashboard tab whenever you want visibility, then run your script via `python train.py` (or whatever you already use). As soon as the trainer reaches `Trainer.fit(...)`, the VS Code tab attaches (see Quickstart), begins streaming metrics, and accepts dashboard commands. Close the tab whenever you like; the script keeps running until you stop it with `Ctrl+C`.

`attach_lightning(...)` simply wraps the Lightning trainer so you can keep calling `trainer.fit(...)` exactly as before. Pass the dataloaders you want available in the dashboard plus a callable loss function; everything else is optional.

OnTheFly `Trainer` skips validation unless you pass `val_every_n_epochs`. Set it to the cadence you need (e.g., `1` for every epoch); omit or set `0` to disable validation entirely. When `do_test_after=True`, the automatic evaluation runs once the stop condition hits, and then the trainer keeps streaming so you can continue interacting with the run from VS Code.

> **Sessions & storage**
>
> Starting a new session wipes the previous session’s storage. Export a session whenever you want to keep it around.

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
- OnTheFly owns training process for any `torch.nn.Module` (including custom ones) and standard `DataLoader`s.
- Actions like pauses, forks, resumes, and loaded sessions are deterministic, with a surfaced determinism “health check” for monitoring.
- Supports Automatically Mixed Precision (AMP) training.
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

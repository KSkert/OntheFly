# 🪰 OnTheFly

[![PyPI](https://img.shields.io/pypi/v/onthefly-ai)](https://pypi.org/project/onthefly-ai/0.1.3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#requirements)

OnTheFly is a **VS Code extension + Python package** for interactive PyTorch training. Run your training script exactly as you do today; while it trains, use the dashboard to:

- stream per-sample loss, metrics, logs, and runtime stats
- pause/resume training and trigger tests
- export/import sessions (with optimizer state) for reproducible resumes
- fork short specialists on rough regions and merge improvements

Everything is local/offline with no accounts or external services. Sessions are ephemeral until you export them, so saving or exporting is how you keep a run around.

> [!IMPORTANT]
> **Project status: Beta.** APIs, UI flows, and file formats may change before v1.0. Expect rough edges and please report issues. Currently, the console only supports PyTorch modules and Lightning trainers, in addition to our native trainer. Not yet supported: multi-node DDP, ZeRO/FSDP sharded optimizers, `torch.compile` graphs, or third-party trainer abstractions outside PyTorch/Lightning.

![On-the-Fly overview](./docs/images/onthefly_dashboard.png)

---

## Contents
- [When should you use?](#when-should-you-use)
- [Getting Started](#-getting-up-and-running-in-2-minutes)
  - [Install](#-install)
  - [Quickstart](#-quickstart)
  - [Minimal PyTorch script](#-minimal-pytorch-script)
  - [Minimal Lightning script](#-minimal-lightning-script)
  - [Distributed / DDP runs](#-distributed--ddp-runs)
  - [Gradient Accumulation](#gradient-accumulation)
  - [Validation](#validation)
- [Interactive Training Loop](#-using-the-interactive-training-loop)
- [License](#license)
- [Citation](#citation)

---

## 🕹 OnTheFly turns model development into a tight, iterative loop

As new data arrives, any previous session can be resumed with its full optimizer state, enabling controlled continuation rather than full retrains. Real-time visibility into pain points makes continuous improvement a measurable, iterative experimentation workflow rather than a one-off job.

---

## When should you use?

OnTheFly can make your life easier if you:

- train **PyTorch models** (classification, regression, etc.) and want more actionability than TensorBoard/print logs
- are using either a Lightning trainer or no trainer
- prefer a **local, offline** workflow inside VS Code rather than cloud dashboards
- are training large models on high-dimensional, possibly sensitive data

---

## 🚀 Getting Up and Running in 2 minutes

### 👇 Install

#### 1) VS Code extension
- Install “[OnTheFly](https://marketplace.visualstudio.com/items?itemName=OnTheFly.onthefly)” from the VS Code Marketplace.

#### 2) Python package

```bash
pip install onthefly-ai
```

Optional extras (quote the spec so your shell doesn’t expand the brackets):
- Data Explorer downloads (`pandas>=2.0`, `scikit-learn>=1.3`, `umap-learn>=0.5`): `pip install "onthefly-ai[explorer]"`
- Surfacing GPU metrics requires (`pynvml>=11.5`): `pip install "onthefly-ai[metrics]"`

#### Requirements

* Visual Studio Code 1.102+
* Python ≥ 3.9
* PyTorch ≥ 2.2 (CUDA 12.x optional)
* OS: Linux, macOS, or Windows

### ✅ Quickstart

1. Launch **OnTheFly: Show Dashboard** from the Command Palette (`Cmd/Ctrl+Shift+P`).
2. `pip install onthefly-ai` inside the same Python environment as your training script.
3. Run your script exactly as you do today; as soon as it calls `Trainer.fit(...)` or `attach_lightning(...)`, the VS Code dashboard listens on `localhost:47621` and attaches automatically.



> [!NOTE]
> **Storage:** To support rapid model development and keep the app lightweight, we don't currently store metadata in cloud. That means you are responsible for exporting sessions that you want to save. Starting a new session or resetting the current one will clean out the previous session’s storage.


The Python backend prints `[onthefly] dashboard connected on tcp://127.0.0.1:47621` when the dashboard is available. You can open the dashboard before or after launching the script—the session backfills metrics and keeps streaming so you can pause, resume, and trigger tests at any time.


### 🤏 Minimal PyTorch script

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

### 🤏 Minimal Lightning script

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

### 🌐 Distributed / DDP runs

If you're using `torch.distributed`/`torchrun` on a single machine, launch your script the same way—OnTheFly detects PyTorch `DistributedSampler`s, keeps them in place, and tracks a per-rank cursor so pause/resume/fork scans don't disturb ordering. When training is paused, the dashboard's **Distribution Health** command issues collectives to gather metadata and optional weight hashes from all ranks so you can confirm parity before resuming.

```bash
torchrun --nproc_per_node=4 train.py --run-name ddp-baseline
```

### Gradient Accumulation

The native trainer zeroes gradients and steps every batch by default, so gradient accumulation is up to your `training_step(...)` override. Register a custom step that mirrors the pattern you already use in vanilla PyTorch—divide the loss, delay `optimizer.step()`, and only update the scaler when the accumulation window completes. Whatever dict you return becomes the metrics stream.

<details>
<summary>Show example</summary>

```python
trainer = Trainer(project="demo", run_name="accum", max_epochs=5)

@trainer.training_step
def train_step(batch, state):
    accum = 4  # 4 micro-batches per optimizer step
    idx = train_step.counter % accum
    if idx == 0:
        state["optimizer"].zero_grad(set_to_none=True)
    x, y = batch
    with state["autocast"]:
        logits = state["model"](x.to(state["device"]))
        loss = state["loss_fn"](logits, y.to(state["device"])) / accum
    state["scaler"].scale(loss).backward()
    if idx == accum - 1:
        state["scaler"].step(state["optimizer"])
        state["scaler"].update()
    train_step.counter += 1
    return {"loss": loss.detach()}

train_step.counter = 0
trainer.fit(model=..., optimizer=..., loss_fn=..., train_loader=..., val_loader=...)
```

</details>

Lightning integrations just keep using Lightning's built-in accumulation settings (`accumulate_grad_batches`, manual strategies, etc.) since `attach_lightning(...)` doesn't override the optimizer loop.

### Validation

OnTheFly `Trainer` skips validation unless you pass `val_every_n_epochs`. Set it to the cadence you need (e.g., `1` for every epoch); omit or set `0` to disable validation entirely. When `do_test_after=True`, the automatic evaluation runs once the stop condition hits, and then the trainer keeps streaming so you can continue interacting with the run from VS Code.

### Avoiding Lightning Test Overwrites

When you attach a Lightning trainer, every manual **Run Test** request and every automatic post-train test now executes Lightning’s real `trainer.test`, so your module’s `test_step` and `on_test_end` hooks always fire. Because those hooks own the file paths, they will overwrite previous artifacts unless you append an identifier yourself.

OnTheFly exposes the active test label via the `ONTHEFLY_TEST_LABEL` environment variable (`test_1`, `test_2`, …) before each test run. Reading it is optional, but highly recommended if your hook writes to fixed paths:

```python
import os

class LitGAN(L.LightningModule):
    def on_test_end(self):
        label = os.environ.get("ONTHEFLY_TEST_LABEL", "test_latest")
        csv_path = f"logs/{self.results_prefix}_{label}.csv"
```

If you skip this, your original file destinations remain untouched and each successive test overwrites the last run’s outputs.

---

## 🎛 Using the Interactive Training Loop

**Train → Observe → Pause → Focus → Compare → Merge → Export/Resume**  
Use all of OnTheFly, or just the parts you want (forking is optional).

### 1) Observe training in real time
- Compute **per-sample loss** and export subsets for visibility, on demand.
- Track metrics, logs, and runtime stats from inside VS Code (no cloud, no accounts)

### 2) Intervene safely mid-run
- **Pause/Resume** anytime to take a clean snapshot and avoid “hope-and-pray” long runs
- Trigger **mid-run tests** and **health checks** (determinism, gradients, instability signals) before committing more budget

### 3) Focus on failure regions (optional)
- **Mine hard samples** (loss tails / residual clusters) and fork short-budget specialists
- Export slice indices/rows to **CSV / Parquet / JSON** for notebook debugging or dataset fixes

### 4) Compare specialists and choose what to keep
- Evaluate experts side-by-side on target slices
- Inspect lineage (parent/children) before you commit to a merge

### 5) Merge improvements into one model
- Merge via **SWA, distillation, Fisher Soup, or adapter fusion**
- Resume training from the merged model without restarting the whole run

### 6) Make runs reproducible and portable
- **Export sessions** (model + optimizer state) for controlled continuation instead of full retrains
- **Import sessions** later to run tests, generate reports, or extend training

**Works with:** PyTorch `nn.Module` + standard `DataLoader`s (single GPU or torch.distributed/DDP), and Lightning via `attach_lightning(...)`  
**Also:** AMP support • deterministic actions (pause/fork/resume/load) • fully local/offline

---

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

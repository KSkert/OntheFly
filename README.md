# On the Fly

A **VS Code dashboard** and Python API for orchestrating a hard-sample–guided Mixture-of-Experts (MoE) training workflow. From Python, call `quickstart(...)` to open a session; monitoring and controls live in the dashboard.

![On-the-Fly overview](./docs/images/onthefly-dashboard.png)

---

## Overview

- Start a run from Python; control and inspect it in VS Code.
- Pause/resume safely and snapshot metrics/checkpoints/buffers.
- Fork short-budget specialists from hard samples or slices.
- Merge improvements back using SWA, distillation, Fisher Soup, or adapter fusion.
- Run fully manual or enable an automated planner with safeguards.

---

> [!IMPORTANT]
> **Project status: Beta.** APIs, UI flows, and file formats may change without notice before v1.0.
> Expect rough edges and please report issues and ideas.

---

## Features

- **Mid-training analysis:** Inspect per-sample loss histograms, slice reports, cluster previews, gate loads, and expert ancestry without waiting for full epochs.
- **Fork & merge tools:** Create specialists from residual clusters or loss tails; compare experts side-by-side before merging.
- **Data export:** Download indices or rows for any slice/cluster/high-loss tail to CSV/Parquet/JSON for notebooks, SQL, or BI tools.
- **Planner cards:** Suggested Fork/Merge actions appear as editable “plan cards” (apply / snooze / ignore) with tunable LR/WD, budgets, and recipes.

---

## Modes

### Manual mode (analyst-in-the-loop)

Keep full control with **Automode OFF**. You can pause runs, inspect evidence, export subsets, then decide when to fork or merge.

**What you can do**

- **Pause/Resume** at any time to take a clean snapshot.
- **Inspect before acting**: loss tails, clusters (auto-k), slices, and routing/gate loads.
- **Export subsets** for offline analysis.
- **Approve or edit plan cards** prior to execution.
- **Compare experts** on target slices.
- **Merge on your terms** via SWA / Distill / Fisher-Soup / Adapter-Fuse.

**Typical manual loop**

1. Pause when drift or a weak slice appears.
2. Inspect loss tails, clusters, and slice deltas.
3. Export a subset (e.g., `region=APAC & volatility>p90`) for a quick notebook check.
4. Fork a short-budget specialist.
5. Evaluate on target slices; iterate if needed.
6. Merge improvements and resume training.

### Automode (automated planner)

With **Automode ON**, plans execute immediately with built-in safeguards. You can intervene at any time.

Automode monitors the run and proposes/executes plans such as:

- **Stabilize on instability:** Detect NaNs/Inf, loss/grad spikes, sharpness/GNS alerts; try LR↓, WD↑, gradient clipping, bf16; optional SAM/EMA.
- **Mine hard samples:** Stream per-sample loss (optionally grad norm, margin, small embeddings) with robust quantiles.
- **Cluster residuals (auto-k):** Tag high-loss clusters; fall back to top-q loss tails when appropriate.
- **Specialize with budgets:** Launch short ASHA/Successive-Halving rungs for candidate specialists.
- **Route with a gate:** Train a small router (`switch_softmax` by default; temperature + load-balance aux).
- **Explore on plateaus:** Trend tests (Theil–Sen + Mann–Kendall; Page–Hinkley aware) trigger small HPO sweeps with early stop.
- **Target weak slices:** If you report per-slice validation metrics, persistent underperformers trigger slice-focused forks.
- **Merge on cadence:** Periodically unify via SWA / Distill / Fisher-Soup / Adapter-Fuse; parent remains catch-all.
- **Pacing & limits:** Adaptive cooldowns; cap parallel children; show lightweight diagnostics in the UI.

**Planner card (summary fields)**

```

action: fork | merge
reason: instability_spike | residual_cluster | high_loss_tail | loss_plateau | slice_underperformance
selection: {kind: all | quantile | kmeans | indices, ...}
training_recipe: small set of variants + early stopping
gate_recipe: tiny router (if specializing)
merge_recipe: swa | distill | fisher_soup | adapter_fuse
budget_steps: short (ASHA first rung)
cooldown_steps: adaptive
diagnostics: lightweight numbers for the UI

````

**Selection cheatsheet**

- `all` — global stabilize/explore
- `quantile` — top-q loss tail (e.g., 0.85–1.0)
- `kmeans` — cluster IDs from auto-k residual clustering
- `indices` — exact sample IDs (if provided)

---

## Method (at a glance)

> Train a generalist, mine and cluster hard samples into regimes, train specialists, learn a gating network, and export a unified MoE for inference.

1. Train a compact **generalist** on all data.
2. **Hard-sample mining** flags high-loss examples online.
3. **Clustering** groups hard samples into candidate regimes.
4. Train **per-regime specialists**.
5. Learn a **gating network** to mix experts.
6. **Benchmark fairly** against a monolithic baseline with matched compute.

---

## Installation

```bash
pip install onthefly-ai
````

> If your published import name differs from the PyPI package, adjust the imports below.

---

## Quickstart (runnable)

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from onthefly import quickstart

# toy dataset
X = torch.randn(4096, 28*28)
y = (X[:, :50].sum(dim=1) > 0).long()
ds = TensorDataset(X, y)
train = DataLoader(ds, batch_size=128, shuffle=True)
val = DataLoader(ds, batch_size=256)
test = DataLoader(ds, batch_size=256)

# tiny model
model = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

quickstart(
    project="mnist-demo",
    run_name="baseline",
    model=model,
    optimizer=opt,
    loss_fn=loss,
    train_loader=train,
    val_loader=val,
    test_loader=test,
    max_epochs=1,
    do_test_after=True,
)
```

---

## VS Code dashboard

1. Open VS Code → Command Palette (`Ctrl/Cmd + Shift + P`).
2. Run **“On the Fly: Show Dashboard.”**
3. Select your Python interpreter and training script.
4. Press **▶ Run** to start/monitor training, inspect clusters, and compare experts.

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.2 (CUDA 12.x optional)
* OS: Linux, macOS, or Windows
* VS Code optional (for the dashboard)

---

## Reproducible examples

Example training scripts are provided in `examples/` to illustrate the workflow without hyper-parameter tuning.

---

## Troubleshooting

* **No GPU visible:** set `CUDA_VISIBLE_DEVICES` or install matching CUDA wheels.
* **VS Code command missing:** ensure the extension is enabled; restart VS Code.

---

## License

[Apache-2.0](LICENSE)

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{onthefly2025,
  title        = {On-the-Fly: Hard-Sample–Guided Mixture-of-Experts},
  author       = {Luke Skertich},
  year         = {2025},
  url          = {https://github.com/KSkert/onthefly}
}
```

## On the Fly

The tool is a **VS Code dashboard** that orchestrates a new kind of ML workflow. Your Python side just calls `quickstart(...)` to open a session; **everything else happens in the dashboard**.

![On-the-Fly overview](./docs/images/onthefly-dashboard.png)

* **One-click control:** ▶ Play, ⏸ Pause, ⟲ Refresh to clean everything out. Complementing tools like W&B or MLFlow, this beta version aims to help you achieve better performance as quickly as possible. This doesn't eradicate effort in hypertuning and model choice, but it can be the third factor that pushes the model over the gap. You no longer have to train models all the way to the end to analyze model performance. Performance analysis happens mid-training.
* **Fork & merge without scripts:** Manually fork from loss regions or accept **AutoFork** plan cards; merge specialists back via **SWA / Distill / Fisher-Soup / Adapter-Fuse**.
# Manual Mode (Analyst-in-the-Loop)

Keep full control. With **Automode OFF**, the dashboard turns into a surgical console: you can **pause runs**, **inspect evidence**, **download data subsets** for offline analysis, then **fork and merge** based on what you learn.

## What you can do in Manual Mode

* **Pause/Resume anytime** — freeze the trainer to take a clean snapshot of metrics, checkpoints, and buffers.
* **Inspect before you act** — dive into per-sample loss histograms, slice reports, cluster previews (auto-k), gate loads, and expert ancestry.
* **Download subsets for analysis** — export **indices or rows** for any slice/cluster/high-loss tail to **CSV/Parquet/JSON** for notebooks, SQL, or BI tools.
* **Approve or edit plan cards** — each suggested **Fork/Merge** arrives as a card you can **Apply / Snooze / Ignore** (and tweak LR/WD, budgets, or recipes).
* **Compare experts side-by-side** — validate lifts on your target slices before you merge.
* **Merge on your terms** — choose **SWA / Distill / Fisher-Soup / Adapter-Fuse** with configurable cadence and safeguards.

## Typical analyst loop

1. **Pause** when you spot drift or a stubborn slice.
2. **Inspect** loss tails, clusters, and slice deltas to form a hypothesis.
3. **Download** a subset (e.g., `region=APAC & volatility>p90`) for a quick notebook deep-dive.
4. **Fork** a specialist from that subset (short budget).
5. **Evaluate** on target slices; iterate if needed.
6. **Merge** improvements back into the generalist and **resume** training.

## Automode toggle (rewrite)

* **OFF — Manual Mode:** Plans appear as cards you can **Apply / Snooze / Ignore**. You can **Pause**, inspect views, and **download subsets** before deciding to **Fork** or **Merge**.
* **ON — Automode:** Plans execute immediately with built-in safeguards; you can still intervene at any time.



# Hard-Sample–Guided Mixture-of-Experts

> This is a training workflow that finds hard samples on the fly, clusters them into regimes, trains specialists, and learns a gating network—yielding a unified Mixture-of-Experts (MoE) that adapts to regime shifts.

* Mines hard examples mid-training
* Clusters them into candidate regimes
* Trains per-regime specialists
* Learns a gating network to route inputs
* Exports a single MoE for inference

---

## Why

Most models (from ARIMA to LSTMs) assume one set of dynamics. Real data (macro/markets/climate) shifts: booms vs. recessions, volatility spikes vs. calm, structural breaks. A single global model smooths over structure. **On-the-Fly makes regime specialization explicit.**

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.2 (CUDA 12.x optional)
* OS: Linux, macOS, or Windows
* VS Code optional (for the dashboard)

---

## Install

```bash
pip install onthefly-ai
```

> **Note:** If your published import name differs from the PyPI package, adjust the `import` statements below.

---

## TL;DR quickstart (runnable)

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from onthefly import quickstart

# tiny toy dataset
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

## Dashboard (VS Code)

1. Open VS Code → Command Palette (`Ctrl/Cmd + Shift + P`).
2. Run **“On the Fly: Show Dashboard.”**
3. Select your Python interpreter and training script.
4. Press **▶ Run** to start/monitor training, inspect clusters, and compare experts.

> Prefer headless? Use the CLI:

```bash
onthefly run --script path/to/train.py --python /path/to/python
# or drive with a config file:
onthefly quickstart --config configs/mnist.yaml
```

---

## Method (at a glance)

1. **Generalist** — Train a compact base model on all data.
2. **Hard-sample mining** — Flag examples with high loss online.
3. **Clustering** — Group hard samples (e.g., K-Means) into regimes.
4. **Specialists** — Train one model per cluster.
5. **Gating network** — An LSTM (or transformer) reads recent history and outputs mixture weights.
6. **Fair comparison** — Benchmark vs. a monolithic baseline with matched compute.

> Each component (hard-example mining, curriculum/self-paced learning, dynamic routing) is known; **On-the-Fly** combines them into a practical workflow for regime-switching data.

In the dashboard automode, this will look like:

1. Press play to begin training the baseline.
2. On the dashboard, all of this is customizable, and under the hood, **AutoFork**:

* **Watching for instability.** If it sees NaNs/Inf, sharp loss/grad spikes, or sharpness/GNS alarms, it auto-spawns **stabilize** variants (LR↓, WD↑, grad clip, bf16; optional SAM/EMA) and keeps the best.
* **Mining hard samples.** It streams per-sample loss (optionally grad norm, margin, tiny embeddings) and maintains robust quantiles (t-digest fallback to rolling).
* **Clustering residuals (auto-k).** It clusters the pool (built-in K-Means; optional sklearn KMeans/GMM / HDBSCAN) and tags **dead/high-loss clusters**. If none clearly stand out, it targets the **high-loss tail** (e.g., top-q quantile).
* **Specializing.** For each targeted cluster/tail, it launches a **specialist** child with a short, budgeted schedule (ASHA/Successive-Halving first rung).
* **Routing.** When a child is viable, it trains a tiny **gate** (`switch_softmax` by default; temp & load-balance aux) to mix the parent with its new specialists.
* **Exploring on plateaus.** When trend tests (Theil–Sen + Mann–Kendall; Page–Hinkley aware) say you’re flat, it runs a **small HPO sweep** (optimizer/LR/WD, optional SAM) with early stop.
* **Fixing weak slices.** If you report `val` metrics per slice, persistent underperformers trigger slice-focused forks.
* **Merging on cadence.** It periodically **re-unifies** (or earlier if a child plateaus/improves ≥0.5% vs its first eval), using **SWA / Distill / Fisher-Soup / Adapter-Fuse**. Parent stays as catch-all.
* **Pacing itself.** Cooldowns expand/contract with regime change; parallel children are capped; diagnostics (z-scores, trends, budgets) are shown in the panel.

**What a plan looks like (summarized in the dashboard card):**

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
```

**Selection cheatsheet**

* `all` — global stabilize/explore.
* `quantile` — top-q loss tail (e.g., 0.85–1.0).
* `kmeans` — target cluster IDs from auto-k residual clustering.
* `indices` — exact sample IDs (if you pass them; enables precise forks).

That’s it—the dashboard executes these automatically with **Automode ON**, or shows them as plan suggestion cards with **Automode OFF**.


---

## Reproducible benchmarks

To get an idea of how this methodology works and how effective it can be with—**no hyper-tuning**, 
there are example training scripts providd in examples/.

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
  author       = {Your Name and Coauthors},
  year         = {2025},
  url          = {https://github.com/yourorg/onthefly}
}
```

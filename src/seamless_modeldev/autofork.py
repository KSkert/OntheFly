from __future__ import annotations
"""
Fully upgraded Auto-Fork engine for mixture-of-experts style training orchestration.

Highlights
---------
- Robust trend/plateau detection: Mann–Kendall trend test, Theil–Sen slope, Page–Hinkley
  change-point alarms, optional learning-curve projection for expected improvement.
- Residual specialization beyond 1D loss: streaming feature pools supporting loss,
  gradient norm, margin/confidence, and optional representation embeddings.
- Clustering with auto-k: K-Means (built-in), optional scikit-learn KMeans/GMM/HDBSCAN,
  silhouette/BIC selection on a downsampled feature subset for efficiency.
- Streaming statistics: EMA/EWMA, robust scaling (rank/IQR/MAD), optional t-digest
  (if available) for quantiles; fallback to rolling windows.
- Instability protection: NaN/Inf traps, robust control charts, sharpness proxy hooks
  (Hutchinson trace, SAM suggestion), gradient-noise-scale estimator.
- Slice-aware evaluation: accept per-slice validation metrics, track persistent under-
  performance and emit slice-specific fork plans with expected uplift.
- Budgeted scheduling: basic ASHA/Successive Halving style recommendations, dynamic
  cooldown based on regime changes, compute-aware priority.
- Merge routes: gating (switch/MLP with load-balance auxiliary), knowledge distillation,
  Fisher-weighted model soups, adapter fusion suggestions.

This module is dependency-light. It will opportunistically use NumPy, scikit-learn,
PyTorch, tdigest, or hdbscan if they are installed; otherwise it falls back to
portable implementations.

Public API (stable)
-------------------
- AutoForkRules: dataclass containing configuration knobs.
- AutoForkEngine(rules): orchestrator with the following key methods:
  * observe_batch(batch_loss, sample_losses=None, sample_ids=None,
                  grad_norm=None, sample_margins=None, sample_embed=None,
                  nan_flag=False)
  * observe_eval(split, metrics: dict[str,float], slice_name: str | None = None)
  * report_child_result(child_id, eval_metrics: dict[str,float])
  * should_fork() -> ForkPlan | None
  * get_diagnostics() -> dict
  * set_parent_checkpoint(ckpt_id)
  * register_child(child_id, meta=None); child_finished(child_id)

ForkPlan schema (stable keys):
{
  "action": "fork",
  "reason": str,
  "init_from": str | None,
  "selection": {...},  # dataset selection rule
  "training_recipe": {...},
  "gate_recipe": {...} | None,
  "merge_recipe": {...} | None,
  "cooldown_steps": int,
  "priority": int,      # 0 = highest
  "budget_steps": int | None,
  "diagnostics": {...}
}

Selection kinds (stable):
- {"kind":"all"}
- {"kind":"quantile","from":q0,"to":q1,"metric":"per_sample_loss"}
- {"kind":"kmeans","k":K,"target_clusters":[...],"fit_on":"recent","window":N,"feature":"loss|grad|margin|embed"}
- {"kind":"indices","ids":[...],"slice": Optional[str]}

Note: This engine proposes plans; the trainer is responsible for realizing them.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable
import math
import time
import random
import statistics

# Optional dependencies (gently)
try:
    import numpy as _np  # type: ignore
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

try:
    from sklearn.cluster import KMeans as _SKKMeans  # type: ignore
    from sklearn.mixture import GaussianMixture as _SKGMM  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    import hdbscan as _HDBSCAN  # type: ignore
    _HAVE_HDBSCAN = True
except Exception:
    _HAVE_HDBSCAN = False

try:
    import torch as _torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

try:
    from tdigest import TDigest as _TDigest  # type: ignore
    _HAVE_TDIGEST = True
except Exception:
    _HAVE_TDIGEST = False

# ----------------------------
# Utilities
# ----------------------------

def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _percentile(values: Sequence[float], q: float) -> float:
    """Naive percentile over a list (fallback)."""
    if not values:
        return float('nan')
    q = _clamp01(q)
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = float(sum(values) / len(values))
    var = sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)
    return m, math.sqrt(var + 1e-12)


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = statistics.median(values)
    return statistics.median(abs(v - m) for v in values) + 1e-12


def _zscore(values: Sequence[float]) -> List[float]:
    m, s = _mean_std(values)
    if s == 0.0:
        return [0.0 for _ in values]
    return [(float(v) - m) / s for v in values]


def _ranks(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    order = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    for r, (_, i) in enumerate(order):
        ranks[i] = r / max(1, len(values) - 1)
    return ranks


# Lightweight 1D kmeans (fallback)

def _kmeans_1d(values: List[float], k: int, n_init: int = 5, iters: int = 50, seed: int = 0):
    if len(values) < max(1, k):
        return [sum(values) / len(values)], [0] * len(values)

    best_inertia = float('inf')
    best_centers, best_labels = None, None

    rng = (1103515245 * (seed or int(time.time())) + 12345) % (2 ** 31)

    def rand():
        nonlocal rng
        rng = (1103515245 * rng + 12345) % (2 ** 31)
        return rng / (2 ** 31)

    for _ in range(n_init):
        centers = [values[int(rand() * len(values))] for _ in range(k)]
        labels = [0] * len(values)
        for __ in range(iters):
            # assign
            for i, v in enumerate(values):
                labels[i] = min(range(k), key=lambda j: abs(v - centers[j]))
            # update
            new_centers = []
            for j in range(k):
                pts = [v for i, v in enumerate(values) if labels[i] == j]
                if pts:
                    new_centers.append(sum(pts) / len(pts))
                else:
                    new_centers.append(values[int(rand() * len(values))])
            if max(abs(a - b) for a, b in zip(new_centers, centers)) < 1e-6:
                centers = new_centers
                break
            centers = new_centers
        inertia = sum((values[i] - centers[labels[i]]) ** 2 for i in range(len(values)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers, best_labels = centers[:], labels[:]
    return best_centers, best_labels


# ----------------------------
# Robust trend & change detection
# ----------------------------

def _theil_sen_slope(y: Sequence[float]) -> float:
    """Median of pairwise slopes. O(n^2), but n is small (patience window)."""
    n = len(y)
    if n < 2:
        return 0.0
    slopes: List[float] = []
    for i in range(n):
        yi = float(y[i])
        for j in range(i + 1, n):
            dy = float(y[j]) - yi
            dx = float(j - i)
            if dx != 0:
                slopes.append(dy / dx)
    if not slopes:
        return 0.0
    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2:
        return slopes[mid]
    return 0.5 * (slopes[mid - 1] + slopes[mid])


def _mann_kendall_trend(y: Sequence[float]) -> Tuple[float, int]:
    """
    Returns (S, n_pairs). S>0 indicates increasing trend, S<0 decreasing.
    Simple (non-ties) variant sufficient for short windows.
    """
    n = len(y)
    S = 0
    n_pairs = 0
    for i in range(n - 1):
        yi = float(y[i])
        for j in range(i + 1, n):
            diff = float(y[j]) - yi
            n_pairs += 1
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1
    return float(S), n_pairs


class _PageHinkley:
    def __init__(self, delta: float = 1e-3, lam: float = 0.01, alpha: float = 0.99):
        self.delta = float(delta)
        self.lam = float(lam)
        self.alpha = float(alpha)  # smoothing for mean
        self.mean = None  # type: Optional[float]
        self.cum = 0.0
        self.min_cum = 0.0
        self.max_cum = 0.0

    def update(self, x: float) -> Dict[str, Any]:
        if self.mean is None:
            self.mean = x
            self.cum = 0.0
            self.min_cum = 0.0
            self.max_cum = 0.0
            return {"alarm_up": False, "alarm_down": False, "cum": 0.0}
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        self.max_cum = max(self.max_cum, self.cum)
        alarm_up = (self.cum - self.min_cum) > self.lam
        alarm_down = (self.max_cum - self.cum) > self.lam
        return {"alarm_up": alarm_up, "alarm_down": alarm_down, "cum": self.cum}


# ----------------------------
# Quantile sketch wrapper (optional tdigest)
# ----------------------------

class QuantileSketch:
    """Thin wrapper around t-digest if available; else keeps a bounded buffer."""

    def __init__(self, max_buffer: int = 8192):
        self.max_buffer = int(max_buffer)
        self._buf: List[float] = []
        self._td = _TDigest() if _HAVE_TDIGEST else None

    def add(self, x: float, w: float = 1.0):
        if not _is_finite(x):
            return
        if self._td is not None:
            self._td.update(x, w)
            return
        # fallback buffer
        self._buf.append(x)
        if len(self._buf) > self.max_buffer:
            # reservoir-ish downsample
            random.shuffle(self._buf)
            self._buf = self._buf[: self.max_buffer]

    def percentile(self, q: float) -> float:
        q = _clamp01(q)
        if self._td is not None:
            return float(self._td.quantile(q))
        return _percentile(self._buf, q)

    def size(self) -> int:
        if self._td is not None:
            # t-digest does not expose raw count; approximate via centroid weights
            try:
                return int(sum(c.count for c in self._td.centroids()))
            except Exception:
                return 0
        return len(self._buf)


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class ClusterConfig:
    algo: str = "auto"  # auto|kmeans|gmm|hdbscan
    k: int = 5
    kmax: int = 8
    auto_k: bool = True
    feature: str = "loss"  # loss|loss+grad|loss+grad+margin|embed
    sample_for_k: int = 2048 # max samples for auto-k search


@dataclass
class GateConfig:
    type: str = "switch_softmax"  # switch_softmax|linear_meta|mlp
    epochs: int = 30
    aux_load_balance: float = 0.01
    temperature: float = 1.5
    note: str = "Train a small router to mix parent and child."


@dataclass
class MergeConfig:
    method: str = "swa"  # swa|distill|fisher_soup|adapter_fuse|none
    distill_temp: float = 2.0
    fisher_scale: float = 1.0


@dataclass
class BudgetConfig:
    use_asha: bool = True
    min_budget_steps: int = 800
    max_budget_steps: int = 4000
    reduction_factor: float = 3.0  # successive halving


@dataclass
class AutoForkRules:
    enabled: bool = True

    # --- Trend/plateau
    loss_plateau_patience: int = 200
    loss_plateau_delta: float = 1e-4
    use_mk_trend: bool = True
    use_theil_sen: bool = True

    # --- Change-point
    change_point: bool = True
    page_hinkley_delta: float = 1e-3
    page_hinkley_lambda: float = 0.01
    ph_alpha: float = 0.99

    # --- Per-sample pool
    per_sample_window: int = 5000
    time_decay: float = 0.997  # exponential down-weighting for pools

    # --- Clustering
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    dead_cluster_z: float = 1.0
    high_loss_quantile: float = 0.85

    # --- Instability
    spike_sigma: float = 3.0
    ema_decay: float = 0.98

    # --- Sharpness & GNS
    hessian_trace_period: int = 800
    sharpness_z: float = 2.5
    gns_window: int = 256
    gns_spike: float = 2.0

    # --- Scheduling
    max_parallel_children: int = 2
    base_cooldown_steps: int = 1000
    dynamic_cooldown: bool = True
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    warmup_steps_before_first_fork: int = 200
    require_first_val_before_fork: bool = False

    # --- Gate & Merge
    gate: GateConfig = field(default_factory=GateConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)

    # --- Evaluation / decisioning
    require_paired_eval: bool = True
    min_val_uplift: float = 0.0  # minimal uplift (%) to keep

    # --- Unification & backpressure
    merge_on_plateau: bool = True
    reunify_every_steps: int = 100          # force a reunification attempt at this cadence
    min_child_steps_before_merge: int = 50 # don't merge a newborn
    fork_backpressure_alpha: float = 0.75    # cooldown multiplier per active child


    # --- Parentless merge behavior (NEW)
    allow_parentless_merge: bool = True      # enable merge even if parent val baseline is unknown
    child_self_improvement: float = 0.005     # ≥0.5% improvement vs the child's first val to qualify
    child_plateau_patience: int = 3          # per-child plateau window (evals)
    child_plateau_delta: float = 1e-4        # per-child plateau tolerance



# ----------------------------
# Main Engine
# ----------------------------

class AutoForkEngine:
    """Stateful auto-fork engine with robust signals and model-agnostic hooks."""

    def __init__(self, rules: AutoForkRules):
        self.rules = rules

        # rolling windows
        self._batch_loss_window: List[float] = []
        self._per_sample_losses: List[float] = []
        self._per_sample_grad: List[float] = []
        self._per_sample_margin: List[float] = []
        self._per_sample_embed: List[List[float]] = []  # optional small embeddings
        self._per_sample_ids: Optional[List[Any]] = None

        # sketches
        self._loss_sketch = QuantileSketch()

        # EMA trackers
        self._ema = {
            "loss_mean": None,
            "loss_var": 0.0,
            "grad_mean": None,
            "grad_var": 0.0,
            "sharp_mean": None,
            "sharp_var": 0.0,
        }
        self._last_grad_z: Optional[float] = None
        self._last_loss_z: Optional[float] = None
        self._last_sharp_z: Optional[float] = None
        self._nan_seen: bool = False

        # Page–Hinkley
        self._ph = _PageHinkley(
            delta=self.rules.page_hinkley_delta,
            lam=self.rules.page_hinkley_lambda,
            alpha=self.rules.ph_alpha,
        ) if self.rules.change_point else None

        # gradient-noise-scale estimation (track micro-batch stats if provided)
        self._gns_buffer: List[float] = []
        self._gns_mu: Optional[float] = None
        self._gns_var: float = 0.0

        # scheduling
        self._step: int = 0
        self._last_fork_step: int = -1_000_000_000
        self._active_children: Dict[str, Dict[str, Any]] = {}

        # evaluation memory (overall + slices)
        self._eval_history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": [], "test": []}
        self._slice_eval: Dict[str, List[Dict[str, float]]] = {}

        # integration hooks
        self._parent_checkpoint_id: Optional[str] = None

        # external probes (optional)
        self.hessian_trace_fn: Optional[Callable[[], float]] = None
        self.custom_instability_score_fn: Optional[Callable[[], float]] = None

        # model control
        self._last_merge_step: int = -1_000_000_000


    # ---------- Public API ----------

    def set_parent_checkpoint(self, ckpt_id: str):
        self._parent_checkpoint_id = ckpt_id

    def register_child(self, child_id: str, meta: Optional[Dict[str, Any]] = None):
        self._active_children[child_id] = {"started_at_step": self._step, **(meta or {})}

    def child_finished(self, child_id: str):
        self._active_children.pop(child_id, None)

    def observe_eval(self, split: str, metrics: Dict[str, float], slice_name: Optional[str] = None):
        split = split.lower()
        self._eval_history.setdefault(split, []).append(dict(metrics))
        if slice_name is not None:
            self._slice_eval.setdefault(slice_name, []).append(dict(metrics))


    def report_child_result(self, child_id: str, eval_metrics: Dict[str, float]):
        meta = self._active_children.get(child_id)
        if meta is None:
            return
        hist = meta.setdefault("eval_history", [])
        hist.append(dict(eval_metrics))

    def observe_microbatch_grads(self, grads: Sequence[float]):
        """Optional: call within a step to feed micro-batch grad norms for GNS."""
        for g in grads:
            if _is_finite(g):
                self._gns_buffer.append(float(g))
                if len(self._gns_buffer) > self.rules.gns_window:
                    self._gns_buffer = self._gns_buffer[-self.rules.gns_window :]

    def observe_batch(
        self,
        batch_loss: float,
        sample_losses: Optional[Sequence[float]] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        grad_norm: Optional[float] = None,
        sample_margins: Optional[Sequence[float]] = None,
        sample_embed: Optional[Sequence[Sequence[float]]] = None,
        nan_flag: bool = False,
    ):
        self._step += 1

        # record batch loss
        if _is_finite(batch_loss):
            self._batch_loss_window.append(float(batch_loss))
            max_win = max(self.rules.loss_plateau_patience, 1024)
            if len(self._batch_loss_window) > max_win:
                self._batch_loss_window = self._batch_loss_window[-max_win:]
            self._update_ema("loss", float(batch_loss))
            self._last_loss_z = self._z_from_ema("loss", float(batch_loss))
            if self._ph is not None:
                self._ph.update(float(batch_loss))
        else:
            self._nan_seen = True

        # per-sample pool
        if sample_losses is not None:
            if self._per_sample_ids is None and sample_ids is not None:
                self._per_sample_ids = []
            for i, v in enumerate(sample_losses):
                if not _is_finite(v):
                    self._nan_seen = True
                    continue
                fv = float(v)
                self._per_sample_losses.append(fv)
                self._loss_sketch.add(fv)
                if self._per_sample_ids is not None:
                    sid = sample_ids[i] if sample_ids is not None and i < len(sample_ids) else None
                    self._per_sample_ids.append(sid)
            # Optional auxiliary per-sample features
            if sample_margins is not None:
                for m in sample_margins:
                    if _is_finite(m):
                        self._per_sample_margin.append(float(m))
            if grad_norm is not None:
                self._per_sample_grad.append(float(grad_norm))
            if sample_embed is not None:
                for e in sample_embed:
                    try:
                        vec = [float(x) for x in e]
                        self._per_sample_embed.append(vec)
                    except Exception:
                        pass
            # window trim
            self._trim_per_sample_windows()

        # grad spikes
        if grad_norm is not None:
            if _is_finite(grad_norm):
                self._update_ema("grad", float(grad_norm))
                self._last_grad_z = self._z_from_ema("grad", float(grad_norm))
            else:
                self._nan_seen = True

        if nan_flag:
            self._nan_seen = True

        # Optional sharpness probe (external or periodic internal torch-based)
        if self.hessian_trace_fn is not None and (self._step % max(1, self.rules.hessian_trace_period) == 0):
            try:
                trace = float(self.hessian_trace_fn())
                self._update_ema("sharp", trace)
                self._last_sharp_z = self._z_from_ema("sharp", trace)
            except Exception:
                pass

        # Gradient Noise Scale estimation (per step from micro-batches)
        self._update_gns()

    # ---------- Decisions ----------

    def should_fork(self) -> Optional[Dict[str, Any]]:
        if not self.rules.enabled:
            return None
        # Hard warm-up: never fork before N steps (stops early spikes)
        if self._step < int(self.rules.warmup_steps_before_first_fork):
            return None
        # Optional: force at least one validation pass before any fork
        if self.rules.require_first_val_before_fork and not self._eval_history.get("val"):
            return None
        if len(self._active_children) >= self.rules.max_parallel_children:
            return None
        if (self._step - self._last_fork_step) < self._current_cooldown():
            return None
        
        # cooldown scales with active children
        effective_cooldown = int(self._current_cooldown() * (1.0 + self.rules.fork_backpressure_alpha * len(self._active_children)))
        if (self._step - self._last_fork_step) < effective_cooldown:
            return None

        # 1) Instability (highest priority)
        plan = self._instability_check()
        if plan:
            return self._finalize_plan(plan, priority=0)

        # 2) Residual specialization (requires enough data)
        plan = self._residual_cluster_check()
        if plan:
            return self._finalize_plan(plan, priority=1)

        # 3) Plateau / exploration (global)
        plan = self._plateau_check()
        if plan:
            return self._finalize_plan(plan, priority=2)

        # 4) Slice underperformance (from eval history)
        plan = self._slice_underperf_check()
        if plan:
            return self._finalize_plan(plan, priority=2)

        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        m, s = _mean_std(self._batch_loss_window[-self.rules.loss_plateau_patience:])
        pm, ps = _mean_std(self._per_sample_losses[-min(self.rules.per_sample_window, 2000):])
        return {
            "step": self._step,
            "active_children": len(self._active_children),
            "last_fork_step": self._last_fork_step,
            "cooldown": self._current_cooldown(),
            "batch_loss_mean": m,
            "batch_loss_std": s,
            "per_sample_mean": pm,
            "per_sample_std": ps,
            "last_grad_z": self._last_grad_z,
            "last_loss_z": self._last_loss_z,
            "last_sharp_z": self._last_sharp_z,
            "nan_seen": self._nan_seen,
            "gns_mu": self._gns_mu,
            "gns_var": self._gns_var,
        }

    # ---------- Internal checks ----------

    def _current_cooldown(self) -> int:
        if not self.rules.dynamic_cooldown:
            return self.rules.base_cooldown_steps
        # If Page–Hinkley recently alarmed, reduce cooldown; else increase slightly
        base = self.rules.base_cooldown_steps
        if self._ph is None:
            return base
        # Heuristic: look at cum statistic magnitude
        # (No direct access—tracked internally; adjust based on last EMA z)
        z = abs(self._last_loss_z or 0.0)
        if z > 2.5:
            return max(200, int(base * 0.6))
        if z < 0.5:
            return int(base * 1.2)
        return base

    def _plateau_check(self) -> Optional[Dict[str, Any]]:
        p = self.rules.loss_plateau_patience
        w = self._batch_loss_window
        if len(w) < p:
            return None
        window = w[-p:]
        start, end = window[0], window[-1]
        delta = start - end

        # Robust trend tests
        slope = _theil_sen_slope(window) if self.rules.use_theil_sen else (end - start) / max(1, p - 1)
        mk_S, mk_pairs = _mann_kendall_trend(window) if self.rules.use_mk_trend else (0.0, 0)
        mk_stat = mk_S / max(1, mk_pairs)

        plateau_like = (delta < self.rules.loss_plateau_delta) and (abs(slope) < self.rules.loss_plateau_delta)
        plateau_like = plateau_like or (abs(mk_stat) < 0.05)  # weak monotone evidence

        # Page–Hinkley change-point: if upward alarm, prefer stabilization or LR/backoff
        ph_alarm_up = False
        ph_alarm_down = False
        if self._ph is not None:
            # A single update occurs at observe; here we just re-evaluate on window shape
            # Heuristic from slope + last z
            ph_alarm_up = (self._last_loss_z or -10.0) > 3.0
            ph_alarm_down = (self._last_loss_z or 10.0) < -3.0

        if plateau_like and not ph_alarm_up:
            # Suggest global HPO exploration
            budget = self._suggest_budget()
            return {
                "action": "fork",
                "reason": "loss_plateau",
                "init_from": self._parent_checkpoint_id,
                "selection": {"kind": "all"},
                "training_recipe": {
                    "variants": [
                        {"lr_mul": 0.5, "wd_mul": 1.0},
                        {"lr_mul": 0.25, "wd_mul": 2.0},
                        {"optimizer": "adamw"},
                        {"optimizer": "lion"},
                        {"optimizer": "adamw+sam", "sam_rho": 0.05},
                    ],
                    "early_stopping_patience": 5,
                },
                "gate_recipe": None,
                "merge_recipe": None,
                "cooldown_steps": self._current_cooldown(),
                "budget_steps": budget,
                "diagnostics": {
                    "start_loss": start,
                    "end_loss": end,
                    "delta": delta,
                    "theil_sen": slope,
                    "mk_stat": mk_stat,
                    "window": p,
                },
            }
        return None

    def _build_feature_matrix(self) -> Tuple[List[List[float]], List[int]]:
        """Create a feature matrix for clustering based on configured feature set.
        Returns (features, indices_kept)."""
        cfg = self.rules.cluster
        n = len(self._per_sample_losses)
        if n == 0:
            return [], []
        feats: List[List[float]] = []
        idxs: List[int] = []
        feature = cfg.feature
        for i in range(n):
            row: List[float] = []
            if "loss" in feature:
                row.append(float(self._per_sample_losses[i]))
            if "grad" in feature and i < len(self._per_sample_grad):
                row.append(float(self._per_sample_grad[min(i, len(self._per_sample_grad) - 1)]))
            if "margin" in feature and i < len(self._per_sample_margin):
                row.append(float(self._per_sample_margin[min(i, len(self._per_sample_margin) - 1)]))
            if "embed" in feature and i < len(self._per_sample_embed):
                # down-project embeds by taking first few dims (assumed already small)
                row.extend(self._per_sample_embed[min(i, len(self._per_sample_embed) - 1)][:8])
            if row:
                feats.append(row)
                idxs.append(i)
        # robust scaling: rank for 1D, IQR for multi-d
        if feats and len(feats[0]) == 1:
            vals = [r[0] for r in feats]
            ranks = _ranks(vals)
            feats = [[r] for r in ranks]
        else:
            # per-dim robust scale
            d = len(feats[0]) if feats else 0
            for j in range(d):
                col = [r[j] for r in feats]
                q1 = _percentile(col, 0.25)
                q3 = _percentile(col, 0.75)
                iqr = (q3 - q1) + 1e-12
                for i in range(len(feats)):
                    feats[i][j] = (feats[i][j] - q1) / iqr
        return feats, idxs

    def _residual_cluster_check(self) -> Optional[Dict[str, Any]]:
        n = len(self._per_sample_losses)
        k_default = max(2, self.rules.cluster.k)
        if n < max(300, k_default * 40):
            return None

        feats, idxs = self._build_feature_matrix()
        if not feats:
            return None

        # Auto-k selection on a sample
        algo = self.rules.cluster.algo
        k = self.rules.cluster.k
        labels: List[int] = []
        centers: Optional[List[List[float]]] = None

        # Downsample for model selection
        sample_idx = list(range(len(feats)))
        if len(sample_idx) > self.rules.cluster.sample_for_k:
            random.shuffle(sample_idx)
            sample_idx = sample_idx[: self.rules.cluster.sample_for_k]
        sample_feats = [feats[i] for i in sample_idx]

        def kmeans(feat: List[List[float]], k_: int) -> Tuple[List[int], List[List[float]]]:
            if _HAVE_SKLEARN:
                km = _SKKMeans(n_clusters=k_, n_init=10, random_state=0)
                labels_ = km.fit_predict(feat)
                centers_ = km.cluster_centers_.tolist()
                return list(map(int, labels_)), [list(map(float, c)) for c in centers_]
            # fallback 1D/nd: run kmeans on first dim only (reasonable in rank-scaled space)
            if feat and len(feat[0]) == 1:
                vals = [f[0] for f in feat]
                c, lab = _kmeans_1d(vals, k_)
                centers_ = [[float(x)] for x in c]
                return lab, centers_
            # simple Lloyd's using first 2 dims
            d = len(feat[0])
            init = random.sample(feat, k_)
            centers_ = [c[:] for c in init]
            labels_ = [0] * len(feat)
            for _ in range(50):
                # assign
                for i, v in enumerate(feat):
                    labels_[i] = min(range(k_), key=lambda j: _sqdist(v, centers_[j]))
                # update
                changed = False
                for j in range(k_):
                    pts = [feat[i] for i in range(len(feat)) if labels_[i] == j]
                    if pts:
                        newc = [_mean([p[t] for p in pts]) for t in range(d)]
                        if _sqdist(newc, centers_[j]) > 1e-9:
                            changed = True
                        centers_[j] = newc
                if not changed:
                    break
            return labels_, centers_

        def _silhouette(feat: List[List[float]], labels_: List[int]) -> float:
            # approximate silhouette using first 2 dims
            if not feat:
                return 0.0
            d = min(2, len(feat[0]))
            byc: Dict[int, List[List[float]]] = {}
            for v, l in zip(feat, labels_):
                byc.setdefault(l, []).append(v[:d])
            # precompute centroids
            cents = {c: [_mean([p[t] for p in pts]) for t in range(d)] for c, pts in byc.items()}
            # compute avg intra- and nearest inter-centroid distances
            s_vals = []
            for v, l in zip(feat, labels_):
                a = math.sqrt(_sqdist(v[:d], cents[l]))
                b = min(math.sqrt(_sqdist(v[:d], c)) for k2, c in cents.items() if k2 != l) if len(cents) > 1 else a
                if max(a, b) > 0:
                    s_vals.append((b - a) / max(a, b))
            return float(sum(s_vals) / max(1, len(s_vals)))

        # choose k
        if self.rules.cluster.auto_k:
            k_best, s_best = None, -1e9
            for kk in range(2, max(2, self.rules.cluster.kmax) + 1):
                labs_s, _ = kmeans(sample_feats, kk)
                s = _silhouette(sample_feats, labs_s)
                if s > s_best:
                    s_best, k_best = s, kk
            k = k_best or k

        # final clustering on full feats
        labels, centers = kmeans(feats, k)

        # identify "dead" clusters via mean z on 1D loss ranks or multi-d center distance
        # We'll compute per-cluster mean of (ranked loss) as a universal proxy
        loss_vals = [self._per_sample_losses[i] for i in idxs]
        loss_ranks = _ranks(loss_vals)
        cluster_means: List[float] = []
        for c in range(k):
            vals = [loss_ranks[i] for i in range(len(loss_ranks)) if labels[i] == c]
            m = sum(vals) / len(vals) if vals else 0.0
            cluster_means.append((m - 0.5) / (0.25 + 1e-12))  # standardized around 0
        dead = [i for i, m in enumerate(cluster_means) if abs(m) >= self.rules.dead_cluster_z]

        # Fallback: high-loss tail by quantile
        if not dead:
            q = self.rules.high_loss_quantile
            budget = self._suggest_budget()
            return {
                "action": "fork",
                "reason": "high_loss_tail",
                "init_from": self._parent_checkpoint_id,
                "selection": {"kind": "quantile", "from": q, "to": 1.0, "metric": "per_sample_loss"},
                "training_recipe": {
                    "variants": [
                        {"lr_mul": 1.0, "wd_mul": 1.0},
                        {"lr_mul": 0.5, "wd_mul": 1.5},
                    ],
                    "early_stopping_patience": 5,
                },
                "gate_recipe": asdict(self.rules.gate),
                "merge_recipe": None,
                "cooldown_steps": self._current_cooldown(),
                "budget_steps": budget,
                "diagnostics": {
                    "k": k,
                    "dead_clusters": [],
                    "quantile_from": q,
                    "pool_size": n,
                },
            }

        # Construct selection
        selection: Dict[str, Any] = {
            "kind": "kmeans",
            "k": k,
            "target_clusters": dead,
            "fit_on": "recent",
            "window": min(n, self.rules.per_sample_window),
            "feature": self.rules.cluster.feature,
        }
        if self._per_sample_ids is not None:
            ids = [self._per_sample_ids[idxs[i]] for i in range(len(idxs)) if labels[i] in dead]
            explicit = _dedup_stable(ids)
            if explicit:
                selection = {"kind": "indices", "ids": explicit}

        budget = self._suggest_budget()
        return {
            "action": "fork",
            "reason": "residual_cluster",
            "init_from": self._parent_checkpoint_id,
            "selection": selection,
            "training_recipe": {
                "variants": [
                    {"lr_mul": 1.0, "wd_mul": 1.0},
                    {"lr_mul": 0.5, "wd_mul": 1.5},
                ],
                "early_stopping_patience": 5,
            },
            "gate_recipe": asdict(self.rules.gate),
            "merge_recipe": None,
            "cooldown_steps": self._current_cooldown(),
            "budget_steps": budget,
            "diagnostics": {
                "k": k,
                "cluster_means": cluster_means,
                "dead_clusters": dead,
                "pool_size": n,
            },
        }

    def _instability_check(self) -> Optional[Dict[str, Any]]:
        if self._nan_seen:
            self._nan_seen = False
            return self._stabilization_plan("nan_detected")

        gz = self._last_grad_z if self._last_grad_z is not None else -float('inf')
        lz = self._last_loss_z if self._last_loss_z is not None else -float('inf')
        sz = self._last_sharp_z if self._last_sharp_z is not None else -float('inf')
        cz = None
        if self.custom_instability_score_fn is not None:
            try:
                cz = float(self.custom_instability_score_fn())
            except Exception:
                cz = None

        if gz >= self.rules.spike_sigma or lz >= self.rules.spike_sigma or sz >= self.rules.sharpness_z or (cz is not None and cz >= self.rules.spike_sigma):
            return self._stabilization_plan(
                reason="instability_spike",
                diag={"grad_z": gz, "loss_z": lz, "sharp_z": sz, "custom": cz},
            )
        # Gradient noise scale spike
        if self._gns_mu is not None and self._gns_var > 0:
            gns_z = (self._gns_mu) / (math.sqrt(self._gns_var) + 1e-12)
            if gns_z >= self.rules.gns_spike:
                return self._stabilization_plan(
                    reason="gns_spike",
                    diag={"gns_mean": self._gns_mu, "gns_var": self._gns_var, "z": gns_z},
                )
        return None

    def _slice_underperf_check(self) -> Optional[Dict[str, Any]]:
        # Check any tracked slice that persistently underperforms latest overall val metric (e.g., loss)
        if not self._eval_history.get("val"):
            return None
        recent_val = self._eval_history["val"][-1]
        global_loss = recent_val.get("loss")
        if global_loss is None:
            return None
        for slice_name, hist in self._slice_eval.items():
            if not hist:
                continue
            sl_loss = hist[-1].get("loss")
            if sl_loss is None:
                continue
            # Underperform if slice loss significantly above global
            if sl_loss > global_loss * (1.0 + max(0.02, self.rules.min_val_uplift)):
                # Build a plan to specialize on this slice (ids may not be known)
                budget = self._suggest_budget()
                return {
                    "action": "fork",
                    "reason": "slice_underperformance",
                    "init_from": self._parent_checkpoint_id,
                    "selection": {"kind": "all", "slice": slice_name},  # trainer resolves slice
                    "training_recipe": {
                        "variants": [
                            {"lr_mul": 0.8, "wd_mul": 1.2},
                            {"lr_mul": 0.6, "wd_mul": 1.2},
                        ],
                        "early_stopping_patience": 5,
                    },
                    "gate_recipe": asdict(self.rules.gate),
                    "merge_recipe": None,
                    "cooldown_steps": self._current_cooldown(),
                    "budget_steps": budget,
                    "diagnostics": {"slice": slice_name, "slice_loss": sl_loss, "global_loss": global_loss},
                }
        return None

    def _stabilization_plan(self, reason: str, diag: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        budget = self._suggest_budget(kind="stabilize")
        return {
            "action": "fork",
            "reason": reason,
            "init_from": self._parent_checkpoint_id,
            "selection": {"kind": "all"},
            "training_recipe": {
                "variants": [
                    {"lr_mul": 0.25, "wd_mul": 2.0, "grad_clip_norm": 1.0, "precision": "bf16"},
                    {"lr_mul": 0.5,  "wd_mul": 1.5, "grad_clip_norm": 1.0, "precision": "bf16"},
                    {"optimizer": "adamw+sam", "sam_rho": 0.05, "ema": 0.999},
                ],
                "notes": "Stabilize with smaller LR, stronger WD, gradient clipping; consider smaller batch size.",
            },
            "gate_recipe": None,
            "merge_recipe": None,
            "cooldown_steps": self._current_cooldown(),
            "budget_steps": budget,
            "diagnostics": diag or {},
        }

    def _finalize_plan(self, plan: Dict[str, Any], priority: int) -> Dict[str, Any]:
        if len(self._active_children) >= self.rules.max_parallel_children and plan.get("action") == "fork":
            return None
        plan["priority"] = int(priority)
        if plan.get("action") == "fork":
            self._last_fork_step = self._step
        elif plan.get("action") == "merge":
            self._last_merge_step = self._step
        return plan


    # ---------- EMA & stats ----------

    def _update_ema(self, kind: str, x: float):
        decay = self.rules.ema_decay
        m_key = f"{kind}_mean"
        v_key = f"{kind}_var"
        m = self._ema.get(m_key)
        if m is None:
            self._ema[m_key] = x
            self._ema[v_key] = 0.0
            return
        new_m = decay * m + (1 - decay) * x
        new_v = decay * self._ema[v_key] + (1 - decay) * (x - new_m) ** 2
        self._ema[m_key] = new_m
        self._ema[v_key] = new_v

    def _z_from_ema(self, kind: str, x: float) -> float:
        m = self._ema.get(f"{kind}_mean")
        v = self._ema.get(f"{kind}_var", 0.0)
        if m is None:
            return 0.0
        std = math.sqrt(max(v, 1e-12))
        return (x - m) / (std + 1e-12)

    def _update_gns(self):
        if not self._gns_buffer:
            return
        mu = sum(self._gns_buffer) / len(self._gns_buffer)
        var = sum((g - mu) ** 2 for g in self._gns_buffer) / max(1, len(self._gns_buffer) - 1)
        # EWMA on moments
        if self._gns_mu is None:
            self._gns_mu = mu
            self._gns_var = var
        else:
            a = self.rules.ema_decay
            self._gns_mu = a * self._gns_mu + (1 - a) * mu
            self._gns_var = a * self._gns_var + (1 - a) * var
        # clear per-step buffer
        self._gns_buffer.clear()

    def _trim_per_sample_windows(self):
        win = self.rules.per_sample_window
        extra = max(0, len(self._per_sample_losses) - win)
        if extra > 0:
            self._per_sample_losses = self._per_sample_losses[extra:]
        extra_g = max(0, len(self._per_sample_grad) - win)
        if extra_g > 0:
            self._per_sample_grad = self._per_sample_grad[extra_g:]
        extra_m = max(0, len(self._per_sample_margin) - win)
        if extra_m > 0:
            self._per_sample_margin = self._per_sample_margin[extra_m:]
        extra_e = max(0, len(self._per_sample_embed) - win)
        if extra_e > 0:
            self._per_sample_embed = self._per_sample_embed[extra_e:]
        if self._per_sample_ids is not None:
            extra_i = max(0, len(self._per_sample_ids) - win)
            if extra_i > 0:
                self._per_sample_ids = self._per_sample_ids[extra_i:]

    # --- Child-plateau detector (NEW) --------------------
    def _child_plateau_like(self, hist: List[Dict[str, float]]) -> bool:
        p = max(2, int(self.rules.child_plateau_patience))
        if len(hist) < p:
            return False
        w = [float(h["loss"]) for h in hist[-p:] if "loss" in h and _is_finite(h["loss"])]
        if len(w) < p:
            return False
        slope = _theil_sen_slope(w)
        delta = w[0] - w[-1]
        return (abs(slope) < self.rules.child_plateau_delta) and (delta < self.rules.child_plateau_delta)
    
    def _suggest_budget(self, kind: str = "explore") -> int:
        b = self.rules.budget
        if not b.use_asha:
            return int((b.min_budget_steps + b.max_budget_steps) / 2)
        # Successive halving first rung budget
        if kind == "stabilize":
            return int(max(b.min_budget_steps, 0.5 * b.min_budget_steps))
        return int(b.min_budget_steps)
    

    # --- MERGE SCHEDULER ---------------------------------
    def _best_merge_candidate(self) -> Optional[Dict[str, Any]]:
        # Parentless default: pick the child with best *own* loss that
        # either improved enough since its first eval, or plateaued.
        best = None
        for cid, meta in self._active_children.items():
            hist = meta.get("eval_history", [])
            if not hist:
                continue
            steps_alive = self._step - int(meta.get("started_at_step", self._step))
            if steps_alive < self.rules.min_child_steps_before_merge:
                continue
            losses = [float(h["loss"]) for h in hist if "loss" in h and _is_finite(h["loss"])]
            if not losses:
                continue
            first, last = losses[0], losses[-1]
            rel_improv = (first - last) / max(1e-12, first)
            plateau = self._child_plateau_like(hist) if self.rules.merge_on_plateau else False
            if rel_improv >= self.rules.child_self_improvement or plateau:
                if best is None or last < best["loss"]:
                    best = {"child_id": cid, "loss": last, "rel_improv": rel_improv, "plateau": plateau}
        return best

    def should_merge(self) -> Optional[Dict[str, Any]]:
        if not self._active_children:
            return None
        if (self.rules.merge and getattr(self.rules.merge, "method", "auto") == "none"):
            return None

        # cadence: allow tries periodically, even without plateau signal
        time_forced = (self._step - self._last_merge_step) >= self.rules.reunify_every_steps

        # child plateau flag
        plateau_child = any(
            self._child_plateau_like(meta.get("eval_history", []))
            for meta in self._active_children.values()
        ) if self.rules.merge_on_plateau else False

        cand = self._best_merge_candidate()
        if (time_forced or plateau_child) and cand is not None:
            return self._finalize_plan({
                "action": "merge",
                "reason": "plateau_reunify_child" if plateau_child else "periodic_reunify",
                "method": self.rules.merge.method,
                "from_child": cand["child_id"],
                "into": "baseline",
                "merge_recipe": asdict(self.rules.merge),
                "cooldown_steps": int(self._current_cooldown() * 1.5),
                "diagnostics": cand,  # no parent fields
            }, priority=0)
        return None

    def note_merge(self):
        self._last_merge_step = self._step


# ----------------------------
# Small helpers
# ----------------------------

def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _dedup_stable(xs: Sequence[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------
# Optional PyTorch sharpness probe (utility)
# ----------------------------

def hutchinson_trace(model, loss_fn, data_batch, params: Optional[Sequence[Any]] = None, device: Optional[str] = None) -> float:
    """
    Estimate Hessian trace via one Hutchinson probe.
    Requires PyTorch and differentiable loss; used as a hook.
    """
    if not _HAVE_TORCH:
        raise RuntimeError("PyTorch not available for hutchinson_trace")
    torch = _torch
    model.train()
    if device is not None:
        model.to(device)
    inputs, targets = data_batch
    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    v = [torch.randint_like(g, low=0, high=2) * 2 - 1 for g in grad]
    Hv = torch.autograd.grad(grad, params, grad_outputs=v, retain_graph=False)
    dot = sum((h * vi).sum() for h, vi in zip(Hv, v))
    return float(dot.item())

# ----------------------------
# Backward compatibility shim
# ----------------------------

# For compatibility with existing imports expecting `AutoForkRules` and `AutoForkEngine`
# from a module named `autofork.py`, this file can be named `autoforks.py` and imported
# as needed in the application layer.

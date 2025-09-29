import math
import numpy as np

def moving_block_indices(n, block_len, rng, circular=False):
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n - block_len + 1, size=math.ceil(n / block_len)) if not circular else rng.integers(0, n, size=math.ceil(n / block_len))
    idx = []
    for s in starts:
        if circular:
            idx.extend([(s + j) % n for j in range(block_len)])
        else:
            end = min(s + block_len, n)
            idx.extend(list(range(s, end)))
    return np.array(idx[:n], dtype=int)

def block_bootstrap_ci(y_true, pred_a, pred_b, metric_fn, n_boot=1000, block_len=None, seed=0, circular=False, block_frac=0.05):
    y_true = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    n = len(y_true)
    if block_len is None:
        block_len = max(2, min(int(round(block_frac*n)), n//2 or 1))
    rng = np.random.default_rng(seed)
    base_delta = metric_fn(y_true, b) - metric_fn(y_true, a)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = moving_block_indices(n, block_len, rng, circular=circular)
        deltas[i] = metric_fn(y_true[idx], b[idx]) - metric_fn(y_true[idx], a[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return base_delta, (float(lo), float(hi))

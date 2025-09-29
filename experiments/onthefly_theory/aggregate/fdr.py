import numpy as np
from scipy.stats import norm
import pandas as pd

def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if not np.any(mask): return q
    p_nonan = p[mask]; m = p_nonan.size
    order = np.argsort(p_nonan, kind="mergesort")
    ranks = np.empty_like(order); ranks[order] = np.arange(1, m + 1)
    q_raw = p_nonan * m / ranks
    q_sorted = np.minimum.accumulate(q_raw[order][::-1])[::-1]
    q_adj = np.minimum(q_sorted, 1.0)
    q[mask] = q_adj[np.argsort(order)]
    return q

def stouffer(pvals: np.ndarray) -> float:
    p = np.asarray([x for x in pvals if np.isfinite(x) and not np.isnan(x)])
    if p.size == 0: return np.nan
    z = norm.isf(p)
    z_sum = np.sum(z)
    z_comb = z_sum / np.sqrt(len(z))
    return float(norm.sf(z_comb))

def add_fdr_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    dm_p_cols = [c for c in results_df.columns if c.startswith("dm_p_")]
    if not dm_p_cols: return results_df
    df = results_df.copy()
    for c in dm_p_cols:
        qcol = c.replace("dm_p_", "dm_q_")
        df[qcol] = bh_adjust(df[c].values)
    for c in dm_p_cols:
        qcol_ds = c.replace("dm_p_", "dm_q_ds_")
        df[qcol_ds] = np.nan
        for ds, idx in df.groupby("dataset").groups.items():
            block = df.loc[idx, c]
            df.loc[idx, qcol_ds] = bh_adjust(block.values)
    return df

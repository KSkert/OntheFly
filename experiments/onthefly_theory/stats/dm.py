import math
import numpy as np
from scipy.stats import norm

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2,
                         alternative: str = "two-sided", lag: int = None):
    e1 = np.asarray(e1); e2 = np.asarray(e2)
    assert e1.shape == e2.shape
    if power == 1:  d = np.abs(e1) - np.abs(e2)
    elif power == 2: d = (e1**2) - (e2**2)
    else:           d = (np.abs(e1)**power) - (np.abs(e2)**power)
    T = len(d); dbar = float(np.mean(d))
    if lag is None:
        lag = max(h - 1, int(np.floor(1.5 * (T ** (1/3)))))
    def autocov(x, k):
        x0 = x[:-k] if k > 0 else x
        xk = x[k:]  if k > 0 else x
        return float(np.cov(x0, xk, ddof=0)[0, 1]) if k > 0 else float(np.var(x, ddof=0))
    lrv = autocov(d, 0)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma = autocov(d, k)
        lrv += 2.0 * w * gamma
    denom = math.sqrt(lrv / T) if lrv > 0 else np.inf
    dm = dbar / (denom if denom > 0 else np.inf)
    hln = dm * math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(hln)))
    elif alternative == "greater":
        p = 1 - norm.cdf(hln)
    else:
        p = norm.cdf(hln)
    return {"dm_stat": float(hln), "p_value": float(p), "lag": int(lag), "lrv": float(lrv), "mean_diff": float(dbar), "T": int(T)}

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..warnings import hush_statsmodels

def infer_seasonal_period(series_index: pd.DatetimeIndex):
    if series_index.inferred_freq is None:
        deltas = np.diff(series_index.values).astype('timedelta64[D]').astype(int)
        if len(deltas) == 0: return None
        md = int(np.median(deltas))
        if 27 <= md <= 31: return 12
        if 85 <= md <= 95: return 4
        return None
    freq = series_index.inferred_freq.upper()
    if freq.startswith("M"): return 12
    if freq.startswith("Q"): return 4
    if freq.startswith("W"): return 52
    if freq.startswith("D"): return 7
    return None

def sarima_forecast(train, steps, m):
    best_aic, best_fit, best_cfg = np.inf, None, None
    for p in [0,1,2]:
        for d in [0,1]:
            for q in [0,1,2]:
                for P in [0,1]:
                    for D in [0,1]:
                        for Q in [0,1]:
                            try:
                                with hush_statsmodels():
                                    mod = SARIMAX(train, order=(p,d,q),
                                                  seasonal_order=(P,D,Q,m) if m else (0,0,0,0),
                                                  enforce_stationarity=False, enforce_invertibility=False)
                                    fit = mod.fit(disp=False)
                                if fit.aic < best_aic:
                                    best_aic, best_fit, best_cfg = fit.aic, fit, (p,d,q,P,D,Q)
                            except Exception:
                                pass
    if best_fit is None:
        return np.repeat(train[-1], steps), {"m": m, "order": None, "seasonal_order": None, "aic": None, "note":"fallback"}
    fc = best_fit.forecast(steps=steps)
    p,d,q,P,D,Q = best_cfg
    return np.asarray(fc, float), {"m": m, "order": (p,d,q), "seasonal_order": (P,D,Q,m if m else 0), "aic": float(best_aic)}

def ets_forecast(train, steps, m):
    try:
        trend = "add" if m is None else "add"
        seasonal = "add" if m else None
        with hush_statsmodels():
            model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=m).fit()
        fc = model.forecast(steps)
        return np.asarray(fc, dtype=float), {"m": m, "model": "ETS"}
    except Exception:
        return np.repeat(train[-1], steps), {"m": m, "model": "ETS-fallback"}

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sktime.datasets import load_airline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..warnings import hush_statsmodels
from .yfinance_cache import yf_close_series

def to_series(obj, value_hint=None, index_hint=None, year_to_datetime=False, name=None):
    if isinstance(obj, pd.Series):
        s = obj.astype(float)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if index_hint and index_hint in df.columns:
            idx = df[index_hint]
            if year_to_datetime:
                idx = pd.to_datetime(idx.astype(int).astype(str) + "-12-31")
            else:
                try: idx = pd.to_datetime(idx)
                except Exception: pass
            df = df.drop(columns=[index_hint]); df.index = idx
        if value_hint and value_hint in df.columns:
            s = df[value_hint].astype(float)
        else:
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                raise ValueError("No numeric column to coerce to Series.")
            s = num.iloc[:, 0].astype(float)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
    if not isinstance(s.index, pd.DatetimeIndex):
        try: s.index = pd.to_datetime(s.index)
        except Exception: pass
    s = s.sort_index()
    if name: s = s.rename(name)
    return s

def year_month_wide_to_series(df, year_col="YEAR", name="value"):
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    if year_col not in df.columns:
        raise ValueError(f"{year_col} not found.")
    long = df.melt(id_vars=[year_col], var_name="month", value_name=name)
    long = long[long["month"].isin(months)].sort_values([year_col, "month"])
    mnum = {m: i+1 for i, m in enumerate(months)}
    dt = pd.to_datetime(long[year_col].astype(int).astype(str) + "-" + long["month"].map(mnum).astype(int).astype(str) + "-01")
    s = pd.Series(long[name].astype(float).values, index=dt, name=name).sort_index()
    return s.dropna()

def make_supervised_from_series(series: pd.Series, max_lag: int):
    s = pd.Series(series).astype(float).sort_index().dropna()
    values = s.values
    X, y = [], []
    for t in range(max_lag, len(values)):
        X.append(values[t-max_lag:t]); y.append(values[t])
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    return X, y, max_lag, X.shape[1]

def simple_time_split(X, y, test_ratio=0.2, val_ratio=0.2):
    n = len(y)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    return (X[:val_start], X[val_start:test_start], X[test_start:], y[:val_start], y[val_start:test_start], y[test_start:])

def rolling_origin_splits(X, y, k_folds=3, val_size=0.2, test_ratio=0.2):
    n = len(y)
    test_len = int(math.ceil(n * test_ratio))
    end_test = n
    start_test = n - test_len
    fold_size = max(1, test_len // k_folds)
    for i in range(k_folds):
        test_end_i = start_test + (i + 1) * fold_size if i < k_folds - 1 else end_test
        test_start_i = start_test + i * fold_size
        trainval_end = test_start_i
        val_len = int(max(1, round(val_size * trainval_end)))
        val_start = max(0, trainval_end - val_len)
        yield ((0, val_start), (val_start, trainval_end), (test_start_i, test_end_i))

def scale_after_split(X_tr, X_val, X_te, y_tr, y_val, y_te, scaler_X_cls=MinMaxScaler, scaler_y_cls=MinMaxScaler):
    scaler_X = scaler_X_cls().fit(X_tr)
    scaler_y = scaler_y_cls().fit(y_tr.reshape(-1, 1))
    X_train = scaler_X.transform(X_tr)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_te)
    y_train = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_te.reshape(-1, 1)).ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

def build_ts_datasets(hp) -> list:
    ASSET_END_DATE = "2024-12-31"
    datasets = []

    macro_obj = sm.datasets.macrodata.load_pandas().data
    year = macro_obj["year"].astype(int); q = macro_obj["quarter"].astype(int)
    idx = pd.PeriodIndex(year=year, quarter=q, freq="Q").to_timestamp(how="end")
    cols = ["realgdp", "realcons", "realinv", "unemp", "infl"]
    macro = macro_obj[cols].copy(); macro.index = idx; macro = macro.sort_index()
    realgdp = pd.Series(macro["realgdp"].astype(float).values, index=macro.index, name="realgdp").dropna()
    realgdp_qoq = realgdp.pct_change().mul(100.0).rename("realgdp_qoq").dropna()

    co2_obj = sm.datasets.co2.load_pandas().data
    co2 = to_series(co2_obj, value_hint="co2", name="co2").dropna()
    eln_obj = sm.datasets.elnino.load_pandas().data
    eln = year_month_wide_to_series(eln_obj, year_col="YEAR", name="elnino_sst").dropna()
    nile_obj = sm.datasets.nile.load_pandas().data
    nile = to_series(nile_obj, value_hint="volume", index_hint="year", year_to_datetime=True, name="nile").dropna()

    try:
        airline = to_series(load_airline(), name="airline").dropna()
    except Exception:
        airline = None

    if hp.use_ts_airline and airline is not None: datasets.append(("Airline", airline, 12, False))
    if hp.use_ts_co2: datasets.append(("CO2", co2, 12, False))
    if hp.use_ts_elnino: datasets.append(("ElNino", eln, 12, False))
    if hp.use_ts_nile: datasets.append(("Nile", nile, 2, False))
    if hp.use_ts_gdp_qoq: datasets.append(("US_RealGDP_QoQ_%", realgdp_qoq, 8, False))
    if hp.use_ts_gdp_level: datasets.append(("US_RealGDP_level", realgdp, 8, False))

    if hp.use_ts_btc:
        try:
            btc = yf_close_series("BTC-USD", start="2014-01-01", end=ASSET_END_DATE, auto_adjust=True, cache_dir="cache")
            if hp.use_returns_for_assets:
                btc = np.log(btc).diff().dropna().rename("BTCUSD_logret"); lag_btc = 60; is_ret = True
            else:
                lag_btc = 60; is_ret = False
            datasets.append(("BTC-USD", btc, lag_btc, is_ret))
        except Exception:
            pass

    if hp.use_ts_ng:
        try:
            ng = yf_close_series("NG=F", start="2000-01-01", end=ASSET_END_DATE, auto_adjust=True, cache_dir="cache").rename("NG=F")
            if hp.use_returns_for_assets:
                ng = np.log(ng).diff().dropna().rename("NG_logret"); lag_ng = 60; is_ret = True
            else:
                lag_ng = 60; is_ret = False
            datasets.append(("NaturalGas", ng, lag_ng, is_ret))
        except Exception:
            pass

    if hp.use_ts_sp_csv:
        import os
        try:
            path = "data/sp.csv"
            sp = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
            target_col = "Adj. Close" if "Adj. Close" in sp.columns else "Close"
            sp_series = pd.Series(sp[target_col].astype(float).values, index=sp["Date"], name="SPY").sort_index().dropna()
            if hp.use_returns_for_assets:
                sp_series = np.log(sp_series).diff().dropna().rename("SPY_logret"); lag_sp = 30; is_ret = True
            else:
                lag_sp = 30; is_ret = False
            datasets.append(("S&P", sp_series, lag_sp, is_ret))
        except Exception:
            pass

    return datasets

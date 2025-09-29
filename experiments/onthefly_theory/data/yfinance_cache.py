import os, hashlib
import numpy as np
import pandas as pd
import yfinance as yf

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def yf_close_series(ticker, start="2000-01-01", end=None, auto_adjust=True,
                    cache_dir=None, force_refresh: bool=False):
    os.makedirs(cache_dir, exist_ok=True) if cache_dir else None
    cache_name = ticker.replace('=','_') + (f"_{end}" if end else "") + ".csv"
    cache_path = os.path.join(cache_dir, cache_name) if cache_dir else None

    if cache_path and os.path.exists(cache_path) and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
        src = "cache"
    else:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=auto_adjust, progress=False)
        if df is None or df.empty:
            raise ValueError(f"Empty download for {ticker}")
        df = df.sort_index()
        if cache_path:
            tmp = df.reset_index()[["Date","Close"]]; tmp.to_csv(cache_path, index=False)
        src = "download"

    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"]
        if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
            close = close.iloc[:, 0]
    else:
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").astype(float).dropna()
    close.index = pd.to_datetime(close.index).sort_values()
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    if not np.isfinite(close.values).all():
        raise ValueError(f"Non-finite values for {ticker}")

    last_date = str(close.index[-1].date())
    close.attrs["last_date"] = last_date
    close.attrs["source"] = src
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            close.attrs["cache_sha256"] = _sha256_bytes(fh.read())
        close.attrs["cache_path"] = cache_path
    return close

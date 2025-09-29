# %% [markdown]

import warnings
# Be selective: keep helpful warnings visible; hush only a few noisy ones.
warnings.filterwarnings("ignore", message=".*is a deprecated alias.*")             # numpy alias noise
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")          # temp scaling LBFGS on tiny val sets
warnings.filterwarnings("ignore", message=".*non-invertible starting MA.*")        # ARIMA grid search edge cases


import math
import random
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from sktime.datasets import load_airline
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # <-- needed for MoE prob mixing
from torch.utils.data import Dataset, DataLoader

import sklearn, statsmodels
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml, fetch_california_housing, fetch_covtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


from tqdm.auto import tqdm

# =========================
# Repro & device
# =========================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# =========================
# Hyperparameters + toggles
# =========================


@dataclass
class HParams:
    # ---------- Optimization ----------
    batch_size: int = 128
    weight_decay: float = 5e-5
    lr_generalist: float = 3e-4
    lr_specialist: float = 5e-4
    lr_gate: float = 1e-3          # give the gate real learning capacity
    lr_baseline: float = 3e-4
    grad_clip: float = 1.0
    gate_patience: int = 8
    bootstrap_block_frac: float = 0.10  # tighter CI on deltas for longer series

    # ---------- Budgets (bigger, MoE gets majority) ----------
    # Time series: multiple datasets + BTC/NG
    ts_epochs_total: int = 60
    ts_frac_warmup: float = 0.30
    ts_frac_spec: float = 0.50
    ts_frac_gate: float = 0.20

    # Tabular regression: more room for specialists
    tr_epochs_total: int = 40
    tr_frac_warmup: float = 0.30
    tr_frac_spec: float = 0.50
    tr_frac_gate: float = 0.20

    # Tabular classification
    tc_epochs_total: int = 40
    tc_frac_warmup: float = 0.30
    tc_frac_spec: float = 0.50
    tc_frac_gate: float = 0.20

    # ---------- Nets (larger, still manageable) ----------
    generalist_hidden: int = 384
    specialist_hidden: int = 320
    baseline_hidden: int = 256
    dropout: float = 0.15

    # LSTM (TS)
    lstm_hidden: int = 96
    lstm_layers: int = 2

    # CNN (tabular classification)
    cnn_channels: int = 64
    cnn_kernel: int = 3
    cnn_hidden: int = 256

    # ---------- Hard mining (make specialists actually specialize) ----------
    hard_fraction: float = 0.60   # mine the hardest 60% → enough data per expert
    n_specialists: int = 4        # still friendly to the gate, but nontrivial

    # ---------- Evaluation ----------
    seeds: Tuple[int, ...] = (13, 42, 202, 777)
    rolling_folds: int = 3        # rolling-origin eval for TS
    test_ratio: float = 0.20
    val_ratio: float = 0.20

    # ---------- DATASET SWITCHES ----------
    # time series
    use_ts_airline: bool = True
    use_ts_btc: bool = True
    use_ts_co2: bool = True
    use_ts_elnino: bool = True
    use_ts_nile: bool = True
    use_ts_gdp_qoq: bool = True
    use_ts_gdp_level: bool = False
    use_ts_sp_csv: bool = False    # flip to True only if data/sp.csv exists
    use_ts_ng: bool = True
    use_returns_for_assets: bool = True

    # tabular regression
    use_tr_california: bool = True
    use_tr_airfoil: bool = True
    use_tr_concrete: bool = True
    use_tr_energy: bool = True
    use_tr_yacht: bool = True
    use_tr_ccpp: bool = True

    # tabular classification
    use_tc_adult: bool = True
    use_tc_covtype: bool = True

    # ---------- MODEL SWITCHES ----------
    # Time series
    run_ts_mlp: bool = True
    run_ts_moe_mlp_hard: bool = True
    run_ts_moe_mlp_rand: bool = True
    run_ts_lstm: bool = True
    run_ts_moe_lstm_hard: bool = True
    run_ts_moe_lstm_rand: bool = True
    run_ts_equal_params: bool = True   # fairness check vs equal-param MLP
    run_ts_naive_ens: bool = True      # naive ensemble baseline
    run_ts_rw: bool = True
    run_ts_arima: bool = True
    run_ts_sarima_ets: bool = True  # SARIMA + ETS

    # Tabular regression
    run_tr_mlp: bool = True
    run_tr_moe_mlp_hard: bool = True
    run_tr_moe_mlp_rand: bool = True

    # Tabular classification
    run_tc_mlp: bool = True
    run_tc_cnn: bool = True
    run_tc_rf: bool = True
    run_tc_moe_mlp_hard: bool = True
    run_tc_moe_mlp_rand: bool = True
    run_tc_moe_cnn_hard: bool = True
    run_tc_moe_cnn_rand: bool = True

# =========================
# Dataset helpers (TS)
# =========================

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
    dt = pd.to_datetime(long[year_col].astype(int).astype(str) + "-" +
                        long["month"].map(mnum).astype(int).astype(str) + "-01")
    s = pd.Series(long[name].astype(float).values, index=dt, name=name).sort_index()
    return s.dropna()

# =========================
# PATCH: yfinance reproducibility
# =========================
import hashlib

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def yf_close_series(ticker, start="2000-01-01", end=None, auto_adjust=True,
                    cache_dir: Optional[str]=None, force_refresh: bool=False):
    """
    Frozen snapshot by default: if cache exists, read it; else download once and cache.
    If 'end' is provided, download strictly up to that date. Audit gets file hash + last date.
    """
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
            tmp = df.reset_index()[["Date","Close"]]
            tmp.to_csv(cache_path, index=False)
        src = "download"

    # handle multiindex
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

    # audit attrs
    last_date = str(close.index[-1].date())
    close.attrs["last_date"] = last_date
    close.attrs["source"] = src
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            close.attrs["cache_sha256"] = _sha256_bytes(fh.read())
        close.attrs["cache_path"] = cache_path
    return close


# =========================
# TS supervised frames
# =========================

def make_supervised_from_series(series: pd.Series, max_lag: int):
    s = pd.Series(series).astype(float).sort_index().dropna()
    values = s.values
    X, y = [], []
    for t in range(max_lag, len(values)):
        X.append(values[t-max_lag:t])
        y.append(values[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, max_lag, X.shape[1]

def simple_time_split(X, y, test_ratio=0.2, val_ratio=0.2):
    n = len(y)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    return (
        X[:val_start], X[val_start:test_start], X[test_start:],
        y[:val_start], y[val_start:test_start], y[test_start:],
    )

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

def scale_after_split(X_tr, X_val, X_te, y_tr, y_val, y_te,
                      scaler_X_cls=MinMaxScaler, scaler_y_cls=MinMaxScaler):
    scaler_X = scaler_X_cls().fit(X_tr)
    scaler_y = scaler_y_cls().fit(y_tr.reshape(-1, 1))
    X_train = scaler_X.transform(X_tr)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_te)
    y_train = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_te.reshape(-1, 1)).ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

# =========================
# Datasets: Tabular loaders
# =========================

def load_openml_safe(name: Optional[str]=None, data_id: Optional[int]=None, version=None, as_frame=True):
    try:
        if name is not None:
            ds = fetch_openml(name=name, version=version, as_frame=as_frame)
        else:
            ds = fetch_openml(data_id=data_id, as_frame=as_frame)
        X = ds.data.copy()
        y = ds.target.copy()
        return X, y
    except Exception as e:
        print(f"[load_openml_safe] Skipping {name or data_id}: {e}")
        return None, None

def prep_regression_frame(X: pd.DataFrame, y: pd.Series):
    X = X.copy()
    y = pd.to_numeric(pd.Series(y).astype(float), errors="coerce")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Use sparse=True for width; downstream dense models will handle densification/SVD if needed
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True) if "sparse_output" in OneHotEncoder().get_params() \
          else OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,   # keep sparse if any transformer yields sparse
        n_jobs=None
    )
    return pre

def prep_classification_frame(X: pd.DataFrame, y: pd.Series):
    X = X.copy()
    y = pd.Series(y)
    if not pd.api.types.is_integer_dtype(y):
        y = y.astype("category").cat.codes
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True) if "sparse_output" in OneHotEncoder().get_params() \
          else OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None
    )
    return pre, y.astype(int).values

# =========================
# Models & utils (TS + Tabular)
# =========================

class ArrayDataset(Dataset):
    def __init__(self, X, y, task="reg"):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "reg":
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
        self.task = task
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x): return self.net(x)  # logits

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, n_classes, channels=32, kernel=3, hidden=128, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
        self.n_features = n_features
    def forward(self, x):
        x1 = x.view(x.size(0), 1, self.n_features)  # (B,1,F)
        h = self.conv(x1)
        return self.head(h)  # logits

class LSTMRegressor(nn.Module):
    def __init__(self, seq_len, lstm_hidden=32, lstm_layers=2, dropout=0.2, head_hidden=128):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
    def forward(self, x_full):
        x_seq = x_full[:, :self.seq_len].unsqueeze(-1)  # (B, T, 1)
        out, _ = self.lstm(x_seq)
        h_last = out[:, -1, :]
        return self.head(h_last)

# Gates
class LSTMGate(nn.Module):
    def __init__(self, seq_len, num_experts, lstm_hidden=32, hidden_dim=128, dropout=0.2, lstm_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
    def forward(self, x_full):
        x_seq = x_full[:, :self.seq_len].unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]
        logits = self.fc(h)
        return F.softmax(logits, dim=1)

class TabularGateMLP(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_equal_params_mlp(input_dim: int, target_params: int, dropout: float=0.2) -> MLPRegressor:
    lo, hi = 8, 4096
    best_H = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        tmp = MLPRegressor(input_dim, hidden_dim=mid, dropout=dropout)
        p = count_params(tmp)
        if p <= target_params:
            best_H = mid; lo = mid + 1
        else:
            hi = mid - 1
    return MLPRegressor(input_dim, hidden_dim=best_H, dropout=dropout)

# =========================
# Training helpers
# =========================

def train_epoch(model, loader, criterion, optimizer, grad_clip=None):
    model.train(); total = 0.0; n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = yb.size(0); total += loss.item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(predict_fn, loader, criterion):
    total = 0.0; n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb)
        bs = yb.size(0); total += criterion(pred, yb).item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def per_sample_losses(model, X_np, y_np, batch_size=2048, desc="per-loss", task="reg"):
    model.eval()
    N = len(y_np)
    losses = np.zeros(N, dtype=np.float32)
    if task == "reg":
        crit = nn.MSELoss(reduction="none")
    else:
        crit = nn.CrossEntropyLoss(reduction="none")
    for start in tqdm(range(0, N, batch_size), desc=desc, leave=False):
        end = min(start + batch_size, N)
        Xb = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
        if task == "reg":
            yb = torch.tensor(y_np[start:end], dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(Xb)
            losses[start:end] = crit(pred, yb).squeeze(1).cpu().numpy()
        else:
            yb = torch.tensor(y_np[start:end], dtype=torch.long, device=device)
            logits = model(Xb)
            losses[start:end] = crit(logits, yb).cpu().numpy()
    return losses

@torch.no_grad()
def moe_predict_reg(gate, experts, xb):
    ex_outs = [ex(xb) for ex in experts]
    w = gate(xb)
    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
    return out

# === FIXED: probability mixing for classification ===
@torch.no_grad()
def moe_predict_cls(gate, experts, xb):
    w = gate(xb)  # (B,E)
    probs = [F.softmax(ex(xb), dim=1) for ex in experts]  # list of (B,C)
    mix = sum(w[:, [j]] * probs[j] for j in range(len(experts)))  # (B,C)
    return torch.log(mix.clamp_min(1e-12))  # return log-probs as "logits" for downstream metrics

# === Gate training: NLL on mixed log-probs (classification) ===
def train_gate_cls(gate, experts, train_loader, val_loader, hp: HParams, n_classes: int, max_epochs: int):
    for ex in experts:
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)

    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    crit = nn.NLLLoss()
    best_val = float("inf"); best_state = None; patience = 0

    def mixed_logprobs(xb):
        w = gate(xb)  # (B,E)
        logps = [F.log_softmax(ex(xb), dim=1) for ex in experts]  # list of (B,C)
        stack = torch.stack(logps, dim=2)           # (B,C,E)
        m = torch.amax(stack, dim=2, keepdim=True)  # (B,C,1)
        mix = (w.unsqueeze(1) * torch.exp(stack - m)).sum(dim=2)  # (B,C)
        return (m.squeeze(2) + torch.log(mix.clamp_min(1e-12)))   # (B,C)

    # budget for gate: from *_epochs_total fractions handled by caller; we just loop given max epochs
    pbar = tqdm(range(1, max_epochs + 1), desc="gate(cls)", leave=False)
    for ep in pbar:
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logp = mixed_logprobs(Xb)
            loss = crit(logp, yb)
            optg.zero_grad(); loss.backward()
            if hp.grad_clip: nn.utils.clip_grad_norm_(gate.parameters(), hp.grad_clip)
            optg.step()

        gate.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logp = mixed_logprobs(Xb)
                bs = yb.size(0); val_loss += crit(logp, yb).item() * bs; n += bs
        val_loss /= max(n, 1)

        pbar.set_postfix(val_nll=f"{val_loss:.5f}", patience=f"{patience}/{hp.gate_patience}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience: pbar.close(); break

    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate

def train_gate_reg(gate, experts, train_loader, val_loader, hp: HParams, criterion, max_epochs: int):
    for ex in experts:
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)
    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    best_val = float("inf"); best_state = None; patience = 0
    pbar = tqdm(range(1, max_epochs + 1), desc="gate(reg)", leave=False)
    for ep in pbar:
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                ex_outs = [ex(Xb) for ex in experts]
            w = gate(Xb)
            out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
            loss = criterion(out, yb)
            optg.zero_grad(); loss.backward()
            if hp.grad_clip: nn.utils.clip_grad_norm_(gate.parameters(), hp.grad_clip)
            optg.step()
        # validate
        gate.eval(); val_loss = 0.0; n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                ex_outs = [ex(Xb) for ex in experts]
                w = gate(Xb)
                out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                bs = yb.size(0); val_loss += criterion(out, yb).item() * bs; n += bs
        val_loss /= max(n, 1)
        pbar.set_postfix(val_mse=f"{val_loss:.6f}", patience=f"{patience}/{hp.gate_patience}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience:
                pbar.close(); break
    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate

# =========================
# Stats utilities
# =========================

def rmse(y, p): return float(mean_squared_error(y, p, squared=False))
def mae(y, p):  return float(mean_absolute_error(y, p))

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
    base_delta = metric_fn(y_true, b) - metric_fn(y_true, a)  # NOTE: negative = improvement of 'b' vs 'a'
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = moving_block_indices(n, block_len, rng, circular=circular)
        deltas[i] = metric_fn(y_true[idx], b[idx]) - metric_fn(y_true[idx], a[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return base_delta, (float(lo), float(hi))

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2,
                         alternative: str = "two-sided", lag: int = None):
    from math import sqrt
    from scipy.stats import norm
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
    dm = dbar / denom if denom > 0 else 0.0
    hln = dm * math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(hln)))
    elif alternative == "greater":
        p = 1 - norm.cdf(hln)
    else:
        p = norm.cdf(hln)
    return {"dm_stat": float(hln), "p_value": float(p), "lag": int(lag), "lrv": float(lrv), "mean_diff": float(dbar), "T": int(T)}

# =========================
# Helper: temperature scaling for classification
# =========================

def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    # Find T > 0 minimizing NLL on validation logits
    T = torch.tensor([1.0], requires_grad=True, device=device)
    y = torch.tensor(y_val, dtype=torch.long, device=device)
    L = torch.tensor(logits_val, dtype=torch.float32, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        scaled = L / T.clamp_min(1e-6)
        loss = F.cross_entropy(scaled, y)
        loss.backward()
        return loss
    opt.step(_closure)
    return float(T.detach().cpu().item())

def apply_temperature(logits_np: np.ndarray, T: float) -> np.ndarray:
    return logits_np / max(T, 1e-6)

def fit_temperature_logprobs(logp_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Probability-domain temperature for log-probs:
      p_T(k|x) ∝ p(k|x)^(1/T). Optimize T on NLL.
    """
    P = np.exp(np.asarray(logp_val))
    P = np.clip(P, 1e-12, 1.0)
    y = torch.tensor(y_val, dtype=torch.long, device=device)
    P_t = torch.tensor(P, dtype=torch.float32, device=device)

    T = torch.tensor([1.0], requires_grad=True, device=device)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        invT = 1.0 / T.clamp_min(1e-6)
        logpT = (invT * torch.log(P_t) - torch.logsumexp(invT * torch.log(P_t), dim=1, keepdim=True))
        # Gather -log p_T(y)
        nll = -torch.gather(logpT, 1, y.view(-1,1)).mean()
        nll.backward()
        return nll

    opt.step(_closure)
    return float(T.detach().cpu().item())

def apply_temperature_logprobs(logp_np: np.ndarray, T: float) -> np.ndarray:
    P = np.exp(np.asarray(logp_np))
    P = np.clip(P, 1e-12, 1.0)
    invT = 1.0 / max(T, 1e-6)
    logpT = invT * np.log(P) - np.log(np.sum(P**invT, axis=1, keepdims=True))
    return logpT  # still log-probs

# =========================
# PATCH: SARIMA/ETS helpers
# =========================
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def infer_seasonal_period(series_index: pd.DatetimeIndex) -> Optional[int]:
    if series_index.inferred_freq is None:
        # heuristic by median delta
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
        model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=m).fit()
        fc = model.forecast(steps)
        return np.asarray(fc, dtype=float), {"m": m, "model": "ETS"}
    except Exception:
        return np.repeat(train[-1], steps), {"m": m, "model": "ETS-fallback"}


# =========================
# TS core experiment (one split) — arch-matched MoE
# =========================

def run_ts_once(name, series, max_lag, hp: HParams, seed: int,
                split_ranges: Optional[Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]] = None,
                is_asset_returns: bool=False):
    set_seed(seed)
    X_raw, y_raw, seq_len, feat_dim = make_supervised_from_series(series, max_lag)

    if split_ranges is None:
        X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw = simple_time_split(
            X_raw, y_raw, test_ratio=hp.test_ratio, val_ratio=hp.val_ratio
        )
        tr0, tr1 = 0, len(y_tr_raw)
        va0, va1 = tr1, tr1 + len(y_val_raw)
        te0, te1 = va1, va1 + len(y_te_raw)
    else:
        (tr0,tr1),(va0,va1),(te0,te1) = split_ranges
        X_tr_raw, y_tr_raw = X_raw[tr0:tr1], y_raw[tr0:tr1]
        X_val_raw, y_val_raw = X_raw[va0:va1], y_raw[va0:va1]
        X_te_raw, y_te_raw = X_raw[te0:te1], y_raw[te0:te1]

    # scaler choice: StandardScaler for returns; MinMax for levels
    scaler_cls = StandardScaler if is_asset_returns else MinMaxScaler
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = scale_after_split(
        X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw,
        scaler_X_cls=scaler_cls, scaler_y_cls=scaler_cls
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train, y_train, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val,   y_val,   task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test,  y_test,  task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    crit = nn.MSELoss()

    # === Budget allocation (TS) ===
    E_TOTAL = hp.ts_epochs_total
    E_WARM  = max(1, int(E_TOTAL * hp.ts_frac_warmup))
    E_SPEC_TOTAL = max(0, int(E_TOTAL * hp.ts_frac_spec))
    E_GATE  = max(1, E_TOTAL - E_WARM - E_SPEC_TOTAL)
    E_SPEC  = max(1, E_SPEC_TOTAL // max(1, hp.n_specialists))

    results = {
        "task":"time_series","dataset":name,"seed":seed,"fold":"single" if split_ranges is None else f"{te0}:{te1}",
        "n_train":len(y_tr_raw),"n_val":len(y_val_raw),"n_test":len(y_te_raw),
        "budget_total":E_TOTAL,"budget_warmup":E_WARM,"budget_spec_total":E_SPEC_TOTAL,"budget_spec_each":E_SPEC,"budget_gate":E_GATE
    }

    audit = {}  # later dumped to JSON

    def gather_preds_targets(predict_fn, loader):
        preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(device)
                preds.append(predict_fn(Xb).cpu().numpy())
                targs.append(yb.cpu().numpy())
        preds = np.vstack(preds); targs = np.vstack(targs)
        preds_inv = scaler_y.inverse_transform(preds)
        targs_inv = scaler_y.inverse_transform(targs)
        return preds_inv.squeeze(), targs_inv.squeeze()

    # -----------------
    # Baseline MLP (TS) — uses E_TOTAL budget
    # -----------------
    if hp.run_ts_mlp:
        baseline_mlp = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt_b = optim.Adam(baseline_mlp.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        best_val = float("inf"); best_state = None; patience = 0; best_epoch = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} baseline-MLP(s={seed})", leave=False):
            _ = train_epoch(baseline_mlp, train_loader, crit, opt_b, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: baseline_mlp(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in baseline_mlp.state_dict().items()}
                patience = 0; best_epoch = ep
            else:
                patience += 1
                if patience > hp.gate_patience:
                    break
        audit["baseline_mlp_params"] = count_params(baseline_mlp)
        audit["baseline_mlp_epochs_ran"] = best_epoch
        if best_state is not None:
            baseline_mlp.load_state_dict(best_state)
        base_p, y_test_inv = gather_preds_targets(lambda xb: baseline_mlp(xb), test_loader)
        results["baseline_rmse"] = rmse(y_test_inv, base_p); results["baseline_mae"] = mae(y_test_inv, base_p)
        results["baseline_best_epoch"] = int(best_epoch)

    # -----------------
    # Generalist warmup + specialists + gates (regression MoE)
    # -----------------
    def train_moe_reg(expert_arch="mlp", tag="MLP"):
        if expert_arch == "mlp":
            generalist = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
            expert_factory = lambda: MLPRegressor(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
        else:  # LSTM experts
            generalist = LSTMRegressor(seq_len=seq_len, lstm_hidden=hp.lstm_hidden,
                                       lstm_layers=hp.lstm_layers, dropout=hp.dropout,
                                       head_hidden=hp.baseline_hidden).to(device)
            expert_factory = lambda: LSTMRegressor(seq_len=seq_len, lstm_hidden=hp.lstm_hidden,
                                                   lstm_layers=hp.lstm_layers, dropout=hp.dropout,
                                                   head_hidden=hp.baseline_hidden).to(device)

        # warmup generalist
        opt_g = optim.Adam(generalist.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(E_WARM), desc=f"{name} warmup-{tag}(s={seed})", leave=False):
            _ = train_epoch(generalist, train_loader, crit, opt_g, grad_clip=hp.grad_clip)

        # hard mining
        losses = per_sample_losses(generalist, X_train, y_train, batch_size=2048, desc=f"{name} per-loss-{tag}(s={seed})", task="reg")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        hard_X = X_train[hard_idx]; hard_y = y_train[hard_idx]
        km = KMeans(n_clusters=hp.n_specialists, random_state=seed, n_init=10)
        hard_labels = km.fit_predict(hard_X)

        specialists = []
        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = hard_X[mask]; yc = hard_y[mask]
            if len(Xc) == 0: Xc, yc = hard_X, hard_y
            dl_c = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = expert_factory()
            opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} spec-{tag}{c+1}/{hp.n_specialists}(s={seed})", leave=False):
                _ = train_epoch(spec, dl_c, crit, opt_s, grad_clip=hp.grad_clip)
            specialists.append(spec.eval())
        experts = [generalist.eval()] + specialists

        # random specialists
        rng = np.random.default_rng(seed)
        rand_idx = rng.choice(N, size=k, replace=False)
        rand_X = X_train[rand_idx]; rand_y = y_train[rand_idx]
        kmR = KMeans(n_clusters=hp.n_specialists, random_state=seed, n_init=10)
        rand_labels = kmR.fit_predict(rand_X)
        specialists_rand = []
        for c in range(hp.n_specialists):
            mask = (rand_labels == c)
            Xc = rand_X[mask]; yc = rand_y[mask]
            if len(Xc) == 0: Xc, yc = rand_X, rand_y
            dl_c = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = expert_factory()
            opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} specR-{tag}{c+1}/{hp.n_specialists}(s={seed})", leave=False):
                _ = train_epoch(spec, dl_c, crit, opt_s, grad_clip=hp.grad_clip)
            specialists_rand.append(spec.eval())
        experts_rand = [generalist.eval()] + specialists_rand

        # train gates
        gate_hard = LSTMGate(seq_len=seq_len, num_experts=len(experts),
                             lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                             dropout=hp.dropout, lstm_layers=hp.lstm_layers).to(device)
        gate_hard = train_gate_reg(gate_hard, experts, train_loader, val_loader, hp, crit, max_epochs=E_GATE)

        gate_rand = LSTMGate(seq_len=seq_len, num_experts=len(experts_rand),
                             lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                             dropout=hp.dropout, lstm_layers=hp.lstm_layers).to(device)
        gate_rand = train_gate_reg(gate_rand, experts_rand, train_loader, val_loader, hp, crit, max_epochs=E_GATE)

        # predictions
        moe_p, y_inv = gather_preds_targets(lambda xb: moe_predict_reg(gate_hard, experts, xb), test_loader)
        moeR_p, _    = gather_preds_targets(lambda xb: moe_predict_reg(gate_rand, experts_rand, xb), test_loader)

        return {
            f"moe_{tag}_rmse": rmse(y_inv, moe_p),
            f"moe_{tag}_mae":  mae(y_inv, moe_p),
            f"moe_{tag}_pred": moe_p,
            f"moeR_{tag}_rmse": rmse(y_inv, moeR_p),
            f"moeR_{tag}_mae":  mae(y_inv, moeR_p),
            f"moeR_{tag}_pred": moeR_p,
            "y_test_inv": y_inv,
            "generalist_params": count_params(generalist),
            "experts_params_sum": sum(count_params(e) for e in experts)
        }

    # ---- MoE-MLP
    out_mlp = None
    if hp.run_ts_moe_mlp_hard or hp.run_ts_moe_mlp_rand or hp.run_ts_mlp:
        out_mlp = train_moe_reg(expert_arch="mlp", tag="mlp")
        results.update({
            "moe_rmse": out_mlp["moe_mlp_rmse"], "moe_mae": out_mlp["moe_mlp_mae"],
            "moeR_rmse": out_mlp["moeR_mlp_rmse"], "moeR_mae": out_mlp["moeR_mlp_mae"],
        })
        y_test_inv = out_mlp["y_test_inv"]

    # ---- Equal-params & Naive ensemble (vs baseline)
    if hp.run_ts_equal_params and hp.run_ts_mlp and y_test_inv is not None and out_mlp is not None:
        dummy_gate = LSTMGate(seq_len=seq_len, num_experts=1+hp.n_specialists,
                              lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                              dropout=hp.dropout, lstm_layers=hp.lstm_layers)
        mlp_generalist = MLPRegressor(feat_dim, hp.generalist_hidden, hp.dropout)
        mlp_specialists = [MLPRegressor(feat_dim, hp.specialist_hidden, hp.dropout) for _ in range(hp.n_specialists)]
        n_params_moe = count_params(mlp_generalist) + sum(count_params(s) for s in mlp_specialists) + count_params(dummy_gate)
        eqp_model = make_equal_params_mlp(feat_dim, target_params=n_params_moe, dropout=hp.dropout).to(device)
        opt_eqp = optim.Adam(eqp_model.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        best_val = float("inf"); best_state = None; patience = 0
        for ep in tqdm(range(1, hp.ts_epochs_total + 1), desc=f"{name} equal-params(s={seed})", leave=False):
            _ = train_epoch(eqp_model, train_loader, crit, opt_eqp, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: eqp_model(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in eqp_model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None:
            eqp_model.load_state_dict(best_state)
        eqp_p, _ = gather_preds_targets(lambda xb: eqp_model(xb), test_loader)
        results["eqp_rmse"] = rmse(y_test_inv, eqp_p); results["eqp_mae"] = mae(y_test_inv, eqp_p)
        audit["equal_params_params"] = count_params(eqp_model)

        # naive ensemble (quick)
        gen = MLPRegressor(feat_dim, hp.generalist_hidden, hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(max(1, E_WARM//4)), desc=f"{name} naive-gen(s={seed})", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)
        specs = []
        for i in range(hp.n_specialists):
            spec = MLPRegressor(feat_dim, hp.specialist_hidden, hp.dropout).to(device)
            opts = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(max(1, E_SPEC//4)), desc=f"{name} naive-spec{i+1}(s={seed})", leave=False):
                _ = train_epoch(spec, train_loader, crit, opts, grad_clip=hp.grad_clip)
            specs.append(spec.eval())
        @torch.no_grad()
        def naive_predict(xb):
            outs = [gen(xb)] + [sp(xb) for sp in specs]
            return sum(outs) / float(len(outs))
        nae_p, _ = gather_preds_targets(lambda xb: naive_predict(xb), test_loader)
        results["moeN_rmse"] = rmse(y_test_inv, nae_p); results["moeN_mae"] = mae(y_test_inv, nae_p)

    # -----------------
    # LSTM baseline (TS) — same E_TOTAL budget as baseline MLP
    # -----------------
    if hp.run_ts_lstm:
        lstm_model = LSTMRegressor(seq_len=seq_len, lstm_hidden=hp.lstm_hidden,
                                   lstm_layers=hp.lstm_layers, dropout=hp.dropout,
                                   head_hidden=hp.baseline_hidden).to(device)
        opt_lstm = optim.Adam(lstm_model.parameters(), lr=min(hp.lr_baseline, 5e-4), weight_decay=hp.weight_decay)
        best_val = float("inf"); best_state = None; patience = 0; best_epoch = 0
        for ep in tqdm(range(1, hp.ts_epochs_total + 1), desc=f"{name} LSTM(s={seed})", leave=False):
            _ = train_epoch(lstm_model, train_loader, crit, opt_lstm, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: lstm_model(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in lstm_model.state_dict().items()}
                patience = 0; best_epoch = ep
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None:
            lstm_model.load_state_dict(best_state)
        lstm_p, y_test_inv2 = gather_preds_targets(lambda xb: lstm_model(xb), test_loader)
        results["lstm_rmse"] = rmse(y_test_inv2, lstm_p); results["lstm_mae"] = mae(y_test_inv2, lstm_p)
        audit["lstm_params"] = count_params(lstm_model)
        audit["lstm_epochs_ran"] = best_epoch

    # ---- MoE-LSTM
    out_lstm = None
    if (hp.run_ts_moe_lstm_hard or hp.run_ts_moe_lstm_rand or hp.run_ts_lstm):
        out_lstm = train_moe_reg(expert_arch="lstm", tag="lstm")
        results.update({
            "moeLSTM_rmse": out_lstm["moe_lstm_rmse"], "moeLSTM_mae": out_lstm["moe_lstm_mae"],
            "moeLSTMR_rmse": out_lstm["moeR_lstm_rmse"], "moeLSTMR_mae": out_lstm["moeR_lstm_mae"],
        })
        y_test_inv = out_lstm["y_test_inv"]

    # -----------------
    # RW / ARIMA (TS only)
    # -----------------
    if hp.run_ts_rw:
        rw_p = X_te_raw[:, -1].astype(float)
        results["rw_rmse"] = rmse(y_te_raw, rw_p); results["rw_mae"] = mae(y_te_raw, rw_p)
    if hp.run_ts_arima:
        # ARIMA order selection on train+val via AIC, refit on train+val, forecast TEST ONLY.
        full_vals = pd.Series(series).dropna().values.astype(float)
        # Supervised framing starts at index `max_lag`; the supervised lengths align with y_* sizes:
        trainval_len = max_lag + len(y_tr_raw) + len(y_val_raw)
        trainval_series = full_vals[:trainval_len]
        test_h = len(y_te_raw)
        best_aic, best_order = np.inf, (1,1,0)
        for p in [0,1,2]:
            for d in [0,1,2]:
                for q in [0,1,2]:
                    try:
                        fit = ARIMA(trainval_series, order=(p,d,q)).fit()
                        if fit.aic < best_aic:
                            best_aic, best_order = fit.aic, (p,d,q)
                    except Exception:
                        continue
        try:
            fitted = ARIMA(trainval_series, order=best_order).fit()
            fc = fitted.forecast(steps=test_h)
            arima_test = np.asarray(fc, dtype=float)
        except Exception:
            arima_test = np.full((test_h,), float(trainval_series[-1]))
        results["arima_rmse"] = rmse(y_te_raw, arima_test); results["arima_mae"] = mae(y_te_raw, arima_test)
        audit["arima_best_order"] = best_order
        audit["arima_aic"] = best_aic

    if hp.run_ts_sarima_ets:
            # SARIMA / ETS
        try:
            # Build the same train+val slice used for ARIMA
            full_vals = pd.Series(series).dropna().values.astype(float)
            trainval_len = max_lag + len(y_tr_raw) + len(y_val_raw)
            trainval_series = full_vals[:trainval_len]
            m = infer_seasonal_period(pd.Series(series).dropna().index)
            sarima_pred, sarima_info = sarima_forecast(trainval_series, len(y_te_raw), m)
            results["sarima_rmse"] = rmse(y_te_raw, sarima_pred); results["sarima_mae"] = mae(y_te_raw, sarima_pred)
            audit["sarima_info"] = sarima_info
        except Exception as e:
            results["sarima_rmse"] = np.nan; results["sarima_mae"] = np.nan; audit["sarima_err"] = str(e)

        try:
            ets_pred, ets_info = ets_forecast(trainval_series, len(y_te_raw), m)
            results["ets_rmse"] = rmse(y_te_raw, ets_pred); results["ets_mae"] = mae(y_te_raw, ets_pred)
            audit["ets_info"] = ets_info
        except Exception as e:
            results["ets_rmse"] = np.nan; results["ets_mae"] = np.nan; audit["ets_err"] = str(e)


    # -----------------
    # CI + DM vs baseline (if baseline exists)
    # -----------------
    if "baseline_rmse" in results:
        base_preds, _ = gather_preds_targets(lambda xb: baseline_mlp(xb), test_loader)
        y_true_inv = y_test_inv
        n_test = len(y_true_inv); blk = max(5, int(round(hp.bootstrap_block_frac * n_test)))

        def add_ci_dm(tag_pred, tag):
            if tag_pred is None: return
            d_rm, ci_rm = block_bootstrap_ci(y_true_inv, base_preds, tag_pred, rmse, n_boot=1000, block_len=blk, seed=seed, block_frac=hp.bootstrap_block_frac)
            d_ma, ci_ma = block_bootstrap_ci(y_true_inv, base_preds, tag_pred, mae,  n_boot=1000, block_len=blk, seed=seed+123, block_frac=hp.bootstrap_block_frac)
            e_base = y_true_inv - base_preds
            e_tag  = y_true_inv - tag_pred
            dm = diebold_mariano_test(e_base, e_tag, h=1, power=2, alternative="two-sided")
            results[f"delta_rmse_{tag}"] = d_rm; results[f"ci95_delta_rmse_{tag}_lo"] = ci_rm[0]; results[f"ci95_delta_rmse_{tag}_hi"] = ci_rm[1]
            results[f"delta_mae_{tag}"]  = d_ma; results[f"ci95_delta_mae_{tag}_lo"]  = ci_ma[0]; results[f"ci95_delta_mae_{tag}_hi"]  = ci_ma[1]
            results[f"dm_stat_{tag}"] = dm["dm_stat"]; results[f"dm_p_{tag}"] = dm["p_value"]

        if out_mlp is not None:
            add_ci_dm(out_mlp.get("moe_mlp_pred"), "moe_mlp")
            add_ci_dm(out_mlp.get("moeR_mlp_pred"), "moeR_mlp")
        if hp.run_ts_equal_params and "eqp_rmse" in results:
            add_ci_dm(eqp_p if 'eqp_p' in locals() else None, "eqp")
        if out_lstm is not None:
            add_ci_dm(out_lstm.get("moe_lstm_pred"), "moe_lstm")
            add_ci_dm(out_lstm.get("moeR_lstm_pred"), "moeR_lstm")
        if hp.run_ts_naive_ens and "moeN_rmse" in results:
            add_ci_dm(nae_p if 'nae_p' in locals() else None, "naive")
        if hp.run_ts_rw and "rw_rmse" in results:
            add_ci_dm(rw_p, "rw")
        if hp.run_ts_arima and "arima_rmse" in results:
            add_ci_dm(arima_test, "arima")
        if "sarima_rmse" in results and np.isfinite(results["sarima_rmse"]):
            add_ci_dm(sarima_pred, "sarima")
        if "ets_rmse" in results and np.isfinite(results["ets_rmse"]):
            add_ci_dm(ets_pred, "ets")


    results["audit"] = audit
    return results

# =========================
# TS high-level runner
# =========================

def run_ts(name, series, max_lag, hp: HParams, is_asset_returns: bool=False):
    results = []
    if hp.rolling_folds and hp.rolling_folds > 0:
        X_raw, y_raw, _, _ = make_supervised_from_series(series, max_lag)
        for seed in hp.seeds:
            for ranges in rolling_origin_splits(X_raw, y_raw, k_folds=hp.rolling_folds,
                                                val_size=hp.val_ratio, test_ratio=hp.test_ratio):
                res = run_ts_once(name, series, max_lag, hp, seed, split_ranges=ranges, is_asset_returns=is_asset_returns)
                results.append(res)
    else:
        for seed in hp.seeds:
            res = run_ts_once(name, series, max_lag, hp, seed, split_ranges=None, is_asset_returns=is_asset_returns)
            results.append(res)
    return pd.DataFrame(results)

# =========================
# Tabular regression runner (MLP + MoE-MLP)
# =========================

def run_tabular_regression(name, X: pd.DataFrame, y: pd.Series, hp: HParams):
    pre = prep_regression_frame(X, y)
    y = pd.to_numeric(pd.Series(y).astype(float), errors="coerce").values
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=hp.test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=hp.val_ratio, random_state=42)

    # 👉 preprocess FIRST (produces possibly-sparse matrices), then optional SVD for Torch models
    X_train = pre.fit_transform(X_train)
    X_val   = pre.transform(X_val)
    X_test  = pre.transform(X_test)

    svd = None
    # In both regression & classification SVD sections:
    if sp.issparse(X_train):
        target_k = 1024 if X_train.shape[1] > 4000 else min(512, X_train.shape[1] - 1)
        svd = TruncatedSVD(n_components=target_k, random_state=42)
        X_train_nn = svd.fit_transform(X_train)
        X_val_nn   = svd.transform(X_val)
        X_test_nn  = svd.transform(X_test)
    else:
        X_train_nn, X_val_nn, X_test_nn = X_train, X_val, X_test

    # Standardize targets for stability
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = sy.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = sy.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s  = sy.transform(y_test.reshape(-1, 1)).ravel()

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train_nn, y_train_s, task="reg"), batch_size=hp.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val_nn,   y_val_s,   task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test_nn,  y_test_s,  task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    crit = nn.MSELoss()


    # budget
    E_TOTAL = hp.tr_epochs_total
    E_WARM  = max(1, int(E_TOTAL * hp.tr_frac_warmup))
    E_SPEC_TOTAL = max(0, int(E_TOTAL * hp.tr_frac_spec))
    E_GATE  = max(1, E_TOTAL - E_WARM - E_SPEC_TOTAL)
    E_SPEC  = max(1, E_SPEC_TOTAL // max(1, hp.n_specialists))

    rows = {"task":"tabular_regression","dataset":name,"n_train":len(y_train),"n_val":len(y_val),"n_test":len(y_test),
            "budget_total":E_TOTAL,"budget_warmup":E_WARM,"budget_spec_each":E_SPEC,"budget_gate":E_GATE}

    # Baseline MLP
    if hp.run_tr_mlp:
        base = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(base.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        best_val = float("inf"); best_state = None; patience = 0; best_epoch = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TR-MLP", leave=False):
            _ = train_epoch(base, train_loader, crit, opt, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: base(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in base.state_dict().items()}
                patience = 0; best_epoch = ep
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None: base.load_state_dict(best_state)
        @torch.no_grad()
        def inv_preds(model):
            preds = []
            for Xb,_ in test_loader:
                preds.append(model(Xb.to(device)).cpu().numpy())
            preds = np.vstack(preds).squeeze()
            return sy.inverse_transform(preds.reshape(-1,1)).ravel()
        bp = inv_preds(base)
        rows["baseline_rmse"] = rmse(y_test, bp); rows["baseline_mae"] = mae(y_test, bp)

    # MoE-MLP (hard + random)
    if hp.run_tr_moe_mlp_hard or hp.run_tr_moe_mlp_rand:
        gen = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(E_WARM), desc=f"{name} TR-warmup", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)

        losses = per_sample_losses(gen, X_train_nn, y_train_s, desc=f"{name} TR-hard", task="reg")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])

        experts, expertsR = [], []
        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = X_train_nn[hard_idx][mask]; yc = y_train_s[hard_idx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[hard_idx], y_train_s[hard_idx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec  = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TR-spec{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            experts.append(spec.eval())

        rng = np.random.default_rng(42)
        ridx = rng.choice(N, size=k, replace=False)
        kmR = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        rlabels = kmR.fit_predict(X_train_nn[ridx])
        for c in range(hp.n_specialists):
            mask = (rlabels == c)
            Xc = X_train_nn[ridx][mask]; yc = y_train_s[ridx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[ridx], y_train_s[ridx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TR-specR{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            expertsR.append(spec.eval())

        gateH = TabularGateMLP(X_train_nn.shape[1], num_experts=1+hp.n_specialists, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        gateR = TabularGateMLP(X_train_nn.shape[1], num_experts=1+hp.n_specialists, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        experts  = [gen.eval()] + experts
        expertsR = [gen.eval()] + expertsR

        gateH = train_gate_reg(gateH, experts,  train_loader, val_loader, hp, crit, max_epochs=E_GATE)
        gateR = train_gate_reg(gateR, expertsR, train_loader, val_loader, hp, crit, max_epochs=E_GATE)

        @torch.no_grad()
        def inv_preds_gate(gate, experts):
            preds = []
            for Xb,_ in test_loader:
                Xb = Xb.to(device)
                preds.append(moe_predict_reg(gate, experts, Xb).cpu().numpy())
            preds = np.vstack(preds).squeeze()
            return sy.inverse_transform(preds.reshape(-1,1)).ravel()

        mp = inv_preds_gate(gateH, experts)
        mr = inv_preds_gate(gateR, expertsR)
        rows["moe_rmse"]  = rmse(y_test, mp); rows["moe_mae"]  = mae(y_test, mp)
        rows["moeR_rmse"] = rmse(y_test, mr); rows["moeR_mae"] = mae(y_test, mr)

    return pd.DataFrame([rows])

# =========================
# Tabular classification runner (MLP, CNN, RF + MoE of MLP/CNN)
# =========================

def run_tabular_classification(name, X: pd.DataFrame, y, hp: HParams):
    pre, y = prep_classification_frame(X, y)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=hp.test_ratio, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=hp.val_ratio, random_state=42, stratify=y_train_full)

    X_train = pre.fit_transform(X_train); X_val = pre.transform(X_val); X_test = pre.transform(X_test)
    # If OneHot blew the width up and matrices are sparse, compress before feeding Torch models.
    svd = None
    # In both regression & classification SVD sections:
    if sp.issparse(X_train):
        target_k = 1024 if X_train.shape[1] > 4000 else min(512, X_train.shape[1] - 1)
        svd = TruncatedSVD(n_components=target_k, random_state=42)
        X_train_nn = svd.fit_transform(X_train)
        X_val_nn   = svd.transform(X_val)
        X_test_nn  = svd.transform(X_test)
    else:
        X_train_nn, X_val_nn, X_test_nn = X_train, X_val, X_test


    n_classes = int(len(np.unique(y)))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train_nn, y_train, task="cls"), batch_size=hp.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val_nn,   y_val,   task="cls"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test_nn,  y_test,  task="cls"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)


    # budget
    E_TOTAL = hp.tc_epochs_total
    E_WARM  = max(1, int(E_TOTAL * hp.tc_frac_warmup))
    E_SPEC_TOTAL = max(0, int(E_TOTAL * hp.tc_frac_spec))
    E_GATE  = max(1, E_TOTAL - E_WARM - E_SPEC_TOTAL)
    E_SPEC  = max(1, E_SPEC_TOTAL // max(1, hp.n_specialists))

    rows = {"task":"tabular_classification","dataset":name,"n_train":len(y_train),"n_val":len(y_val),"n_test":len(y_test),
            "budget_total":E_TOTAL,"budget_warmup":E_WARM,"budget_spec_each":E_SPEC,"budget_gate":E_GATE}

    def logits_to_metrics(logits_np, y_true, *, inputs_are_log_probs: bool = False, n_classes: int = None):
        if inputs_are_log_probs:
            proba = np.exp(np.asarray(logits_np))
        else:
            proba = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
        y_pred = proba.argmax(axis=1)
        return accuracy_score(y_true, y_pred), log_loss(y_true, proba, labels=list(range(n_classes)))

    # Baseline MLP
    if hp.run_tc_mlp:
        mlp = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(mlp.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        best_val = float("inf"); best_state = None; patience = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TC-MLP", leave=False):
            _ = train_epoch(mlp, train_loader, crit, opt, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: mlp(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in mlp.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None: mlp.load_state_dict(best_state)
        # temperature on val
        logits_val = np.vstack([mlp(torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                        for i in range(0, len(X_val_nn), 1024)])
        T = fit_temperature(logits_val, y_val)
        logits_test = np.vstack([mlp(torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                         for i in range(0, len(X_test_nn), 1024)])
        acc, ll = logits_to_metrics(apply_temperature(logits_test, T), y_test)
        rows["mlp_acc"] = acc; rows["mlp_logloss"] = ll; rows["mlp_T"] = T

    # Baseline CNN
    if hp.run_tc_cnn:
        cnn = CNN1DClassifier(n_features=X_train_nn.shape[1], n_classes=n_classes,
                      channels=hp.cnn_channels, kernel=hp.cnn_kernel, hidden=hp.cnn_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(cnn.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        best_val = float("inf"); best_state = None; patience = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TC-CNN", leave=False):
            _ = train_epoch(cnn, train_loader, crit, opt, grad_clip=hp.grad_clip)
            v = eval_epoch(lambda xb: cnn(xb), val_loader, crit)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in cnn.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None: cnn.load_state_dict(best_state)
        logits_val = np.vstack([cnn(torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                        for i in range(0, len(X_val_nn), 1024)])
        T = fit_temperature(logits_val, y_val)
        logits_test = np.vstack([cnn(torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                         for i in range(0, len(X_test_nn), 1024)])
        acc, ll = logits_to_metrics(apply_temperature(logits_test, T), y_test)
        rows["cnn_acc"] = acc; rows["cnn_logloss"] = ll; rows["cnn_T"] = T

    # RandomForest (standalone baseline)
    if hp.run_tc_rf:
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)
        y_pred = proba.argmax(axis=1)
        rows["rf_acc"] = accuracy_score(y_test, y_pred); rows["rf_logloss"] = log_loss(y_test, proba, labels=list(range(n_classes)))

    # MoE-MLP (hard/random)
    if hp.run_tc_moe_mlp_hard or hp.run_tc_moe_mlp_rand:
        gen = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        for _ in tqdm(range(E_WARM), desc=f"{name} TC-MLP-warmup", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)

        losses = per_sample_losses(gen, X_train_nn, y_train, desc=f"{name} TC-MLP-hard", task="cls")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])
        experts, expertsR = [], []

        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = X_train_nn[hard_idx][mask]; yc = y_train[hard_idx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[hard_idx], y_train[hard_idx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="cls"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec  = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TC-MLP-spec{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            experts.append(spec.eval())

        rng = np.random.default_rng(42)
        ridx = rng.choice(N, size=k, replace=False)
        kmR = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        rlabels = kmR.fit_predict(X_train_nn[ridx])
        for c in range(hp.n_specialists):
            mask = (rlabels == c)
            Xc = X_train_nn[ridx][mask]; yc = y_train[ridx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[ridx], y_train[ridx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="cls"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TC-MLP-specR{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            expertsR.append(spec.eval())

        experts  = [gen.eval()] + experts
        expertsR = [gen.eval()] + expertsR
        gateH = TabularGateMLP(X_train_nn.shape[1], num_experts=len(experts),  hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        gateR = TabularGateMLP(X_train_nn.shape[1], num_experts=len(expertsR), hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)

        # share gate epochs budget via global var used in train_gate_cls
        global hp_gate_epochs
        hp_gate_epochs = E_GATE
        gateH = train_gate_cls(gateH, experts,  train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)
        gateR = train_gate_cls(gateR, expertsR, train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)


        def collect_logits_calibrated(model_gate, exps, X_val_nn, y_val, X_test_nn):
            # 1) VAL logits → fit T
            logp_val = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_val_nn), 1024)
            ])
            T = fit_temperature_logprobs(logp_val, y_val)

            # 2) TEST logits → apply T
            logp_test = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_test_nn), 1024)
            ])
            logp_test_T = apply_temperature_logprobs(logp_test, T)
            return logp_test_T, T


        logitsH_test, TH = collect_logits_calibrated(gateH, experts,  X_val_nn, y_val, X_test_nn)
        accH, llH = logits_to_metrics(logitsH_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)

        logitsR_test, TR = collect_logits_calibrated(gateR, expertsR, X_val_nn, y_val, X_test_nn)
        accR, llR = logits_to_metrics(logitsR_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)
        rows["moe_mlp_acc"] = accH; rows["moe_mlp_logloss"] = llH; rows["moe_mlp_T"] = TH
        rows["moeR_mlp_acc"] = accR; rows["moeR_mlp_logloss"] = llR; rows["moeR_mlp_T"] = TR

    # MoE-CNN (hard/random)
    if hp.run_tc_cnn and (hp.run_tc_moe_cnn_hard or hp.run_tc_moe_cnn_rand):
        gen = CNN1DClassifier(n_features=X_train_nn.shape[1], n_classes=n_classes,
                      channels=hp.cnn_channels, kernel=hp.cnn_kernel, hidden=hp.cnn_hidden, dropout=hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        for _ in tqdm(range(E_WARM), desc=f"{name} TC-CNN-warmup", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)

        losses = per_sample_losses(gen, X_train_nn, y_train, desc=f"{name} TC-CNN-hard", task="cls")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])
        experts, expertsR = [], []
        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = X_train_nn[hard_idx][mask]; yc = y_train[hard_idx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[hard_idx], y_train[hard_idx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="cls"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec  = CNN1DClassifier(n_features=X_train_nn.shape[1], n_classes=n_classes,
                      channels=hp.cnn_channels, kernel=hp.cnn_kernel, hidden=hp.cnn_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TC-CNN-spec{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            experts.append(spec.eval())

        rng = np.random.default_rng(42)
        ridx = rng.choice(N, size=k, replace=False)
        kmR = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        rlabels = kmR.fit_predict(X_train_nn[ridx])
        for c in range(hp.n_specialists):
            mask = (rlabels == c)
            Xc = X_train_nn[ridx][mask]; yc = y_train[ridx][mask]
            if len(Xc) == 0: Xc, yc = X_train_nn[ridx], y_train[ridx]
            dl = DataLoader(ArrayDataset(Xc, yc, task="cls"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = CNN1DClassifier(n_features=X_train_nn.shape[1], n_classes=n_classes,
                                 channels=hp.cnn_channels, kernel=hp.cnn_kernel, hidden=hp.cnn_hidden, dropout=hp.dropout).to(device)
            opt = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(E_SPEC), desc=f"{name} TC-CNN-specR{c+1}", leave=False):
                _ = train_epoch(spec, dl, crit, opt, grad_clip=hp.grad_clip)
            expertsR.append(spec.eval())

        experts  = [gen.eval()] + experts
        expertsR = [gen.eval()] + expertsR
        gateH = TabularGateMLP(X_train_nn.shape[1], num_experts=len(experts),  hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        gateR = TabularGateMLP(X_train_nn.shape[1], num_experts=len(expertsR), hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        hp_gate_epochs = E_GATE
        gateH = train_gate_cls(gateH, experts,  train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)
        gateR = train_gate_cls(gateR, expertsR, train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)

        def collect_logits_calibrated(model_gate, exps, X_val_nn, y_val, X_test_nn):
            # 1) VAL logits → fit T
            logp_val = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_val_nn), 1024)
            ])
            T = fit_temperature_logprobs(logp_val, y_val)

            # 2) TEST logits → apply T
            logp_test = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_test_nn), 1024)
            ])
            logp_test_T = apply_temperature_logprobs(logp_test, T)
            return logp_test_T, T


        logitsH_test, TH = collect_logits_calibrated(gateH, experts,  X_val_nn, y_val, X_test_nn)
        accH, llH = logits_to_metrics(logitsH_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)

        logitsR_test, TR = collect_logits_calibrated(gateR, expertsR, X_val_nn, y_val, X_test_nn)
        accR, llR = logits_to_metrics(logitsR_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)

        rows["moe_cnn_acc"] = accH; rows["moe_cnn_logloss"] = llH; rows["moe_cnn_T"] = TH
        rows["moeR_cnn_acc"] = accR; rows["moeR_cnn_logloss"] = llR; rows["moeR_cnn_T"] = TR

    return pd.DataFrame([rows])

# =========================
# Prepare TS datasets (native freq only)
# =========================

def build_ts_datasets(hp: HParams) -> List[Tuple[str, pd.Series, int, bool]]:
    ASSET_END_DATE = "2024-12-31"
    datasets: List[Tuple[str, pd.Series, int, bool]] = []
    # Macro (quarterly)
    macro_obj = sm.datasets.macrodata.load_pandas().data
    year = macro_obj["year"].astype(int)
    q = macro_obj["quarter"].astype(int)
    idx = pd.PeriodIndex(year=year, quarter=q, freq="Q").to_timestamp(how="end")
    cols = ["realgdp", "realcons", "realinv", "unemp", "infl"]
    macro = macro_obj[cols].copy()
    macro.index = idx
    macro = macro.sort_index()

    realgdp = pd.Series(macro["realgdp"].astype(float).values, index=macro.index, name="realgdp").sort_index().dropna()
    realgdp_qoq = realgdp.pct_change().mul(100.0).rename("realgdp_qoq").dropna()

    # Monthly/Annual
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

    # BTC / NG / SP CSV
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
        except Exception as e:
            print("Skipping BTC-USD:", e)

    if hp.use_ts_ng:
        try:
            ng = yf_close_series("NG=F", start="2000-01-01", end=ASSET_END_DATE, auto_adjust=True, cache_dir="cache").rename("NG=F")
            if hp.use_returns_for_assets:
                ng = np.log(ng).diff().dropna().rename("NG_logret"); lag_ng = 60; is_ret = True
            else:
                lag_ng = 60; is_ret = False
            datasets.append(("NaturalGas", ng, lag_ng, is_ret))
        except Exception as e:
            print("Skipping NG=F:", e)

    if hp.use_ts_sp_csv:
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
        except Exception as e:
            print("Skipping S&P:", e)

    return datasets

# =========================
# Prepare Tabular datasets
# =========================

def build_tabular_regression(hp: HParams):
    out = []
    if hp.use_tr_california:
        cal = fetch_california_housing(as_frame=True)
        Xc, yc = cal.frame.drop(columns=["MedHouseVal"]), cal.frame["MedHouseVal"]
        out.append(("CaliforniaHousing", Xc, yc))
    if hp.use_tr_airfoil:
        X, y = load_openml_safe(name="airfoil_self_noise")
        if X is not None: out.append(("AirfoilSelfNoise", X, y))
    if hp.use_tr_concrete:
        X, y = load_openml_safe(name="Concrete Compressive Strength")
        if X is None:
            X, y = load_openml_safe(name="concrete", version=6)
        if X is not None: out.append(("ConcreteStrength", X, y))
    if hp.use_tr_energy:
        X, y = load_openml_safe(name="energy-efficiency")
        if X is not None:
            if "Y1" in X.columns:
                y1 = X["Y1"]; X1 = X.drop(columns=["Y1","Y2"], errors="ignore")
                out.append(("EnergyEfficiency_Y1", X1, y1))
            else:
                out.append(("EnergyEfficiency", X, y))
    if hp.use_tr_yacht:
        X, y = load_openml_safe(name="yacht_hydrodynamics")
        if X is not None: out.append(("YachtHydrodynamics", X, y))
    if hp.use_tr_ccpp:
        X, y = load_openml_safe(name="Combined Cycle Power Plant")
        if X is None:
            X, y = load_openml_safe(name="CCPP")
        if X is not None: out.append(("CCPP", X, y))
    return out

def build_tabular_classification(hp: HParams):
    out = []
    if hp.use_tc_adult:
        X, y = load_openml_safe(name="adult")
        if X is not None: out.append(("Adult", X, y))
    if hp.use_tc_covtype:
        cov = fetch_covtype(as_frame=True)
        Xc, yc = cov.frame.drop(columns=["Cover_Type"]), cov.frame["Cover_Type"]
        out.append(("Covertype", Xc, yc))
    return out

# =========================
# Run everything
# =========================

set_seed(42)
hp = HParams(rolling_folds=3)

all_rows = []
audit_rows = []

# Time series
for name, series, lag, is_ret in build_ts_datasets(hp):
    print(f"=== [TS] Running {name} (lag={lag}) ===")
    df = run_ts(name, series, lag, hp, is_asset_returns=is_ret)
    all_rows.append(df)
    # pull audits
    for _, row in df.iterrows():
        if isinstance(row.get("audit"), dict):
            aud = row["audit"].copy()
            aud.update({"task":"time_series","dataset":row["dataset"],"seed":row["seed"],"fold":row["fold"]})
            audit_rows.append(aud)

# Tabular Regression
for name, X, y in build_tabular_regression(hp):
    print(f"=== [TR] Running {name} ===")
    df = run_tabular_regression(name, X, y, hp)
    all_rows.append(df)

# Tabular Classification
for name, X, y in build_tabular_classification(hp):
    print(f"=== [TC] Running {name} ===")
    df = run_tabular_classification(name, X, y, hp)
    all_rows.append(df)

if all_rows:
    results_df = pd.concat(all_rows, ignore_index=True)
else:
    results_df = pd.DataFrame()



def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjustment.
    Returns q-values with right-to-left monotonicity enforced.
    NaNs are ignored and preserved.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    mask = ~np.isnan(p)
    if not np.any(mask):
        return q

    p_nonan = p[mask]
    m = p_nonan.size
    order = np.argsort(p_nonan, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)

    q_raw = p_nonan * m / ranks
    q_sorted = np.minimum.accumulate(q_raw[order][::-1])[::-1]
    q_adj = np.minimum(q_sorted, 1.0)

    q[mask] = q_adj[np.argsort(order)]  # invert the ordering
    return q

# =========================
# PATCH: per-dataset aggregation + BH
# =========================
from scipy.stats import norm

def stouffer(pvals: np.ndarray) -> float:
    p = np.asarray([x for x in pvals if np.isfinite(x) and not np.isnan(x)])
    if p.size == 0: return np.nan
    z = norm.isf(p)  # inverse survival => z for upper tail
    z_sum = np.sum(z)
    z_comb = z_sum / np.sqrt(len(z))
    return float(norm.sf(z_comb))  # combined p (upper tail)

dm_cols = [c for c in results_df.columns if c.startswith("dm_p_")]
if dm_cols:
    ds_groups = results_df.groupby(["task","dataset"], sort=False)
    rows = []
    for (task, ds), block in ds_groups:
        row = {"task": task, "dataset": ds}
        for c in dm_cols:
            p = block[c].values
            row[c.replace("dm_p_", "dm_p_combined_")] = stouffer(p)
        rows.append(row)
    per_dataset = pd.DataFrame(rows)

    # BH across datasets (per combined-DM column)
    for c in [k for k in per_dataset.columns if k.startswith("dm_p_combined_")]:
        qcol = c.replace("dm_p_combined_", "dm_q_combined_")
        per_dataset[qcol] = bh_adjust(per_dataset[c].values)

    # save also
    per_dataset.to_csv("./examples/results/per_dataset_inference.csv", index=False)


# =========================
# FDR control (BH) for DM p-values
# =========================
dm_p_cols = [c for c in results_df.columns if c.startswith("dm_p_")]

if dm_p_cols:
    df = results_df  # alias

    # -- (A) Global BH across all rows for each DM column
    for c in dm_p_cols:
        qcol = c.replace("dm_p_", "dm_q_")
        df[qcol] = bh_adjust(df[c].values)

    # -- (B) Per-dataset BH (optional, useful for per-dataset summaries)
    for c in dm_p_cols:
        qcol_ds = c.replace("dm_p_", "dm_q_ds_")
        df[qcol_ds] = np.nan
        for ds, idx in df.groupby("dataset").groups.items():
            block = df.loc[idx, c]
            df.loc[idx, qcol_ds] = bh_adjust(block.values)

    # OPTIONAL: a quick summary CSV showing how many discoveries per row at q=0.10
    FDR_Q = 0.10
    sig_cols_global = [c.replace("dm_p_", "dm_q_") for c in dm_p_cols if c.replace("dm_p_", "dm_q_") in df.columns]
    if sig_cols_global:
        sig_tbl = (
            df[["task","dataset"] + sig_cols_global]
            .assign(**{f"discoveries@{FDR_Q}": lambda d: (d[sig_cols_global] <= FDR_Q).sum(axis=1)})
            .sort_values(["task","dataset"])
        )
        os.makedirs("./examples/results", exist_ok=True)
        sig_tbl.to_csv("./examples/results/fdr_summary.csv", index=False)

# =========================
# Results to disk (+ audit artifacts)
# =========================

os.makedirs("./examples/results", exist_ok=True)
OUTPUT_TXT_PATH = "./examples/results/resultsE.txt"
OUTPUT_CSV_PATH = "./examples/results/resultsE.csv"
OUTPUT_AUDIT_JSON = "./examples/results/audit.json"

def fmt_ts_block(df):
    base_cols = [
        "task","dataset","seed","fold","n_train","n_val","n_test",
        "budget_total","budget_warmup","budget_spec_total","budget_spec_each","budget_gate",
        "baseline_rmse","baseline_mae",
        "moe_rmse","moe_mae","moeR_rmse","moeR_mae",
        "moeN_rmse","moeN_mae",
        "eqp_rmse","eqp_mae",
        "lstm_rmse","lstm_mae",
        "moeLSTM_rmse","moeLSTM_mae","moeLSTMR_rmse","moeLSTMR_mae",
        "rw_rmse","rw_mae","arima_rmse","arima_mae",
        "delta_rmse_moe_mlp","ci95_delta_rmse_moe_mlp_lo","ci95_delta_rmse_moe_mlp_hi",
        "delta_mae_moe_mlp","ci95_delta_mae_moe_mlp_lo","ci95_delta_mae_moe_mlp_hi",
        "dm_stat_moe_mlp","dm_p_moe_mlp",
        "delta_rmse_moeR_mlp","ci95_delta_rmse_moeR_mlp_lo","ci95_delta_rmse_moeR_mlp_hi",
        "dm_stat_moeR_mlp","dm_p_moeR_mlp",
        "delta_rmse_moe_lstm","ci95_delta_rmse_moe_lstm_lo","ci95_delta_rmse_moe_lstm_hi",
        "dm_stat_moe_lstm","dm_p_moe_lstm",
        "delta_rmse_moeR_lstm","ci95_delta_rmse_moeR_lstm_lo","ci95_delta_rmse_moeR_lstm_hi",
        "dm_stat_moeR_lstm","dm_p_moeR_lstm",
        "delta_rmse_eqp","ci95_delta_rmse_eqp_lo","ci95_delta_rmse_eqp_hi",
        "dm_stat_eqp","dm_p_eqp",
        "delta_rmse_naive","ci95_delta_rmse_naive_lo","ci95_delta_rmse_naive_hi",
        "dm_stat_naive","dm_p_naive",
        "delta_rmse_rw","ci95_delta_rmse_rw_lo","ci95_delta_rmse_rw_hi",
        "dm_stat_rw","dm_p_rw",
        "delta_rmse_arima","ci95_delta_rmse_arima_lo","ci95_delta_rmse_arima_hi",
        "dm_stat_arima","dm_p_arima",
    ]
    # If BH columns exist, add them (global and/or per-dataset)
    q_cols = [c for c in df.columns if c.startswith("dm_q_")]
    cols = [c for c in base_cols + q_cols if c in df.columns]
    return df[cols].sort_values(["dataset","seed","fold"]).to_string(index=False)


def fmt_tr_block(df):
    cols = [c for c in [
        "task","dataset","n_train","n_val","n_test",
        "budget_total","budget_warmup","budget_spec_each","budget_gate",
        "baseline_rmse","baseline_mae",
        "moe_rmse","moe_mae","moeR_rmse","moeR_mae"
    ] if c in df.columns]
    return df[cols].to_string(index=False)

def fmt_tc_block(df):
    cols = [c for c in [
        "task","dataset","n_train","n_val","n_test",
        "budget_total","budget_warmup","budget_spec_each","budget_gate",
        "mlp_acc","mlp_logloss","mlp_T",
        "cnn_acc","cnn_logloss","cnn_T",
        "rf_acc","rf_logloss",
        "moe_mlp_acc","moe_mlp_logloss","moe_mlp_T",
        "moeR_mlp_acc","moeR_mlp_logloss","moeR_mlp_T",
        "moe_cnn_acc","moe_cnn_logloss","moe_cnn_T",
        "moeR_cnn_acc","moeR_cnn_logloss","moeR_cnn_T"
    ] if c in df.columns]
    return df[cols].to_string(index=False)

lines = []
if not results_df.empty:
    ts_df = results_df[results_df["task"] == "time_series"]
    tr_df = results_df[results_df["task"] == "tabular_regression"]
    tc_df = results_df[results_df["task"] == "tabular_classification"]

    legend = [
        "SIGN CONVENTION:",
        "  Bootstrap delta = metric(tag) - metric(base). Negative is better (improvement vs baseline).",
        "  DM test sign: positive DM ⇒ baseline worse than tag; negative DM ⇒ baseline better.",
        ""
    ]
    lines.extend(legend)

    if not ts_df.empty:
        lines.append("=== TIME SERIES (per-run) ===")
        lines.append(fmt_ts_block(ts_df))
        lines.append("")
    if not tr_df.empty:
        lines.append("=== TABULAR REGRESSION (per-run) ===")
        lines.append(fmt_tr_block(tr_df))
        lines.append("")
    if not tc_df.empty:
        lines.append("=== TABULAR CLASSIFICATION (per-run) ===")
        lines.append(fmt_tc_block(tc_df))
        lines.append("")

with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    if lines:
        f.write("\n".join(lines))
    else:
        f.write("No datasets were available to run.\n")

# CSV + JSON audit
results_df.to_csv(OUTPUT_CSV_PATH, index=False)
# === Run header: environment + seeds (inserted once, at the top of audit JSON) ===
try:
    audit_rows.insert(0, {
        "task": "_env",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "statsmodels_version": statsmodels.__version__,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "seeds": list(hp.seeds),
        "bootstrap_block_frac": hp.bootstrap_block_frac,
        "asset_end_date": "2024-12-31"
    })
except Exception as e:
    print("[audit header] skipping env header due to:", e)

with open(OUTPUT_AUDIT_JSON, "w", encoding="utf-8") as jf:
    json.dump(audit_rows, jf, indent=2)

print(f"Wrote results to: {OUTPUT_TXT_PATH}")
print(f"CSV: {OUTPUT_CSV_PATH}")
print(f"Audit JSON: {OUTPUT_AUDIT_JSON}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Macrodata MoE vs baselines with SOTA-style comparisons.

Adds:
- Classical baselines on the *same* target (real GDP log growth, %): Naive, AR(1), AR(p<=4 via AIC).
- Relative errors (RRMSE/RMAE) vs baselines.
- Diebold–Mariano tests (h=1, squared-error loss, HAC variance).
- Optional expanding-window evaluation for all models.

No files, no internet required.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------
# Reproducibility & device
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Hyperparameters & toggles
# ---------------------------

@dataclass
class HParams:
    # features
    max_lag: int = 4  # quarterly growth lags

    # training
    batch_size: int = 32
    weight_decay: float = 1e-4
    lr_generalist: float = 1e-3
    lr_specialist: float = 1e-3
    lr_gate: float = 5e-4
    lr_baseline: float = 1e-3

    # epochs
    warmup_generalist_epochs: int = 60
    specialist_epochs: int = 30
    gate_max_epochs: int = 80
    gate_patience: int = 10

    # models
    generalist_hidden: int = 128
    specialist_hidden: int = 128
    baseline_hidden: int = 128
    dropout: float = 0.2
    lstm_hidden: int = 32
    lstm_layers: int = 2

    # hard-sample mining
    hard_fraction: float = 0.30
    n_specialists: int = 3

    # misc
    seed: int = 42

# Use expanding-window OOS evaluation for baselines (always) and optionally for ML models.
USE_EXPANDING = False  # set True for rolling-origin retraining of ML models (slower, more "SOTA-like")

# ---------------------------
# Data (in-memory Macrodata)
# ---------------------------

def load_macro_df() -> pd.DataFrame:
    d = sm.datasets.macrodata.load_pandas().data
    year = d["year"].astype(int)
    q = d["quarter"].astype(int)
    idx = pd.PeriodIndex(year=year, quarter=q, freq="Q").to_timestamp(how="end")
    cols = ["realgdp", "realcons", "realinv", "unemp", "infl"]
    df = d[cols].copy()
    df.index = idx
    df = df.sort_index()
    return df

def build_features_from_df(df: pd.DataFrame, max_lag=4):
    """
    Target (y): real GDP log growth in percent:
        g_t = 100 * (log(realgdp_t) - log(realgdp_{t-1}))
    Features (leak-free):
      - gdp_growth_lag1..lagN (first max_lag columns -> LSTM gate sequence)
      - gdp_growth_roll3_std (shifted)
      - exogenous lags at t-1: realcons_lag1, realinv_lag1, unemp_lag1, infl_lag1
    """
    required = ["realgdp", "realcons", "realinv", "unemp", "infl"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in macro dataset.")

    df = df.sort_index().copy()
    df["log_realgdp"] = np.log(df["realgdp"].clip(lower=1e-9))
    df["gdp_growth"] = 100.0 * (df["log_realgdp"] - df["log_realgdp"].shift(1))

    for lag in range(1, max_lag + 1):
        df[f"gdp_growth_lag{lag}"] = df["gdp_growth"].shift(lag)

    df["gdp_growth_roll3_std"] = df["gdp_growth"].shift(1).rolling(3).std()

    exog = ["realcons", "realinv", "unemp", "infl"]
    for c in exog:
        df[f"{c}_lag1"] = df[c].shift(1)

    df = df.dropna()

    feature_cols = [f"gdp_growth_lag{lag}" for lag in range(1, max_lag + 1)]
    feature_cols += ["gdp_growth_roll3_std"] + [f"{c}_lag1" for c in exog]

    X = df[feature_cols].values
    y = df["gdp_growth"].values  # growth target (%, real GDP)
    dates = df.index.to_numpy()
    return X, y, dates, max_lag, len(feature_cols), feature_cols

def time_split(X, y, dates, test_ratio=0.2, val_ratio=0.2):
    n = len(y)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    return (
        X[:val_start], X[val_start:test_start], X[test_start:],
        y[:val_start], y[val_start:test_start], y[test_start:],
        dates[:val_start], dates[val_start:test_start], dates[test_start:],
    )

def scale_after_split(X_tr, X_val, X_te, y_tr, y_val, y_te):
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_y = MinMaxScaler().fit(y_tr.reshape(-1, 1))

    X_train = scaler_X.transform(X_tr)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_te)

    y_train = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

# ---------------------------
# Dataset
# ---------------------------

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------------------------
# Models
# ---------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

class GatingNetwork(nn.Module):
    """ LSTM reads the growth-lag slice (first max_lag columns) and outputs softmax weights over experts. """
    def __init__(self, seq_len, num_experts, lstm_hidden=32, hidden_dim=128, dropout=0.2, lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x_seq):
        x = x_seq.unsqueeze(-1)  # [B, seq_len, 1]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logits = self.fc(h)
        return self.softmax(logits)

# ---------------------------
# Train / Eval helpers
# ---------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        bs = yb.size(0); total += loss.item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(predict_fn, loader, criterion):
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb)
        bs = yb.size(0); total += criterion(pred, yb).item() * bs; n += bs
    return total / max(n, 1)

@torch.no_grad()
def per_sample_losses(model, X_np, y_np, batch_size=512):
    model.eval()
    N = len(y_np); losses = np.zeros(N, dtype=np.float32)
    crit = nn.MSELoss(reduction="none")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
        yb = torch.tensor(y_np[start:end], dtype=torch.float32, device=device).unsqueeze(1)
        pred = model(Xb)
        losses[start:end] = crit(pred, yb).squeeze(1).cpu().numpy()
    return losses

@torch.no_grad()
def direction_acc_growth_from_arrays(pred_inv, true_inv):
    return float(((pred_inv >= 0) == (true_inv >= 0)).mean())

@torch.no_grad()
def direction_acc_growth(predict_fn, loader, scaler_y):
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        pred = predict_fn(Xb).cpu().numpy()  # scaled
        true = yb.cpu().numpy()              # scaled
        pred_inv = scaler_y.inverse_transform(pred)
        true_inv = scaler_y.inverse_transform(true)
        correct += ((pred_inv >= 0) == (true_inv >= 0)).sum()
        total += len(true)
    return float(correct) / max(total, 1)

@torch.no_grad()
def moe_predict(gate, experts, xb, seq_len):
    ex_outs = [ex(xb) for ex in experts]
    w = gate(xb[:, :seq_len])
    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
    return out

# ---------------------------
# Classical baselines & stats
# ---------------------------

def naive_forecast(y_train, y_test_len):
    """ Persistence on growth: ŷ_t = y_{t-1}. """
    last = y_train[-1]
    return np.full(y_test_len, last, dtype=float)

def fit_ar_best_aic(y_train, max_lag=4):
    """ Select AR(p) with p<=max_lag by AIC. """
    best = None
    best_p = 1
    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(y_train, lags=p, old_names=False).fit()
            aic = model.aic
            if best is None or aic < best[0]:
                best = (aic, model)
                best_p = p
        except Exception:
            continue
    if best is None:
        # fallback to AR(1)
        best = (np.inf, AutoReg(y_train, lags=1, old_names=False).fit())
        best_p = 1
    return best[1], best_p

def ar_forecast(y_train, y_test_len, p=1):
    model = AutoReg(y_train, lags=p, old_names=False).fit()
    # dynamic='in-sample' one-step ahead
    preds = model.predict(start=len(y_train), end=len(y_train)+y_test_len-1, dynamic=False)
    return np.asarray(preds, dtype=float), model

def diebold_mariano_test(e1, e2, h=1):
    """
    DM test for equal predictive accuracy (squared-error loss).
    Returns (DM statistic, p-value). Uses HAC variance with Bartlett kernel and lag = floor(n**(1/3)).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    assert e1.shape == e2.shape
    d = (e1**2 - e2**2)  # loss differential; positive means model 2 better (lower loss) if we test e1 vs e2
    n = d.shape[0]
    d_mean = d.mean()

    # HAC variance (Bartlett kernel)
    L = int(np.floor(n ** (1/3)))
    gamma0 = np.var(d, ddof=1)
    var_hac = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        cov = np.cov(d[l:], d[:-l], ddof=1)[0, 1]
        var_hac += 2 * w * cov
    var_hac = var_hac / n

    # DM statistic (Harvey small-sample adjustment optional; h=1 -> no extra terms)
    DM = d_mean / np.sqrt(var_hac + 1e-12)
    # two-sided p-value using t_{n-1}
    from scipy.stats import t as student_t
    pval = 2 * (1 - student_t.cdf(abs(DM), df=max(n - 1, 1)))
    return float(DM), float(pval)

# ---------------------------
# MAIN
# ---------------------------

def main():
    hp = HParams()
    set_seed(hp.seed)
    print(f"Device: {device}")

    # Macrodata -> leak-free growth features (no files)
    df = load_macro_df()
    X_raw, y_raw, dates, seq_len, feat_dim, feat_names = build_features_from_df(df, max_lag=hp.max_lag)

    # Split (chronological) then scale ML features/target (no leakage)
    X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw, d_tr, d_val, d_te = time_split(X_raw, y_raw, dates)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_X, scaler_y,
    ) = scale_after_split(X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=hp.batch_size, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(ArrayDataset(X_val,   y_val),   batch_size=hp.batch_size, shuffle=False, pin_memory=pin, num_workers=0)
    test_loader  = DataLoader(ArrayDataset(X_test,  y_test),  batch_size=hp.batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    # ----- Classical baselines on *unscaled* growth (train/test split) -----
    # We fit AR models on TRAIN+VAL (typical when test is held out).
    y_train_val = np.concatenate([y_tr_raw, y_val_raw])
    n_test = len(y_te_raw)

    # Naive (persistence) on growth
    naive_pred = naive_forecast(y_train_val, n_test)

    # AR(1)
    ar1_pred, ar1_model = ar_forecast(y_train_val, n_test, p=1)

    # AR(p) AIC select (p <= 4)
    arAIC_model, best_p = fit_ar_best_aic(y_train_val, max_lag=hp.max_lag)
    arAIC_pred = arAIC_model.predict(start=len(y_train_val), end=len(y_train_val)+n_test-1, dynamic=False).astype(float)

    # Evaluate classical baselines
    def _eval_unscaled(pred, true):
        rmse = mean_squared_error(true, pred, squared=False)
        mae  = mean_absolute_error(true, pred)
        diracc = direction_acc_growth_from_arrays(pred, true)
        return rmse, mae, diracc

    naive_rmse, naive_mae, naive_dir = _eval_unscaled(naive_pred, y_te_raw)
    ar1_rmse, ar1_mae, ar1_dir = _eval_unscaled(ar1_pred, y_te_raw)
    arAIC_rmse, arAIC_mae, arAIC_dir = _eval_unscaled(arAIC_pred, y_te_raw)

    # ----- ML: Generalist -> hard mining -> specialists -> gate -----
    generalist = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
    opt_g = optim.Adam(generalist.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
    crit = nn.MSELoss()

    if not USE_EXPANDING:
        print("Training generalist (warmup for hard-sample mining)...")
        for ep in range(1, hp.warmup_generalist_epochs + 1):
            tr_loss = train_epoch(generalist, train_loader, crit, opt_g)
            if ep % 10 == 0 or ep == hp.warmup_generalist_epochs:
                val_loss = eval_epoch(lambda xb: generalist(xb), val_loader, crit)
                print(f" Gen Ep{ep:3d} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Hard mining on scaled TRAIN only
        losses = per_sample_losses(generalist, X_train, y_train, batch_size=1024)
        N = len(losses)
        k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        hard_X = X_train[hard_idx]; hard_y = y_train[hard_idx]

        print(f"Clustering {k} hard samples into {hp.n_specialists} specialists...")
        km = KMeans(n_clusters=hp.n_specialists, random_state=hp.seed, n_init=10)
        hard_labels = km.fit_predict(hard_X)

        specialists = []
        print("Training specialists...")
        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = hard_X[mask]; yc = hard_y[mask]
            if len(Xc) == 0:
                Xc, yc = hard_X, hard_y
            dl_c = DataLoader(ArrayDataset(Xc, yc), batch_size=hp.batch_size, shuffle=True, pin_memory=pin, num_workers=0)
            spec = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
            opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in range(hp.specialist_epochs): _ = train_epoch(spec, dl_c, crit, opt_s)
            spec.eval(); specialists.append(spec)
            print(f" Specialist {c+1}/{hp.n_specialists} trained on {len(Xc)} samples.")

        experts = [generalist.eval()] + specialists

        gate = GatingNetwork(
            seq_len=seq_len, num_experts=len(experts),
            lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
            dropout=hp.dropout, lstm_layers=hp.lstm_layers
        ).to(device)

        # Train gate on TRAIN, early stop on VAL
        train_loader_gate = DataLoader(ArrayDataset(X_train, y_train), batch_size=hp.batch_size, shuffle=True, pin_memory=pin, num_workers=0)
        val_loader_gate   = DataLoader(ArrayDataset(X_val,   y_val),   batch_size=hp.batch_size, shuffle=False, pin_memory=pin, num_workers=0)
        def train_gate(gate, experts, train_loader, val_loader):
            for ex in experts:
                ex.eval()
                for p in ex.parameters(): p.requires_grad_(False)
            optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
            best_val = float("inf"); best_state = None; patience = 0; epochs = 0
            print("Training gating network...")
            for ep in range(1, hp.gate_max_epochs + 1):
                gate.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    with torch.no_grad(): ex_outs = [ex(Xb) for ex in experts]
                    w = gate(Xb[:, :hp.max_lag])
                    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                    loss = crit(out, yb)
                    optg.zero_grad(); loss.backward(); optg.step()
                # validate
                gate.eval()
                vtot, n = 0.0, 0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(device), yb.to(device)
                        ex_outs = [ex(Xb) for ex in experts]
                        w = gate(Xb[:, :hp.max_lag])
                        out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                        bs = yb.size(0); vtot += crit(out, yb).item() * bs; n += bs
                v = vtot / max(n,1)
                epochs += 1
                print(f" Gate Ep{ep:2d} | Val MSE: {v:.6f}")
                if v < best_val - 1e-8:
                    best_val = v; best_state = gate.state_dict(); patience = 0
                else:
                    patience += 1
                    if patience > hp.gate_patience:
                        print(" Early stopping gating."); break
            if best_state is not None:
                gate.load_state_dict(best_state)
            return gate, epochs

        gate, gate_epochs = train_gate(gate, experts, train_loader_gate, val_loader_gate)

        # Evaluate MoE on TEST (scaled MSE; inverse RMSE/MAE in % growth)
        def moe_pred_fn(xb): return moe_predict(gate, experts, xb, seq_len)
        moe_mse_scaled = eval_epoch(lambda xb: moe_pred_fn(xb), test_loader, crit)

        @torch.no_grad()
        def gather_preds_targets(predict_fn, loader):
            preds, targs = [], []
            for Xb, yb in loader:
                Xb = Xb.to(device)
                preds.append(predict_fn(Xb).cpu().numpy())
                targs.append(yb.cpu().numpy())
            preds = np.vstack(preds); targs = np.vstack(targs)
            preds_inv = scaler_y.inverse_transform(preds)
            targs_inv = scaler_y.inverse_transform(targs)
            return preds_inv.squeeze(), targs_inv.squeeze()

        moe_p, moe_t = gather_preds_targets(lambda xb: moe_pred_fn(xb), test_loader)
        moe_rmse = mean_squared_error(moe_t, moe_p, squared=False)
        moe_mae  = mean_absolute_error(moe_t, moe_p)
        moe_dir  = direction_acc_growth_from_arrays(moe_p, moe_t)

        # Matched-budget baseline MLP
        total_moe_epochs = hp.warmup_generalist_epochs + hp.n_specialists * hp.specialist_epochs + gate_epochs
        baseline = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt_b = optim.Adam(baseline.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        print("Training baseline (matched budget)...")
        for ep in range(1, total_moe_epochs + 1):
            _ = train_epoch(baseline, train_loader, crit, opt_b)
            if ep % 20 == 0 or ep == total_moe_epochs:
                v = eval_epoch(lambda xb: baseline(xb), val_loader, crit)
                print(f" BL Ep{ep:3d} | Val MSE: {v:.6f}")

        # Baseline eval
        base_p_scaled = []
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            base_p_scaled.append(baseline(Xb).detach().cpu().numpy())
        base_p_scaled = np.vstack(base_p_scaled)
        base_p = scaler_y.inverse_transform(base_p_scaled).squeeze()
        base_rmse = mean_squared_error(moe_t, base_p, squared=False)
        base_mae  = mean_absolute_error(moe_t, base_p)
        base_dir  = direction_acc_growth_from_arrays(base_p, moe_t)

    else:
        # ---- Expanding-window option (retrain ML each step) ----
        # We use TRAIN+VAL as initial window and roll through TEST, refitting fast.
        print("Expanding-window evaluation enabled. This may take longer.")
        # Initial fits on TRAIN+VAL
        X_in = np.vstack([X_train, X_val])
        y_in = np.concatenate([y_train, y_val])
        dates_in = np.concatenate([d_tr, d_val])
        X_out = X_test
        y_out = y_test

        def refit_and_predict_one(X_hist, y_hist, x_next):
            # generalist -> specialists -> gate (light versions)
            gen = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
            opt_g = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
            ds = DataLoader(ArrayDataset(X_hist, y_hist), batch_size=hp.batch_size, shuffle=True, num_workers=0)
            for _ in range(20): _ = train_epoch(gen, ds, crit, opt_g)  # shorter for speed
            losses = per_sample_losses(gen, X_hist, y_hist, batch_size=512)
            k = max(1, int(np.ceil(hp.hard_fraction * len(losses))))
            hard_idx = np.argsort(losses)[-k:]
            hard_X = X_hist[hard_idx]; hard_y = y_hist[hard_idx]
            km = KMeans(n_clusters=hp.n_specialists, random_state=hp.seed, n_init=5)
            hard_labels = km.fit_predict(hard_X)
            specs = []
            for c in range(hp.n_specialists):
                mask = (hard_labels == c)
                Xc = hard_X[mask]; yc = hard_y[mask]
                if len(Xc) == 0: Xc, yc = hard_X, hard_y
                dl = DataLoader(ArrayDataset(Xc, yc), batch_size=hp.batch_size, shuffle=True, num_workers=0)
                sp = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
                opt_s = optim.Adam(sp.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
                for _ in range(10): _ = train_epoch(sp, dl, crit, opt_s)
                sp.eval(); specs.append(sp)
            experts = [gen.eval()] + specs
            gate = GatingNetwork(seq_len=seq_len, num_experts=len(experts), lstm_hidden=hp.lstm_hidden,
                                 hidden_dim=hp.baseline_hidden, dropout=hp.dropout, lstm_layers=hp.lstm_layers).to(device)
            # quick gate train
            ds_gate = DataLoader(ArrayDataset(X_hist, y_hist), batch_size=hp.batch_size, shuffle=True, num_workers=0)
            optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
            for _ in range(20):
                for Xb, yb in ds_gate:
                    Xb, yb = Xb.to(device), yb.to(device)
                    with torch.no_grad(): ex_outs = [ex(Xb) for ex in experts]
                    w = gate(Xb[:, :seq_len])
                    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                    loss = crit(out, yb)
                    optg.zero_grad(); loss.backward(); optg.step()
            with torch.no_grad():
                xb = torch.tensor(x_next[None, :], dtype=torch.float32, device=device)
                yhat = moe_predict(gate, experts, xb, seq_len).cpu().numpy().squeeze()
            return yhat

        # Roll
        moe_p_list = []
        base_p_list = []
        for t in range(len(X_out)):
            X_hist = np.vstack([X_in, X_out[:t]])
            y_hist = np.concatenate([y_in, y_out[:t]])
            # Refit baseline MLP briefly
            base = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
            opt_b = optim.Adam(base.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
            dsb = DataLoader(ArrayDataset(X_hist, y_hist), batch_size=hp.batch_size, shuffle=True, num_workers=0)
            for _ in range(20): _ = train_epoch(base, dsb, crit, opt_b)
            with torch.no_grad():
                xb = torch.tensor(X_out[t][None, :], dtype=torch.float32, device=device)
                base_p_list.append(base(xb).cpu().numpy().squeeze())

            # Refit MoE quickly and predict one step
            moe_p_list.append(refit_and_predict_one(X_hist, y_hist, X_out[t]))

        # Gather scaled -> invert
        moe_p_scaled = np

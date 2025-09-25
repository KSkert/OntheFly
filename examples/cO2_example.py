# Device: cpu
# Training generalist (warmup for hard-sample mining)...
#  Gen Ep 10 | Train MSE: 0.005347 | Val MSE: 0.022568
#  Gen Ep 20 | Train MSE: 0.005480 | Val MSE: 0.020377
#  Gen Ep 30 | Train MSE: 0.004190 | Val MSE: 0.018523
#  Gen Ep 40 | Train MSE: 0.003129 | Val MSE: 0.012441
# Clustering 84 hard samples into 3 specialists...
# Training specialists...
#  Specialist 1/3 trained on 39 samples.
#  Specialist 2/3 trained on 25 samples.
#  Specialist 3/3 trained on 20 samples.
# Training gating network...
#  Gate Ep 1 | Val MSE: 0.106500
#  Gate Ep 2 | Val MSE: 0.097594
#  Gate Ep 3 | Val MSE: 0.087503
#  Gate Ep 4 | Val MSE: 0.073850
#  Gate Ep 5 | Val MSE: 0.055270
#  Gate Ep 6 | Val MSE: 0.031318
#  Gate Ep 7 | Val MSE: 0.012222
#  Gate Ep 8 | Val MSE: 0.004896
#  Gate Ep 9 | Val MSE: 0.003461
#  Gate Ep10 | Val MSE: 0.003143
#  Gate Ep11 | Val MSE: 0.003141
#  Gate Ep12 | Val MSE: 0.003261
#  Gate Ep13 | Val MSE: 0.003267
#  Gate Ep14 | Val MSE: 0.003311
#  Gate Ep15 | Val MSE: 0.003368
#  Gate Ep16 | Val MSE: 0.003423
#  Gate Ep17 | Val MSE: 0.003314
#  Gate Ep18 | Val MSE: 0.003394
#  Gate Ep19 | Val MSE: 0.003386
#  Gate Ep20 | Val MSE: 0.003371
#  Early stopping gating.
# ## Total MoE epochs: 135
# MoE Test - MSE (scaled): 0.009182, DirAcc: 0.3619, RMSE: 3.3945, MAE: 3.0200
# Training baseline (matched budget)...
#  BL Ep 20 | Val MSE: 0.017998
#  BL Ep 40 | Val MSE: 0.014707
#  BL Ep 60 | Val MSE: 0.013702
#  BL Ep 80 | Val MSE: 0.025316
#  BL Ep100 | Val MSE: 0.018081
#  BL Ep120 | Val MSE: 0.014900
#  BL Ep135 | Val MSE: 0.012055
# Baseline Test - MSE (scaled): 0.039457, DirAcc: 0.3619, RMSE: 6.8630, MAE: 5.5167
"""
Hard-sample mining and clustering on the Mauna Loa CO₂ dataset:
"""


import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
# Hyperparameters
# ---------------------------

@dataclass
class HParams:
    # features
    max_lag: int = 3

    # training
    batch_size: int = 32
    weight_decay: float = 1e-4
    lr_generalist: float = 1e-3
    lr_specialist: float = 1e-3
    lr_gate: float = 5e-4
    lr_baseline: float = 1e-3

    # epochs
    warmup_generalist_epochs: int = 40
    specialist_epochs: int = 25
    gate_max_epochs: int = 60
    gate_patience: int = 8

    # models
    generalist_hidden: int = 128
    specialist_hidden: int = 128
    baseline_hidden: int = 128
    dropout: float = 0.2
    lstm_hidden: int = 32
    lstm_layers: int = 2  # >=2 so LSTM dropout is effective

    # hard-sample mining
    hard_fraction: float = 0.25  # top 25% hardest train samples
    n_specialists: int = 3

    # misc
    seed: int = 42

# ---------------------------
# Data (in-memory CO₂)
# ---------------------------

def load_CO2_df() -> pd.DataFrame:
    """
    Return a DataFrame with:
      - DatetimeIndex (monthly)
      - single numeric column: 'co2' (ppm)
    No file I/O, no internet.
    """
    d = sm.datasets.co2.load_pandas().data  # Series or DataFrame; index is timestamp
    # Extract a Series with a DatetimeIndex
    if isinstance(d, pd.DataFrame):
        s = d['co2'] if 'co2' in d.columns else d.select_dtypes(include=[np.number]).iloc[:, 0]
    else:
        s = d  # already a Series

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    # Monthly-start frequency; interpolate small gaps smoothly
    s = s.resample('MS').mean().interpolate(method='time')

    df = s.to_frame(name='co2')  # natural column name
    return df

def build_features_from_df(df: pd.DataFrame, max_lag=3):
    """
    Create leak-free features from a DataFrame with:
      - DatetimeIndex
      - column 'co2'
    Features:
      - co2_lag1..co2_lagN
      - co2_roll3_std (shifted by 1 to avoid leakage)
    Returns: X, y, seq_len, feat_dim
    """
    if 'co2' not in df.columns:
        raise ValueError("Expected a 'co2' column in the input DataFrame.")

    df = df.sort_index().copy()

    # Lags of co2
    for lag in range(1, max_lag + 1):
        df[f'co2_lag{lag}'] = df['co2'].shift(lag)

    # Shifted rolling std (uses data up to t-1)
    df['co2_roll3_std'] = df['co2'].shift(1).rolling(3).std()

    df = df.dropna()

    feature_cols = [f'co2_lag{lag}' for lag in range(1, max_lag + 1)] + ['co2_roll3_std']
    X = df[feature_cols].values
    y = df['co2'].values
    return X, y, max_lag, len(feature_cols)

def time_split(X, y, test_ratio=0.2, val_ratio=0.2):
    n = len(y)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    return (
        X[:val_start],
        X[val_start:test_start],
        X[test_start:],
        y[:val_start],
        y[val_start:test_start],
        y[test_start:],
    )

def scale_after_split(X_tr, X_val, X_te, y_tr, y_val, y_te):
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_y = MinMaxScaler().fit(y_tr.reshape(-1, 1))

    X_train = scaler_X.transform(X_tr)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_te)

    y_train = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_te.reshape(-1, 1)).ravel()

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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)

class GatingNetwork(nn.Module):
    """
    LSTM reads the lag slice (first max_lag columns) and outputs softmax weights over experts.
    """
    def __init__(self, seq_len, num_experts, lstm_hidden=32, hidden_dim=128, dropout=0.2, lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
def direction_acc(predict_fn, loader, lag_idx=0):
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb).squeeze(1)
        prev = Xb[:, lag_idx]  # co2_lag1 assumed at index 0
        correct += ((pred >= prev) == (yb.squeeze(1) >= prev)).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)

@torch.no_grad()
def moe_predict(gate, experts, xb, seq_len):
    ex_outs = [ex(xb) for ex in experts]              # each expert sees all features
    w = gate(xb[:, :seq_len])                         # gate uses lag slice
    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
    return out

def train_gate(gate, experts, train_loader, val_loader, hp: HParams, criterion):
    # freeze experts
    for ex in experts:
        ex.eval()
        for p in ex.parameters(): p.requires_grad_(False)

    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    best_val = float("inf"); best_state = None; patience = 0; gate_epochs = 0

    print("Training gating network...")
    for ep in range(1, hp.gate_max_epochs + 1):
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad(): ex_outs = [ex(Xb) for ex in experts]
            w = gate(Xb[:, :hp.max_lag])
            out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
            loss = criterion(out, yb)
            optg.zero_grad(); loss.backward(); optg.step()

        # validate
        gate.eval()
        val_loss = 0.0; n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                ex_outs = [ex(Xb) for ex in experts]
                w = gate(Xb[:, :hp.max_lag])
                out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                bs = yb.size(0); val_loss += criterion(out, yb).item() * bs; n += bs
        val_loss /= max(n, 1)

        gate_epochs += 1
        print(f" Gate Ep{ep:2d} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss; best_state = gate.state_dict(); patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience:
                print(" Early stopping gating."); break

    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate, gate_epochs

# ---------------------------
# MAIN
# ---------------------------

def main():
    hp = HParams()
    set_seed(hp.seed)
    print(f"Device: {device}")

    # CO₂ -> features (no files)
    df = load_CO2_df()  # DatetimeIndex + 'co2'
    X_raw, y_raw, seq_len, feat_dim = build_features_from_df(df, max_lag=hp.max_lag)

    # Split then scale (no leakage)
    X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw = time_split(X_raw, y_raw)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_X, scaler_y,
    ) = scale_after_split(X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val,   y_val),   batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test,  y_test),  batch_size=hp.batch_size, shuffle=False, pin_memory=pin)

    # 1) Generalist
    generalist = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
    opt_g = optim.Adam(generalist.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
    crit = nn.MSELoss()

    print("Training generalist (warmup for hard-sample mining)...")
    for ep in range(1, hp.warmup_generalist_epochs + 1):
        tr_loss = train_epoch(generalist, train_loader, crit, opt_g)
        if ep % 10 == 0 or ep == hp.warmup_generalist_epochs:
            val_loss = eval_epoch(lambda xb: generalist(xb), val_loader, crit)
            print(f" Gen Ep{ep:3d} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f}")

    # 2) Hard-sample mining & clustering
    losses = per_sample_losses(generalist, X_train, y_train, batch_size=1024)
    N = len(losses)
    k = max(1, int(np.ceil(hp.hard_fraction * N)))
    hard_idx = np.argsort(losses)[-k:]  # top-k hardest
    hard_X = X_train[hard_idx]; hard_y = y_train[hard_idx]

    print(f"Clustering {k} hard samples into {hp.n_specialists} specialists...")
    km = KMeans(n_clusters=hp.n_specialists, random_state=hp.seed, n_init=10)
    hard_labels = km.fit_predict(hard_X)

    # 3) Specialists
    specialists = []
    print("Training specialists...")
    for c in range(hp.n_specialists):
        mask = (hard_labels == c)
        Xc = hard_X[mask]; yc = hard_y[mask]
        if len(Xc) == 0:
            Xc, yc = hard_X, hard_y

        dl_c = DataLoader(ArrayDataset(Xc, yc), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
        spec = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
        opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)

        for _ in range(hp.specialist_epochs): _ = train_epoch(spec, dl_c, crit, opt_s)
        spec.eval(); specialists.append(spec)
        print(f" Specialist {c+1}/{hp.n_specialists} trained on {len(Xc)} samples.")

    experts = [generalist.eval()] + specialists
    num_experts = len(experts)

    # 4) Gate
    gate = GatingNetwork(
        seq_len=seq_len, num_experts=num_experts,
        lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
        dropout=hp.dropout, lstm_layers=hp.lstm_layers
    ).to(device)
    gate, gate_epochs = train_gate(gate, experts, train_loader, val_loader, hp, crit)

    total_moe_epochs = hp.warmup_generalist_epochs + hp.n_specialists * hp.specialist_epochs + gate_epochs
    print(f"## Total MoE epochs: {total_moe_epochs}")

    # 5) Evaluate MoE
    for ex in experts: ex.eval()
    gate.eval()

    moe_mse_scaled = eval_epoch(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, crit)
    moe_dir = direction_acc(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, lag_idx=0)

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

    moe_p, moe_t = gather_preds_targets(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader)
    moe_rmse_px = mean_squared_error(moe_t, moe_p, squared=False)
    moe_mae_px = mean_absolute_error(moe_t, moe_p)

    print(f"MoE Test - MSE (scaled): {moe_mse_scaled:.6f}, DirAcc: {moe_dir:.4f}, "
          f"RMSE: {moe_rmse_px:.4f}, MAE: {moe_mae_px:.4f}")

    # 6) Baseline with matched budget
    baseline = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
    print("Training baseline (matched budget)...")
    for ep in range(1, total_moe_epochs + 1):
        _ = train_epoch(baseline, train_loader, crit, opt_b)
        if ep % 20 == 0 or ep == total_moe_epochs:
            v = eval_epoch(lambda xb: baseline(xb), val_loader, crit)
            print(f" BL Ep{ep:3d} | Val MSE: {v:.6f}")

    base_mse_scaled = eval_epoch(lambda xb: baseline(xb), test_loader, crit)
    base_dir = direction_acc(lambda xb: baseline(xb), test_loader, lag_idx=0)
    base_p, base_t = gather_preds_targets(lambda xb: baseline(xb), test_loader)
    base_rmse_px = mean_squared_error(base_t, base_p, squared=False)
    base_mae_px = mean_absolute_error(base_t, base_p)

    print(f"Baseline Test - MSE (scaled): {base_mse_scaled:.6f}, DirAcc: {base_dir:.4f}, "
          f"RMSE: {base_rmse_px:.4f}, MAE: {base_mae_px:.4f}")

if __name__ == "__main__":
    main()
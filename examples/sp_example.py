# Device: cpu
# Training generalist (warmup for hard-sample mining)...
#  Gen Ep 10 | Train MSE: 0.006135 | Val MSE: 0.021172
#  Gen Ep 20 | Train MSE: 0.006207 | Val MSE: 0.012284
#  Gen Ep 30 | Train MSE: 0.004889 | Val MSE: 0.013292
#  Gen Ep 40 | Train MSE: 0.004518 | Val MSE: 0.011078
# Clustering 113 hard samples into 3 specialists...
# Training specialists...
#  Specialist 1/3 trained on 60 samples.
#  Specialist 2/3 trained on 42 samples.
#  Specialist 3/3 trained on 11 samples.
# Training gating network...
#  Gate Ep 1 | Val MSE: 0.082633
#  Gate Ep 2 | Val MSE: 0.073275
#  Gate Ep 3 | Val MSE: 0.062221
#  Gate Ep 4 | Val MSE: 0.049264
#  Gate Ep 5 | Val MSE: 0.034789
#  Gate Ep 6 | Val MSE: 0.020019
#  Gate Ep 7 | Val MSE: 0.007874
#  Gate Ep 8 | Val MSE: 0.002279
#  Gate Ep 9 | Val MSE: 0.001300
#  Gate Ep10 | Val MSE: 0.001125
#  Gate Ep11 | Val MSE: 0.001115
#  Gate Ep12 | Val MSE: 0.001164
#  Gate Ep13 | Val MSE: 0.001128
#  Gate Ep14 | Val MSE: 0.001202
#  Gate Ep15 | Val MSE: 0.001176
#  Gate Ep16 | Val MSE: 0.001176
#  Gate Ep17 | Val MSE: 0.001138
#  Gate Ep18 | Val MSE: 0.001143
#  Gate Ep19 | Val MSE: 0.001125
#  Gate Ep20 | Val MSE: 0.001110
#  Gate Ep21 | Val MSE: 0.001075
#  Gate Ep22 | Val MSE: 0.001041
#  Gate Ep23 | Val MSE: 0.001055
#  Gate Ep24 | Val MSE: 0.001058
#  Gate Ep25 | Val MSE: 0.001099
#  Gate Ep26 | Val MSE: 0.001025
#  Gate Ep27 | Val MSE: 0.001058
#  Gate Ep28 | Val MSE: 0.001061
#  Gate Ep29 | Val MSE: 0.001065
#  Gate Ep30 | Val MSE: 0.001043
#  Gate Ep31 | Val MSE: 0.000968
#  Gate Ep32 | Val MSE: 0.000955
#  Gate Ep33 | Val MSE: 0.000991
#  Gate Ep34 | Val MSE: 0.000865
#  Gate Ep35 | Val MSE: 0.000892
#  Gate Ep36 | Val MSE: 0.000807
#  Gate Ep37 | Val MSE: 0.000848
#  Gate Ep38 | Val MSE: 0.000921
#  Gate Ep39 | Val MSE: 0.001010
#  Gate Ep40 | Val MSE: 0.000983
#  Gate Ep41 | Val MSE: 0.000983
#  Gate Ep42 | Val MSE: 0.000923
#  Gate Ep43 | Val MSE: 0.000959
#  Gate Ep44 | Val MSE: 0.001005
#  Gate Ep45 | Val MSE: 0.001061
#  Early stopping gating.
# ## Total MoE epochs: 160
# MoE Test - MSE (scaled): 0.005153, DirAcc: 0.4113, RMSE: 40.6944, MAE: 35.2698
# Training baseline (matched budget)...
#  BL Ep 20 | Val MSE: 0.013876
#  BL Ep 40 | Val MSE: 0.010625
#  BL Ep 60 | Val MSE: 0.012199
#  BL Ep 80 | Val MSE: 0.013108
#  BL Ep100 | Val MSE: 0.010661
#  BL Ep120 | Val MSE: 0.008216
#  BL Ep140 | Val MSE: 0.008706
#  BL Ep160 | Val MSE: 0.008323
# Baseline Test - MSE (scaled): 0.020845, DirAcc: 0.4681, RMSE: 76.5357, MAE: 60.4087


"""
Hard-sample mining and clustering for stock regression:
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    lr_gate: float = 5e-4  # gates can be twitchy; moderate LR helps
    lr_baseline: float = 1e-3

    # epochs
    warmup_generalist_epochs: int = 40  # to mine hard samples
    specialist_epochs: int = 25         # per specialist
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
    n_specialists: int = 3       # KMeans K

    # misc
    seed: int = 42


# ---------------------------
# Data
# ---------------------------

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)           # keep on CPU
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(path, max_lag=3):
    """
    Features:
      - Close_lag1..lagN
      - Volume_roll3 (mean) up to t-1
      - Close_roll3_std up to t-1
    Target: Close_t
    """
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")

    for lag in range(1, max_lag + 1):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)

    df["Volume_roll3"] = df["Volume"].shift(1).rolling(3).mean()
    df["Close_roll3_std"] = df["Close"].shift(1).rolling(3).std()

    df.dropna(inplace=True)

    features = [f"Close_lag{lag}" for lag in range(1, max_lag + 1)] + [
        "Volume_roll3",
        "Close_roll3_std",
    ]
    X = df[features].values
    y = df["Close"].values
    return X, y, max_lag, len(features)


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

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """
    LSTM reads the lag sequence (first max_lag columns) and outputs softmax
    weights over experts: [generalist + specialists].
    """
    def __init__(self, seq_len, num_experts, lstm_hidden=32, hidden_dim=128, dropout=0.2, lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_seq):
        x = x_seq.unsqueeze(-1)     # [B, seq_len, 1]
        out, _ = self.lstm(x)       # [B, seq_len, H]
        h = out[:, -1, :]           # [B, H]
        logits = self.fc(h)         # [B, E]
        return self.softmax(logits) # [B, E]


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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = yb.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(predict_fn, loader, criterion):
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb)
        bs = yb.size(0)
        total += criterion(pred, yb).item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def per_sample_losses(model, X_np, y_np, batch_size=512):
    """
    Compute MSE per sample in scaled space on CPU numpy arrays.
    """
    model.eval()
    N = len(y_np)
    losses = np.zeros(N, dtype=np.float32)
    crit = nn.MSELoss(reduction="none")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
        yb = torch.tensor(y_np[start:end], dtype=torch.float32, device=device).unsqueeze(1)
        pred = model(Xb)
        l = crit(pred, yb).squeeze(1).detach().cpu().numpy()
        losses[start:end] = l
    return losses


@torch.no_grad()
def direction_acc(predict_fn, loader, lag_idx=0):
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb).squeeze(1)
        prev = Xb[:, lag_idx]
        correct += ((pred >= prev) == (yb.squeeze(1) >= prev)).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def moe_predict(gate, experts, xb, seq_len):
    # experts consume full features; gate consumes lag slice
    ex_outs = [ex(xb) for ex in experts]              # list of [B,1]
    w = gate(xb[:, :seq_len])                         # [B,E]
    out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))  # [B,1]
    return out


def train_gate(gate, experts, train_loader, val_loader, hp: HParams, criterion):
    # freeze experts
    for ex in experts:
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)

    optg = optim.Adam(gate.parameters(), lr=hp.lr_gate, weight_decay=hp.weight_decay)
    best_val = float("inf")
    best_state = None
    patience = 0
    gate_epochs = 0

    print("Training gating network...")
    for ep in range(1, hp.gate_max_epochs + 1):
        gate.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                ex_outs = [ex(Xb) for ex in experts]
            w = gate(Xb[:, :hp.max_lag])
            out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
            loss = criterion(out, yb)
            optg.zero_grad()
            loss.backward()
            optg.step()

        # validate
        gate.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                ex_outs = [ex(Xb) for ex in experts]
                w = gate(Xb[:, :hp.max_lag])
                out = sum(w[:, [j]] * ex_outs[j] for j in range(len(experts)))
                bs = yb.size(0)
                val_loss += criterion(out, yb).item() * bs
                n += bs
        val_loss /= max(n, 1)

        gate_epochs += 1
        print(f" Gate Ep{ep:2d} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = gate.state_dict()
            patience = 0
        else:
            patience += 1
            if patience > hp.gate_patience:
                print(" Early stopping gating.")
                break

    if best_state is not None:
        gate.load_state_dict(best_state)

    return gate, gate_epochs


# ---------------------------
# MAIN
# ---------------------------

def main():
    # Hardcoded path (as requested)
    path = '../data/S&P.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    hp = HParams()
    set_seed(hp.seed)
    print(f"Device: {device}")

    # Load & split
    X_raw, y_raw, seq_len, feat_dim = load_data(path, max_lag=hp.max_lag)
    X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw = time_split(X_raw, y_raw)

    # Scale AFTER split (no leakage)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_X, scaler_y,
    ) = scale_after_split(X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val_raw, y_te_raw)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(StockDataset(X_test,  y_test),  batch_size=hp.batch_size, shuffle=False, pin_memory=pin)

    # ---------------------------------------------------------
    # 1) Train GENERALIST to mine hard samples
    # ---------------------------------------------------------
    generalist = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
    opt_g = optim.Adam(generalist.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
    crit = nn.MSELoss()

    print("Training generalist (warmup for hard-sample mining)...")
    for ep in range(1, hp.warmup_generalist_epochs + 1):
        tr_loss = train_epoch(generalist, train_loader, crit, opt_g)
        if ep % 10 == 0 or ep == hp.warmup_generalist_epochs:
            val_loss = eval_epoch(lambda xb: generalist(xb), val_loader, crit)
            print(f" Gen Ep{ep:3d} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f}")

    # ---------------------------------------------------------
    # 2) Hard-sample mining & clustering
    # ---------------------------------------------------------
    losses = per_sample_losses(generalist, X_train, y_train, batch_size=1024)
    N = len(losses)
    k = max(1, int(np.ceil(hp.hard_fraction * N)))
    hard_idx = np.argsort(losses)[-k:]  # top-k hardest
    hard_X = X_train[hard_idx]
    hard_y = y_train[hard_idx]

    # cluster hard samples into K specialists
    K = hp.n_specialists
    print(f"Clustering {k} hard samples into {K} specialists...")
    km = KMeans(n_clusters=K, random_state=hp.seed, n_init=10)
    hard_labels = km.fit_predict(hard_X)

    # ---------------------------------------------------------
    # 3) Train SPECIALISTS on their clusters
    # ---------------------------------------------------------
    specialists = []
    print("Training specialists...")
    for c in range(K):
        mask = (hard_labels == c)
        Xc = hard_X[mask]
        yc = hard_y[mask]
        if len(Xc) == 0:
            # safeguard: if a cluster is empty, fall back to all hard samples
            Xc, yc = hard_X, hard_y

        ds_c = StockDataset(Xc, yc)
        dl_c = DataLoader(ds_c, batch_size=hp.batch_size, shuffle=True, pin_memory=pin)

        spec = MLPRegressor(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
        opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)

        for ep in range(1, hp.specialist_epochs + 1):
            _ = train_epoch(spec, dl_c, crit, opt_s)
        spec.eval()
        specialists.append(spec)
        print(f" Specialist {c+1}/{K} trained on {len(Xc)} samples.")

    # generalist becomes Expert 0 (already trained)
    generalist.eval()
    experts = [generalist] + specialists
    num_experts = len(experts)

    # ---------------------------------------------------------
    # 4) Train GATE over [generalist + specialists]
    # ---------------------------------------------------------
    gate = GatingNetwork(
        seq_len=seq_len,
        num_experts=num_experts,
        lstm_hidden=hp.lstm_hidden,
        hidden_dim=hp.baseline_hidden,
        dropout=hp.dropout,
        lstm_layers=hp.lstm_layers,
    ).to(device)

    gate, gate_epochs = train_gate(gate, experts, train_loader, val_loader, hp, crit)

    total_moe_epochs = hp.warmup_generalist_epochs + K * hp.specialist_epochs + gate_epochs
    print(f"## Total MoE epochs: {total_moe_epochs}")

    # ---------------------------------------------------------
    # 5) Evaluate MoE (scaled + inverse-scaled metrics)
    # ---------------------------------------------------------
    gate.eval()
    for ex in experts: ex.eval()

    moe_mse_scaled = eval_epoch(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, crit)
    moe_dir = direction_acc(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, lag_idx=0)

    @torch.no_grad()
    def gather_preds_targets(predict_fn, loader):
        preds, targs = [], []
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(predict_fn(Xb).cpu().numpy())
            targs.append(yb.cpu().numpy())
        preds = np.vstack(preds)
        targs = np.vstack(targs)
        preds_inv = scaler_y.inverse_transform(preds)
        targs_inv = scaler_y.inverse_transform(targs)
        return preds_inv.squeeze(), targs_inv.squeeze()

    moe_p, moe_t = gather_preds_targets(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader)
    moe_rmse_px = mean_squared_error(moe_t, moe_p, squared=False)
    moe_mae_px = mean_absolute_error(moe_t, moe_p)

    print(f"MoE Test - MSE (scaled): {moe_mse_scaled:.6f}, DirAcc: {moe_dir:.4f}, "
          f"RMSE: {moe_rmse_px:.4f}, MAE: {moe_mae_px:.4f}")

    # ---------------------------------------------------------
    # 6) Train a fresh BASELINE for the same total epoch budget
    # ---------------------------------------------------------
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

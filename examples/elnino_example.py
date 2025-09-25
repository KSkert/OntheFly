# Device: cpu
# Training generalist (warmup for hard-sample mining)...
#  Gen Ep 10 | Train MSE: 0.008232 | Val MSE: 0.006756
#  Gen Ep 20 | Train MSE: 0.005059 | Val MSE: 0.008688
#  Gen Ep 30 | Train MSE: 0.004598 | Val MSE: 0.004223
#  Gen Ep 40 | Train MSE: 0.004349 | Val MSE: 0.004748
#  Gen Ep 50 | Train MSE: 0.003951 | Val MSE: 0.004849
#  Gen Ep 60 | Train MSE: 0.003434 | Val MSE: 0.004609
# Clustering 138 hard samples into 3 specialists...
# Training specialists...
#  Specialist 1/3 trained on 54 samples.
#  Specialist 2/3 trained on 35 samples.
#  Specialist 3/3 trained on 49 samples.
# Training gating network...
#  Gate Ep 1 | Val MSE: 0.011178
#  Gate Ep 2 | Val MSE: 0.010227
#  Gate Ep 3 | Val MSE: 0.008748
#  Gate Ep 4 | Val MSE: 0.005995
#  Gate Ep 5 | Val MSE: 0.003071
#  Gate Ep 6 | Val MSE: 0.002618
#  Gate Ep 7 | Val MSE: 0.002599
#  Gate Ep 8 | Val MSE: 0.002607
#  Gate Ep 9 | Val MSE: 0.002613
#  Gate Ep10 | Val MSE: 0.002606
#  Gate Ep11 | Val MSE: 0.002608
#  Gate Ep12 | Val MSE: 0.002605
#  Gate Ep13 | Val MSE: 0.002597
#  Gate Ep14 | Val MSE: 0.002597
#  Gate Ep15 | Val MSE: 0.002594
#  Gate Ep16 | Val MSE: 0.002589
#  Gate Ep17 | Val MSE: 0.002591
#  Gate Ep18 | Val MSE: 0.002586
#  Gate Ep19 | Val MSE: 0.002588
#  Gate Ep20 | Val MSE: 0.002578
#  Gate Ep21 | Val MSE: 0.002575
#  Gate Ep22 | Val MSE: 0.002575
#  Gate Ep23 | Val MSE: 0.002571
#  Gate Ep24 | Val MSE: 0.002573
#  Gate Ep25 | Val MSE: 0.002567
#  Gate Ep26 | Val MSE: 0.002563
#  Gate Ep27 | Val MSE: 0.002564
#  Gate Ep28 | Val MSE: 0.002561
#  Gate Ep29 | Val MSE: 0.002563
#  Gate Ep30 | Val MSE: 0.002564
#  Gate Ep31 | Val MSE: 0.002559
#  Gate Ep32 | Val MSE: 0.002556
#  Gate Ep33 | Val MSE: 0.002556
#  Gate Ep34 | Val MSE: 0.002554
#  Gate Ep35 | Val MSE: 0.002553
#  Gate Ep36 | Val MSE: 0.002554
#  Gate Ep37 | Val MSE: 0.002554
#  Gate Ep38 | Val MSE: 0.002551
#  Gate Ep39 | Val MSE: 0.002549
#  Gate Ep40 | Val MSE: 0.002549
#  Gate Ep41 | Val MSE: 0.002548
#  Gate Ep42 | Val MSE: 0.002549
#  Gate Ep43 | Val MSE: 0.002549
#  Gate Ep44 | Val MSE: 0.002545
#  Gate Ep45 | Val MSE: 0.002546
#  Gate Ep46 | Val MSE: 0.002548
#  Gate Ep47 | Val MSE: 0.002548
#  Gate Ep48 | Val MSE: 0.002547
#  Gate Ep49 | Val MSE: 0.002548
#  Gate Ep50 | Val MSE: 0.002546
#  Gate Ep51 | Val MSE: 0.002546
#  Gate Ep52 | Val MSE: 0.002547
#  Gate Ep53 | Val MSE: 0.002544
#  Gate Ep54 | Val MSE: 0.002547
#  Gate Ep55 | Val MSE: 0.002545
#  Gate Ep56 | Val MSE: 0.002546
#  Gate Ep57 | Val MSE: 0.002547
#  Gate Ep58 | Val MSE: 0.002548
#  Gate Ep59 | Val MSE: 0.002544
#  Gate Ep60 | Val MSE: 0.002545
#  Gate Ep61 | Val MSE: 0.002546
#  Gate Ep62 | Val MSE: 0.002545
#  Gate Ep63 | Val MSE: 0.002543
#  Gate Ep64 | Val MSE: 0.002540
#  Gate Ep65 | Val MSE: 0.002542
#  Gate Ep66 | Val MSE: 0.002544
#  Gate Ep67 | Val MSE: 0.002542
#  Gate Ep68 | Val MSE: 0.002541
#  Gate Ep69 | Val MSE: 0.002542
#  Gate Ep70 | Val MSE: 0.002541
#  Gate Ep71 | Val MSE: 0.002541
#  Gate Ep72 | Val MSE: 0.002542
#  Gate Ep73 | Val MSE: 0.002543
#  Gate Ep74 | Val MSE: 0.002543
#  Gate Ep75 | Val MSE: 0.002542
#  Early stopping gating.
# ## Total MoE epochs: 225
# MoE Test - MSE (scaled): 0.002410, DirAcc(level): 0.8958, RMSE(sst): 0.4860, MAE(sst): 0.4026
# Training baseline (matched budget)...
#  BL Ep 20 | Val MSE: 0.007983
#  BL Ep 40 | Val MSE: 0.005224
#  BL Ep 60 | Val MSE: 0.006189
#  BL Ep 80 | Val MSE: 0.003923
#  BL Ep100 | Val MSE: 0.004906
#  BL Ep120 | Val MSE: 0.003549
#  BL Ep140 | Val MSE: 0.003118
#  BL Ep160 | Val MSE: 0.003318
#  BL Ep180 | Val MSE: 0.003325
#  BL Ep200 | Val MSE: 0.003613
#  BL Ep220 | Val MSE: 0.005917
#  BL Ep225 | Val MSE: 0.002855
# Baseline Test - MSE (scaled): 0.003073, DirAcc(level): 0.8750, RMSE(sst): 0.5671, MAE(sst): 0.4423


"""
Hard-sample mining and clustering on the El Niño monthly SST dataset
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
    max_lag: int = 12  # monthly data: use one seasonal cycle of lags

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
    lstm_layers: int = 2  # >=2 so LSTM dropout is effective

    # hard-sample mining
    hard_fraction: float = 0.30  # top 30% hardest train samples
    n_specialists: int = 3

    # misc
    seed: int = 42

# ---------------------------
# Data (in-memory El Niño SST, native)
# ---------------------------

def load_elnino_df() -> pd.DataFrame:
    """
    Return a monthly DatetimeIndex dataframe with a single column 'sst' (°C).
    The original dataset has rows as years and columns JAN..DEC; we melt to monthly series.
    """
    d = sm.datasets.elnino.load_pandas().data  # columns: YEAR, JAN..DEC (floats, some NaNs)
    # Melt year-month grid into long monthly series
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df_long = d.melt(id_vars=["YEAR"], value_vars=months, var_name="month_abbr", value_name="sst")
    # Map month abbreviations to month numbers
    month_map = {m:i+1 for i,m in enumerate(months)}
    df_long["month"] = df_long["month_abbr"].map(month_map)
    # Build a DatetimeIndex at month start
    dt = pd.to_datetime(dict(year=df_long["YEAR"].astype(int), month=df_long["month"].astype(int), day=1))
    sst = pd.Series(df_long["sst"].astype(float).values, index=dt).sort_index()
    # Monthly series: average within month (in case of duplicates), interpolate small gaps
    s = sst.resample("MS").mean().interpolate(method="time")
    return s.to_frame(name="sst")

def build_features_from_df(df: pd.DataFrame, max_lag=12):
    """
    Features (leak-free):
      - sst_lag1..sst_lagN  (these first 'max_lag' columns form the gate's sequence)
      - sst_roll3_std       (shifted)
      - cyclical month encoding: sin_month, cos_month  (shifted is unnecessary for exogenous calendar vars)
    Target: sst level
    Returns: X, y, seq_len, feat_dim
    """
    if "sst" not in df.columns:
        raise ValueError("Expected 'sst' column in the El Niño dataset.")
    df = df.sort_index().copy()

    # Month-of-year features (cyclical)
    mo = df.index.month.values.astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * mo / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * mo / 12.0)

    # SST lags
    for lag in range(1, max_lag + 1):
        df[f"sst_lag{lag}"] = df["sst"].shift(lag)

    # Shifted rolling std of SST (no leakage)
    df["sst_roll3_std"] = df["sst"].shift(1).rolling(3).std()

    df = df.dropna()

    # First max_lag features must be the lags (for LSTM gate slice)
    feature_cols = [f"sst_lag{lag}" for lag in range(1, max_lag + 1)]
    # Then volatility + cyclical month features
    feature_cols += ["sst_roll3_std", "sin_month", "cos_month"]

    X = df[feature_cols].values
    y = df["sst"].values
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
def direction_acc_level(predict_fn, loader, lag_idx=0):
    """
    Direction on *level*: sign(ŷ_t - x_{t, lag1}) vs sign(y_t - x_{t, lag1}),
    where x_{t, lag1} is the first lag feature (sst_{t-1}).
    """
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = predict_fn(Xb).squeeze(1)
        prev = Xb[:, lag_idx]  # sst_lag1 at index 0
        correct += ((pred >= prev) == (yb.squeeze(1) >= prev)).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)

@torch.no_grad()
def moe_predict(gate, experts, xb, seq_len):
    ex_outs = [ex(xb) for ex in experts]              # each expert sees all features
    w = gate(xb[:, :seq_len])                         # gate uses first seq_len = max_lag features (sst lags)
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

    # El Niño SST -> features (no files)
    df = load_elnino_df()  # DatetimeIndex + 'sst'
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

    # 5) Evaluate MoE (scaled loss + inverse-transformed RMSE/MAE on level)
    for ex in experts: ex.eval()
    gate.eval()

    moe_mse_scaled = eval_epoch(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, crit)
    moe_dir = direction_acc_level(lambda xb: moe_predict(gate, experts, xb, seq_len), test_loader, lag_idx=0)

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
    moe_rmse = mean_squared_error(moe_t, moe_p, squared=False)
    moe_mae = mean_absolute_error(moe_t, moe_p)

    print(f"MoE Test - MSE (scaled): {moe_mse_scaled:.6f}, DirAcc(level): {moe_dir:.4f}, "
          f"RMSE(sst): {moe_rmse:.4f}, MAE(sst): {moe_mae:.4f}")

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
    base_dir = direction_acc_level(lambda xb: baseline(xb), test_loader, lag_idx=0)
    base_p, base_t = gather_preds_targets(lambda xb: baseline(xb), test_loader)
    base_rmse = mean_squared_error(base_t, base_p, squared=False)
    base_mae = mean_absolute_error(base_t, base_p)

    print(f"Baseline Test - MSE (scaled): {base_mse_scaled:.6f}, DirAcc(level): {base_dir:.4f}, "
          f"RMSE(sst): {base_rmse:.4f}, MAE(sst): {base_mae:.4f}")

if __name__ == "__main__":
    main()

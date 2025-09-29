import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import HParams
from ..device import get_device
from ..data.tabular import prep_regression_frame
from ..models.base import ArrayDataset
from ..models.mlp import MLPRegressor
from ..models.gates import TabularGateMLP, moe_predict_reg
from ..train.loops import train_epoch, per_sample_losses
from ..train.gate_training import train_gate_reg
from ..stats.metrics import rmse, mae

device = get_device()

def run_tabular_regression(name, X: pd.DataFrame, y: pd.Series, hp: HParams):
    pre = prep_regression_frame(X, y)
    y = pd.to_numeric(pd.Series(y).astype(float), errors="coerce").values
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=hp.test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=hp.val_ratio, random_state=42)

    X_train = pre.fit_transform(X_train); X_val = pre.transform(X_val); X_test = pre.transform(X_test)

    if sp.issparse(X_train):
        target_k = 1024 if X_train.shape[1] > 4000 else min(512, X_train.shape[1] - 1)
        svd = TruncatedSVD(n_components=target_k, random_state=42)
        X_train_nn = svd.fit_transform(X_train); X_val_nn = svd.transform(X_val); X_test_nn = svd.transform(X_test)
    else:
        X_train_nn, X_val_nn, X_test_nn = X_train, X_val, X_test

    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = sy.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s   = sy.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s  = sy.transform(y_test.reshape(-1, 1)).ravel()

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train_nn, y_train_s, task="reg"), batch_size=hp.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val_nn,   y_val_s,   task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test_nn,  y_test_s,  task="reg"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    crit = nn.MSELoss()

    E_TOTAL = hp.tr_epochs_total
    E_WARM  = max(1, int(E_TOTAL * hp.tr_frac_warmup))
    E_SPEC_TOTAL = max(0, int(E_TOTAL * hp.tr_frac_spec))
    E_GATE  = max(1, E_TOTAL - E_WARM - E_SPEC_TOTAL)
    E_SPEC  = max(1, E_SPEC_TOTAL // max(1, hp.n_specialists))

    rows = {"task":"tabular_regression","dataset":name,"n_train":len(y_train),"n_val":len(y_val),"n_test":len(y_test),
            "budget_total":E_TOTAL,"budget_warmup":E_WARM,"budget_spec_each":E_SPEC,"budget_gate":E_GATE}

    if hp.run_tr_mlp:
        base = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(base.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        best_val = float("inf"); best_state = None; patience = 0; best_epoch = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TR-MLP", leave=False):
            _ = train_epoch(base, train_loader, crit, opt, grad_clip=hp.grad_clip)
            with torch.no_grad():
                base.eval()
                vsum, n = 0.0, 0
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = base(Xb); bs = yb.size(0)
                    vs = crit(pred, yb).item() * bs
                    vsum += vs; n += bs
                v = vsum / max(n, 1)
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

    if hp.run_tr_moe_mlp_hard or hp.run_tr_moe_mlp_rand:
        gen = MLPRegressor(X_train_nn.shape[1], hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(E_WARM), desc=f"{name} TR-warmup", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)

        losses = per_sample_losses(gen, X_train_nn, y_train_s, desc=f"{name} TR-hard", task="reg")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])

        experts, expertsR = [], []
        pin = torch.cuda.is_available()
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

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

from ..config import HParams
from ..device import get_device
from ..data.tabular import prep_classification_frame
from ..models.base import ArrayDataset
from ..models.mlp import MLPClassifier
from ..models.cnn import CNN1DClassifier
from ..models.gates import TabularGateMLP, moe_predict_cls
from ..train.loops import train_epoch, per_sample_losses
from ..train.gate_training import train_gate_cls
from ..stats.tempscale import fit_temperature, apply_temperature, fit_temperature_logprobs, apply_temperature_logprobs
from sklearn.metrics import accuracy_score, log_loss

device = get_device()

def _logits_to_metrics(logits_np, y_true, *, inputs_are_log_probs: bool = False, n_classes: int = None):
    if inputs_are_log_probs:
        proba = np.exp(np.asarray(logits_np))
    else:
        proba = torch.softmax(torch.tensor(logits_np), dim=1).numpy()
    y_pred = proba.argmax(axis=1)
    return accuracy_score(y_true, y_pred), log_loss(y_true, proba, labels=list(range(n_classes)))

def run_tabular_classification(name, X: pd.DataFrame, y, hp: HParams):
    pre, y = prep_classification_frame(X, y)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=hp.test_ratio, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=hp.val_ratio, random_state=42, stratify=y_train_full)

    X_train = pre.fit_transform(X_train); X_val = pre.transform(X_val); X_test = pre.transform(X_test)
    if sp.issparse(X_train):
        target_k = 1024 if X_train.shape[1] > 4000 else min(512, X_train.shape[1] - 1)
        svd = TruncatedSVD(n_components=target_k, random_state=42)
        X_train_nn = svd.fit_transform(X_train); X_val_nn = svd.transform(X_val); X_test_nn = svd.transform(X_test)
    else:
        X_train_nn, X_val_nn, X_test_nn = X_train, X_val, X_test

    n_classes = int(len(np.unique(y)))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(ArrayDataset(X_train_nn, y_train, task="cls"), batch_size=hp.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(ArrayDataset(X_val_nn,   y_val,   task="cls"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ArrayDataset(X_test_nn,  y_test,  task="cls"), batch_size=hp.batch_size, shuffle=False, pin_memory=pin)

    E_TOTAL = hp.tc_epochs_total
    E_WARM  = max(1, int(E_TOTAL * hp.tc_frac_warmup))
    E_SPEC_TOTAL = max(0, int(E_TOTAL * hp.tc_frac_spec))
    E_GATE  = max(1, E_TOTAL - E_WARM - E_SPEC_TOTAL)
    E_SPEC  = max(1, E_SPEC_TOTAL // max(1, hp.n_specialists))

    rows = {"task":"tabular_classification","dataset":name,"n_train":len(y_train),"n_val":len(y_val),"n_test":len(y_test),
            "budget_total":E_TOTAL,"budget_warmup":E_WARM,"budget_spec_each":E_SPEC,"budget_gate":E_GATE}

    if hp.run_tc_mlp:
        mlp = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(mlp.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        best_val = float("inf"); best_state = None; patience = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TC-MLP", leave=False):
            _ = train_epoch(mlp, train_loader, crit, opt, grad_clip=hp.grad_clip)
            with torch.no_grad():
                mlp.eval()
                vsum, n = 0.0, 0
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = mlp(Xb); bs = yb.size(0)
                    vs = crit(pred, yb).item() * bs
                    vsum += vs; n += bs
                v = vsum / max(n, 1)
            if v < best_val - 1e-8:
                best_val = v; best_state = {k: v_.detach().cpu().clone() for k, v_ in mlp.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience > hp.gate_patience: break
        if best_state is not None: mlp.load_state_dict(best_state)
        logits_val = np.vstack([mlp(torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                        for i in range(0, len(X_val_nn), 1024)])
        T = fit_temperature(logits_val, y_val)
        logits_test = np.vstack([mlp(torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).detach().cpu().numpy()
                         for i in range(0, len(X_test_nn), 1024)])
        acc, ll = _logits_to_metrics(apply_temperature(logits_test, T), y_test, n_classes=n_classes)
        rows["mlp_acc"] = acc; rows["mlp_logloss"] = ll; rows["mlp_T"] = T

    if hp.run_tc_cnn:
        cnn = CNN1DClassifier(n_features=X_train_nn.shape[1], n_classes=n_classes,
                      channels=hp.cnn_channels, kernel=hp.cnn_kernel, hidden=hp.cnn_hidden, dropout=hp.dropout).to(device)
        opt = optim.Adam(cnn.parameters(), lr=hp.lr_baseline, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        best_val = float("inf"); best_state = None; patience = 0
        for ep in tqdm(range(1, E_TOTAL + 1), desc=f"{name} TC-CNN", leave=False):
            _ = train_epoch(cnn, train_loader, crit, opt, grad_clip=hp.grad_clip)
            with torch.no_grad():
                cnn.eval()
                vsum, n = 0.0, 0
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = cnn(Xb); bs = yb.size(0)
                    vsum += crit(pred, yb).item() * bs; n += bs
                v = vsum / max(n, 1)
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
        acc, ll = _logits_to_metrics(apply_temperature(logits_test, T), y_test, n_classes=n_classes)
        rows["cnn_acc"] = acc; rows["cnn_logloss"] = ll; rows["cnn_T"] = T

    if hp.run_tc_rf:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)
        y_pred = proba.argmax(axis=1)
        from sklearn.metrics import accuracy_score, log_loss as ll_
        rows["rf_acc"] = accuracy_score(y_test, y_pred); rows["rf_logloss"] = ll_(y_test, proba, labels=list(range(n_classes)))

    if hp.run_tc_moe_mlp_hard or hp.run_tc_moe_mlp_rand:
        gen = MLPClassifier(X_train_nn.shape[1], n_classes, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()
        for _ in tqdm(range(E_WARM), desc=f"{name} TC-MLP-warmup", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)

        losses = per_sample_losses(gen, X_train_nn, y_train, desc=f"{name} TC-MLP-hard", task="cls")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])

        experts, expertsR = [], []
        pin = torch.cuda.is_available()
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

        gateH = train_gate_cls(gateH, experts,  train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)
        gateR = train_gate_cls(gateR, expertsR, train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)

        def collect_logits_calibrated(model_gate, exps, X_val_nn, y_val, X_test_nn):
            logp_val = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_val_nn), 1024)
            ])
            T = fit_temperature_logprobs(logp_val, y_val)
            logp_test = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_test_nn), 1024)
            ])
            logp_test_T = apply_temperature_logprobs(logp_test, T)
            return logp_test_T, T

        logitsH_test, TH = collect_logits_calibrated(gateH, experts,  X_val_nn, y_val, X_test_nn)
        accH, llH = _logits_to_metrics(logitsH_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)
        logitsR_test, TR = collect_logits_calibrated(gateR, expertsR, X_val_nn, y_val, X_test_nn)
        accR, llR = _logits_to_metrics(logitsR_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)

        rows["moe_mlp_acc"] = accH; rows["moe_mlp_logloss"] = llH; rows["moe_mlp_T"] = TH
        rows["moeR_mlp_acc"] = accR; rows["moeR_mlp_logloss"] = llR; rows["moeR_mlp_T"] = TR

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
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=hp.n_specialists, random_state=42, n_init=10)
        hard_labels = km.fit_predict(X_train_nn[hard_idx])
        experts, expertsR = [], []
        pin = torch.cuda.is_available()
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
        gateH = train_gate_cls(gateH, experts,  train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)
        gateR = train_gate_cls(gateR, expertsR, train_loader, val_loader, hp, n_classes, max_epochs=E_GATE)

        def collect_logits_calibrated(model_gate, exps, X_val_nn, y_val, X_test_nn):
            logp_val = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_val_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_val_nn), 1024)
            ])
            T = fit_temperature_logprobs(logp_val, y_val)
            logp_test = np.vstack([
                moe_predict_cls(model_gate, exps, torch.tensor(X_test_nn[i:i+1024], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(X_test_nn), 1024)
            ])
            logp_test_T = apply_temperature_logprobs(logp_test, T)
            return logp_test_T, T

        logitsH_test, TH = collect_logits_calibrated(gateH, experts,  X_val_nn, y_val, X_test_nn)
        accH, llH = _logits_to_metrics(logitsH_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)
        logitsR_test, TR = collect_logits_calibrated(gateR, expertsR, X_val_nn, y_val, X_test_nn)
        accR, llR = _logits_to_metrics(logitsR_test, y_test, inputs_are_log_probs=True, n_classes=n_classes)

        rows["moe_cnn_acc"] = accH; rows["moe_cnn_logloss"] = llH; rows["moe_cnn_T"] = TH
        rows["moeR_cnn_acc"] = accR; rows["moeR_cnn_logloss"] = llR; rows["moeR_cnn_T"] = TR

    return pd.DataFrame([rows])

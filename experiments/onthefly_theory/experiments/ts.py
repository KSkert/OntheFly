import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..device import get_device, set_seed
from ..config import HParams
from ..data.time_series import make_supervised_from_series, simple_time_split, rolling_origin_splits, scale_after_split
from ..models.mlp import MLPRegressor as MLPReg
from ..models.lstm import LSTMRegressor
from ..models.gates import LSTMGate, moe_predict_reg
from ..models.base import ArrayDataset, count_params, make_equal_params_mlp
from ..train.loops import train_epoch, eval_epoch, per_sample_losses
from ..train.gate_training import train_gate_reg
from ..stats.metrics import rmse, mae
from ..stats.bootstrap import block_bootstrap_ci
from ..stats.dm import diebold_mariano_test
from ..stats.seasonal import infer_seasonal_period, sarima_forecast, ets_forecast
from statsmodels.tsa.arima.model import ARIMA

device = get_device()

def run_ts_once(name, series, max_lag, hp: HParams, seed: int,
                split_ranges=None, is_asset_returns: bool=False):
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

    scaler_cls = (lambda: None)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    audit = {}

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

    # Baseline MLP
    if hp.run_ts_mlp:
        baseline_mlp = MLPReg(input_dim=feat_dim, hidden_dim=hp.baseline_hidden, dropout=hp.dropout).to(device)
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
                if patience > hp.gate_patience: break
        audit["baseline_mlp_params"] = count_params(baseline_mlp)
        audit["baseline_mlp_epochs_ran"] = best_epoch
        if best_state is not None:
            baseline_mlp.load_state_dict(best_state)
        base_p, y_test_inv = gather_preds_targets(lambda xb: baseline_mlp(xb), test_loader)
        results["baseline_rmse"] = rmse(y_test_inv, base_p); results["baseline_mae"] = mae(y_test_inv, base_p)
        results["baseline_best_epoch"] = int(best_epoch)

    def train_moe_reg(expert_arch="mlp", tag="MLP"):
        if expert_arch == "mlp":
            generalist = MLPReg(input_dim=feat_dim, hidden_dim=hp.generalist_hidden, dropout=hp.dropout).to(device)
            def expert_factory():
                return MLPReg(input_dim=feat_dim, hidden_dim=hp.specialist_hidden, dropout=hp.dropout).to(device)
        else:
            generalist = LSTMRegressor(seq_len=seq_len, lstm_hidden=hp.lstm_hidden,
                                       lstm_layers=hp.lstm_layers, dropout=hp.dropout,
                                       head_hidden=hp.baseline_hidden).to(device)
            def expert_factory():
                return LSTMRegressor(seq_len=seq_len, lstm_hidden=hp.lstm_hidden,
                                     lstm_layers=hp.lstm_layers, dropout=hp.dropout,
                                     head_hidden=hp.baseline_hidden).to(device)

        opt_g = optim.Adam(generalist.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(E_WARM), desc=f"{name} warmup-{tag}(s={seed})", leave=False):
            _ = train_epoch(generalist, train_loader, crit, opt_g, grad_clip=hp.grad_clip)

        losses = per_sample_losses(generalist, X_train, y_train, batch_size=2048, desc=f"{name} per-loss-{tag}(s={seed})", task="reg")
        N = len(losses); k = max(1, int(np.ceil(hp.hard_fraction * N)))
        hard_idx = np.argsort(losses)[-k:]
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=hp.n_specialists, random_state=seed, n_init=10)
        hard_labels = km.fit_predict(X_train[hard_idx])

        pin = torch.cuda.is_available()
        specialists = []
        for c in range(hp.n_specialists):
            mask = (hard_labels == c)
            Xc = X_train[hard_idx][mask]; yc = y_train[hard_idx][mask]
            if len(Xc) == 0: Xc, yc = X_train[hard_idx], y_train[hard_idx]
            dl_c = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = expert_factory()
            opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(max(1, E_SPEC)), desc=f"{name} spec-{tag}{c+1}/{hp.n_specialists}(s={seed})", leave=False):
                _ = train_epoch(spec, dl_c, crit, opt_s, grad_clip=hp.grad_clip)
            specialists.append(spec.eval())
        experts = [generalist.eval()] + specialists

        rng = np.random.default_rng(seed)
        rand_idx = rng.choice(N, size=k, replace=False)
        kmR = KMeans(n_clusters=hp.n_specialists, random_state=seed, n_init=10)
        rand_labels = kmR.fit_predict(X_train[rand_idx])
        specialists_rand = []
        for c in range(hp.n_specialists):
            mask = (rand_labels == c)
            Xc = X_train[rand_idx][mask]; yc = y_train[rand_idx][mask]
            if len(Xc) == 0: Xc, yc = X_train[rand_idx], y_train[rand_idx]
            dl_c = DataLoader(ArrayDataset(Xc, yc, task="reg"), batch_size=hp.batch_size, shuffle=True, pin_memory=pin)
            spec = expert_factory()
            opt_s = optim.Adam(spec.parameters(), lr=hp.lr_specialist, weight_decay=hp.weight_decay)
            for _ in tqdm(range(max(1, E_SPEC)), desc=f"{name} specR-{tag}{c+1}/{hp.n_specialists}(s={seed})", leave=False):
                _ = train_epoch(spec, dl_c, crit, opt_s, grad_clip=hp.grad_clip)
            specialists_rand.append(spec.eval())
        experts_rand = [generalist.eval()] + specialists_rand

        gate_hard = LSTMGate(seq_len=seq_len, num_experts=len(experts),
                             lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                             dropout=hp.dropout, lstm_layers=hp.lstm_layers).to(device)
        gate_hard = train_gate_reg(gate_hard, experts, train_loader, val_loader, hp, crit, max_epochs=E_GATE)

        gate_rand = LSTMGate(seq_len=seq_len, num_experts=len(experts_rand),
                             lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                             dropout=hp.dropout, lstm_layers=hp.lstm_layers).to(device)
        gate_rand = train_gate_reg(gate_rand, experts_rand, train_loader, val_loader, hp, crit, max_epochs=E_GATE)

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

    out_mlp = None
    if hp.run_ts_moe_mlp_hard or hp.run_ts_moe_mlp_rand or hp.run_ts_mlp:
        out_mlp = train_moe_reg(expert_arch="mlp", tag="mlp")
        results.update({
            "moe_rmse": out_mlp["moe_mlp_rmse"], "moe_mae": out_mlp["moe_mlp_mae"],
            "moeR_rmse": out_mlp["moeR_mlp_rmse"], "moeR_mae": out_mlp["moeR_mlp_mae"],
        })
        y_test_inv = out_mlp["y_test_inv"]

    if hp.run_ts_equal_params and hp.run_ts_mlp and out_mlp is not None:
        dummy_gate = LSTMGate(seq_len=seq_len, num_experts=1+hp.n_specialists,
                              lstm_hidden=hp.lstm_hidden, hidden_dim=hp.baseline_hidden,
                              dropout=hp.dropout, lstm_layers=hp.lstm_layers)
        mlp_generalist = MLPReg(feat_dim, hp.generalist_hidden, hp.dropout)
        mlp_specialists = [MLPReg(feat_dim, hp.specialist_hidden, hp.dropout) for _ in range(hp.n_specialists)]
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

        gen = MLPReg(feat_dim, hp.generalist_hidden, hp.dropout).to(device)
        optg = optim.Adam(gen.parameters(), lr=hp.lr_generalist, weight_decay=hp.weight_decay)
        for _ in tqdm(range(max(1, E_WARM//4)), desc=f"{name} naive-gen(s={seed})", leave=False):
            _ = train_epoch(gen, train_loader, crit, optg, grad_clip=hp.grad_clip)
        specs = []
        for i in range(hp.n_specialists):
            spec = MLPReg(feat_dim, hp.specialist_hidden, hp.dropout).to(device)
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

    out_lstm = None
    if (hp.run_ts_moe_lstm_hard or hp.run_ts_moe_lstm_rand or hp.run_ts_lstm):
        out_lstm = train_moe_reg(expert_arch="lstm", tag="lstm")
        results.update({
            "moeLSTM_rmse": out_lstm["moe_lstm_rmse"], "moeLSTM_mae": out_lstm["moe_lstm_mae"],
            "moeLSTMR_rmse": out_lstm["moeR_lstm_rmse"], "moeLSTMR_mae": out_lstm["moeR_lstm_mae"],
        })
        y_test_inv = out_lstm["y_test_inv"]

    if hp.run_ts_rw:
        rw_p = X_te_raw[:, -1].astype(float)
        results["rw_rmse"] = rmse(y_te_raw, rw_p); results["rw_mae"] = mae(y_te_raw, rw_p)

    if hp.run_ts_arima:
        full_vals = pd.Series(series).dropna().values.astype(float)
        trainval_len = max_lag + len(y_tr_raw) + len(y_val_raw)
        trainval_series = full_vals[:trainval_len]
        test_h = len(y_te_raw)
        best_aic, best_order = np.inf, (1,1,0)
        for p in [0,1,2]:
            for d in [0,1,2]:
                for q in [0,1,2]:
                    try:
                        with torch.no_grad():
                            fit = ARIMA(trainval_series, order=(p,d,q)).fit()
                        if fit.aic < best_aic:
                            best_aic, best_order = fit.aic, (p,d,q)
                    except Exception:
                        continue
        try:
            with torch.no_grad():
                fitted = ARIMA(trainval_series, order=best_order).fit()
            fc = fitted.forecast(steps=test_h)
            arima_test = np.asarray(fc, dtype=float)
        except Exception:
            arima_test = np.full((test_h,), float(trainval_series[-1]))
        results["arima_rmse"] = rmse(y_te_raw, arima_test); results["arima_mae"] = mae(y_te_raw, arima_test)
        audit["arima_best_order"] = best_order; audit["arima_aic"] = best_aic

    if hp.run_ts_sarima_ets:
        try:
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
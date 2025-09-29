import os, json
import pandas as pd

def _fmt_cols(df, cols):
    have = [c for c in cols if c in df.columns]
    return df[have]

def fmt_ts_block(df):
    base = [ "task","dataset","seed","fold","n_train","n_val","n_test",
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
             "dm_stat_moe_mlp","dm_p_moe_mlp" ]
    df2 = _fmt_cols(df, base + [c for c in df.columns if c.startswith("dm_q_")])
    return df2.sort_values(["dataset","seed","fold"]).to_string(index=False)

def fmt_tr_block(df):
    cols = ["task","dataset","n_train","n_val","n_test","budget_total","budget_warmup","budget_spec_each","budget_gate",
            "baseline_rmse","baseline_mae","moe_rmse","moe_mae","moeR_rmse","moeR_mae"]
    return _fmt_cols(df, cols).to_string(index=False)

def fmt_tc_block(df):
    cols = ["task","dataset","n_train","n_val","n_test","budget_total","budget_warmup","budget_spec_each","budget_gate",
            "mlp_acc","mlp_logloss","mlp_T","cnn_acc","cnn_logloss","cnn_T",
            "rf_acc","rf_logloss","moe_mlp_acc","moe_mlp_logloss","moe_mlp_T",
            "moeR_mlp_acc","moeR_mlp_logloss","moeR_mlp_T","moe_cnn_acc","moe_cnn_logloss","moe_cnn_T",
            "moeR_cnn_acc","moeR_cnn_logloss","moeR_cnn_T"]
    return _fmt_cols(df, cols).to_string(index=False)

def write_all(results_df: pd.DataFrame, audit_rows, outdir: str):
    txt = []
    if not results_df.empty:
        txt.append("SIGN CONVENTION:\n  Bootstrap delta = metric(tag) - metric(base). Negative is better.\n  Positive DM ⇒ baseline worse than tag.\n")
        ts_df = results_df[results_df["task"] == "time_series"]
        tr_df = results_df[results_df["task"] == "tabular_regression"]
        tc_df = results_df[results_df["task"] == "tabular_classification"]
        if not ts_df.empty: txt += ["=== TIME SERIES (per-run) ===", fmt_ts_block(ts_df), ""]
        if not tr_df.empty: txt += ["=== TABULAR REGRESSION (per-run) ===", fmt_tr_block(tr_df), ""]
        if not tc_df.empty: txt += ["=== TABULAR CLASSIFICATION (per-run) ===", fmt_tc_block(tc_df), ""]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "resultsE.txt"), "w", encoding="utf-8") as f: f.write("\n".join(txt) if txt else "No datasets were available.\n")
    results_df.to_csv(os.path.join(outdir, "resultsE.csv"), index=False)
    if audit_rows:
        with open(os.path.join(outdir, "audit.json"), "w", encoding="utf-8") as jf:
            json.dump(audit_rows, jf, indent=2)

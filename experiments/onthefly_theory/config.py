from dataclasses import dataclass
from typing import Tuple

@dataclass
class HParams:
    # ---------- Optimization ----------
    batch_size: int = 128
    weight_decay: float = 5e-5
    lr_generalist: float = 3e-4
    lr_specialist: float = 5e-4
    lr_gate: float = 1e-3
    lr_baseline: float = 3e-4
    grad_clip: float = 1.0
    gate_patience: int = 8
    bootstrap_block_frac: float = 0.10

    # ---------- Budgets ----------
    ts_epochs_total: int = 60
    ts_frac_warmup: float = 0.30
    ts_frac_spec: float = 0.50
    ts_frac_gate: float = 0.20

    tr_epochs_total: int = 40
    tr_frac_warmup: float = 0.30
    tr_frac_spec: float = 0.50
    tr_frac_gate: float = 0.20

    tc_epochs_total: int = 40
    tc_frac_warmup: float = 0.30
    tc_frac_spec: float = 0.50
    tc_frac_gate: float = 0.20

    # ---------- Nets ----------
    generalist_hidden: int = 384
    specialist_hidden: int = 320
    baseline_hidden: int = 256
    dropout: float = 0.15

    lstm_hidden: int = 96
    lstm_layers: int = 2

    cnn_channels: int = 64
    cnn_kernel: int = 3
    cnn_hidden: int = 256

    # ---------- Hard mining ----------
    hard_fraction: float = 0.60
    n_specialists: int = 4

    # ---------- Evaluation ----------
    seeds: Tuple[int, ...] = (13, 42, 202, 777)
    rolling_folds: int = 3
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
    use_ts_sp_csv: bool = False
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
    run_ts_equal_params: bool = True
    run_ts_naive_ens: bool = True
    run_ts_rw: bool = True
    run_ts_arima: bool = True
    run_ts_sarima_ets: bool = True

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

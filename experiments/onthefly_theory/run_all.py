import os, json
import pandas as pd
import numpy as np
import torch, sklearn, statsmodels
import yfinance as yf

from .config import HParams
from .device import set_seed
from .data.time_series import build_ts_datasets
from .data.tabular import build_tabular_regression, build_tabular_classification
from .experiments.ts import run_ts
from .experiments.tabular_reg import run_tabular_regression
from .experiments.tabular_cls import run_tabular_classification
from .aggregate.fdr import add_fdr_columns, bh_adjust, stouffer
from .aggregate.formatters import write_all

def main(outdir: str = "./experiments/results"):
    os.makedirs(outdir, exist_ok=True)
    set_seed(42)
    hp = HParams(rolling_folds=3)

    all_rows, audit_rows = [], []

    for name, series, lag, is_ret in build_ts_datasets(hp):
        print(f"=== [TS] Running {name} (lag={lag}) ===")
        df = run_ts(name, series, lag, hp, is_asset_returns=is_ret)
        all_rows.append(df)
        for _, row in df.iterrows():
            if isinstance(row.get("audit"), dict):
                aud = dict(row["audit"])
                aud.update({"task":"time_series","dataset":row["dataset"],"seed":row["seed"],"fold":row["fold"]})
                audit_rows.append(aud)

    for name, X, y in build_tabular_regression(hp):
        print(f"=== [TR] Running {name} ===")
        df = run_tabular_regression(name, X, y, hp)
        all_rows.append(df)

    for name, X, y in build_tabular_classification(hp):
        print(f"=== [TC] Running {name} ===")
        df = run_tabular_classification(name, X, y, hp)
        all_rows.append(df)

    results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Per-dataset Stouffer combine (if DM present)
    dm_cols = [c for c in results_df.columns if c.startswith("dm_p_")]
    if not results_df.empty and dm_cols:
        ds_groups = results_df.groupby(["task","dataset"], sort=False)
        rows = []
        from scipy.stats import norm
        def stouffer_local(pvals):
            p = np.asarray([x for x in pvals if np.isfinite(x) and not np.isnan(x)])
            if p.size == 0: return np.nan
            z = norm.isf(p); z_sum = np.sum(z); z_comb = z_sum / np.sqrt(len(z))
            return float(norm.sf(z_comb))
        for (task, ds), block in ds_groups:
            row = {"task": task, "dataset": ds}
            for c in dm_cols:
                p = block[c].values
                row[c.replace("dm_p_", "dm_p_combined_")] = stouffer_local(p)
            rows.append(row)
        per_dataset = pd.DataFrame(rows)
        for c in [k for k in per_dataset.columns if k.startswith("dm_p_combined_")]:
            qcol = c.replace("dm_p_combined_", "dm_q_combined_")
            per_dataset[qcol] = bh_adjust(per_dataset[c].values)
        os.makedirs(outdir, exist_ok=True)
        per_dataset.to_csv(os.path.join(outdir, "per_dataset_inference.csv"), index=False)

    if not results_df.empty:
        results_df = add_fdr_columns(results_df)

    env_header = {
        "task": "_env",
        "python_version": f"{os.sys.version_info[0]}.{os.sys.version_info[1]}.{os.sys.version_info[2]}",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "statsmodels_version": statsmodels.__version__,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "seeds": list(hp.seeds),
        "bootstrap_block_frac": hp.bootstrap_block_frac,
        "asset_end_date": "2024-12-31"
    }
    audit_rows.insert(0, env_header)

    write_all(results_df, audit_rows, outdir)
    print(f"Wrote results to: {outdir}/epxerimentation_results.txt")
    print(f"CSV: {outdir}/experimentation_results.csv")
    print(f"Audit JSON: {outdir}/audit.json")

if __name__ == "__main__":
    main()
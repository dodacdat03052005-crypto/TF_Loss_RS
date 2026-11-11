# -*- coding: utf-8 -*-
"""
Compare DBLoss vs HD_DBLoss (and optional extra variants) on PatchTST.

Examples (Colab / Linux):
    python scripts/compare_losses.py --dataset ETTh1 --horizons 96
    python scripts/compare_losses.py --dataset ETTh1 --horizons 96 192 336 720
    python scripts/compare_losses.py --report-only          # only rebuild table from saved results
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Official hyper-parameters per (dataset, model, horizon) from scripts/multivariate_forecast/
BENCHMARK_CONFIGS = {
    ("ETTh1", "PatchTST"): {
        96: {"d_ff": 2048, "d_model": 512, "e_layers": 1, "factor": 3, "n_headers": 2, "norm": True, "seq_len": 96},
        192: {"d_ff": 512, "d_model": 128, "norm": True, "seq_len": 512},
        336: {"batch_size": 64, "d_ff": 512, "d_model": 128, "lr": 0.0001, "norm": True, "seq_len": 512},
        720: {"batch_size": 64, "d_ff": 512, "d_model": 128, "lr": 0.0001, "norm": True, "seq_len": 512},
    },
    ("ETTh1", "DLinear"): {
        96: {"batch_size": 32, "d_ff": 2048, "d_model": 512, "lr": 0.005, "norm": True, "seq_len": 336},
        192: {"batch_size": 64, "d_ff": 2048, "d_model": 512, "lr": 0.001, "norm": True, "period_len": 24, "seq_len": 336, "station_lr": 1e-05},
        336: {"batch_size": 32, "d_ff": 2048, "d_model": 512, "lr": 0.005, "norm": True, "seq_len": 336},
        720: {"batch_size": 64, "d_ff": 2048, "d_model": 512, "factor": 3, "norm": True, "period_len": 24, "seq_len": 336, "station_lr": 0.0001},
    },
    ("ETTh1", "iTransformer"): {
        96: {"beta": 0.9, "d_ff": 256, "d_model": 256, "e_layers": 2, "norm": True, "seq_len": 96},
        192: {"beta": 0.9, "d_ff": 256, "d_model": 256, "e_layers": 2, "norm": True, "seq_len": 512},
        336: {"beta": 0.9, "d_ff": 128, "d_model": 128, "e_layers": 2, "norm": True, "seq_len": 512},
        720: {"beta": 0.9, "d_ff": 512, "d_model": 512, "e_layers": 2, "norm": True, "seq_len": 96},
    },
    ("ETTh2", "PatchTST"): {
        96: {"beta": 0.2, "d_ff": 128, "d_model": 32, "norm": True, "seq_len": 512},
        192: {"d_ff": 128, "d_model": 32, "norm": True, "seq_len": 512},
        336: {"d_ff": 128, "d_model": 32, "norm": True, "seq_len": 512},
        720: {"beta": 0.2, "d_ff": 128, "d_model": 32, "norm": True, "seq_len": 336},
    },
    ("ETTm1", "PatchTST"): {
        96: {"beta": 0.1, "d_ff": 256, "d_model": 128, "norm": True, "seq_len": 336},
        192: {"beta": 0.1, "d_ff": 256, "d_model": 128, "norm": True, "seq_len": 336},
        336: {"beta": 0.1, "d_ff": 64, "d_model": 32, "norm": True, "seq_len": 512},
        720: {"beta": 0.1, "d_ff": 256, "d_model": 128, "norm": True, "seq_len": 336},
    },
}

# Loss variants to compare (Step-wise Empirical Variance Weighting, fixed alpha=0.2, beta=0.5):
#   1. HD_DBLoss:    Seasonal inverse-variance only (gamma_S=0.5, gamma_T=0.0)
#   2. HD_DBLoss_ST: Seasonal inverse-variance + Trend trajectory retention (gamma_S=0.5, gamma_T=0.25)
LOSS_VARIANTS = {
    "MSE": {"loss": "MSE"},
    "DBLoss": {"loss": "DBLoss"},
    "HD_DBLoss": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma_S": 0.5,
        "gamma_T": 0.0,
        "momentum": 0.95,
    },
    "HD_DBLoss_ST": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma_S": 0.5,
        "gamma_T": 0.25,
        "momentum": 0.95,
    },
}

METRICS = ["mse_norm", "mae_norm"]


def save_path(dataset, model, loss_name, horizon, batch_size=None):
    h_tag = f"H{horizon}" if batch_size is None else f"H{horizon}_bs{batch_size}"
    if model == "PatchTST":
        return f"{dataset}/compare/{loss_name}/{h_tag}"
    return f"{dataset}/{model}/compare/{loss_name}/{h_tag}"


def run_one(dataset, model, loss_name, horizon, args):
    cfg_map = BENCHMARK_CONFIGS.get((dataset, model), BENCHMARK_CONFIGS[("ETTh1", "PatchTST")])
    params = dict(cfg_map[horizon])
    params["horizon"] = horizon
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    loss_cfg = dict(LOSS_VARIANTS[loss_name])
    # Respect dataset/model-specific base beta if defined in official script
    if "beta" in params and loss_name.startswith("HD_DBLoss"):
        loss_cfg["beta"] = params["beta"]
    params.update(loss_cfg)
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "run_benchmark.py"),
        "--config-path", "rolling_forecast_config.json",
        "--data-name-list", f"{dataset}.csv",
        "--strategy-args", json.dumps({"horizon": horizon}),
        "--adapter", "transformer_adapter",
        "--model-name", f"time_series_library.{model}",
        "--model-hyper-params", json.dumps(params),
        "--gpus", str(args.gpu),
        "--num-workers", "2",
        "--timeout", "60000",
        "--save-path", save_path(dataset, model, loss_name, horizon, args.batch_size),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    bs_info = params.get("batch_size", 32)
    print(f"\n===== {loss_name} | {dataset} | {model} | horizon={horizon} | batch_size={bs_info} =====", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)


def read_result(dataset, model, loss_name, horizon, batch_size=None):
    pattern = os.path.join(ROOT, "result", save_path(dataset, model, loss_name, horizon, batch_size), "test_report*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None
    df = pd.read_csv(files[-1])
    out = {}
    if "metric_name" in df.columns:
        score_cols = [c for c in df.columns if c not in ["strategy_args", "metric_name"]]
        if not score_cols:
            return None
        score_col = score_cols[-1]
        for m in METRICS:
            stem = m.replace("_norm", "")
            match = df[df["metric_name"].astype(str).str.contains(stem, case=False, na=False)]
            if not match.empty:
                val = match.iloc[0][score_col]
                try:
                    out[m] = float(val)
                except (ValueError, TypeError):
                    out[m] = float("nan")
            else:
                out[m] = float("nan")
    else:
        row = df.iloc[0]
        for m in METRICS:
            stem = m.replace("_norm", "")
            col = next((c for c in df.columns if stem in c.lower()), None)
            if col is not None:
                try:
                    out[m] = float(row[col])
                except (ValueError, TypeError):
                    out[m] = float("nan")
            else:
                out[m] = float("nan")
    return out


def build_table(dataset, model, horizons, losses, batch_size=None):
    rows = []
    for h in horizons:
        res = {l: read_result(dataset, model, l, h, batch_size) for l in losses}
        for m in METRICS:
            row = {"horizon": str(h), "metric": m.replace("_norm", "").upper()}
            for l in losses:
                row[l] = res[l][m] if res[l] else float("nan")
            base = row.get("DBLoss")
            for l in losses:
                if l != "DBLoss" and base and pd.notna(row[l]) and pd.notna(base):
                    row[f"{l} vs DBLoss (%)"] = (base - row[l]) / base * 100
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(horizons) > 1 and not df.empty:
        avg_rows = []
        for m_label in ["MSE", "MAE"]:
            sub = df[df["metric"] == m_label]
            avg_row = {"horizon": "AVG", "metric": m_label}
            for l in losses:
                avg_row[l] = sub[l].mean()
            base = avg_row.get("DBLoss")
            for l in losses:
                if l != "DBLoss" and base and pd.notna(avg_row[l]) and pd.notna(base):
                    avg_row[f"{l} vs DBLoss (%)"] = (base - avg_row[l]) / base * 100
            avg_rows.append(avg_row)
        df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--model", default="PatchTST", choices=["PatchTST", "DLinear", "iTransformer"])
    p.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336])
    p.add_argument(
        "--losses",
        nargs="+",
        default=["DBLoss", "HD_DBLoss", "HD_DBLoss_ST"],
        choices=list(LOSS_VARIANTS),
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override training batch_size (e.g. 128 or 256 for faster GPU runs)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run DBLoss even if cached result exists")
    args = p.parse_args()
    args.dataset = args.dataset.replace(".csv", "")

    if not args.report_only:
        for h in args.horizons:
            for l in args.losses:
                if l == "DBLoss" and not args.force and read_result(args.dataset, args.model, l, h, args.batch_size) is not None:
                    print(f"\n===== Skipping {l} | {args.dataset} | {args.model} | horizon={h} (already exists) =====", flush=True)
                    continue
                run_one(args.dataset, args.model, l, h, args)

    table = build_table(args.dataset, args.model, args.horizons, args.losses, args.batch_size)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    suffix = f"{args.dataset}" if args.model == "PatchTST" else f"{args.dataset}_{args.model}"
    out_csv = os.path.join(ROOT, "result", f"compare_{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()


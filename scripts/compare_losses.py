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

# PatchTST hyper-parameters per horizon for ETTh1 (taken from scripts/multivariate_forecast/ETTh1_script/PatchTST.sh)
ETTH1_PATCHTST = {
    96: {"d_ff": 2048, "d_model": 512, "e_layers": 1, "factor": 3, "n_headers": 2, "norm": True, "seq_len": 96},
    192: {"d_ff": 512, "d_model": 128, "norm": True, "seq_len": 512},
    336: {"batch_size": 64, "d_ff": 512, "d_model": 128, "lr": 0.0001, "norm": True, "seq_len": 512},
    720: {"batch_size": 64, "d_ff": 512, "d_model": 128, "lr": 0.0001, "norm": True, "seq_len": 512},
}

# Loss variants to compare: Horizon-Adaptive HD_DBLoss (scaling hyperparameters with H)
LOSS_VARIANTS = {
    "DBLoss": {"loss": "DBLoss"},
    "HD_DBLoss_H1": {
        # 1. Horizon-adaptive gamma(H) + alpha(H) (keeps beta=0.5)
        # H=96: gamma=0.85, alpha=0.20 | H=192: gamma=0.33, alpha=0.15 | H=336: gamma=-0.09, alpha=0.13
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma": 0.85,
        "gamma_season": 0.85,
        "tau": 3.5,
        "h_ref": 96.0,
        "kappa_gamma": 0.75,
        "eta_alpha": 0.35,
        "kappa_beta": 0.0,
    },
    "HD_DBLoss_H2": {
        # 2. Horizon-adaptive gamma(H) + low beta (Trend-focused like ETTm1/ETTh2 beta=0.2)
        "loss": "HD_DBLoss",
        "alpha": 0.15,
        "beta": 0.25,
        "gamma": 0.85,
        "gamma_season": 0.85,
        "tau": 3.5,
        "h_ref": 96.0,
        "kappa_gamma": 0.75,
        "eta_alpha": 0.0,
        "kappa_beta": -0.05,
    },
    "HD_DBLoss_H3": {
        # 3. Horizon-adaptive gamma(H) + high beta (Season-focused like ETTh1 iTransformer beta=0.8)
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.75,
        "gamma": 0.85,
        "gamma_season": 0.85,
        "tau": 3.5,
        "h_ref": 96.0,
        "kappa_gamma": 0.75,
        "eta_alpha": 0.35,
        "kappa_beta": 0.05,
    },
}

METRICS = ["mse_norm", "mae_norm"]


def save_path(dataset, loss_name, horizon):
    return f"{dataset}/compare/{loss_name}/H{horizon}"


def run_one(dataset, loss_name, horizon, args):
    params = dict(ETTH1_PATCHTST[horizon])
    params["horizon"] = horizon
    params.update(LOSS_VARIANTS[loss_name])
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "run_benchmark.py"),
        "--config-path", "rolling_forecast_config.json",
        "--data-name-list", f"{dataset}.csv",
        "--strategy-args", json.dumps({"horizon": horizon}),
        "--adapter", "transformer_adapter",
        "--model-name", "time_series_library.PatchTST",
        "--model-hyper-params", json.dumps(params),
        "--gpus", str(args.gpu),
        "--num-workers", "1",
        "--timeout", "60000",
        "--save-path", save_path(dataset, loss_name, horizon),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print(f"\n===== {loss_name} | {dataset} | horizon={horizon} =====", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)


def read_result(dataset, loss_name, horizon):
    pattern = os.path.join(ROOT, "result", save_path(dataset, loss_name, horizon), "test_report*.csv")
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


def build_table(dataset, horizons, losses):
    rows = []
    for h in horizons:
        res = {l: read_result(dataset, l, h) for l in losses}
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
    p.add_argument("--horizons", type=int, nargs="+", default=[96])
    p.add_argument(
        "--losses",
        nargs="+",
        default=["DBLoss", "HD_DBLoss_H1", "HD_DBLoss_H2", "HD_DBLoss_H3"],
        choices=list(LOSS_VARIANTS),
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run DBLoss even if cached result exists")
    args = p.parse_args()

    if not args.report_only:
        for h in args.horizons:
            for l in args.losses:
                if l == "DBLoss" and not args.force and read_result(args.dataset, l, h) is not None:
                    print(f"\n===== Skipping {l} | {args.dataset} | horizon={h} (already exists) =====", flush=True)
                    continue
                run_one(args.dataset, l, h, args)

    table = build_table(args.dataset, args.horizons, args.losses)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    out_csv = os.path.join(ROOT, "result", f"compare_{args.dataset}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()


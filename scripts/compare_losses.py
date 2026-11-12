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

# Loss variants to compare (First-Principles Stationary Seasonal vs Non-Stationary Trend Precision Ratio):
#   w_T(h) = w_min + (1 - w_min) * (1 + (c_step * (h - 1)) ** p_order) ** (-gamma),   w_S(h) = 1.0
#   beta(h) = beta / (beta + (1 - beta) * w_T(h))  [Depends ONLY on step h, NEVER on horizon H!]
LOSS_VARIANTS = {
    "MSE": {"loss": "MSE"},
    "DBLoss": {"loss": "DBLoss"},
    "HD_DBLoss": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma": 1.0,
        "c_step": 0.015,
        "p_order": 1.0,
        "w_min": 0.0,
        "step_decay": 0.0,
    },
    "HD_DBLoss_BW": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma": 1.0,
        "c_step": 0.011,
        "p_order": 2.0,
        "w_min": 0.05,
        "step_decay": 0.0,
    },
    "HD_DBLoss_BW3": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma": 1.0,
        "c_step": 0.009,
        "p_order": 3.0,
        "w_min": 0.05,
        "step_decay": 0.0,
    },
    "HD_DBLoss_s2": {
        "loss": "HD_DBLoss",
        "alpha": 0.2,
        "beta": 0.5,
        "gamma": 1.2,
        "c_step": 0.020,
        "p_order": 1.0,
        "w_min": 0.0,
        "step_decay": 0.0,
    },
}

METRICS = ["mse_norm", "mae_norm"]


def save_path(dataset, model, loss_name, horizon, batch_size=None, seed=None):
    h_tag = f"H{horizon}"
    if batch_size is not None:
        h_tag += f"_bs{batch_size}"
    if seed is not None:
        h_tag += f"_s{seed}"
    if model == "PatchTST":
        return f"{dataset}/compare/{loss_name}/{h_tag}"
    return f"{dataset}/{model}/compare/{loss_name}/{h_tag}"


def run_one(dataset, model, loss_name, horizon, args, seed_tag=None):
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
        "--save-path", save_path(dataset, model, loss_name, horizon, args.batch_size, seed_tag),
    ]
    run_seed = seed_tag if seed_tag is not None else args.seed
    if run_seed is not None:
        cmd += ["--seed", str(run_seed)]
    bs_info = params.get("batch_size", 32)
    print(
        f"\n===== {loss_name} | {dataset} | {model} | horizon={horizon} | batch_size={bs_info} | seed={run_seed} =====",
        flush=True,
    )
    subprocess.run(cmd, cwd=ROOT, check=False)


def read_result(dataset, model, loss_name, horizon, batch_size=None, seed=None):
    pattern = os.path.join(
        ROOT, "result", save_path(dataset, model, loss_name, horizon, batch_size, seed), "test_report*.csv"
    )
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


def build_table(dataset, model, horizons, losses, batch_size=None, seeds=(None,)):
    """
    Build comparison table. With multiple seeds, each cell is the mean over seeds,
    `<loss>_std` is the across-seed standard deviation, and the AVG row is computed
    per seed first (mean over horizons) and then aggregated across seeds.
    """
    seeds = list(seeds)
    multi = len(seeds) > 1
    # vals[loss][metric][horizon] -> list of per-seed values (NaN-free)
    vals = {l: {m: {h: [] for h in horizons} for m in METRICS} for l in losses}
    for h in horizons:
        for l in losses:
            for s in seeds:
                res = read_result(dataset, model, l, h, batch_size, s)
                if res:
                    for m in METRICS:
                        if pd.notna(res[m]):
                            vals[l][m][h].append(res[m])

    def summarize(values):
        if not values:
            return float("nan"), float("nan"), 0
        ser = pd.Series(values, dtype=float)
        std = float(ser.std(ddof=1)) if len(ser) > 1 else float("nan")
        return float(ser.mean()), std, len(ser)

    def add_cells(row, per_loss_values):
        for l in losses:
            mean, std, n = summarize(per_loss_values[l])
            row[l] = mean
            if multi:
                row[f"{l}_std"] = std
                row[f"{l}_n"] = n
        base = row.get("DBLoss")
        for l in losses:
            if l != "DBLoss" and base and pd.notna(row[l]) and pd.notna(base):
                row[f"{l} vs DBLoss (%)"] = (base - row[l]) / base * 100

    rows = []
    for h in horizons:
        for m in METRICS:
            row = {"horizon": str(h), "metric": m.replace("_norm", "").upper()}
            add_cells(row, {l: vals[l][m][h] for l in losses})
            rows.append(row)

    if len(horizons) > 1:
        for m in METRICS:
            row = {"horizon": "AVG", "metric": m.replace("_norm", "").upper()}
            per_loss = {}
            for l in losses:
                # Per-seed average across horizons (only seeds present for all horizons)
                n_common = min(len(vals[l][m][h]) for h in horizons)
                per_loss[l] = [
                    sum(vals[l][m][h][i] for h in horizons) / len(horizons) for i in range(n_common)
                ]
            add_cells(row, per_loss)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--model", default="PatchTST", choices=["PatchTST", "DLinear", "iTransformer"])
    p.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336])
    p.add_argument(
        "--losses",
        nargs="+",
        default=["DBLoss", "HD_DBLoss_BW", "HD_DBLoss_BW3"],
        choices=list(LOSS_VARIANTS),
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override training batch_size (e.g. 128 or 256 for faster GPU runs)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="Single seed (results stored in the legacy, untagged path)")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run every loss with several seeds (stored in seed-tagged paths) and report mean/std",
    )
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run cached losses even if result exists")
    args = p.parse_args()
    args.dataset = args.dataset.replace(".csv", "")

    seed_tags = args.seeds if args.seeds else [None]

    if not args.report_only:
        for s in seed_tags:
            for h in args.horizons:
                for l in args.losses:
                    cached = read_result(args.dataset, args.model, l, h, args.batch_size, s) is not None
                    if not args.force and cached:
                        print(
                            f"\n===== Skipping {l} | {args.dataset} | {args.model} | horizon={h} | seed={s} (already exists) =====",
                            flush=True,
                        )
                        continue
                    run_one(args.dataset, args.model, l, h, args, s)

    table = build_table(args.dataset, args.model, args.horizons, args.losses, args.batch_size, seed_tags)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    suffix = f"{args.dataset}" if args.model == "PatchTST" else f"{args.dataset}_{args.model}"
    out_csv = os.path.join(ROOT, "result", f"compare_{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()


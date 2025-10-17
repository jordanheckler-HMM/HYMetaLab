#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from validation.utils import fit_ols, log_safe

OUTDIR = Path("validation")
OUTDIR.mkdir(exist_ok=True)


def discover_and_merge_data(data_root: Path):
    """Find runs_phase*_summary.csv in data_root and merge them into one DataFrame.
    Returns (df, saved_path) where saved_path is None if nothing was written.
    """
    pattern = str(data_root / "runs_phase*_summary.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, None
    dfs = []
    for p in matches:
        try:
            d = pd.read_csv(p)
            d["_source_file"] = os.path.basename(p)
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return None, None
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    outp = OUTDIR / "runs_summary_merged.csv"
    merged.to_csv(outp, index=False)
    return merged, outp


def main():
    p = argparse.ArgumentParser(
        description="Refine resilience equation from run summaries"
    )
    p.add_argument(
        "--input",
        "-i",
        help="Path to runs_summary.csv (optional). If omitted, the script will try to discover per-phase summaries under data/ and merge them.",
    )
    p.add_argument(
        "--floor_eps",
        type=float,
        default=None,
        help="If provided, floor openness (epsilon) values below this to this floor before logs",
    )
    p.add_argument(
        "--floor_lam",
        type=float,
        default=None,
        help="If provided, floor lam values below this to this floor before logs",
    )
    args = p.parse_args()

    # Load
    df = None
    input_path = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: provided input path does not exist: {input_path}")
            return
        df = pd.read_csv(input_path)
    else:
        # prefer a canonical runs_summary.csv in CWD
        if Path("runs_summary.csv").exists():
            input_path = Path("runs_summary.csv")
            df = pd.read_csv(input_path)
        else:
            # try data/ under repo root
            data_dir = ROOT / "data"
            merged, outp = discover_and_merge_data(data_dir)
            if merged is not None:
                print(f"Discovered and merged {outp}")
                df = merged
                input_path = outp
            else:
                # try discovery_results for any runs_summary.csv files
                dr_pattern = str(ROOT / "discovery_results" / "**" / "runs_summary.csv")
                dr_matches = sorted(glob.glob(dr_pattern, recursive=True))
                if dr_matches:
                    # take first
                    input_path = Path(dr_matches[0])
                    print(f"Using discovered runs_summary at {input_path}")
                    df = pd.read_csv(input_path)
                else:
                    print(
                        "ERROR: No input runs_summary.csv found; provide --input or put per-phase summaries under data/"
                    )
                    return

    # Validate required columns; if missing, try discovery_results for per-experiment runs_summary files
    initial_n = len(df)
    req = ["R_mean", "kE", "kI", "n_agents", "openness"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        # try discovery_results/**/data/runs_summary.csv
        dr_pattern = str(
            ROOT / "discovery_results" / "**" / "data" / "runs_summary.csv"
        )
        dr_matches = sorted(glob.glob(dr_pattern, recursive=True))
        chosen = None
        for p in dr_matches:
            try:
                head = pd.read_csv(p, nrows=0)
                cols = set(head.columns.tolist())
                if all(c in cols for c in req):
                    chosen = p
                    break
            except Exception:
                continue
        if chosen:
            print(f"Found discovery runs_summary with required cols: {chosen}")
            df = pd.read_csv(chosen)
            input_path = Path(chosen)
        else:
            print(f"ERROR: missing required columns in {input_path}: {missing}")
            return

    # Drop missing
    before = len(df)
    df = df.dropna(subset=req).copy()
    dropped = before - len(df)

    # Optional flooring for small epsilon / lam values (to avoid -inf in logs)
    if args.floor_eps is not None and "openness" in df.columns:
        try:
            df["openness"] = pd.to_numeric(df["openness"], errors="coerce").fillna(0.0)
            df.loc[df["openness"] < args.floor_eps, "openness"] = args.floor_eps
        except Exception:
            pass
    if args.floor_lam is not None and "lam" in df.columns:
        try:
            df["lam"] = pd.to_numeric(df["lam"], errors="coerce")
            df.loc[df["lam"].isna(), "lam"] = args.floor_lam
            df.loc[df["lam"] < args.floor_lam, "lam"] = args.floor_lam
        except Exception:
            pass

    # Safety-net: floor exact zero values to tiny positives before logs
    if "openness" in df.columns:
        try:
            df["openness"] = df["openness"].replace(0, 1e-5)
        except Exception:
            pass
    if "lam" in df.columns:
        try:
            df["lam"] = df["lam"].replace(0, 0.001)
        except Exception:
            pass

    # Create logs and guards
    df["log_R"] = log_safe(df["R_mean"], "R_mean")
    df["log_E"] = log_safe(df["kE"], "kE")
    df["log_I"] = log_safe(df["kI"], "kI")
    df["log_N"] = log_safe(df["n_agents"], "n_agents")
    # convert zeros/negatives to NaN then log
    df["log_eps"] = log_safe(df["openness"].replace(0, np.nan), "openness")
    if "lam" in df.columns:
        df["log_lam"] = log_safe(df["lam"].replace(0, np.nan), "lam")
    else:
        df["log_lam"] = np.nan

    # regime flags
    df["regime"] = np.where(
        df["openness"] <= 1e-4,
        "closed",
        np.where(df["openness"] <= 1e-2, "min_open", "open"),
    )

    # shock events
    if os.path.exists("shock_events.csv"):
        se = pd.read_csv("shock_events.csv")
        shock_ids = se["condition_id"].unique().tolist()
        df["shock_flag"] = (
            df.get("condition_id", pd.Series()).isin(shock_ids).astype(int)
        )
    else:
        df["shock_flag"] = 0

    # Noise ceiling
    config_keys = [
        c
        for c in ["n_agents", "openness", "kE", "kI", "lam", "eta", "epochs"]
        if c in df.columns
    ]
    noise_ceiling = None
    if "seed" in df.columns and config_keys:
        grp = df.groupby(config_keys, dropna=False)
        noise_var = grp["R_mean"].var().dropna()
        total_var = df["R_mean"].var()
        if len(noise_var) and not pd.isna(total_var) and total_var != 0:
            noise_ceiling = float(1 - noise_var.mean() / total_var)

    with open(OUTDIR / "noise_ceiling.json", "w") as f:
        json.dump({"noise_ceiling_estimate": noise_ceiling}, f, indent=2)

    # Optional openness events features
    extra_feats = []
    if os.path.exists("openness_events.csv"):
        ev = pd.read_csv("openness_events.csv")
        agg = (
            ev.groupby("condition_id", dropna=False)
            .agg(
                openness_event_count=("event_type", "count"),
                exergy_in_sum=("exergy_in", "sum"),
                waste_out_sum=("waste_out", "sum"),
            )
            .reset_index()
        )
        df = df.merge(agg, on="condition_id", how="left")
        df[["openness_event_count", "exergy_in_sum", "waste_out_sum"]] = df[
            ["openness_event_count", "exergy_in_sum", "waste_out_sum"]
        ].fillna(0)
        df["log_exergy_in_sum"] = log_safe(
            df["exergy_in_sum"].replace(0, np.nan), "exergy_in_sum"
        )
        extra_feats += ["log_exergy_in_sum", "openness_event_count", "waste_out_sum"]

    # Safety: assert no non-positive values where logs were taken
    # Count rows dropped due to non-positive or NaNs in logs
    before_logs = len(df)
    df = df.dropna(subset=["log_R", "log_E", "log_I", "log_N"])  # essential
    after_logs = len(df)
    dropped_logs = before_logs - after_logs

    # Models
    summaries = {}

    # helper plotter
    def plot_preds(model, X_cols, tag):
        valid = df[X_cols + ["log_R"]].dropna().index
        X = df.loc[valid, X_cols].astype(float)
        X = sm.add_constant(X)
        y_true = df.loc[valid, "log_R"].astype(float)
        y_pred = model.predict(X)
        plt.figure(figsize=(6, 6))
        plt.scatter(np.exp(y_true), np.exp(y_pred), s=10)
        mn = min(np.exp(y_true.min()), np.exp(y_pred.min()))
        mx = max(np.exp(y_true.max()), np.exp(y_pred.max()))
        plt.plot([mn, mx], [mn, mx], color="k", linewidth=0.8)
        plt.xlabel("Actual R_mean")
        plt.ylabel("Predicted R_mean")
        plt.title(f"Predicted vs Actual ({tag})")
        plt.savefig(OUTDIR / f"preds_actual_{tag}.png", dpi=160)
        plt.close()

    # M0
    X0 = ["log_E", "log_I", "log_N"]
    m0, s0, c0 = fit_ols(df.dropna(subset=X0 + ["log_R"]), "log_R", X0)
    if c0 is not None and len(c0):
        c0.to_csv(OUTDIR / "coefficients_M0.csv", index=False)
    summaries["M0"] = s0
    if m0 is not None:
        plot_preds(m0, X0, "M0")

    # M1
    X1 = ["log_E", "log_I", "log_N", "log_eps", "log_lam"]
    m1, s1, c1 = fit_ols(df.dropna(subset=X1 + ["log_R"]), "log_R", X1)
    if c1 is not None and len(c1):
        c1.to_csv(OUTDIR / "coefficients_M1.csv", index=False)
    summaries["M1"] = s1
    if m1 is not None:
        plot_preds(m1, X1, "M1")

    # M2
    df["log_E_eps"] = df["log_E"] * df["log_eps"]
    df["log_I_eps"] = df["log_I"] * df["log_eps"]
    df["log_N_eps"] = df["log_N"] * df["log_eps"]
    X2 = [
        "log_E",
        "log_I",
        "log_N",
        "log_eps",
        "log_lam",
        "log_E_eps",
        "log_I_eps",
        "log_N_eps",
    ] + extra_feats
    m2, s2, c2 = fit_ols(df.dropna(subset=X2 + ["log_R"]), "log_R", X2)
    if c2 is not None and len(c2):
        c2.to_csv(OUTDIR / "coefficients_M2.csv", index=False)
    summaries["M2"] = s2
    if m2 is not None:
        plot_preds(m2, X2, "M2")

    # M3: regime split
    reg_results = {}
    for reg in ["closed", "min_open", "open"]:
        sub = df[df["regime"] == reg].dropna(subset=X2 + ["log_R"])
        if len(sub) < 20:
            reg_results[reg] = {"note": "insufficient data", "n": len(sub)}
            continue
        mR, sR, cR = fit_ols(sub, "log_R", X2)
        cR.to_csv(OUTDIR / f"coefficients_M3_{reg}.csv", index=False)
        reg_results[reg] = sR
        # preds
        Xc = sm.add_constant(sub[X2].astype(float))
        y_pred = np.exp(mR.predict(Xc))
        y_true = np.exp(sub["log_R"])
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=10)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], color="k", linewidth=0.8)
        plt.xlabel("Actual R_mean")
        plt.ylabel("Predicted R_mean")
        plt.title(f"Predicted vs Actual (M3 - {reg})")
        plt.savefig(OUTDIR / f"preds_actual_M3_{reg}.png", dpi=160)
        plt.close()

    # Save summary
    with open(OUTDIR / "fit_summary.json", "w") as f:
        json.dump(
            {
                "noise_ceiling": noise_ceiling,
                "M0": summaries.get("M0"),
                "M1": summaries.get("M1"),
                "M2": summaries.get("M2"),
                "M3": reg_results,
            },
            f,
            indent=2,
        )

    # report
    with open(OUTDIR / "report.md", "w") as f:
        f.write("# Equation Refinement Report (v1)\n\n")
        f.write(
            f"Rows read: {initial_n}, dropped missing required: {dropped}, dropped invalid logs: {dropped_logs}\n\n"
        )
        f.write("## Noise Ceiling\n")
        f.write(
            f"- Estimated upper R² bound (due to seed noise): **{noise_ceiling}**\n\n"
        )
        f.write("## Model Fit Summary\n")
        for k, v in summaries.items():
            if not v:
                f.write(f"- **{k}**: no fit performed\n")
                continue
            r2 = v.get("r2")
            r2_adj = v.get("r2_adj")
            n = v.get("n")
            r2s = f"{r2:.3f}" if r2 is not None else "NA"
            r2as = f"{r2_adj:.3f}" if r2_adj is not None else "NA"
            f.write(f"- **{k}**: R²={r2s}, adjR²={r2as}, n={n}\n")
        f.write("\n### Regime Split (M3)\n")
        for reg, v in reg_results.items():
            if isinstance(v, dict) and "note" in v:
                f.write(f"- {reg}: {v['note']} (n={v.get('n',0)})\n")
            else:
                r2 = v.get("r2")
                r2_adj = v.get("r2_adj")
                n = v.get("n")
                r2s = f"{r2:.3f}" if r2 is not None else "NA"
                r2as = f"{r2_adj:.3f}" if r2_adj is not None else "NA"
                f.write(f"- {reg}: R²={r2s}, adjR²={r2as}, n={n}\n")
        f.write("\n## Recommendation\n")
        f.write(
            "Prefer M2 (interactions) unless regime splits show significantly better fits; consider piecewise by openness regime.\n"
        )

    print(f"Validation complete. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()

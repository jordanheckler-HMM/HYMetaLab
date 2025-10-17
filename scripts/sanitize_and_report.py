#!/usr/bin/env python3
"""
Sanitize run CSVs  enforce numeric dtypes, compute per-test linear fits,
and emit validation summaries that won't be null/NaN.

Usage examples:
  python scripts/sanitize_and_report.py --domain civ --runs path/to/runs_summary.csv --outdir discovery_results/civ_tests_<stamp>/data
  python scripts/sanitize_and_report.py --domain agents --runs path/to/runs_summary.csv --outdir discovery_results/agent_tests_<stamp>/data
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_PARAM_COLS = ["epsilon", "cci", "eta", "inv_eta"]
NUMERIC_METRIC_COLS = [
    "final_CCI",
    "stability_CCI_mean",
    "stability_hazard_mean",
    "survival_rate",
]
OPTIONAL_ID_COLS = ["test", "seed", "domain", "mechanism"]

FIT_X_CANDIDATES = ["epsilon", "cci", "inv_eta", "eta"]
FIT_Y_CANDIDATES = [
    "stability_CCI_mean",
    "final_CCI",
    "survival_rate",
    "stability_hazard_mean",
]


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_inv_eta(df: pd.DataFrame) -> pd.DataFrame:
    # Create inv_eta if eta exists and inv_eta missing
    if "inv_eta" not in df.columns and "eta" in df.columns:
        inv = pd.to_numeric(df["eta"], errors="coerce")
        inv = inv.where(inv > 0)
        df["inv_eta"] = 1.0 / inv
    return df


def has_variation(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.nunique() >= 2


def linfit(x: np.ndarray, y: np.ndarray):
    # simple least squares: y = m x + b
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return float(m), float(b), r2


def compute_fits(df: pd.DataFrame, domain: str) -> list[dict[str, Any]]:
    out = []
    if df.empty:
        return out
    tests = (
        df["test"].dropna().unique().tolist() if "test" in df.columns else ["__all__"]
    )
    for t in tests:
        sub = df[df["test"] == t] if t != "__all__" else df.copy()
        for xcol in FIT_X_CANDIDATES:
            if xcol not in sub.columns:
                continue
            for ycol in FIT_Y_CANDIDATES:
                if ycol not in sub.columns:
                    continue
                X = pd.to_numeric(sub[xcol], errors="coerce")
                Y = pd.to_numeric(sub[ycol], errors="coerce")
                mask = ~(X.isna() | Y.isna())
                X, Y = X[mask], Y[mask]
                if len(X) < 3 or not has_variation(X) or not has_variation(Y):
                    # not enough info to fit
                    out.append(
                        {
                            "domain": domain,
                            "test": t,
                            "x": xcol,
                            "y": ycol,
                            "slope": float("nan"),
                            "intercept": float("nan"),
                            "r2": float("nan"),
                            "n": int(len(X)),
                        }
                    )
                    continue
                m, b, r2 = linfit(X.values, Y.values)
                out.append(
                    {
                        "domain": domain,
                        "test": t,
                        "x": xcol,
                        "y": ycol,
                        "slope": m,
                        "intercept": b,
                        "r2": r2,
                        "n": int(len(X)),
                    }
                )
    return out


def summarize_project(df: pd.DataFrame, runs_file: str) -> dict[str, Any]:
    # Project-level rollups
    mean_survival = (
        pd.to_numeric(df.get("survival_rate"), errors="coerce").dropna().mean()
    )
    # prefer final_CCI; fallback to stability_CCI_mean
    cci_series = pd.to_numeric(df.get("final_CCI"), errors="coerce")
    if cci_series is None or cci_series.dropna().empty:
        cci_series = pd.to_numeric(df.get("stability_CCI_mean"), errors="coerce")
    mean_cci = cci_series.dropna().mean() if cci_series is not None else np.nan

    # creation success = fraction with survival_rate > 0 (or final_CCI > 0 if no survival col)
    sr = pd.to_numeric(df.get("survival_rate"), errors="coerce")
    if sr is not None and not sr.dropna().empty:
        creation_success_frac = float((sr.fillna(0) > 0).mean())
    else:
        fc = pd.to_numeric(df.get("final_CCI"), errors="coerce")
        creation_success_frac = (
            float((fc.fillna(0) > 0).mean()) if fc is not None else 0.0
        )

    # Validation rule (simple, adjustable):
    # - mean_survival > 0.05 OR mean_cci > 0.02
    # - and at least some non-null rows exist
    nonnull_rows = int(len(df.dropna(how="all")))
    validated = bool(
        nonnull_rows > 0
        and ((mean_survival and mean_survival > 0.05) or (mean_cci and mean_cci > 0.02))
    )

    reason = "ok"
    if not validated:
        reasons = []
        if not nonnull_rows:
            reasons.append("no data rows")
        if not (mean_survival and mean_survival > 0.05):
            reasons.append("low or null mean_survival")
        if not (mean_cci and mean_cci > 0.02):
            reasons.append("low or null mean_cci")
        reason = "; ".join(reasons) if reasons else "no criteria matched"

    return {
        "file": runs_file,
        "mean_CCI": (
            None if (mean_cci is None or np.isnan(mean_cci)) else float(mean_cci)
        ),
        "mean_survival": (
            0.0
            if (mean_survival is None or np.isnan(mean_survival))
            else float(mean_survival)
        ),
        "creation_success_frac": float(creation_success_frac),
        "validated": bool(validated),
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        required=True,
        choices=["civ", "agents"],
        help="Domain label for outputs.",
    )
    ap.add_argument(
        "--runs",
        required=True,
        help="Path to runs_summary.csv or any CSV of per-run rows.",
    )
    ap.add_argument(
        "--outdir", required=True, help="Directory to write json summaries."
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs_path = args.runs
    if not os.path.exists(runs_path):
        print(f"[ERROR] runs file not found: {runs_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(runs_path)
    # Keep id/test columns if present
    keep_cols = [c for c in OPTIONAL_ID_COLS if c in df.columns]
    # sanitize numerics
    df = ensure_inv_eta(df)
    df = coerce_numeric(df, NUMERIC_PARAM_COLS + NUMERIC_METRIC_COLS)

    # Minimal guarantee: preserve test column; if absent, synthesize
    if "test" not in df.columns:
        df["test"] = "unspecified"

    # Write cleaned CSV next to original (optional)
    cleaned_path = os.path.join(args.outdir, f"runs_cleaned_{args.domain}.csv")
    df.to_csv(cleaned_path, index=False)

    # Compute per-test fits
    fits = compute_fits(df, args.domain)
    stamp = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
    summary_json = {"timestamp": stamp, "fits": fits}
    with open(os.path.join(args.outdir, f"summary_{args.domain}.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # Project-level summary
    proj = summarize_project(df, runs_path)
    with open(
        os.path.join(
            args.outdir, f"project_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        ),
        "w",
    ) as f:
        json.dump([proj], f, indent=2)

    # Integrity status file (mirrors your prior style)
    integrity_lines = [
        "# Research Integrity Status",
        f"Study: {args.domain}_tests  validated={proj['validated']}",
        f"Reason: {proj['reason']}",
        "",
    ]
    with open(os.path.join(args.outdir, "Research_Integrity_Prompt.md"), "w") as f:
        f.write("\n".join(integrity_lines))

    print(
        "[OK] Wrote:",
        os.path.join(args.outdir, f"summary_{args.domain}.json"),
        os.path.join(args.outdir, "project_summary_*"),
        os.path.join(args.outdir, "Research_Integrity_Prompt.md"),
        cleaned_path,
    )


if __name__ == "__main__":
    main()

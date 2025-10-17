#!/usr/bin/env python3
"""
Phase19b — Validation Evaluator
Parses Phase19 outputs (zip or folder), checks laws vs theory bands, and writes a markdown summary + plots.
"""
import glob
import json
import math
import random
import statistics
import zipfile
from datetime import datetime
from pathlib import Path

# Optional deps
try:
    import numpy as np
    import pandas as pd

    HAVE_PD = True
except Exception:
    HAVE_PD = False

try:
    import matplotlib.pyplot as plt

    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/Phase19b_Evaluator_{TS}")
OUTDIR.mkdir(parents=True, exist_ok=True)

FORCE_INPUT_PATH = ""
THEORY_BOUNDS = {
    "rhoN_min": 10.0,
    "rhoN_max": 15.0,
    "slope_target": -1.0,
    "slope_tol": 0.35,
    "beta_alpha_min": 6.0,
    "beta_alpha_max": 7.0,
    "epsilon_min": 0.004,
    "epsilon_max": 0.010,
    "residual_max": 0.05,
    "lambda_star": 0.90,
    "lambda_tol": 0.05,
    "tvar_max_at_lambda_star": 0.10,
    "cci_slope_abs_max": 0.001,
}


def find_input_path():
    if FORCE_INPUT_PATH:
        return FORCE_INPUT_PATH
    zips = sorted(
        glob.glob("phase19_validation_*.zip")
        + glob.glob("./**/phase19_validation_*.zip", recursive=True)
    )
    if zips:
        return zips[-1]
    dirs = sorted(
        [p for p in glob.glob("./**/", recursive=True) if "Phase19_Validation_" in p]
    )
    return dirs[-1] if dirs else ""


INPUT_PATH = find_input_path()
if not INPUT_PATH:
    print(
        "❌ Could not find Phase 19 validation outputs. Set FORCE_INPUT_PATH to your ZIP/folder."
    )
    raise SystemExit(1)

WORKDIR = OUTDIR / "extracted"
WORKDIR.mkdir(exist_ok=True)


def extract_if_zip(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(WORKDIR)
        return str(WORKDIR)
    return path


ROOT = Path(extract_if_zip(INPUT_PATH))


def load_json_candidates(glob_pattern_list):
    for pat in glob_pattern_list:
        for f in ROOT.glob(pat):
            try:
                return json.loads(Path(f).read_text())
            except Exception:
                continue
    return None


def load_csv_candidates(glob_pattern_list):
    frames = []
    if not HAVE_PD:
        return frames
    for pat in glob_pattern_list:
        for f in ROOT.glob(pat):
            try:
                df = pd.read_csv(f)
                if len(df):
                    frames.append(df)
            except Exception:
                pass
    return frames


def bootstrap_ci(data, nboot=500, func=statistics.mean, alpha=0.05, seed=42):
    rng = random.Random(seed)
    if not data:
        return (math.nan, (math.nan, math.nan))
    boots = []
    for _ in range(nboot):
        sample = [data[rng.randrange(0, len(data))] for __ in range(len(data))]
        try:
            boots.append(func(sample))
        except Exception:
            continue
    boots.sort()
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[int((1 - alpha / 2) * len(boots)) - 1]
    return (func(data), (lo, hi))


def linregress(x, y):
    n = len(x)
    if n < 2:
        return math.nan, math.nan
    xm = sum(x) / n
    ym = sum(y) / n
    num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    den = sum((xi - xm) ** 2 for xi in x)
    if den == 0:
        return math.nan, math.nan
    slope = num / den
    intercept = ym - slope * xm
    return slope, intercept


report_lines = []


def add_line(s=""):
    report_lines.append(s)


# ---------- 1) Dunbar ----------
dunbar_json = load_json_candidates(
    [
        "**/dunbar_debug/*summary*.json",
        "**/dunbar_debug/*_summary.json",
        "**/dunbar_debug/**/summary.json",
    ]
)
dunbar_csvs = load_csv_candidates(
    [
        "**/dunbar_debug/*.csv",
        "**/dunbar_debug/**/runs_summary.csv",
        "**/dunbar_debug/**/trajectories*.csv",
    ]
)

rhoN_vals = []
slope_est = math.nan
dunbar_pass = False
dunbar_notes = []

if HAVE_PD and dunbar_csvs:
    df = pd.concat(dunbar_csvs, ignore_index=True).copy()
    possible_rho = [c for c in df.columns if "rho" in c.lower() and "*" not in c]
    possible_N = [c for c in df.columns if c.lower() in ("n", "agents", "agent_count")]
    if not possible_N and "params_json" in df.columns:
        try:
            df["N"] = df["params_json"].apply(lambda s: json.loads(s).get("N"))
            possible_N = ["N"]
        except Exception:
            pass
    rho_col = possible_rho[0] if possible_rho else None
    N_col = possible_N[0] if possible_N else None
    if rho_col and N_col:
        sub = df[[rho_col, N_col]].dropna()
        rhoN_vals = list((sub[rho_col] * sub[N_col]).astype(float))
        try:
            x = list(1.0 / sub[N_col].astype(float))
            y = list(sub[rho_col].astype(float))
            slope_est, _ = linregress(x, y)
        except Exception:
            pass

rhoN_mean, (rhoN_lo, rhoN_hi) = (
    bootstrap_ci(rhoN_vals) if rhoN_vals else (math.nan, (math.nan, math.nan))
)
slope_ok = (not math.isnan(slope_est)) and (
    abs(slope_est - THEORY_BOUNDS["slope_target"]) <= THEORY_BOUNDS["slope_tol"]
)
rhoN_ok = (not math.isnan(rhoN_mean)) and (
    THEORY_BOUNDS["rhoN_min"] <= rhoN_mean <= THEORY_BOUNDS["rhoN_max"]
)

dunbar_pass = slope_ok and rhoN_ok
if not rhoN_vals:
    dunbar_notes.append("No ρ★·N data found.")
if math.isnan(slope_est):
    dunbar_notes.append("Could not estimate slope(ρ★ vs 1/N).")

# ---------- 2) Energy–Information Equivalence ----------
ei_json = load_json_candidates(
    [
        "**/energy_info_equiv/*summary*.json",
        "**/energy_info_equiv/**/summary.json",
        "**/energy_info_equiv/**/stats.json",
    ]
)
ei_csvs = load_csv_candidates(
    [
        "**/energy_info_equiv/*.csv",
        "**/energy_info_equiv/**/runs_summary.csv",
        "**/energy_info_equiv/**/trajectories*.csv",
    ]
)

beta_alpha_vals = []
if HAVE_PD and ei_csvs:
    df = pd.concat(ei_csvs, ignore_index=True)
    candidates = [c for c in df.columns if "beta" in c.lower() and "alpha" in c.lower()]
    if candidates:
        col = candidates[0]
        try:
            beta_alpha_vals = list(
                pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
            )
        except Exception:
            pass

beta_alpha_mean, (ba_lo, ba_hi) = (
    bootstrap_ci(beta_alpha_vals)
    if beta_alpha_vals
    else (math.nan, (math.nan, math.nan))
)
ei_pass = (not math.isnan(beta_alpha_mean)) and (
    THEORY_BOUNDS["beta_alpha_min"]
    <= beta_alpha_mean
    <= THEORY_BOUNDS["beta_alpha_max"]
)

# ---------- 3) Openness ε Calibration ----------
op_json = load_json_candidates(
    [
        "**/openness_fit/*summary*.json",
        "**/openness_fit/**/summary.json",
        "**/openness_fit/**/fit.json",
    ]
)
op_csvs = load_csv_candidates(
    [
        "**/openness_fit/*.csv",
        "**/openness_fit/**/runs_summary.csv",
        "**/openness_fit/**/trajectories*.csv",
    ]
)

best_eps = math.nan
best_resid = math.nan

if HAVE_PD and op_csvs:
    df = pd.concat(op_csvs, ignore_index=True)
    eps_col = next(
        (c for c in df.columns if "epsilon" in c.lower() or c.lower() == "eps"), None
    )
    resid_col = next(
        (c for c in df.columns if "resid" in c.lower() or "error" in c.lower()), None
    )
    if eps_col and resid_col:
        sdf = df[[eps_col, resid_col]].dropna()
        if len(sdf):
            row = sdf.loc[sdf[resid_col].astype(float).idxmin()]
            best_eps = float(row[eps_col])
            best_resid = float(row[resid_col])

eps_pass = (not math.isnan(best_eps)) and (
    THEORY_BOUNDS["epsilon_min"] <= best_eps <= THEORY_BOUNDS["epsilon_max"]
)
resid_pass = (not math.isnan(best_resid)) and (
    best_resid <= THEORY_BOUNDS["residual_max"]
)
openness_pass = eps_pass and resid_pass

# ---------- 4) Temporal Feedback (λ*) ----------
tf_csvs = load_csv_candidates(
    [
        "**/temporal_feedback/*.csv",
        "**/temporal_feedback/**/runs_summary.csv",
        "**/temporal_feedback/**/trajectories*.csv",
    ]
)

lambda_vals = []
tvar_by_lambda = {}
cci_slope_vals = []

if HAVE_PD and tf_csvs:
    df = pd.concat(tf_csvs, ignore_index=True)
    lam_col = next(
        (c for c in df.columns if c.lower() in ("lambda", "λ", "lambda_val")), None
    )
    tvar_col = next(
        (c for c in df.columns if "time" in c.lower() and "var" in c.lower()), None
    )
    cci_col = next((c for c in df.columns if c.lower() == "cci"), None)
    epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)

    if lam_col and tvar_col:
        for lam, grp in df.groupby(lam_col):
            v = pd.to_numeric(grp[tvar_col], errors="coerce").dropna().values
            if len(v):
                tvar_by_lambda[float(lam)] = float(np.median(v))

    if lam_col and cci_col and epoch_col:
        for lam, grp in df.groupby(lam_col):
            g2 = grp[[epoch_col, cci_col]].dropna()
            if len(g2) >= 10:
                k = max(10, int(0.2 * len(g2)))
                tail = g2.tail(k)
                try:
                    x = (
                        pd.to_numeric(tail[epoch_col], errors="coerce")
                        .astype(float)
                        .tolist()
                    )
                    y = (
                        pd.to_numeric(tail[cci_col], errors="coerce")
                        .astype(float)
                        .tolist()
                    )
                    slope, _ = linregress(x, y)
                    if not math.isnan(slope):
                        cci_slope_vals.append(slope)
                except Exception:
                    pass

lambda_star = THEORY_BOUNDS["lambda_star"]
nearest_lambda = (
    min(tvar_by_lambda.keys(), key=lambda L: abs(L - lambda_star))
    if tvar_by_lambda
    else math.nan
)
tvar_at_star = (
    tvar_by_lambda.get(nearest_lambda, math.nan)
    if not math.isnan(nearest_lambda)
    else math.nan
)
cci_slope_abs = (
    statistics.median([abs(s) for s in cci_slope_vals]) if cci_slope_vals else math.nan
)

lambda_pass = (
    (not math.isnan(nearest_lambda))
    and abs(nearest_lambda - lambda_star) <= THEORY_BOUNDS["lambda_tol"]
    and (not math.isnan(tvar_at_star))
    and tvar_at_star <= THEORY_BOUNDS["tvar_max_at_lambda_star"]
    and (not math.isnan(cci_slope_abs))
    and cci_slope_abs <= THEORY_BOUNDS["cci_slope_abs_max"]
)


def pf(b):
    return "PASS" if b else "FAIL"


add_line("# Phase 19b — Validation Evaluator\n")
add_line(f"Input: `{INPUT_PATH}` (extracted: `{ROOT}`)\n")
add_line("## Summary Table\n")
add_line("| Law | Metric(s) | Result | Details |")
add_line("|---|---|---|---|")

add_line(
    f"| Dunbar ρ★ | ρ★·N in [10,15]; slope ≈ -1±{THEORY_BOUNDS['slope_tol']} | **{pf(dunbar_pass)}** | "
    f"ρ★·N mean={rhoN_mean:.2f} (CI {rhoN_lo:.2f}–{rhoN_hi:.2f}); slope={slope_est:.2f}; {'; '.join(dunbar_notes)} |"
)

add_line(
    f"| Energy–Info | β/α in [6,7] | **{pf(ei_pass)}** | β/α mean={beta_alpha_mean:.2f} (CI {ba_lo:.2f}–{ba_hi:.2f}) |"
)

add_line(
    f"| Openness ε | ε∈[0.004,0.010], residual≤{THEORY_BOUNDS['residual_max']} | **{pf(openness_pass)}** | "
    f"best ε={best_eps:.4f}; residual={best_resid:.3f} |"
)

add_line(
    f"| Temporal λ* | λ≈0.90±0.05; low t_var; flat CCI | **{pf(lambda_pass)}** | "
    f"nearest λ={nearest_lambda if not math.isnan(nearest_lambda) else 'nan'}; "
    f"t_var@λ≈{tvar_at_star if not math.isnan(tvar_at_star) else 'nan'}; "
    f"median |CCI slope|={cci_slope_abs if not math.isnan(cci_slope_abs) else 'nan'} |"
)

overall = all([dunbar_pass, ei_pass, openness_pass, lambda_pass])
add_line("\n---\n")
add_line(
    f"**Overall Verdict:** {'✅ VALIDATED' if overall else '⚠️ Partial — needs review'}\n"
)

report_path = OUTDIR / "phase19b_summary.md"
Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
print(f"📝 Wrote summary → {report_path}")

if HAVE_PLT:
    try:
        if HAVE_PD and dunbar_csvs and rhoN_vals:
            df = pd.concat(dunbar_csvs, ignore_index=True)
            rho_col = [c for c in df.columns if "rho" in c.lower() and "*" not in c][0]
            N_col = [
                c
                for c in df.columns
                if c.lower() in ("n", "agents", "agent_count", "N")
            ][0]
            sub = df[[rho_col, N_col]].dropna().copy()
            sub["invN"] = 1.0 / pd.to_numeric(sub[N_col], errors="coerce")
            x = sub["invN"].astype(float).values
            y = pd.to_numeric(sub[rho_col], errors="coerce").astype(float).values
            plt.figure()
            plt.scatter(x, y)
            m, b = linregress(x, y)
            xx = np.linspace(min(x), max(x), 100)
            yy = m * xx + b
            plt.plot(xx, yy)
            plt.xlabel("1/N")
            plt.ylabel("rho_star")
            plt.title(f"Dunbar: rho vs 1/N (slope={m:.2f})")
            plt.tight_layout()
            p = OUTDIR / "plot_dunbar_rho_vs_invN.png"
            plt.savefig(p, dpi=160)
            print(f"📈 Saved {p}")
    except Exception:
        pass

    try:
        if HAVE_PD and op_csvs and not math.isnan(best_resid):
            df = pd.concat(op_csvs, ignore_index=True)
            eps_col = next(
                (c for c in df.columns if "epsilon" in c.lower() or c.lower() == "eps"),
                None,
            )
            resid_col = next(
                (c for c in df.columns if "resid" in c.lower() or "error" in c.lower()),
                None,
            )
            if eps_col and resid_col:
                sdf = df[[eps_col, resid_col]].dropna().copy()
                plt.figure()
                plt.scatter(sdf[eps_col].astype(float), sdf[resid_col].astype(float))
                plt.axvspan(
                    THEORY_BOUNDS["epsilon_min"],
                    THEORY_BOUNDS["epsilon_max"],
                    alpha=0.2,
                )
                plt.axhline(THEORY_BOUNDS["residual_max"])
                plt.xlabel("epsilon")
                plt.ylabel("residual")
                plt.title("Openness Fit Residuals")
                plt.tight_layout()
                p = OUTDIR / "plot_openness_residuals.png"
                plt.savefig(p, dpi=160)
                print(f"📈 Saved {p}")
    except Exception:
        pass

    try:
        if tvar_by_lambda:
            lam_vals = sorted(tvar_by_lambda.keys())
            tv_vals = [tvar_by_lambda[L] for L in lam_vals]
            plt.figure()
            plt.plot(lam_vals, tv_vals, marker="o")
            plt.axvline(THEORY_BOUNDS["lambda_star"])
            plt.axvspan(
                THEORY_BOUNDS["lambda_star"] - THEORY_BOUNDS["lambda_tol"],
                THEORY_BOUNDS["lambda_star"] + THEORY_BOUNDS["lambda_tol"],
                alpha=0.2,
            )
            plt.axhline(THEORY_BOUNDS["tvar_max_at_lambda_star"])
            plt.xlabel("lambda")
            plt.ylabel("time variance")
            plt.title("Temporal Feedback: t_var vs lambda")
            plt.tight_layout()
            p = OUTDIR / "plot_temporal_tvar_vs_lambda.png"
            plt.savefig(p, dpi=160)
            print(f"📈 Saved {p}")
    except Exception:
        pass

print("✅ Phase 19b evaluator complete.")
print(f"Summary: {report_path}")
print(f"Artifacts: {OUTDIR}")

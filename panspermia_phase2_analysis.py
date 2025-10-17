#!/usr/bin/env python3
"""
Phase II analysis for Panspermia / Openness experiment.
- Loads previous outputs (conditions_summary.json, runs_summary.csv, openness_events.csv, bayes_summary.json)
- Inherits baseline params from panspermia_orchestrator.py
- Computes a normalized collapse-risk per run using:
    1) min-max normalization of raw collapse risk
    2) openness decay factor: exp(-decay_k * cumulative_openness_exposure)
    3) astrophysical anchor factor from bayes_summary.combined_bf
  Final normalized_cr = cr_norm * openness_decay * anchor_factor
- Produces runs_summary_normalized.csv, conditions_summary_normalized.json and comparison plots.

Assumptions (defaults): decay_k = 1e-3. If you want a different function or parameterization, tell me and I'll update.
"""

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
# find latest panspermia run folder
runs = sorted((ROOT / "discovery_results").glob("panspermia_*"))
if not runs:
    raise SystemExit("No panspermia_* run folders found under discovery_results/")
LATEST = runs[-1]
print("Using run folder:", LATEST)

# files
conds_f = LATEST / "conditions_summary.json"
runs_f = LATEST / "runs_summary.csv"
open_f = LATEST / "openness_events.csv"
bayes_f = LATEST / "bayes_summary.json"
orch_f = Path("panspermia_orchestrator.py")

# load data
conds = json.loads(conds_f.read_text())
df = pd.read_csv(runs_f)
df_open = pd.read_csv(open_f)
bayes = json.loads(bayes_f.read_text())

# inherit baseline params by reading orchestrator source
src = orch_f.read_text() if orch_f.exists() else ""
get_param = lambda name, default: (
    int(re.search(rf"{name}\s*=\s*(\d+)", src).group(1))
    if re.search(rf"{name}\s*=\s*(\d+)", src)
    else default
)
BASELINE_STEPS = get_param("BASELINE_STEPS", 80)
LONG_STEPS = get_param("LONG_STEPS", 5000)
THIN_STRIDE = get_param("THIN_STRIDE", 10)
STABILITY_WINDOW = get_param("STABILITY_WINDOW", 200)
print(
    "Inherited params: BASELINE_STEPS=%d LONG_STEPS=%d THIN_STRIDE=%d STABILITY_WINDOW=%d"
    % (BASELINE_STEPS, LONG_STEPS, THIN_STRIDE, STABILITY_WINDOW)
)

# map condition->epsilon (parse condition names)


def parse_epsilon_from_condition(name):
    m = re.search(r"eps([0-9\.eE+-]+)", name)
    if m:
        try:
            return float(m.group(1))
        except:
            return 0.0
    return 0.0


cond_eps = {c["condition"]: parse_epsilon_from_condition(c["condition"]) for c in conds}
print("Condition epsilons:", cond_eps)

# compute per-run decayed openness exposure using openness_events.csv
# We'll decay past events toward the run end (LONG_STEPS) with an exponential kernel
OPENNESS_LAMBDA = 1e-3  # decay rate per epoch for temporal weighting of openness events
decay_k = 1e-3  # used to convert exposure -> multiplicative decay factor

# build mapping epsilon -> condition names (may be many-to-one; choose matching eps)
eps_to_conditions = {}
for c, eps in cond_eps.items():
    eps_to_conditions.setdefault(eps, []).append(c)

# precompute events per (seed, epsilon)
events = df_open[["time", "seed", "epsilon"]].copy()


# compute decayed exposure for each (condition,seed)
def compute_decayed_exposure(cond, seed):
    eps = cond_eps.get(cond, 0.0)
    if eps == 0.0:
        return 0.0
    # select events matching this seed and epsilon (match float equality — eps values are simple here)
    ev = events[(events.seed == int(seed)) & (events.epsilon == float(eps))]
    if ev.empty:
        # fallback: assume continuous exposure
        return float(eps * LONG_STEPS)
    # decay weights relative to run end
    T = LONG_STEPS
    times = ev.time.astype(float).values
    weights = np.exp(-OPENNESS_LAMBDA * (T - times))
    # exposure = sum(epsilon * weight)
    return float((eps * weights).sum())


# compute exposures column
exposures = df.apply(
    lambda row: compute_decayed_exposure(row["condition"], row["seed"]), axis=1
)

# raw collapse risk
raw_cr = df["long_collapse_risk"].astype(float)
# min-max normalization across all runs
min_cr = raw_cr.min()
max_cr = raw_cr.max()
if max_cr - min_cr > 0:
    cr_norm = (raw_cr - min_cr) / (max_cr - min_cr)
else:
    cr_norm = pd.Series(0.0, index=df.index)

# openness_decay factor from exposures
openness_decay = np.exp(-decay_k * exposures)

# astrophysical anchor using bayes_summary combined_bf
combined_bf = bayes.get("combined_bf", 1.0)
# anchor factor chosen as 1 / (1 + log1p(combined_bf)) to moderate extreme BFs
anchor_factor = (
    1.0 / (1.0 + math.log1p(abs(combined_bf))) if combined_bf is not None else 1.0
)

# final normalized collapse risk (pre-clamp)
pre_clamp = cr_norm * openness_decay * anchor_factor
# clamp to [0,1]
clamped = pre_clamp.clip(lower=0.0, upper=1.0)

df["collapse_risk_raw"] = raw_cr
df["collapse_risk_minmax"] = cr_norm
df["collapse_risk_preclamp"] = pre_clamp
df["collapse_risk_normalized"] = clamped
df["openness_exposure_decayed"] = exposures
df["openness_decay"] = openness_decay
df["astro_anchor"] = anchor_factor

# compute stability metrics from trajectories_long.csv if available
traj_f = LATEST / "trajectories_long.csv"
if traj_f.exists():
    df_traj = pd.read_csv(traj_f)
    # window threshold (use last STABILITY_WINDOW epochs)
    min_epoch = max(0, LONG_STEPS - STABILITY_WINDOW)
    # because trajectories may be thinned, pick rows with epoch >= min_epoch
    df_window = df_traj[df_traj.epoch >= min_epoch]
    # compute per-run means
    stab = (
        df_window.groupby(["condition", "seed"])
        .agg({"cci": "mean", "hazard": "mean"})
        .reset_index()
    )
    stab = stab.rename(
        columns={
            "cci": "stability_cci_from_traj",
            "hazard": "stability_hazard_from_traj",
        }
    )
    # merge into df
    df = pd.merge(df, stab, how="left", on=["condition", "seed"])
else:
    # fallback: use existing long_stability_cci in runs if present
    if "long_stability_cci" in df.columns:
        df["stability_cci_from_traj"] = df["long_stability_cci"]
    else:
        df["stability_cci_from_traj"] = np.nan
    df["stability_hazard_from_traj"] = np.nan

# update conditions summary with normalized means
conds_norm = []
for cond in df.condition.unique():
    sub = df[df.condition == cond]
    conds_norm.append(
        {
            "condition": cond,
            "mean_collapse_risk_normalized": float(
                sub["collapse_risk_normalized"].mean()
            ),
            "mean_collapse_risk_raw": float(sub["long_collapse_risk"].mean()),
            "mean_openness_exposure": float(sub["openness_exposure_decayed"].mean()),
        }
    )

# Save outputs
out_dir = LATEST
df.to_csv(out_dir / "runs_summary_normalized.csv", index=False)
with open(out_dir / "conditions_summary_normalized.json", "w") as f:
    json.dump(conds_norm, f, indent=2)

# Correlations: collapse_risk_normalized vs stability metrics
cor_results = {}
if "stability_cci_from_traj" in df.columns:
    valid = df[
        [
            "collapse_risk_normalized",
            "stability_cci_from_traj",
            "stability_hazard_from_traj",
        ]
    ].dropna()
    if not valid.empty:
        cor_results["corr_cr_cci"] = float(
            valid["collapse_risk_normalized"].corr(valid["stability_cci_from_traj"])
        )
        cor_results["corr_cr_hazard"] = float(
            valid["collapse_risk_normalized"].corr(valid["stability_hazard_from_traj"])
        )
    else:
        cor_results["corr_cr_cci"] = None
        cor_results["corr_cr_hazard"] = None
else:
    cor_results["corr_cr_cci"] = None
    cor_results["corr_cr_hazard"] = None

with open(out_dir / "correlations.json", "w") as f:
    json.dump(cor_results, f, indent=2)

# Scatter plots
if cor_results.get("corr_cr_cci") is not None:
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["stability_cci_from_traj"], df["collapse_risk_normalized"], alpha=0.7
    )
    plt.xlabel("Stability CCI (traj)")
    plt.ylabel("collapse_risk_normalized")
    plt.title("Normalized collapse risk vs Stability CCI")
    plt.tight_layout()
    plt.savefig(out_dir / "cr_vs_cci.png")
    plt.close()

if cor_results.get("corr_cr_hazard") is not None:
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["stability_hazard_from_traj"], df["collapse_risk_normalized"], alpha=0.7
    )
    plt.xlabel("Stability Hazard (traj)")
    plt.ylabel("collapse_risk_normalized")
    plt.title("Normalized collapse risk vs Stability Hazard")
    plt.tight_layout()
    plt.savefig(out_dir / "cr_vs_hazard.png")
    plt.close()

# Plots: raw vs normalized collapse risk by condition
plt.figure(figsize=(8, 4))
for cond in df.condition.unique():
    sub = df[df.condition == cond]
    plt.scatter(
        [cond] * len(sub), sub["long_collapse_risk"], alpha=0.6, label=f"{cond} raw"
    )
    plt.scatter(
        [cond] * len(sub),
        sub["collapse_risk_normalized"],
        alpha=0.6,
        marker="x",
        label=f"{cond} norm",
    )
plt.ylabel("collapse risk")
plt.xticks(rotation=20)
plt.title("Raw vs Normalized collapse risk")
plt.tight_layout()
plt.savefig(out_dir / "collapse_risk_comparison.png")
plt.close()

# Save a short report
rep = out_dir / "report" / "panspermia_phase2.md"
with open(rep, "w") as f:
    f.write("# Panspermia Phase II Normalization Report\n\n")
    f.write(
        "Parameters inherited from orchestrator: BASELINE_STEPS=%d, LONG_STEPS=%d, THIN_STRIDE=%d, STABILITY_WINDOW=%d\n\n"
        % (BASELINE_STEPS, LONG_STEPS, THIN_STRIDE, STABILITY_WINDOW)
    )
    f.write("Normalization recipe:\n")
    f.write("- raw_cr -> min-max across runs.\n")
    f.write(
        "- openness exposure = epsilon * LONG_STEPS, openness decay = exp(-decay_k * exposure) with decay_k=%g.\n"
        % decay_k
    )
    f.write(
        "- astrophysical anchor factor = 1/(1+log1p(combined_bf)) with combined_bf=%s\n\n"
        % str(combined_bf)
    )
    f.write("Results: conditions summary normalized:\n\n")
    f.write(pd.DataFrame(conds_norm).to_string())

print("Phase II normalization complete. Outputs written to", out_dir)

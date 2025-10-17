"""
Openness–Entropy Trade-off (v1)
Author: Heck Yeah Lab
Runtime: ~30–45 seconds on a typical laptop

What this tests
---------------
We sweep openness ε in [0, 1] and simulate shock→recovery dynamics.
For each ε, we run N Monte Carlo trajectories and compute:
- AURC           : Area Under (Health) Recovery Curve (higher is better)
- T80            : Time to recover to 0.80 health (lower is better); NaN if never reaches
- min_dip        : Lowest health reached (higher is better)
- survival_pct   : % of runs whose health never fell below collapse_threshold
- entropy_rate   : Mean positive rate of disorder increase (lower is better)

Simple definitions (explicit + reproducible)
--------------------------------------------
- Health h_t ∈ [0, 1.2], starts at 1.0.
- Disorder d_t := 1 - clip(h_t, 0, 1).
- Entropy rate = mean of positive deltas Δd_t across t (average increase of disorder).
- Effective shock = S * (1-ε) * (0.5 + 0.5*C) * (1-R), applied during [t0, t0+D).

"Resilience Score" (composite, unitless)
----------------------------------------
Scaled combination to pick ε*:
  score = w1 * norm(AURC) + w2 * (1 - norm(entropy_rate)) + w3 * norm(survival_pct) + w4 * (1 - norm(T80))
Weights (edit if desired): w = (0.35, 0.30, 0.20, 0.15)

Outputs
-------
- CSV  : discovery_results/openness_entropy/openness_entropy_tradeoff_<timestamp>.csv
- PNGs : metric plots + best-vs-worst overlay
- JSON : small summary with epsilon_star and headline metrics

Tweak knobs in CONFIG below. Keep other code unchanged for comparability.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------
# CONFIG (edit here)
# -----------------
SEED = 42
EPOCHS = 120  # total time steps
T0 = 20  # shock starts at t = T0
DUR = 10  # shock duration
S = 0.60  # shock magnitude (0..1)
R = 0.30  # redundancy (buffers) (0..1)
C = 0.50  # coupling (0..1)
G = 0.05  # baseline growth toward health
KAPPA = 0.30  # damping back toward baseline
SIGMA = 0.01  # stochastic noise std
EPSILON_GRID = np.round(np.linspace(0.00, 1.00, 21), 2)  # 0.00, 0.05, ..., 1.00
N_RUNS = 64  # Monte Carlo runs per epsilon
COLLAPSE_THRESHOLD = 0.05  # below -> considered collapsed

# Composite score weights
W_AURC, W_ENTROPY, W_SURVIVAL, W_T80 = 0.35, 0.30, 0.20, 0.15

# -----------------
# Helpers
# -----------------
rng = np.random.default_rng(SEED)
random.seed(SEED)


def simulate_once(epsilon: float) -> dict[str, float]:
    """Simulate one trajectory; return metrics (AURC, T80, min_dip, survival, entropy_rate)."""
    h = 1.0
    h_hist = np.zeros(EPOCHS, dtype=float)
    disorder = np.zeros(EPOCHS, dtype=float)

    for t in range(EPOCHS):
        # Shock active?
        shock = S if (T0 <= t < T0 + DUR) else 0.0
        # Effective shock with openness, coupling, redundancy
        eff_shock = shock * (1 - epsilon) * (0.5 + 0.5 * C) * (1 - R)

        # Update health
        # Pull toward 1 via growth G; damping pulls back if >1
        noise = rng.normal(0.0, SIGMA)
        h = h - eff_shock + G * (1 - h) - KAPPA * max(0.0, h - 1.0) + noise
        # Clamp
        h = max(0.0, min(1.2, h))
        h_hist[t] = h
        disorder[t] = 1.0 - min(1.0, max(0.0, h))

    # Metrics
    aurc = float(np.trapz(h_hist, dx=1) / EPOCHS)  # normalized AURC (0..~1.1)
    # T80
    t80 = np.nan
    above = np.where(h_hist >= 0.80)[0]
    if len(above) > 0:
        t80 = float(above[0])
    min_dip = float(np.min(h_hist))
    survival = float(np.all(h_hist >= COLLAPSE_THRESHOLD))

    # Entropy rate: mean positive change in disorder
    dd = np.diff(disorder)
    pos_increases = dd[dd > 0]
    entropy_rate = float(np.mean(pos_increases)) if pos_increases.size > 0 else 0.0

    return {
        "AURC": aurc,
        "T80": t80,
        "min_dip": min_dip,
        "survival": survival,
        "entropy_rate": entropy_rate,
    }


def run_sweep() -> pd.DataFrame:
    rows = []
    for eps in EPSILON_GRID:
        metrics = [simulate_once(float(eps)) for _ in range(N_RUNS)]
        # Aggregate
        aurc = np.array([m["AURC"] for m in metrics])
        t80 = np.array([m["T80"] for m in metrics], dtype=float)
        min_dip = np.array([m["min_dip"] for m in metrics])
        surv = np.array([m["survival"] for m in metrics], dtype=float)
        ent = np.array([m["entropy_rate"] for m in metrics])

        row = {
            "epsilon": float(eps),
            "AURC_mean": float(np.mean(aurc)),
            "AURC_std": float(np.std(aurc)),
            "T80_mean": float(np.nanmean(t80)),
            "T80_nan_frac": float(np.isnan(t80).mean()),
            "min_dip_mean": float(np.mean(min_dip)),
            "survival_pct": float(np.mean(surv) * 100.0),
            "entropy_rate_mean": float(np.mean(ent)),
            "entropy_rate_std": float(np.std(ent)),
            "runs": int(N_RUNS),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(series)))
    return (series - lo) / (hi - lo)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize components for composite score
    n_aurc = minmax_norm(df["AURC_mean"])
    n_entropy = minmax_norm(df["entropy_rate_mean"])  # lower is better
    n_surv = minmax_norm(df["survival_pct"])
    n_t80 = minmax_norm(df["T80_mean"])  # lower is better

    score = (
        W_AURC * n_aurc
        + W_ENTROPY * (1 - n_entropy)
        + W_SURVIVAL * n_surv
        + W_T80 * (1 - n_t80)
    )
    out = df.copy()
    out["score"] = score
    # Pick epsilon* (break ties by higher AURC)
    best_idx = int(out["score"].idxmax())
    out["is_optimal"] = False
    out.loc[best_idx, "is_optimal"] = True
    return out


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("discovery_results", "openness_entropy")
    os.makedirs(out_dir, exist_ok=True)

    df = run_sweep()
    df = add_scores(df)

    csv_path = os.path.join(out_dir, f"openness_entropy_tradeoff_{ts}.csv")
    df.to_csv(csv_path, index=False)

    # Summary JSON
    best = df.loc[df["is_optimal"]].iloc[0].to_dict()
    summary = {
        "timestamp": ts,
        "epsilon_star": best["epsilon"],
        "score_star": best["score"],
        "AURC_mean_at_star": best["AURC_mean"],
        "T80_mean_at_star": best["T80_mean"],
        "survival_pct_at_star": best["survival_pct"],
        "entropy_rate_mean_at_star": best["entropy_rate_mean"],
        "config": {
            "EPOCHS": EPOCHS,
            "T0": T0,
            "DUR": DUR,
            "S": S,
            "R": R,
            "C": C,
            "G": G,
            "KAPPA": KAPPA,
            "SIGMA": SIGMA,
            "N_RUNS": N_RUNS,
            "EPSILON_GRID": EPSILON_GRID.tolist(),
            "COLLAPSE_THRESHOLD": COLLAPSE_THRESHOLD,
            "weights": {
                "AURC": W_AURC,
                "entropy": W_ENTROPY,
                "survival": W_SURVIVAL,
                "T80": W_T80,
            },
        },
        "paths": {"csv": csv_path},
    }
    json_path = os.path.join(out_dir, f"openness_entropy_summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    def save_plot(x, y, ylabel, fname):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Openness ε")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Openness ε")
        plt.grid(True, alpha=0.3)
        p = os.path.join(out_dir, f"{fname}_{ts}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        return p

    p1 = save_plot(df["epsilon"], df["AURC_mean"], "AURC (mean)", "aurc_vs_epsilon")
    p2 = save_plot(
        df["epsilon"],
        df["entropy_rate_mean"],
        "Entropy Rate (mean)",
        "entropy_rate_vs_epsilon",
    )
    p3 = save_plot(
        df["epsilon"], df["survival_pct"], "Survival %", "survival_vs_epsilon"
    )
    p4 = save_plot(df["epsilon"], df["T80_mean"], "T80 (mean epochs)", "t80_vs_epsilon")
    p5 = save_plot(df["epsilon"], df["score"], "Composite Score", "score_vs_epsilon")

    # Best vs worst overlay (AURC & Entropy)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epsilon"], df["AURC_mean"], marker="o", label="AURC (↑ better)")
    plt.plot(
        df["epsilon"],
        df["entropy_rate_mean"],
        marker="s",
        label="Entropy Rate (↓ better)",
    )
    plt.axvline(x=best["epsilon"], linestyle="--", label=f"ε* = {best['epsilon']:.2f}")
    plt.xlabel("Openness ε")
    plt.title("Best ε marker with AURC & Entropy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p6 = os.path.join(out_dir, f"overlay_best_{ts}.png")
    plt.tight_layout()
    plt.savefig(p6, dpi=150)
    plt.close()

    print("=== Openness–Entropy Trade-off (v1) complete ===")
    print(f"CSV     : {csv_path}")
    print(f"Summary : {json_path}")
    print("Plots   :")
    for p in (p1, p2, p3, p4, p5, p6):
        print("  -", p)
    print(f"\nε* (argmax score) ≈ {best['epsilon']:.2f}")
    print("You can tweak CONFIG at the top and re-run for Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Openness	6Entropy Trade-off (v1)
Author: Heck Yeah Lab
Runtime: ~30	645 seconds on a typical laptop

What this tests
---------------
We sweep openness 	5 in [0, 1] and simulate shock	6recovery dynamics.
For each 	5, we run N Monte Carlo trajectories and compute:
- AURC           : Area Under (Health) Recovery Curve (higher is better)
- T80            : Time to recover to 0.80 health (lower is better); NaN if never reaches
- min_dip        : Lowest health reached (higher is better)
- survival_pct   : % of runs whose health never fell below collapse_threshold
- entropy_rate   : Mean positive rate of disorder increase (lower is better)

Simple definitions (explicit + reproducible)
--------------------------------------------
- Health h_t 	5 [0, 1.2], starts at 1.0.
- Disorder d_t := 1 - clip(h_t, 0, 1).
- Entropy rate = mean of positive deltas 	6d_t across t (average increase of disorder).
- Effective shock = S * (1-\u03b5) * (0.5 + 0.5*C) * (1-R), applied during [t0, t0+D).

"Resilience Score" (composite, unitless)
----------------------------------------
Scaled combination to pick 	5*:
  score = w1 * norm(AURC) + w2 * (1 - norm(entropy_rate)) + w3 * norm(survival_pct) + w4 * (1 - norm(T80))
Weights (edit if desired): w = (0.35, 0.30, 0.20, 0.15)

Outputs
-------
- CSV  : discovery_results/openness_entropy/openness_entropy_tradeoff_<timestamp>.csv
- PNGs : metric plots + best-vs-worst overlay
- JSON : small summary with epsilon_star and headline metrics

Tweak knobs in CONFIG below. Keep other code unchanged for comparability.
"""

from __future__ import annotations

import random
import sys

import numpy as np

# -----------------
# CONFIG (edit here)
# -----------------
SEED = 42
EPOCHS = 120  # total time steps
T0 = 20  # shock starts at t = T0
DUR = 10  # shock duration
S = 0.60  # shock magnitude (0..1)
R = 0.30  # redundancy (buffers) (0..1)
C = 0.50  # coupling (0..1)
G = 0.05  # baseline growth toward health
KAPPA = 0.30  # damping back toward baseline
SIGMA = 0.01  # stochastic noise std
EPSILON_GRID = np.round(np.linspace(0.00, 1.00, 21), 2)  # 0.00, 0.05, ..., 1.00
N_RUNS = 64  # Monte Carlo runs per epsilon
COLLAPSE_THRESHOLD = 0.05  # below -> considered collapsed

# Composite score weights
W_AURC, W_ENTROPY, W_SURVIVAL, W_T80 = 0.35, 0.30, 0.20, 0.15

# -----------------
# Helpers
# -----------------
rng = np.random.default_rng(SEED)
random.seed(SEED)


def simulate_once(epsilon: float) -> dict[str, float]:
    """Simulate one trajectory; return metrics (AURC, T80, min_dip, survival, entropy_rate)."""
    h = 1.0
    h_hist = np.zeros(EPOCHS, dtype=float)
    disorder = np.zeros(EPOCHS, dtype=float)

    for t in range(EPOCHS):
        # Shock active?
        shock = S if (T0 <= t < T0 + DUR) else 0.0
        # Effective shock with openness, coupling, redundancy
        eff_shock = shock * (1 - epsilon) * (0.5 + 0.5 * C) * (1 - R)

        # Update health
        # Pull toward 1 via growth G; damping pulls back if >1
        noise = rng.normal(0.0, SIGMA)
        h = h - eff_shock + G * (1 - h) - KAPPA * max(0.0, h - 1.0) + noise
        # Clamp
        h = max(0.0, min(1.2, h))
        h_hist[t] = h
        disorder[t] = 1.0 - min(1.0, max(0.0, h))

    # Metrics
    aurc = float(np.trapz(h_hist, dx=1) / EPOCHS)  # normalized AURC (0..~1.1)
    # T80
    t80 = np.nan
    above = np.where(h_hist >= 0.80)[0]
    if len(above) > 0:
        t80 = float(above[0])
    min_dip = float(np.min(h_hist))
    survival = float(np.all(h_hist >= COLLAPSE_THRESHOLD))

    # Entropy rate: mean positive change in disorder
    dd = np.diff(disorder)
    pos_increases = dd[dd > 0]
    entropy_rate = float(np.mean(pos_increases)) if pos_increases.size > 0 else 0.0

    return {
        "AURC": aurc,
        "T80": t80,
        "min_dip": min_dip,
        "survival": survival,
        "entropy_rate": entropy_rate,
    }


def run_sweep() -> pd.DataFrame:
    rows = []
    for eps in EPSILON_GRID:
        metrics = [simulate_once(float(eps)) for _ in range(N_RUNS)]
        # Aggregate
        aurc = np.array([m["AURC"] for m in metrics])
        t80 = np.array([m["T80"] for m in metrics], dtype=float)
        min_dip = np.array([m["min_dip"] for m in metrics])
        surv = np.array([m["survival"] for m in metrics], dtype=float)
        ent = np.array([m["entropy_rate"] for m in metrics])

        row = {
            "epsilon": float(eps),
            "AURC_mean": float(np.mean(aurc)),
            "AURC_std": float(np.std(aurc)),
            "T80_mean": float(np.nanmean(t80)),
            "T80_nan_frac": float(np.isnan(t80).mean()),
            "min_dip_mean": float(np.mean(min_dip)),
            "survival_pct": float(np.mean(surv) * 100.0),
            "entropy_rate_mean": float(np.mean(ent)),
            "entropy_rate_std": float(np.std(ent)),
            "runs": int(N_RUNS),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(series)))
    return (series - lo) / (hi - lo)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize components for composite score
    n_aurc = minmax_norm(df["AURC_mean"])
    n_entropy = minmax_norm(df["entropy_rate_mean"])  # lower is better
    n_surv = minmax_norm(df["survival_pct"])
    n_t80 = minmax_norm(df["T80_mean"])  # lower is better

    score = (
        W_AURC * n_aurc
        + W_ENTROPY * (1 - n_entropy)
        + W_SURVIVAL * n_surv
        + W_T80 * (1 - n_t80)
    )
    out = df.copy()
    out["score"] = score
    # Pick epsilon* (break ties by higher AURC)
    best_idx = int(out["score"].idxmax())
    out["is_optimal"] = False
    out.loc[best_idx, "is_optimal"] = True
    return out


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("discovery_results", "openness_entropy")
    os.makedirs(out_dir, exist_ok=True)

    df = run_sweep()
    df = add_scores(df)

    csv_path = os.path.join(out_dir, f"openness_entropy_tradeoff_{ts}.csv")
    df.to_csv(csv_path, index=False)

    # Summary JSON
    best = df.loc[df["is_optimal"]].iloc[0].to_dict()
    summary = {
        "timestamp": ts,
        "epsilon_star": best["epsilon"],
        "score_star": best["score"],
        "AURC_mean_at_star": best["AURC_mean"],
        "T80_mean_at_star": best["T80_mean"],
        "survival_pct_at_star": best["survival_pct"],
        "entropy_rate_mean_at_star": best["entropy_rate_mean"],
        "config": {
            "EPOCHS": EPOCHS,
            "T0": T0,
            "DUR": DUR,
            "S": S,
            "R": R,
            "C": C,
            "G": G,
            "KAPPA": KAPPA,
            "SIGMA": SIGMA,
            "N_RUNS": N_RUNS,
            "EPSILON_GRID": EPSILON_GRID.tolist(),
            "COLLAPSE_THRESHOLD": COLLAPSE_THRESHOLD,
            "weights": {
                "AURC": W_AURC,
                "entropy": W_ENTROPY,
                "survival": W_SURVIVAL,
                "T80": W_T80,
            },
        },
        "paths": {"csv": csv_path},
    }
    json_path = os.path.join(out_dir, f"openness_entropy_summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    def save_plot(x, y, ylabel, fname):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Openness \u03b5")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Openness \u03b5")
        plt.grid(True, alpha=0.3)
        p = os.path.join(out_dir, f"{fname}_{ts}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        return p

    p1 = save_plot(df["epsilon"], df["AURC_mean"], "AURC (mean)", "aurc_vs_epsilon")
    p2 = save_plot(
        df["epsilon"],
        df["entropy_rate_mean"],
        "Entropy Rate (mean)",
        "entropy_rate_vs_epsilon",
    )
    p3 = save_plot(
        df["epsilon"], df["survival_pct"], "Survival %", "survival_vs_epsilon"
    )
    p4 = save_plot(df["epsilon"], df["T80_mean"], "T80 (mean epochs)", "t80_vs_epsilon")
    p5 = save_plot(df["epsilon"], df["score"], "Composite Score", "score_vs_epsilon")

    # Best vs worst overlay (AURC & Entropy)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epsilon"], df["AURC_mean"], marker="o", label="AURC (↑ better)")
    plt.plot(
        df["epsilon"],
        df["entropy_rate_mean"],
        marker="s",
        label="Entropy Rate (↓ better)",
    )
    plt.axvline(
        x=best["epsilon"], linestyle="--", label=f"\u03b5* = {best['epsilon']:.2f}"
    )
    plt.xlabel("Openness \u03b5")
    plt.title("Best \u03b5 marker with AURC & Entropy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p6 = os.path.join(out_dir, f"overlay_best_{ts}.png")
    plt.tight_layout()
    plt.savefig(p6, dpi=150)
    plt.close()

    print("=== Openness	6Entropy Trade-off (v1) complete ===")
    print(f"CSV     : {csv_path}")
    print(f"Summary : {json_path}")
    print("Plots   :")
    for p in (p1, p2, p3, p4, p5, p6):
        print("  -", p)
    print(f"\n\u03b5* (argmax score) 7 {best['epsilon']:.2f}")
    print("You can tweak CONFIG at the top and re-run for Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

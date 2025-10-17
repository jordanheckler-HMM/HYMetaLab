#!/usr/bin/env python3
"""
open_universe_index.py
Builds an "Effective Openness Index" (epsilon) from astrophysical constants and
runs a small illustrative experiment set mapping those epsilons into the simulation.

Outputs:
 - data/effective_openness_index.json
 - discovery_results/open_universe_index_<stamp>/ (results.csv, plots, summary)

Assumptions:
 - Uses simple min-max normalization across provided universe examples.
 - epsilon range is 0 .. epsilon_max (default 0.0030)
 - This is illustrative; constants and scaling are negotiable.
"""

import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from meaning_experiment import run_experiment_grid
except Exception:
    run_experiment_grid = None

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DATA = Path("./data")
DATA.mkdir(exist_ok=True)
OUT = Path(f"./discovery_results/open_universe_index_{STAMP}")
OUT.mkdir(parents=True, exist_ok=True)

# Embedded astrophysical constants for several example 'universes'.
# Values are illustrative and come from rough real-world estimates or hypothetical variants.
UNIVERSES = {
    "Real_Universe": {
        "H0_km_s_Mpc": 67.4,  # Planck estimate ~67.4 km/s/Mpc
        "Omega_Lambda": 0.685,  # Dark energy fraction
        "entropy_prod_rel": 1.0,  # normalized baseline (arbitrary units)
    },
    "High_Lambda": {"H0_km_s_Mpc": 70.0, "Omega_Lambda": 0.85, "entropy_prod_rel": 1.2},
    "Low_Lambda": {"H0_km_s_Mpc": 65.0, "Omega_Lambda": 0.50, "entropy_prod_rel": 0.8},
    "High_Entropy_Production": {
        "H0_km_s_Mpc": 67.4,
        "Omega_Lambda": 0.685,
        "entropy_prod_rel": 3.0,
    },
    "Cold_Slow_Expanding": {
        "H0_km_s_Mpc": 50.0,
        "Omega_Lambda": 0.60,
        "entropy_prod_rel": 0.3,
    },
}

# Mapping weights (tunable). They indicate how each constant contributes to openness.
WEIGHTS = {"H0_km_s_Mpc": 0.3, "Omega_Lambda": 0.4, "entropy_prod_rel": 0.3}

EPSILON_MAX = 0.0030


def minmax_normalize(values):
    arr = np.array(values, dtype=float)
    lo = arr.min()
    hi = arr.max()
    if hi - lo <= 0:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)


def compute_effective_epsilon(universes, weights, epsilon_max=EPSILON_MAX):
    # collect arrays
    H = [u["H0_km_s_Mpc"] for u in universes.values()]
    O = [u["Omega_Lambda"] for u in universes.values()]
    S = [u["entropy_prod_rel"] for u in universes.values()]

    Hn = minmax_normalize(H)
    On = minmax_normalize(O)
    Sn = minmax_normalize(S)

    keys = list(universes.keys())
    results = {}
    for i, k in enumerate(keys):
        score = (
            weights["H0_km_s_Mpc"] * Hn[i]
            + weights["Omega_Lambda"] * On[i]
            + weights["entropy_prod_rel"] * Sn[i]
        )
        # clamp and scale
        eps = float(max(0.0, min(1.0, score))) * epsilon_max
        results[k] = {
            "raw_score": float(score),
            "epsilon": eps,
            "constants": universes[k],
        }
    return results


def save_index(results, path=DATA / "effective_openness_index.json"):
    with open(path, "w") as f:
        json.dump(
            {"generated": STAMP, "epsilon_max": EPSILON_MAX, "results": results},
            f,
            indent=2,
        )
    return path


def run_mapped_simulations(results):
    if run_experiment_grid is None:
        print("meaning_experiment.run_experiment_grid not available; skipping sims.")
        return None
    sim_rows = []
    for name, meta in results.items():
        eps = meta["epsilon"]
        label = f"OPENIDX_{name}"
        openness = {"mechanism": "agent_io", "epsilon": eps, "period": 40}
        print(f"Running sim for {name}: epsilon={eps:.6f} (label={label})")
        # small smoke parameters
        df = run_experiment_grid(
            label=label,
            agents_list=[200],
            shocks=[0.3],
            stress_duration=["chronic"],
            goal_diversity=[3],
            noise_list=[0.05],
            replicates=3,
        )
        # annotate df with mapping
        if df is not None:
            df["openidx_name"] = name


if __name__ == "__main__":
    results = compute_effective_epsilon(UNIVERSES, WEIGHTS)
    path = save_index(results)
    print("Saved Effective Openness Index to", path)
    # Friendly table
    for k, v in results.items():
        print(f"{k}: score={v['raw_score']:.3f} -> epsilon={v['epsilon']:.6f}")

    # run sims (smoke)
    run_mapped_simulations(results)

    # write a quick CSV summary
    rows = []
    for k, v in results.items():
        rows.append(
            {"universe": k, "epsilon": v["epsilon"], "raw_score": v["raw_score"]}
        )
    pd.DataFrame(rows).to_csv(OUT / "open_universe_index_summary.csv", index=False)
    # plot
    names = [r["universe"] for r in rows]
    eps = [r["epsilon"] for r in rows]
    plt.figure(figsize=(6, 3))
    plt.bar(names, eps)
    plt.title("Effective Openness Index (epsilon)")
    plt.ylabel("epsilon")
    plt.tight_layout()
    plt.savefig(OUT / "open_universe_index.png")
    print("Wrote summary CSV and plot to", OUT)

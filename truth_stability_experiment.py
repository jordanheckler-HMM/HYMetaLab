#!/usr/bin/env python3
"""
Truth Stability Experiment
Runs a lightweight deterministic stability test and writes a small results bundle.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Helpers ----------


def wilson_ci(k, n, z=1.96):
    """Return (p, low, high) Wilson score interval for k successes out of n trials."""
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return p, max(0.0, low), min(1.0, high)


def simulate_cycle(n, pA, pB, pC, rng):
    """Vectorized simulate n trials with probabilities pA/pB/pC using rng."""
    A = rng.random(n) < pA
    B = rng.random(n) < pB
    C = rng.random(n) < pC
    # epsilon noise
    eps = rng.normal(loc=0.0, scale=0.08, size=n)
    S = 0.55 * A.astype(float) + 0.30 * B.astype(float) - 0.35 * C.astype(float) + eps
    S = np.clip(S, 0.0, 1.0)
    return pd.DataFrame(
        {"A": A.astype(int), "B": B.astype(int), "C": C.astype(int), "S": S}
    )


def evaluate_claims(cycle_stats, tau):
    """Given per-cycle stats (list of DataFrames), evaluate each claim per cycle.
    Returns dict of claim_name -> list of booleans (pass per cycle).
    """
    n_cycles = len(cycle_stats)
    results = {i: [] for i in range(1, 8)}

    for df in cycle_stats:
        # precompute means and var
        mean_B1 = df.loc[df["B"] == 1, "S"].mean() if (df["B"] == 1).any() else np.nan
        mean_B0 = df.loc[df["B"] == 0, "S"].mean() if (df["B"] == 0).any() else np.nan
        mean_C1 = df.loc[df["C"] == 1, "S"].mean() if (df["C"] == 1).any() else np.nan
        mean_A0 = df.loc[df["A"] == 0, "S"].mean() if (df["A"] == 0).any() else np.nan
        mean_S = df["S"].mean()
        var_S = df["S"].var()
        # correlation A,S
        if df["A"].nunique() > 1:
            corr_AS = np.corrcoef(df["A"].values, df["S"].values)[0, 1]
        else:
            corr_AS = 0.0

        # Claim 1: Feedback dominates noise
        c1 = (
            (mean_B1 - mean_C1) > 0.10
            if not np.isnan(mean_B1) and not np.isnan(mean_C1)
            else False
        )

        # Claim 2: Noise harms more than scarcity
        c2 = (
            (mean_C1 <= (mean_A0 - 0.05))
            if not np.isnan(mean_C1) and not np.isnan(mean_A0)
            else False
        )

        # Claim 3: Abundance alone determines outcomes
        c3 = corr_AS >= 0.40

        # Claim 4: Open feedback always wins (mean B1 > mean B0 every cycle)
        c4 = (
            (mean_B1 > mean_B0)
            if not np.isnan(mean_B1) and not np.isnan(mean_B0)
            else False
        )

        # Claim 5: Noise doesn't matter
        c5 = (
            (
                abs(
                    mean_C1
                    - (
                        df.loc[df["C"] == 0, "S"].mean()
                        if (df["C"] == 0).any()
                        else np.nan
                    )
                )
                < 0.02
            )
            if (df["C"].nunique() > 1)
            else False
        )

        # Claim 6: Everything balances out
        c6 = var_S < 0.01

        # Claim 7: Null/random baseline
        c7 = mean_S > tau

        results[1].append(bool(c1))
        results[2].append(bool(c2))
        results[3].append(bool(c3))
        results[4].append(bool(c4))
        results[5].append(bool(c5))
        results[6].append(bool(c6))
        results[7].append(bool(c7))

    return results


def render_figures(summary_df, pass_matrix, outdir):
    # Bar chart of T with CI
    claims = summary_df["claim"].values
    T = summary_df["T"].values
    lows = summary_df["ci_low"].values
    highs = summary_df["ci_high"].values
    errs = np.vstack([T - lows, highs - T])

    plt.figure(figsize=(10, 5))
    colors = ["C0"] * len(claims)
    # highlight null claim (assumed last)
    colors[-1] = "C3"
    x = np.arange(len(claims))
    plt.bar(x, T, yerr=errs, color=colors, capsize=4)
    plt.xticks(x, [f"C{i}" for i in range(1, len(claims) + 1)])
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Truth Stability T")
    plt.title("Truth Stability Index per Claim (with 95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_survival.png"), dpi=150)
    plt.close()

    # Cycle curves: per-cycle pass rates for each claim
    n_cycles = pass_matrix.shape[1]
    plt.figure(figsize=(10, 5))
    for i in range(pass_matrix.shape[0]):
        plt.plot(
            range(1, n_cycles + 1),
            pass_matrix[i].astype(float),
            marker="o",
            label=f"C{i+1}",
        )
    plt.xlabel("Cycle")
    plt.ylabel("Pass (1) / Fail (0)")
    plt.title("Per-cycle Pass (1) / Fail (0) for Each Claim")
    plt.ylim(-0.1, 1.1)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cycle_curves.png"), dpi=150)
    plt.close()


def write_bundle(summary_df, pass_matrix, params, outdir):
    summary_df.to_csv(os.path.join(outdir, "claims_summary.csv"), index=False)
    render_figures(summary_df, pass_matrix, outdir)
    # README
    readme = []
    readme.append("# Truth Stability Experiment")
    readme.append(f"Timestamp: {params['timestamp']}")
    readme.append(f"Seed: {params['seed']}")
    readme.append(f"Trials: {params['trials']}  |  Cycles: {params['cycles']}")
    readme.append("")
    readme.append(summary_df.to_markdown(index=False))
    readme.append("")
    readme.append(f"**Experiment Result:** {params['experiment_result']}")
    readme.append("")
    readme.append(
        "Notes: toy world; claims are heuristics; run time <60s; deterministic seed."
    )
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RANDOM_SEED = int(args.seed)
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    per_cycle = max(1, TRIALS // CYCLES)

    rng = np.random.default_rng(RANDOM_SEED)

    # design per-cycle varying probabilities (deterministic schedule)
    cycle_ps = []
    base = np.linspace(0.2, 0.8, CYCLES)
    for i in range(CYCLES):
        # shift patterns: pA, pB, pC vary anti/sync to create contradictions
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    # simulate cycles
    cycle_dfs = []
    start_time = datetime.now()
    for pA, pB, pC in cycle_ps:
        df = simulate_cycle(per_cycle, pA, pB, pC, rng)
        cycle_dfs.append(df)

    # Null baseline tau drawn from Uniform(0.3,0.7) on first cycle deterministically from rng
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    # evaluate claims per cycle
    per_cycle_results = evaluate_claims(cycle_dfs, tau)

    # aggregate
    rows = []
    pass_matrix = np.zeros((7, CYCLES), dtype=int)
    for i in range(1, 8):
        passes = int(np.sum(per_cycle_results[i]))
        fails = CYCLES - passes
        p, low, high = wilson_ci(passes, CYCLES)
        T = 1.0 - (fails / CYCLES)
        # diagnostic: likely true if lower CI > 0.5; false if upper CI < 0.5; else mixed
        if low > 0.5:
            diag = "likely true"
        elif high < 0.5:
            diag = "false"
        else:
            diag = "mixed"
        rows.append(
            {
                "claim": f"C{i}",
                "passes": passes,
                "fails": fails,
                "T": round(T, 3),
                "ci_low": round(low, 3),
                "ci_high": round(high, 3),
                "diagnostic": diag,
            }
        )
        pass_matrix[i - 1, :] = np.array(per_cycle_results[i], dtype=int)

    summary_df = pd.DataFrame(rows)

    # Experiment PASS/FAIL gate vs null (claim 7 is null)
    null_row = summary_df.loc[summary_df["claim"] == "C7"].iloc[0]
    null_T = null_row["T"]
    null_low = null_row["ci_low"]
    null_high = null_row["ci_high"]

    experiment_pass = False
    for _, r in summary_df[summary_df["claim"] != "C7"].iterrows():
        if (r["T"] - null_T) >= 0.20 and (r["ci_low"] > null_high):
            experiment_pass = True
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_test_run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    params = {
        "timestamp": timestamp,
        "seed": RANDOM_SEED,
        "trials": TRIALS,
        "cycles": CYCLES,
        "experiment_result": "PASS" if experiment_pass else "FAIL",
    }

    write_bundle(summary_df, pass_matrix, params, outdir)

    # save raw params + cycle schedule
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(
            {
                "seed": RANDOM_SEED,
                "trials": TRIALS,
                "cycles": CYCLES,
                "per_cycle": per_cycle,
                "cycle_ps": cycle_ps,
                "tau": tau,
            },
            f,
            indent=2,
        )

    # Console summary
    print("\n=== Truth Stability Experiment Summary ===")
    print(f"Results folder: {outdir}")
    top3 = summary_df.sort_values("T", ascending=False).head(3)
    print("Top 3 claims by T:")
    for _, r in top3.iterrows():
        print(
            f"  {r['claim']}: T={r['T']} CI=[{r['ci_low']},{r['ci_high']}] diag={r['diagnostic']}"
        )
    print(f"Experiment PASS?: {'YES' if experiment_pass else 'NO'}")
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Elapsed (s): {elapsed:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()

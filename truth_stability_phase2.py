#!/usr/bin/env python3
"""
Truth Stability — Phase 2 (Dual-World Regimes)
Deterministic, fast (<60s), vectorized.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return p, max(0.0, low), min(1.0, high)


def simulate_cycle_W1(n, pA, pB, pC, rng):
    A = rng.random(n) < pA
    B = rng.random(n) < pB
    C = rng.random(n) < pC
    eps = rng.normal(0.0, 0.08, size=n)
    S = 0.55 * A.astype(float) + 0.30 * B.astype(float) - 0.35 * C.astype(float) + eps
    S = np.clip(S, 0.0, 1.0)
    return A.astype(int), B.astype(int), C.astype(int), S


def simulate_cycle_W2(n, pA, pB, pC, rng):
    A = rng.random(n) < pA
    B = rng.random(n) < pB
    C = rng.random(n) < pC
    eps = rng.normal(0.0, 0.08, size=n)
    # W2: different linear weights plus interactions
    S = (
        0.15 * A.astype(float)
        + 0.50 * B.astype(float)
        - 0.10 * C.astype(float)
        + 0.20 * (A.astype(float) * B.astype(float))
        - 0.25 * (B.astype(float) * C.astype(float))
        + eps
    )
    S = np.clip(S, 0.0, 1.0)
    return A.astype(int), B.astype(int), C.astype(int), S


def evaluate_claims_per_cycle(A, B, C, S, tau, min_cond=20):
    """Return list of length 7 with: 1=pass,0=fail,np.nan=NA"""
    out = [np.nan] * 7
    n = len(S)
    mean_B1 = np.nan
    mean_B0 = np.nan
    mean_C1 = np.nan
    mean_C0 = np.nan
    mean_A0 = np.nan

    # compute group masks and counts
    mask_B1 = B == 1
    mask_B0 = B == 0
    mask_C1 = C == 1
    mask_C0 = C == 0
    mask_A0 = A == 0

    cnt_B1 = int(mask_B1.sum())
    cnt_B0 = int(mask_B0.sum())
    cnt_C1 = int(mask_C1.sum())
    cnt_C0 = int(mask_C0.sum())
    cnt_A0 = int(mask_A0.sum())

    if cnt_B1 >= min_cond:
        mean_B1 = S[mask_B1].mean()
    if cnt_B0 >= min_cond:
        mean_B0 = S[mask_B0].mean()
    if cnt_C1 >= min_cond:
        mean_C1 = S[mask_C1].mean()
    if cnt_C0 >= min_cond:
        mean_C0 = S[mask_C0].mean()
    if cnt_A0 >= min_cond:
        mean_A0 = S[mask_A0].mean()

    mean_S = float(np.mean(S))
    var_S = float(np.var(S, ddof=1))

    # C1: Feedback dominates noise -> mean(S|B=1) - mean(S|C=1) > 0.10
    if (mean_B1 is not np.nan) and (mean_C1 is not np.nan):
        out[0] = 1 if (mean_B1 - mean_C1) > 0.10 else 0

    # C2: Noise harms more than scarcity -> mean(S|C=1) <= mean(S|A=0) - 0.05
    if (mean_C1 is not np.nan) and (mean_A0 is not np.nan):
        out[1] = 1 if (mean_C1 <= (mean_A0 - 0.05)) else 0

    # C3: Abundance alone determines outcomes -> corr(A,S) >= 0.40
    if n > 1 and A.sum() >= min_cond and (n - A.sum()) >= min_cond:
        try:
            corr = np.corrcoef(A.astype(float), S)[0, 1]
            out[2] = 1 if (corr >= 0.40) else 0
        except Exception:
            out[2] = 0

    # C4: Open feedback always wins -> mean(S|B=1) > mean(S|B=0) in that cycle
    if (mean_B1 is not np.nan) and (mean_B0 is not np.nan):
        out[3] = 1 if (mean_B1 > mean_B0) else 0

    # C5: Noise doesn't matter -> |mean(S|C=1) - mean(S|C=0)| < 0.02
    if (mean_C1 is not np.nan) and (mean_C0 is not np.nan):
        out[4] = 1 if (abs(mean_C1 - mean_C0) < 0.02) else 0

    # C6: Everything balances out -> var(S) < 0.01
    out[5] = 1 if (var_S < 0.01) else 0

    # C7: Null/random baseline -> mean(S) > tau
    out[6] = 1 if (mean_S > tau) else 0

    return out


def aggregate_and_score(per_cycle_status, cycles_split):
    """per_cycle_status: list of status lists per cycle (values 0/1/np.nan)
    cycles_split: index where W1 ends (exclusive). Returns DataFrame summary."""
    per_cycle_array = np.array(per_cycle_status, dtype=float)  # shape (C,7)
    C = per_cycle_array.shape[0]
    W1_idx = np.arange(0, cycles_split)
    W2_idx = np.arange(cycles_split, C)

    rows = []
    for claim_idx in range(7):
        col = per_cycle_array[:, claim_idx]
        # W1
        w1_col = col[W1_idx]
        w2_col = col[W2_idx]
        # non-NA cycles
        w1_mask = ~np.isnan(w1_col)
        w2_mask = ~np.isnan(w2_col)
        N_w1 = int(w1_mask.sum())
        N_w2 = int(w2_mask.sum())
        passes_w1 = int(np.nansum(w1_col[w1_mask] == 1)) if N_w1 > 0 else 0
        passes_w2 = int(np.nansum(w2_col[w2_mask] == 1)) if N_w2 > 0 else 0
        fails_w1 = int(np.nansum(w1_col[w1_mask] == 0)) if N_w1 > 0 else 0
        fails_w2 = int(np.nansum(w2_col[w2_mask] == 0)) if N_w2 > 0 else 0

        # T = 1 - F/N = passes/N
        T_w1 = None
        T_w2 = None
        T_overall = None
        ci_w1 = (np.nan, np.nan)
        ci_w2 = (np.nan, np.nan)
        ci_over = (np.nan, np.nan)

        if N_w1 > 0:
            p1, l1, h1 = wilson_ci(passes_w1, N_w1)
            T_w1 = p1
            ci_w1 = (l1, h1)
        else:
            T_w1 = np.nan

        if N_w2 > 0:
            p2, l2, h2 = wilson_ci(passes_w2, N_w2)
            T_w2 = p2
            ci_w2 = (l2, h2)
        else:
            T_w2 = np.nan

        # overall
        overall_mask = ~np.isnan(col)
        N_over = int(overall_mask.sum())
        passes_over = int(np.nansum(col[overall_mask] == 1)) if N_over > 0 else 0
        if N_over > 0:
            pO, lO, hO = wilson_ci(passes_over, N_over)
            T_overall = pO
            ci_over = (lO, hO)
        else:
            T_overall = np.nan

        # delta
        delta = None
        if (not np.isnan(T_w1)) and (not np.isnan(T_w2)):
            delta = float(T_w2 - T_w1)
        else:
            delta = np.nan

        # diagnosis rules
        diagnosis = "mixed/false"
        # universal: both CIs lower bounds >= 0.7
        try:
            if (
                (not np.isnan(T_w1))
                and (not np.isnan(T_w2))
                and (ci_w1[0] >= 0.7)
                and (ci_w2[0] >= 0.7)
            ):
                diagnosis = "universal"
            elif (not np.isnan(T_w1)) and (ci_w1[0] >= 0.7):
                diagnosis = "regime-W1"
            elif (not np.isnan(T_w2)) and (ci_w2[0] >= 0.7):
                diagnosis = "regime-W2"
        except Exception:
            diagnosis = "mixed/false"

        rows.append(
            {
                "claim": f"C{claim_idx+1}",
                "passes_w1": passes_w1,
                "fails_w1": fails_w1,
                "N_w1": N_w1,
                "T_w1": round(float(T_w1), 3) if (not np.isnan(T_w1)) else np.nan,
                "ci_w1_low": round(ci_w1[0], 3) if (not np.isnan(ci_w1[0])) else np.nan,
                "ci_w1_high": (
                    round(ci_w1[1], 3) if (not np.isnan(ci_w1[1])) else np.nan
                ),
                "passes_w2": passes_w2,
                "fails_w2": fails_w2,
                "N_w2": N_w2,
                "T_w2": round(float(T_w2), 3) if (not np.isnan(T_w2)) else np.nan,
                "ci_w2_low": round(ci_w2[0], 3) if (not np.isnan(ci_w2[0])) else np.nan,
                "ci_w2_high": (
                    round(ci_w2[1], 3) if (not np.isnan(ci_w2[1])) else np.nan
                ),
                "T_overall": (
                    round(float(T_overall), 3) if (not np.isnan(T_overall)) else np.nan
                ),
                "ci_over_low": (
                    round(ci_over[0], 3) if (not np.isnan(ci_over[0])) else np.nan
                ),
                "ci_over_high": (
                    round(ci_over[1], 3) if (not np.isnan(ci_over[1])) else np.nan
                ),
                "delta": round(float(delta), 3) if (not np.isnan(delta)) else np.nan,
                "diagnosis": diagnosis,
            }
        )

    df_summary = pd.DataFrame(rows)
    return df_summary


def render_figures_phase2(df_summary, per_cycle_status, cycles_split, outdir):
    claims = df_summary["claim"].tolist()
    n = len(claims)

    # grouped bar chart: W1, W2, Overall
    ind = np.arange(n)
    width = 0.25

    T_w1 = df_summary["T_w1"].values.astype(float)
    T_w2 = df_summary["T_w2"].values.astype(float)
    T_over = df_summary["T_overall"].values.astype(float)

    # CI errors
    err_w1 = np.vstack(
        [
            (T_w1 - df_summary["ci_w1_low"].values),
            (df_summary["ci_w1_high"].values - T_w1),
        ]
    )
    err_w2 = np.vstack(
        [
            (T_w2 - df_summary["ci_w2_low"].values),
            (df_summary["ci_w2_high"].values - T_w2),
        ]
    )
    err_o = np.vstack(
        [
            (T_over - df_summary["ci_over_low"].values),
            (df_summary["ci_over_high"].values - T_over),
        ]
    )

    plt.figure(figsize=(12, 5))
    plt.bar(ind - width, T_w1, width, yerr=err_w1, capsize=4)
    plt.bar(ind, T_w2, width, yerr=err_w2, capsize=4)
    plt.bar(ind + width, T_over, width, yerr=err_o, capsize=4)
    plt.xticks(ind, claims)
    plt.ylabel("T (Truth Stability)")
    plt.ylim(-0.05, 1.05)
    plt.title("Truth Stability by Regime (W1, W2, Overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_by_regime.png"), dpi=150)
    plt.close()

    # cycle curves: per-cycle pass/fail lines
    per_cycle_array = np.array(per_cycle_status, dtype=float)  # shape (C,7)
    C = per_cycle_array.shape[0]
    plt.figure(figsize=(12, 5))
    for j in range(7):
        series = per_cycle_array[:, j]
        # plot with nan gaps
        plt.plot(range(1, C + 1), series, marker="o", label=f"C{j+1}")
    # vertical dashed line between regimes
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Pass (1) / Fail (0) / NA (gap)")
    plt.ylim(-0.1, 1.1)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.title("Per-cycle Claim Outcomes (Phase 2)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cycle_curves_phase2.png"), dpi=150)
    plt.close()


def write_bundle_phase2(df_summary, per_cycle_status, params, outdir):
    df_summary.to_csv(os.path.join(outdir, "claims_summary.csv"), index=False)
    render_figures_phase2(df_summary, per_cycle_status, params["cycles_split"], outdir)
    # README
    readme = []
    readme.append("# Truth Stability — Phase 2 (Dual-World)")
    readme.append(f"Timestamp: {params['timestamp']}")
    readme.append(f"Seed: {params['seed']}")
    readme.append(
        f"Trials: {params['trials']} | Cycles: {params['cycles']} | W1 cycles: 1-{params['cycles_split']} | W2 cycles: {params['cycles_split']+1}-{params['cycles']}"
    )
    readme.append("")
    readme.append(df_summary.to_markdown(index=False))
    readme.append("")
    readme.append(f"**Experiment Result:** {params['experiment_result']}")
    readme.append("")
    readme.append(
        "- Notes: toy worlds; Phase-2 introduces a regime shift to detect universals vs regime-bound claims."
    )
    readme.append("- Deterministic seed ensures reproducibility.")
    readme.append("- NA indicates insufficient conditioning samples (<20) in a cycle.")

    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))


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

    # per-cycle probabilities schedule
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    # draw tau deterministically
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    # simulate cycles; first half W1, second half W2
    cycles_split = CYCLES // 2
    per_cycle_status = []
    start_time = datetime.now()
    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)
        status = evaluate_claims_per_cycle(A, B, C, S, tau, min_cond=20)
        per_cycle_status.append(status)

    # aggregate
    df_summary = aggregate_and_score(per_cycle_status, cycles_split)

    # Experiment PASS gate checks
    # Null is C7
    null_row = df_summary.loc[df_summary["claim"] == "C7"].iloc[0]
    null_w1_T = null_row["T_w1"]
    null_w2_T = null_row["T_w2"]
    null_w1_low = null_row["ci_w1_low"]
    null_w1_high = null_row["ci_w1_high"]
    null_w2_low = null_row["ci_w2_low"]
    null_w2_high = null_row["ci_w2_high"]

    experiment_pass = False

    # condition 1: at least one non-null claim is universal and exceeds null's T in both regimes with non-overlapping CIs
    cond1 = False
    for _, r in df_summary[df_summary["claim"] != "C7"].iterrows():
        if r["diagnosis"] == "universal":
            # check exceeds null's T in both regimes with non-overlapping CIs
            try:
                # for regime W1: r.ci_w1_low > null_w1_high
                if (r["ci_w1_low"] > null_w1_high) and (r["ci_w2_low"] > null_w2_high):
                    cond1 = True
                    break
            except Exception:
                continue

    # condition 2: at least one claim shows regime specificity: |Δ| >= 0.30 with non-overlapping CIs across regimes
    cond2 = False
    for _, r in df_summary[df_summary["claim"] != "C7"].iterrows():
        if not np.isnan(r["delta"]):
            if abs(r["delta"]) >= 0.30:
                # non-overlap: either ci_w1_high < ci_w2_low or vice versa
                if (r["ci_w1_high"] < r["ci_w2_low"]) or (
                    r["ci_w2_high"] < r["ci_w1_low"]
                ):
                    cond2 = True
                    break

    experiment_pass = cond1 and cond2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase2_run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    params = {
        "timestamp": timestamp,
        "seed": RANDOM_SEED,
        "trials": TRIALS,
        "cycles": CYCLES,
        "per_cycle": per_cycle,
        "cycles_split": cycles_split,
        "cycle_ps": cycle_ps,
        "tau": tau,
        "experiment_result": "PASS" if experiment_pass else "FAIL",
    }

    write_bundle_phase2(df_summary, per_cycle_status, params, outdir)

    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Console summary
    print("\n=== Truth Stability Phase 2 Summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    # Top universal candidates
    universals = df_summary[df_summary["diagnosis"] == "universal"]
    if not universals.empty:
        top_univ = universals.sort_values("T_overall", ascending=False).head(3)
        print("\nTop universal candidates:")
        for _, r in top_univ.iterrows():
            print(
                f"  {r['claim']}: T_w1={r['T_w1']} CI=[{r['ci_w1_low']},{r['ci_w1_high']}] | T_w2={r['T_w2']} CI=[{r['ci_w2_low']},{r['ci_w2_high']}]"
            )
    else:
        print("\nTop universal candidates: None")

    # regime-specific
    df_regime_spec = df_summary[
        (~df_summary["delta"].isna()) & (abs(df_summary["delta"]) >= 0.30)
    ]
    if not df_regime_spec.empty:
        print("\nRegime-specific candidates (|Δ|>=0.30):")
        for _, r in df_regime_spec.iterrows():
            print(
                f"  {r['claim']}: Δ={r['delta']} (T_w1={r['T_w1']}, T_w2={r['T_w2']})"
            )
    else:
        print("\nRegime-specific candidates: None")

    print(f"\nExperiment PASS?: {'YES' if experiment_pass else 'NO'}")
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Elapsed (s): {elapsed:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()

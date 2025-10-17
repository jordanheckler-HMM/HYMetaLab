#!/usr/bin/env python3
"""
Truth Stability — Phase 3 (Adaptive Agents)
Deterministic, vectorized, fast.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Helpers ----------


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss_grad(w, X, y):
    # returns gradient of sum logloss wrt w
    p = sigmoid(X.dot(w))
    grad = X.T.dot(p - y) / X.shape[0]
    return grad, p


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


def bin_reliability(y_true, p_pred, bins=10):
    # returns bin_centers, mean_pred, mean_obs, counts
    inds = np.minimum((p_pred * bins).astype(int), bins - 1)
    mean_pred = np.zeros(bins)
    mean_obs = np.zeros(bins)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        mask = inds == b
        cnt = int(mask.sum())
        counts[b] = cnt
        if cnt > 0:
            mean_pred[b] = p_pred[mask].mean()
            mean_obs[b] = y_true[mask].mean()
        else:
            mean_pred[b] = np.nan
            mean_obs[b] = np.nan
    # bin centers
    centers = (np.arange(bins) + 0.5) / bins
    return centers, mean_pred, mean_obs, counts


# ---------- Simulation functions (W1/W2) ----------


def simulate_cycle_W1(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0.0, 0.08, size=n)
    S = 0.55 * A + 0.30 * B - 0.35 * C + eps
    S = np.clip(S, 0.0, 1.0)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


def simulate_cycle_W2(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0.0, 0.08, size=n)
    S = 0.15 * A + 0.50 * B - 0.10 * C + 0.20 * (A * B) - 0.25 * (B * C) + eps
    S = np.clip(S, 0.0, 1.0)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


# ---------- Agent utilities ----------


def design_matrix(A, B, C, wide=False):
    # returns X matrix shape (n, d)
    ones = np.ones_like(A, dtype=float)
    if not wide:
        X = np.vstack([ones, A.astype(float), B.astype(float), C.astype(float)]).T
    else:
        AB = (A * B).astype(float)
        BC = (B * C).astype(float)
        X = np.vstack(
            [ones, A.astype(float), B.astype(float), C.astype(float), AB, BC]
        ).T
    return X


def online_update(w, X, y, lr=0.3, momentum=None):
    # simple gradient step on logistic loss using batch gradient; momentum is previous delta
    grad, p = logloss_grad(w, X, y)
    delta = -lr * grad
    if momentum is not None:
        delta = (
            0.0 * delta + (1 - momentum) * delta + momentum * momentum
        )  # keep shape but simple
        # Note: using simple non-standard momentum placeholder to keep implementation tiny
    w_new = w + delta
    return w_new, p


# ---------- Static claims evaluator (copied/adapted from Phase2) ----------


def evaluate_static_claims_per_cycle(A, B, C, S, tau, min_cond=20):
    n = len(S)
    out = [np.nan] * 7
    # masks
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

    mean_B1 = S[mask_B1].mean() if cnt_B1 >= min_cond else np.nan
    mean_B0 = S[mask_B0].mean() if cnt_B0 >= min_cond else np.nan
    mean_C1 = S[mask_C1].mean() if cnt_C1 >= min_cond else np.nan
    mean_C0 = S[mask_C0].mean() if cnt_C0 >= min_cond else np.nan
    mean_A0 = S[mask_A0].mean() if cnt_A0 >= min_cond else np.nan
    mean_S = float(np.mean(S))

    # C1
    if (not np.isnan(mean_B1)) and (not np.isnan(mean_C1)):
        out[0] = 1 if (mean_B1 - mean_C1) > 0.10 else 0
    # C2
    if (not np.isnan(mean_C1)) and (not np.isnan(mean_A0)):
        out[1] = 1 if (mean_C1 <= (mean_A0 - 0.05)) else 0
    # C3
    if n > 1 and A.sum() >= min_cond and (n - A.sum()) >= min_cond:
        try:
            corr = np.corrcoef(A.astype(float), S)[0, 1]
            out[2] = 1 if (corr >= 0.40) else 0
        except Exception:
            out[2] = 0
    # C4
    if (not np.isnan(mean_B1)) and (not np.isnan(mean_B0)):
        out[3] = 1 if (mean_B1 > mean_B0) else 0
    # C5
    if (not np.isnan(mean_C1)) and (not np.isnan(mean_C0)):
        out[4] = 1 if (abs(mean_C1 - mean_C0) < 0.02) else 0
    # C6
    var_S = float(np.var(S, ddof=1))
    out[5] = 1 if (var_S < 0.01) else 0
    # C7
    out[6] = 1 if (mean_S > tau) else 0

    return out


# ---------- Main Experiment ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta", type=float, default=0.65)
    args = parser.parse_args()

    RANDOM_SEED = int(args.seed)
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    THETA = float(args.theta)
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

    # tau
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    cycles_split = CYCLES // 2

    # Agents init
    agents = {
        "Simple": {
            "wide": False,
            "lr": 0.4,
            "w": None,
            "momentum": None,
            "prev_delta": None,
        },
        "Wide": {
            "wide": True,
            "lr": 0.3,
            "w": None,
            "momentum": None,
            "prev_delta": None,
        },
        "Sticky": {
            "wide": True,
            "lr": 0.2,
            "w": None,
            "momentum": 0.8,
            "prev_delta": None,
        },
    }
    # init weights small zeros
    for name, a in agents.items():
        d = 4 if not a["wide"] else 6
        a["w"] = np.zeros(d, dtype=float)
        a["prev_delta"] = np.zeros(d, dtype=float)

    per_cycle_agent_stats = {name: [] for name in agents}
    per_cycle_static = []

    start_time = datetime.now()

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        # evaluate static claims
        static_status = evaluate_static_claims_per_cycle(A, B, C, S, tau, min_cond=20)
        per_cycle_static.append(static_status)

        # For each agent: predict, evaluate, update
        for name, a in agents.items():
            X = design_matrix(A, B, C, wide=a["wide"])
            w = a["w"]
            # predict probs
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            brier = float(((p - Y) ** 2).mean())
            # avg predicted prob for Y=1 and Y=0
            mean_p1 = float(p[Y == 1].mean()) if (Y == 1).sum() > 0 else np.nan
            mean_p0 = float(p[Y == 0].mean()) if (Y == 0).sum() > 0 else np.nan
            # claim pass/fail
            claim_pass = 1 if (acc >= THETA) else 0
            # update weights via one gradient step
            grad, _ = logloss_grad(w, X, Y)
            delta = -a["lr"] * grad
            if a.get("momentum") is not None:
                # simple momentum: combine prev_delta
                delta = (1 - a["momentum"]) * delta + a["momentum"] * a["prev_delta"]
            w = w + delta
            a["w"] = w
            a["prev_delta"] = delta

            per_cycle_agent_stats[name].append(
                {
                    "acc": acc,
                    "brier": brier,
                    "p_mean1": mean_p1,
                    "p_mean0": mean_p0,
                    "claim_pass": claim_pass,
                }
            )

    # Aggregate agent metrics
    agent_rows = []
    for name, stats in per_cycle_agent_stats.items():
        accs = np.array([s["acc"] for s in stats], dtype=float)
        briers = np.array([s["brier"] for s in stats], dtype=float)
        passes = np.array([s["claim_pass"] for s in stats], dtype=float)
        # regimes
        w1_mask = np.arange(CYCLES) < cycles_split
        w2_mask = ~w1_mask
        # counts per regime (exclude NA concept not needed here)
        N_w1 = int(w1_mask.sum())
        N_w2 = int(w2_mask.sum())
        passes_w1 = int(passes[w1_mask].sum())
        passes_w2 = int(passes[w2_mask].sum())
        fails_w1 = int(N_w1 - passes_w1)
        fails_w2 = int(N_w2 - passes_w2)
        # T metrics
        p1, l1, h1 = wilson_ci(passes_w1, N_w1)
        p2, l2, h2 = wilson_ci(passes_w2, N_w2)
        # overall
        pO, lO, hO = wilson_ci(int(passes.sum()), CYCLES)
        # accuracy means
        acc_w1_mean = float(accs[w1_mask].mean())
        acc_w2_mean = float(accs[w2_mask].mean())
        brier_w1 = float(briers[w1_mask].mean())
        brier_w2 = float(briers[w2_mask].mean())
        # Update latency L: cycles after split until acc >= (acc_w1_mean - 0.05)
        threshold = acc_w1_mean - 0.05
        L = CYCLES
        for offset, idx in enumerate(np.where(w2_mask)[0]):
            if accs[idx] >= threshold:
                L = int(offset)
                break
        # Forgetting F: acc_w1_mean - mean of first two W2 cycles
        first_two_w2 = (
            accs[np.where(w2_mask)[0][:2]]
            if np.where(w2_mask)[0].size >= 1
            else np.array([np.nan])
        )
        acc_first_two_w2_mean = float(np.nanmean(first_two_w2))
        F = float(max(0.0, acc_w1_mean - acc_first_two_w2_mean))
        F = min(F, 1.0)
        # Calibration per regime
        # collect per-regime predictions and truths by replaying cycles to get probs
        # For simplicity, re-simulate per cycle to gather probs using final weights? We'll instead approximate using stored last p via quick replay using stored states not saved.
        # Instead: reconstruct per-cycle predicted probs by running agents with stored progression not saved; to keep code small, approximate calibration by binning mean predicted probabilities p_mean1/p_mean0
        # We'll compute calibration by aggregating per-cycle p_mean1/p_mean0 as proxies
        # Note: this is a lightweight proxy, not strict reliability diagram on trial-level
        p_means = np.array(
            [s["p_mean1"] if not np.isnan(s["p_mean1"]) else 0.5 for s in stats]
        )
        # For regime-level calibration, compute slope/intercept from regressed observed acc vs predicted mean
        try:
            # W1
            y_w1 = np.array([s["acc"] for i, s in enumerate(stats) if w1_mask[i]])
            x_w1 = np.array([p_means[i] for i in range(len(stats)) if w1_mask[i]])
            if x_w1.size >= 2:
                A_mat = np.vstack([np.ones_like(x_w1), x_w1]).T
                coef_w1, _, _, _ = np.linalg.lstsq(A_mat, y_w1, rcond=None)
                int_w1, slope_w1 = float(coef_w1[0]), float(coef_w1[1])
            else:
                slope_w1, int_w1 = np.nan, np.nan
            # W2
            y_w2 = np.array([s["acc"] for i, s in enumerate(stats) if w2_mask[i]])
            x_w2 = np.array([p_means[i] for i in range(len(stats)) if w2_mask[i]])
            if x_w2.size >= 2:
                A_mat = np.vstack([np.ones_like(x_w2), x_w2]).T
                coef_w2, _, _, _ = np.linalg.lstsq(A_mat, y_w2, rcond=None)
                int_w2, slope_w2 = float(coef_w2[0]), float(coef_w2[1])
            else:
                slope_w2, int_w2 = np.nan, np.nan
        except Exception:
            slope_w1 = slope_w2 = int_w1 = int_w2 = np.nan

        agent_rows.append(
            {
                "agent": name,
                "acc_w1_mean": round(acc_w1_mean, 3),
                "acc_w2_mean": round(acc_w2_mean, 3),
                "brier_w1": round(brier_w1, 3),
                "brier_w2": round(brier_w2, 3),
                "T_w1": round(p1, 3),
                "ci_w1_low": round(l1, 3),
                "ci_w1_high": round(h1, 3),
                "T_w2": round(p2, 3),
                "ci_w2_low": round(l2, 3),
                "ci_w2_high": round(h2, 3),
                "T_overall": round(pO, 3),
                "ci_over_low": round(lO, 3),
                "ci_over_high": round(hO, 3),
                "L": int(L),
                "F": round(F, 3),
                "calib_slope_w1": (
                    round(slope_w1, 3) if not np.isnan(slope_w1) else np.nan
                ),
                "calib_int_w1": round(int_w1, 3) if not np.isnan(int_w1) else np.nan,
                "calib_slope_w2": (
                    round(slope_w2, 3) if not np.isnan(slope_w2) else np.nan
                ),
                "calib_int_w2": round(int_w2, 3) if not np.isnan(int_w2) else np.nan,
            }
        )

    df_agents = pd.DataFrame(agent_rows)

    # Static claims summary: recompute Phase-2 style summary locally to keep this file self-contained
    # Reuse aggregate_and_score logic defined earlier in this file? We didn't implement it. For simplicity, evaluate summary using Phase2-like counting
    # Implement quick aggregator for static claims based on per_cycle_static
    per_cycle_array = np.array(per_cycle_static, dtype=float)

    # helper aggregator
    def agg_static(per_cycle_array, cycles_split):
        rows = []
        C = per_cycle_array.shape[0]
        W1_idx = np.arange(0, cycles_split)
        W2_idx = np.arange(cycles_split, C)
        for claim_idx in range(7):
            col = per_cycle_array[:, claim_idx]
            w1_col = col[W1_idx]
            w2_col = col[W2_idx]
            w1_mask = ~np.isnan(w1_col)
            w2_mask = ~np.isnan(w2_col)
            N_w1 = int(w1_mask.sum())
            N_w2 = int(w2_mask.sum())
            passes_w1 = int(np.nansum(w1_col[w1_mask] == 1)) if N_w1 > 0 else 0
            passes_w2 = int(np.nansum(w2_col[w2_mask] == 1)) if N_w2 > 0 else 0
            fails_w1 = int(np.nansum(w1_col[w1_mask] == 0)) if N_w1 > 0 else 0
            fails_w2 = int(np.nansum(w2_col[w2_mask] == 0)) if N_w2 > 0 else 0
            p1, l1, h1 = wilson_ci(passes_w1, N_w1)
            p2, l2, h2 = wilson_ci(passes_w2, N_w2)
            overall_mask = ~np.isnan(col)
            N_over = int(overall_mask.sum())
            passes_over = int(np.nansum(col[overall_mask] == 1)) if N_over > 0 else 0
            pO, lO, hO = wilson_ci(passes_over, N_over)
            delta = float(p2 - p1) if (N_w1 > 0 and N_w2 > 0) else np.nan
            rows.append(
                {
                    "claim": f"C{claim_idx+1}",
                    "passes_w1": passes_w1,
                    "fails_w1": fails_w1,
                    "N_w1": N_w1,
                    "T_w1": round(p1, 3),
                    "ci_w1_low": round(l1, 3),
                    "ci_w1_high": round(h1, 3),
                    "passes_w2": passes_w2,
                    "fails_w2": fails_w2,
                    "N_w2": N_w2,
                    "T_w2": round(p2, 3),
                    "ci_w2_low": round(l2, 3),
                    "ci_w2_high": round(h2, 3),
                    "T_overall": round(pO, 3),
                    "ci_over_low": round(lO, 3),
                    "ci_over_high": round(hO, 3),
                    "delta": round(delta, 3),
                }
            )
        return pd.DataFrame(rows)

    df_claims = agg_static(per_cycle_array, cycles_split)

    # Write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase3_run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    df_agents.to_csv(os.path.join(outdir, "agents_summary.csv"), index=False)
    df_claims.to_csv(os.path.join(outdir, "claims_summary.csv"), index=False)

    # agent_curves.png
    plt.figure(figsize=(10, 5))
    for name, stats in per_cycle_agent_stats.items():
        accs = np.array([s["acc"] for s in stats])
        plt.plot(range(1, CYCLES + 1), accs, marker="o", label=name)
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy")
    plt.title("Per-cycle Agent Accuracy (Phase 3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "agent_curves.png"), dpi=150)
    plt.close()

    # calibration plots: approximate by reusing per-cycle p_mean proxies
    for regime, mask in [
        ("w1", np.arange(CYCLES) < cycles_split),
        ("w2", np.arange(CYCLES) >= cycles_split),
    ]:
        plt.figure(figsize=(6, 6))
        # collect bin aggregated points across cycles using proxies
        # build arrays of per-cycle predicted means and observed acc
        preds = []
        obs = []
        for name, stats in per_cycle_agent_stats.items():
            for i, s in enumerate(stats):
                if mask[i]:
                    if not np.isnan(s["p_mean1"]):
                        preds.append(s["p_mean1"])
                        obs.append(s["acc"])
        if len(preds) == 0:
            centers = np.linspace(0.05, 0.95, 10)
            mean_pred = np.full(10, np.nan)
            mean_obs = np.full(10, np.nan)
        else:
            preds = np.array(preds)
            obs = np.array(obs)
            centers, mean_pred, mean_obs, counts = bin_reliability(obs, preds, bins=10)
        plt.plot([0, 1], [0, 1], "k--")
        plt.scatter(mean_pred, mean_obs)
        plt.xlabel("Predicted (bin mean)")
        plt.ylabel("Observed (bin mean)")
        plt.title(f"Calibration {regime.upper()} (per-cycle proxies)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"calibration_{regime}.png"), dpi=150)
        plt.close()

    # truth_bars_agents.png
    ind = np.arange(len(df_agents))
    width = 0.25
    T_w1 = df_agents["T_w1"].values.astype(float)
    T_w2 = df_agents["T_w2"].values.astype(float)
    T_over = df_agents["T_overall"].values.astype(float)
    err_w1 = np.vstack(
        [
            (T_w1 - df_agents["ci_w1_low"].values),
            (df_agents["ci_w1_high"].values - T_w1),
        ]
    )
    err_w2 = np.vstack(
        [
            (T_w2 - df_agents["ci_w2_low"].values),
            (df_agents["ci_w2_high"].values - T_w2),
        ]
    )
    err_o = np.vstack(
        [
            (T_over - df_agents["ci_over_low"].values),
            (df_agents["ci_over_high"].values - T_over),
        ]
    )
    plt.figure(figsize=(10, 5))
    plt.bar(ind - width, T_w1, width, yerr=err_w1, capsize=4)
    plt.bar(ind, T_w2, width, yerr=err_w2, capsize=4)
    plt.bar(ind + width, T_over, width, yerr=err_o, capsize=4)
    plt.xticks(ind, df_agents["agent"].values)
    plt.ylabel("T")
    plt.title("Agent Truth Stability (W1, W2, Overall)")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_agents.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Truth Stability — Phase 3 (Adaptive Agents)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {RANDOM_SEED}")
    readme.append(
        f"Trials: {TRIALS} | Cycles: {CYCLES} | W1: 1-{cycles_split} | W2: {cycles_split+1}-{CYCLES} | theta: {THETA}"
    )
    readme.append("")
    readme.append(df_agents.to_markdown(index=False))
    readme.append("")
    readme.append(
        "**Experiment Result:** PASS"
        if ((df_agents["T_w1"] >= 0.7).any() and (df_agents["T_w2"] >= 0.7).any())
        else "**Experiment Result:** FAIL"
    )
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RANDOM_SEED,
        "trials": TRIALS,
        "cycles": CYCLES,
        "per_cycle": per_cycle,
        "cycles_split": cycles_split,
        "cycle_ps": cycle_ps,
        "tau": tau,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Console summary
    print("\n=== Truth Stability Phase 3 Summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    for _, r in df_agents.iterrows():
        print(
            f"{r['agent']}: acc_w1={r['acc_w1_mean']} acc_w2={r['acc_w2_mean']} L={r['L']} F={r['F']} T_w1/T_w2={r['T_w1']}/{r['T_w2']} CI_w1=[{r['ci_w1_low']},{r['ci_w1_high']}] CI_w2=[{r['ci_w2_low']},{r['ci_w2_high']}]"
        )
    # PASS/FAIL gate (strict)
    gate = False
    for _, r in df_agents.iterrows():
        if (
            (r["T_w1"] >= 0.7)
            and (r["T_w2"] >= 0.7)
            and (r["L"] <= 2)
            and (r["F"] <= 0.15)
        ):
            gate = True
            break
    print(f"Experiment PASS?: {'YES' if gate else 'NO'}")
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Elapsed (s): {elapsed:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()

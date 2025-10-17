#!/usr/bin/env python3
"""
Truth Stability — Phase 4.5 (Adaptive Communication α)
Runs two modes: fixed-alpha baseline and adaptive per-agent alpha, compares recovery after misinformation.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- helpers ----------


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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


def design_matrix(A, B, C):
    ones = np.ones_like(A, dtype=float)
    AB = (A * B).astype(float)
    BC = (B * C).astype(float)
    X = np.vstack([ones, A.astype(float), B.astype(float), C.astype(float), AB, BC]).T
    return X


def build_topology(n_agents, topology, rng):
    adj = np.zeros((n_agents, n_agents), dtype=float)
    if topology == "ring":
        for i in range(n_agents):
            adj[i, (i - 1) % n_agents] = 1
            adj[i, (i + 1) % n_agents] = 1
    elif topology == "ws":
        k = 4
        beta = 0.2
        for i in range(n_agents):
            for j in range(1, k // 2 + 1):
                nbr = (i + j) % n_agents
                adj[i, nbr] = 1
                adj[nbr, i] = 1
        for i in range(n_agents):
            for j in range(1, k // 2 + 1):
                if rng.random() < beta:
                    old = (i + j) % n_agents
                    possible = list(range(n_agents))
                    possible.remove(i)
                    new = int(rng.choice(possible))
                    adj[i, old] = 0
                    adj[old, i] = 0
                    adj[i, new] = 1
                    adj[new, i] = 1
    elif topology == "er":
        p = 0.15
        rand = rng.random((n_agents, n_agents))
        adj = (rand < p).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj) + np.triu(adj).T
    else:
        raise ValueError("Unknown topology")
    np.fill_diagonal(adj, 0)
    return adj


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


# ---------- core runner for a single mode ----------


def run_mode(
    seed,
    mode,
    N,
    topology,
    alpha_fixed,
    alpha_min,
    alpha_max,
    tau,
    k_up,
    k_down,
    lambda_mix,
    beta,
    gamma,
    frac,
    TRIALS,
    CYCLES,
    theta,
    T_stop,
):
    rng = np.random.default_rng(int(seed))
    per_cycle = max(1, TRIALS // CYCLES)
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    adj = build_topology(N, topology, rng)
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    d = 6
    Ws = np.zeros((N, d), dtype=float)

    # per-cycle storage
    agent_accs = np.zeros((CYCLES, N), dtype=float)
    agent_briers = np.zeros((CYCLES, N), dtype=float)
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)
    mean_alpha = np.zeros(CYCLES, dtype=float)
    min_alpha = np.zeros(CYCLES, dtype=float)
    max_alpha = np.zeros(CYCLES, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    m = int(round(frac * N))
    misinfo_set = []

    # init alphas for adaptive mode
    if mode == "adaptive":
        alphas = np.full(N, alpha_fixed, dtype=float)  # start from alpha_fixed
    else:
        alphas = np.full(N, alpha_fixed, dtype=float)

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        # world
        if ci < 6:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        X = design_matrix(A, B, C)

        # per-agent update (gradient step)
        # vectorize predictions per agent: compute logits SAMP x N
        for i in range(N):
            w = Ws[i]
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            brier = float(((p - Y) ** 2).mean())
            agent_accs[ci, i] = acc
            agent_briers[ci, i] = brier
            grad = X.T.dot(p - Y) / X.shape[0]
            Ws[i] = w - 0.3 * grad

        # determine misinfo at start attack (cycle 7 index 6)
        if ci == 6:
            misinfo_set = (
                list(rng.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )

        # apply misinfo bias during attack (indices 6..T_stop-1)
        if (ci >= 6) and (ci <= (T_stop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # Adaptive alpha: compute disagreement D_i at end of data update before consensus
        if mode == "adaptive":
            # compute p_mat: shape (samples, N)
            logits = X.dot(Ws.T)  # (S, N)
            p_mat = sigmoid(logits)
            # d_pred_i: mean over neighbors and samples of |p_i - p_j|
            d_pred = np.zeros(N, dtype=float)
            for i in range(N):
                nbrs = neighbors[i]
                if len(nbrs) == 0:
                    d_pred[i] = 0.0
                else:
                    # for each neighbor, compute mean over samples of abs diff
                    diffs = [np.mean(np.abs(p_mat[:, i] - p_mat[:, j])) for j in nbrs]
                    d_pred[i] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
            # d_param_i: ||w_i - mean(w_neighbors)||
            d_param = np.zeros(N, dtype=float)
            for i in range(N):
                nbrs = neighbors[i]
                if len(nbrs) == 0:
                    d_param[i] = 0.0
                else:
                    mean_reg = Ws[nbrs].mean(axis=0)
                    d_param[i] = float(np.linalg.norm(Ws[i] - mean_reg))

            # zscore safely
            def safe_z(x):
                m = x.mean()
                s = x.std()
                if s == 0:
                    return np.zeros_like(x)
                return (x - m) / s

            z_pred = safe_z(d_pred)
            z_param = safe_z(d_param)
            D = lambda_mix * z_pred + (1.0 - lambda_mix) * z_param
            # update alphas
            # alpha_i_next = clip(alpha_i * exp(-k_up * D_i) + k_down*(alpha_max - alpha_i)*1{D_i < tau}, [alpha_min, alpha_max])
            grow_mask = D < tau
            new_alphas = alphas * np.exp(-k_up * D)
            new_alphas = new_alphas + (k_down * (alpha_max - alphas) * grow_mask)
            new_alphas = np.clip(new_alphas, alpha_min, alpha_max)
            alphas = new_alphas

        # consensus step: per-agent alpha used
        W_new = Ws.copy()
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                mean_nbr = Ws[i]
            else:
                nbrs = np.array(nbrs)
                mis_nbrs = (
                    np.intersect1d(nbrs, misinfo_set)
                    if len(misinfo_set) > 0
                    else np.array([], dtype=int)
                )
                reg_nbrs = np.setdiff1d(nbrs, mis_nbrs)
                if reg_nbrs.size > 0:
                    mean_reg = Ws[reg_nbrs].mean(axis=0)
                else:
                    mean_reg = np.zeros(d)
                if mis_nbrs.size > 0:
                    mean_mis = Ws[mis_nbrs].mean(axis=0)
                else:
                    mean_mis = np.zeros(d)
                # during attack, misinfo neighbors given boost gamma
                if (ci >= 6) and (ci <= (T_stop - 1)) and (mis_nbrs.size > 0):
                    if reg_nbrs.size > 0:
                        weighted = (mean_reg + gamma * mean_mis) / (1.0 + gamma)
                    else:
                        weighted = mean_mis
                else:
                    if reg_nbrs.size > 0:
                        weighted = mean_reg
                    elif mis_nbrs.size > 0:
                        weighted = mean_mis
                    else:
                        weighted = Ws[i]
                alpha_i = float(alphas[i]) if mode == "adaptive" else float(alpha_fixed)
                W_new[i] = (1 - alpha_i) * Ws[i] + alpha_i * weighted
        Ws = W_new

        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))
        mean_alpha[ci] = float(np.mean(alphas))
        min_alpha[ci] = float(np.min(alphas))
        max_alpha[ci] = float(np.max(alphas))

    # compute phase metrics
    w1_idx = list(range(0, 6))
    attack_idx = list(range(6, T_stop))
    recovery_idx = list(range(T_stop, CYCLES))

    group_mean_acc = agent_accs.mean(axis=1)

    def T_for_idx(idxs):
        if len(idxs) == 0:
            return 0.0, 0.0, 0.0
        passes = int((group_mean_acc[idxs] >= theta).sum())
        p, lo, hi = wilson_ci(passes, len(idxs))
        return p, lo, hi

    T_w1, lo1, hi1 = T_for_idx(w1_idx)
    T_attack, loA, hiA = T_for_idx(attack_idx)
    T_recovery, loR, hiR = T_for_idx(recovery_idx)

    # A_attack: sum_{t=7..10} max(0, theta - mean_acc(t))
    A_attack = (
        float(np.sum(np.maximum(0.0, theta - group_mean_acc[attack_idx])))
        if len(attack_idx) > 0
        else 0.0
    )

    # Lr: first t>=11 with mean_acc >= (W1_mean_acc - 0.03)
    W1_mean_acc = float(group_mean_acc[w1_idx].mean())
    Lr = float("inf")
    target = W1_mean_acc - 0.03
    for offset, idx in enumerate(recovery_idx, start=1):
        if group_mean_acc[idx] >= target:
            Lr = offset
            break
    Lr_val = int(Lr) if Lr != float("inf") else None

    last_acc = float(group_mean_acc[-1])
    R = float(np.clip(W1_mean_acc - last_acc, 0.0, 1.0))
    O = (
        float(max(0.0, np.max(group_mean_acc[recovery_idx] - W1_mean_acc)))
        if len(recovery_idx) > 0
        else 0.0
    )

    mean_alpha_w1 = float(np.mean(mean_alpha[w1_idx]))
    mean_alpha_attack = (
        float(np.mean(mean_alpha[attack_idx])) if len(attack_idx) > 0 else float(np.nan)
    )
    mean_alpha_recovery = (
        float(np.mean(mean_alpha[recovery_idx]))
        if len(recovery_idx) > 0
        else float(np.nan)
    )

    consensus_min_post = (
        float(np.min(consensus_ts[recovery_idx])) if len(recovery_idx) > 0 else 0.0
    )
    disp_recovery_mean = (
        float(np.mean(dispersion[recovery_idx])) if len(recovery_idx) > 0 else 0.0
    )

    # per-agent summary
    agent_rows = []
    for i in range(N):
        acc_w1 = float(agent_accs[w1_idx, i].mean())
        acc_attack = (
            float(agent_accs[attack_idx, i].mean())
            if len(attack_idx) > 0
            else float("nan")
        )
        acc_recovery = (
            float(agent_accs[recovery_idx, i].mean())
            if len(recovery_idx) > 0
            else float("nan")
        )
        agent_rows.append(
            {
                "agent": i,
                "is_misinfo": (i in misinfo_set),
                "acc_w1_mean": round(acc_w1, 3),
                "acc_attack_mean": (
                    round(acc_attack, 3) if not np.isnan(acc_attack) else ""
                ),
                "acc_recovery_mean": (
                    round(acc_recovery, 3) if not np.isnan(acc_recovery) else ""
                ),
            }
        )

    group_row = {
        "T_group_w1": round(T_w1, 3),
        "ci_w1_lo": round(lo1, 3),
        "ci_w1_hi": round(hi1, 3),
        "T_group_attack": round(T_attack, 3),
        "ci_attack_lo": round(loA, 3),
        "ci_attack_hi": round(hiA, 3),
        "T_group_recovery": round(T_recovery, 3),
        "ci_rec_lo": round(loR, 3),
        "ci_rec_hi": round(hiR, 3),
        "A_attack": round(A_attack, 4),
        "Lr": (Lr_val if Lr_val is not None else "inf"),
        "Residual_R": round(R, 4),
        "Overshoot_O": round(O, 4),
        "mean_alpha_w1": round(mean_alpha_w1, 4),
        "mean_alpha_attack": (
            round(mean_alpha_attack, 4) if not np.isnan(mean_alpha_attack) else ""
        ),
        "mean_alpha_recovery": (
            round(mean_alpha_recovery, 4) if not np.isnan(mean_alpha_recovery) else ""
        ),
        "dispersion_mean_recovery": round(disp_recovery_mean, 6),
        "consensus_min_post": round(consensus_min_post, 6),
    }

    return {
        "agent_rows": agent_rows,
        "group_row": group_row,
        "consensus_ts": consensus_ts,
        "dispersion": dispersion,
        "mean_alpha": mean_alpha,
        "min_alpha": min_alpha,
        "max_alpha": max_alpha,
        "group_mean_acc": group_mean_acc,
        "experiment_pass": None,
    }


# ---------- CLI & orchestration ----------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=18)
    p.add_argument("--topology", type=str, default="ws")
    p.add_argument("--trials", type=int, default=6000)
    p.add_argument("--cycles", type=int, default=16)
    p.add_argument("--theta", type=float, default=0.65)
    p.add_argument("--alpha_fixed", type=float, default=0.15)
    p.add_argument("--alpha_min", type=float, default=0.05)
    p.add_argument("--alpha_max", type=float, default=0.30)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--kup", type=float, default=1.2)
    p.add_argument("--kdown", type=float, default=0.05)
    p.add_argument("--lambda_mix", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--frac", type=float, default=0.11)
    p.add_argument("--tstop", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    RAND = int(args.seed)
    N = int(args.agents)
    topology = args.topology
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    theta = float(args.theta)
    alpha_fixed = float(args.alpha_fixed)
    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    tau = float(args.tau)
    k_up = float(args.kup)
    k_down = float(args.kdown)
    lambda_mix = float(args.lambda_mix)
    beta = float(args.beta)
    gamma = float(args.gamma)
    frac = float(args.frac)
    T_stop = int(args.tstop)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_adaptive_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # run fixed baseline
    res_fixed = run_mode(
        RAND,
        "fixed",
        N,
        topology,
        alpha_fixed,
        alpha_min,
        alpha_max,
        tau,
        k_up,
        k_down,
        lambda_mix,
        beta,
        gamma,
        frac,
        TRIALS,
        CYCLES,
        theta,
        T_stop,
    )
    # run adaptive
    res_adapt = run_mode(
        RAND + 123,
        "adaptive",
        N,
        topology,
        alpha_fixed,
        alpha_min,
        alpha_max,
        tau,
        k_up,
        k_down,
        lambda_mix,
        beta,
        gamma,
        frac,
        TRIALS,
        CYCLES,
        theta,
        T_stop,
    )

    # write summaries
    df_fixed_agents = pd.DataFrame(res_fixed["agent_rows"])
    df_adapt_agents = pd.DataFrame(res_adapt["agent_rows"])
    df_fixed_agents.to_csv(os.path.join(outdir, "agents_fixed.csv"), index=False)
    df_adapt_agents.to_csv(os.path.join(outdir, "agents_adaptive.csv"), index=False)
    pd.DataFrame([res_fixed["group_row"]]).to_csv(
        os.path.join(outdir, "summary_fixed.csv"), index=False
    )
    pd.DataFrame([res_adapt["group_row"]]).to_csv(
        os.path.join(outdir, "summary_adaptive.csv"), index=False
    )

    # compare summary
    keys = [
        "A_attack",
        "Lr",
        "Residual_R",
        "Overshoot_O",
        "T_group_w1",
        "T_group_attack",
        "T_group_recovery",
    ]
    compare = {
        k: (res_adapt["group_row"].get(k), res_fixed["group_row"].get(k)) for k in keys
    }
    df_compare = pd.DataFrame(
        [
            {
                ("adaptive_" + k): res_adapt["group_row"].get(k)
                for k in res_adapt["group_row"].keys()
            },
            {
                ("fixed_" + k): res_fixed["group_row"].get(k)
                for k in res_fixed["group_row"].keys()
            },
        ]
    )
    df_compare.to_csv(os.path.join(outdir, "compare_summary.csv"), index=False)

    # plots
    cycles = np.arange(1, CYCLES + 1)
    g_fixed = res_fixed["group_mean_acc"]
    g_adapt = res_adapt["group_mean_acc"]

    # curves_truth_compare.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, g_fixed, marker="o")
    plt.plot(cycles, g_adapt, marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Group mean accuracy")
    plt.title("Group mean accuracy: Fixed vs Adaptive")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_truth_compare.png"), dpi=150)
    plt.close()

    # curves_alpha_adaptive.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, res_adapt["mean_alpha"], marker="o")
    plt.fill_between(cycles, res_adapt["min_alpha"], res_adapt["max_alpha"], alpha=0.2)
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("alpha (mean)")
    plt.title("Adaptive alpha: mean and range across agents")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_alpha_adaptive.png"), dpi=150)
    plt.close()

    # curves_consensus_compare.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, res_fixed["consensus_ts"], marker="o")
    plt.plot(cycles, res_adapt["consensus_ts"], marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Consensus (mean ||w_i - w_bar||)")
    plt.title("Consensus: Fixed vs Adaptive")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_consensus_compare.png"), dpi=150)
    plt.close()

    # curves_dispersion_compare.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, res_fixed["dispersion"], marker="o")
    plt.plot(cycles, res_adapt["dispersion"], marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Dispersion (std acc)")
    plt.title("Dispersion: Fixed vs Adaptive")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_dispersion_compare.png"), dpi=150)
    plt.close()

    # truth_bars_compare.png
    labels = ["W1", "Attack", "Recovery"]
    T_fixed = [
        res_fixed["group_row"]["T_group_w1"],
        res_fixed["group_row"]["T_group_attack"],
        res_fixed["group_row"]["T_group_recovery"],
    ]
    lo_fixed = [
        res_fixed["group_row"]["ci_w1_lo"],
        res_fixed["group_row"]["ci_attack_lo"],
        res_fixed["group_row"]["ci_rec_lo"],
    ]
    hi_fixed = [
        res_fixed["group_row"]["ci_w1_hi"],
        res_fixed["group_row"]["ci_attack_hi"],
        res_fixed["group_row"]["ci_rec_hi"],
    ]
    T_ad = [
        res_adapt["group_row"]["T_group_w1"],
        res_adapt["group_row"]["T_group_attack"],
        res_adapt["group_row"]["T_group_recovery"],
    ]
    lo_ad = [
        res_adapt["group_row"]["ci_w1_lo"],
        res_adapt["group_row"]["ci_attack_lo"],
        res_adapt["group_row"]["ci_rec_lo"],
    ]
    hi_ad = [
        res_adapt["group_row"]["ci_w1_hi"],
        res_adapt["group_row"]["ci_attack_hi"],
        res_adapt["group_row"]["ci_rec_hi"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(
        x - width / 2,
        T_fixed,
        width,
        yerr=np.vstack(
            [
                np.array(T_fixed) - np.array(lo_fixed),
                np.array(hi_fixed) - np.array(T_fixed),
            ]
        ),
        capsize=4,
    )
    plt.bar(
        x + width / 2,
        T_ad,
        width,
        yerr=np.vstack(
            [np.array(T_ad) - np.array(lo_ad), np.array(hi_ad) - np.array(T_ad)]
        ),
        capsize=4,
    )
    plt.xticks(x, labels)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("T_group")
    plt.title("T_group by phase: Fixed vs Adaptive")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_compare.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Truth Stability — Phase 4.5 (Adaptive α)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(
        f"Seed: {RAND} | N: {N} | topology: {topology} | cycles: {CYCLES} | trials: {TRIALS}"
    )
    readme.append(
        f"alpha_fixed: {alpha_fixed} | alpha_min: {alpha_min} | alpha_max: {alpha_max} | tau: {tau} | k_up: {k_up} | k_down: {k_down} | lambda_mix: {lambda_mix}"
    )
    readme.append(f"beta: {beta} | gamma: {gamma} | frac: {frac} | T_stop: {T_stop}")
    readme.append("")
    readme.append("Fixed summary:")
    readme.append(pd.DataFrame([res_fixed["group_row"]]).to_markdown(index=False))
    readme.append("")
    readme.append("Adaptive summary:")
    readme.append(pd.DataFrame([res_adapt["group_row"]]).to_markdown(index=False))
    readme.append("")
    # Win criteria
    A_fixed = res_fixed["group_row"]["A_attack"]
    A_ad = res_adapt["group_row"]["A_attack"]
    Lr_fixed = res_fixed["group_row"]["Lr"]
    Lr_ad = res_adapt["group_row"]["Lr"]
    Trec_fixed = res_fixed["group_row"]["T_group_recovery"]
    Trec_ad = res_adapt["group_row"]["T_group_recovery"]
    win = (
        (A_ad < A_fixed)
        and (
            (Lr_ad == "inf" and Lr_fixed == "inf")
            or (
                isinstance(Lr_ad, int)
                and (Lr_ad <= (Lr_fixed if isinstance(Lr_fixed, int) else float("inf")))
            )
        )
        and (Trec_ad >= Trec_fixed)
        and (0.02 <= res_adapt["group_row"]["dispersion_mean_recovery"] <= 0.12)
    )
    readme.append(f'Adaptive wins vs Fixed? {"YES" if win else "NO"}')
    readme.append("")
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RAND,
        "N": N,
        "topology": topology,
        "trials": TRIALS,
        "cycles": CYCLES,
        "theta": theta,
        "alpha_fixed": alpha_fixed,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "tau": tau,
        "k_up": k_up,
        "k_down": k_down,
        "lambda_mix": lambda_mix,
        "beta": beta,
        "gamma": gamma,
        "frac": frac,
        "T_stop": T_stop,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\n=== Adaptive vs Fixed summary ===")

    def short_summary(prefix, g):
        return f"{prefix} | {g['T_group_w1']} / {g['T_group_attack']} / {g['T_group_recovery']} | A_attack={g['A_attack']} | Lr={g['Lr']} | R={g['Residual_R']} | O={g['Overshoot_O']}"

    print(short_summary("FIXED", res_fixed["group_row"]))
    print(short_summary("ADAPT", res_adapt["group_row"]))
    print(
        f"Winner: {'ADAPTIVE' if win else 'FIXED' if (not win and (A_fixed < A_ad)) else 'TIE'}"
    )
    print(f"Results folder: {os.path.abspath(outdir)}")
    print("=================================\n")


if __name__ == "__main__":
    main()

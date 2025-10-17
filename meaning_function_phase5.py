#!/usr/bin/env python3
"""
Meaning Function — Phase 5
Compute Meaning Index M_t = w_T*T_t + w_D*D_t + w_V*V_t using adaptive-alpha networked learners.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Helpers


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def wilson_ci_for_mean(data, n_boot=1000, rng=None):
    # bootstrap 95% CI for the mean
    if rng is None:
        rng = np.random.default_rng(0)
    data = np.array(data)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=data.size, replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(data.mean()), float(lo), float(hi)


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


# Main experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--topology", type=str, default="ws")
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--frac", type=float, default=0.11)
    parser.add_argument("--tstop", type=int, default=10)
    parser.add_argument("--wT", type=float, default=0.4)
    parser.add_argument("--wD", type=float, default=0.3)
    parser.add_argument("--wV", type=float, default=0.3)
    args = parser.parse_args()

    RNG = np.random.default_rng(int(args.seed))
    seed = int(args.seed)
    N = int(args.agents)
    topology = args.topology
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    theta = float(args.theta)
    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    beta = float(args.beta)
    gamma = float(args.gamma)
    frac = float(args.frac)
    T_stop = int(args.tstop)
    wT = float(args.wT)
    wD = float(args.wD)
    wV = float(args.wV)

    per_cycle = max(1, TRIALS // CYCLES)

    # schedule
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    adj = build_topology(N, topology, RNG)
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    d = 6
    Ws = np.zeros((N, d), dtype=float)

    # value vectors u_i
    u = RNG.random((N, 3))  # curiosity, cooperation, coherence weights in [0,1]

    # storage
    group_mean_acc = np.zeros(CYCLES, dtype=float)
    sigma_acc = np.zeros(CYCLES, dtype=float)
    T_t = np.zeros(CYCLES, dtype=float)
    D_t = np.zeros(CYCLES, dtype=float)
    V_t = np.zeros(CYCLES, dtype=float)
    M_t = np.zeros(CYCLES, dtype=float)

    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)
    drift_to_false = np.zeros(CYCLES, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + RNG.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    m = int(round(frac * N))
    misinfo_set = []

    # adaptive alphas start at midpoint
    alphas = np.full(N, (alpha_min + alpha_max) / 2.0)
    # controller params (reuse Phase4.5 defaults)
    tau = 0.0
    k_up = 1.2
    k_down = 0.05
    lambda_mix = 0.5

    sigma_opt = 0.07

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        # world
        if ci < 6:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, RNG)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, RNG)

        X = design_matrix(A, B, C)

        # per-agent predictions and updates
        p_mat = np.zeros((per_cycle, N), dtype=float)
        accs = np.zeros(N, dtype=float)
        for i in range(N):
            p = sigmoid(X.dot(Ws[i]))
            p_mat[:, i] = p
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            accs[i] = acc
            grad = X.T.dot(p - Y) / X.shape[0]
            Ws[i] = Ws[i] - 0.3 * grad

        group_mean_acc[ci] = float(accs.mean())
        sigma_acc[ci] = float(accs.std())
        T_t[ci] = float(group_mean_acc[ci])  # continuous truth index
        dispersion[ci] = sigma_acc[ci]

        # determine misinfo set at start of attack
        if ci == 6:
            misinfo_set = (
                list(RNG.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )

        # apply misinfo during attack cycles
        if (ci >= 6) and (ci <= (T_stop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # adaptive alpha controller: compute disagreement signals using p_mat and Ws
        # d_pred: mean abs diff between agent and neighbors over batch
        d_pred = np.zeros(N, dtype=float)
        d_param = np.zeros(N, dtype=float)
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                d_pred[i] = 0.0
                d_param[i] = 0.0
            else:
                diffs = [np.mean(np.abs(p_mat[:, i] - p_mat[:, j])) for j in nbrs]
                d_pred[i] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
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
        grow_mask = D < tau
        new_alphas = alphas * np.exp(-k_up * D)
        new_alphas = new_alphas + (k_down * (alpha_max - alphas) * grow_mask)
        new_alphas = np.clip(new_alphas, alpha_min, alpha_max)
        alphas = new_alphas

        # consensus with possible misinfo broadcast boost
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
                alpha_i = float(alphas[i])
                W_new[i] = (1 - alpha_i) * Ws[i] + alpha_i * weighted
        Ws = W_new

        # consensus metric and drift
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        sims = []
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            if wi_norm == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(wi, v) / (wi_norm * v_norm)))
        drift_to_false[ci] = float(np.mean(sims))

        # compute value components per agent
        c_i = p_mat.var(axis=0)  # curiosity as variance of predictions
        p_i = np.zeros(N, dtype=float)
        r_i = np.zeros(N, dtype=float)
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            nbrs = neighbors[i]
            if len(nbrs) == 0 or wi_norm == 0:
                p_i[i] = 0.0
            else:
                mean_nb = Ws[nbrs].mean(axis=0)
                p_i[i] = 1.0 - (np.linalg.norm(wi - mean_nb) / wi_norm)
            r_i[i] = 1.0 - abs(accs.mean() - theta)
        # V_i = mean( u_i · [c,p,r] ) normalized by 3
        dot = (u * np.vstack([c_i, p_i, r_i]).T).sum(axis=1)
        V_i = dot / 3.0
        V_i = np.clip(V_i, 0.0, 1.0)
        V_t[ci] = float(np.mean(V_i))

        # D_t normalized dispersion -> 1 - |σ_acc - σ_opt| / σ_opt
        D_raw = 1.0 - (abs(sigma_acc[ci] - sigma_opt) / sigma_opt)
        D_t[ci] = float(np.clip(D_raw, 0.0, 1.0))

        # M_t
        M_t[ci] = wT * T_t[ci] + wD * D_t[ci] + wV * V_t[ci]

    # phases
    w1_idx = list(range(0, 6))
    attack_idx = list(range(6, T_stop))
    recovery_idx = list(range(T_stop, CYCLES))

    rng_bs = np.random.default_rng(seed + 1234)
    M_w1_mean, M_w1_lo, M_w1_hi = wilson_ci_for_mean(
        M_t[w1_idx], n_boot=1000, rng=rng_bs
    )
    M_att_mean, M_att_lo, M_att_hi = wilson_ci_for_mean(
        M_t[attack_idx], n_boot=1000, rng=rng_bs
    )
    M_rec_mean, M_rec_lo, M_rec_hi = wilson_ci_for_mean(
        M_t[recovery_idx], n_boot=1000, rng=rng_bs
    )

    # Derived metrics
    EM = float((M_rec_mean - M_att_mean) / (M_w1_mean if M_w1_mean != 0 else 1.0))
    # Meaning Loss Area A_M = sum max(0, M_w1_mean - M_t) over attack cycles
    A_M = (
        float(np.sum(np.maximum(0.0, M_w1_mean - M_t[attack_idx])))
        if len(attack_idx) > 0
        else 0.0
    )
    R_M = float(M_rec_mean / (M_w1_mean if M_w1_mean != 0 else 1.0))

    disp_recovery_mean = (
        float(np.mean(sigma_acc[recovery_idx])) if len(recovery_idx) > 0 else 0.0
    )

    pass_cond = (
        (M_rec_mean >= 0.9 * M_w1_mean)
        and (EM >= 1.0)
        and (R_M >= 0.9)
        and (0.02 <= disp_recovery_mean <= 0.12)
    )

    # write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"meaning_phase5_run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # meaning_curves.png
    cycles = np.arange(1, CYCLES + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, T_t, marker="o", label="T_t")
    plt.plot(cycles, D_t, marker="o", label="D_t")
    plt.plot(cycles, V_t, marker="o", label="V_t")
    plt.plot(cycles, M_t, marker="o", label="M_t")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Index")
    plt.title("Meaning components and M_t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "meaning_curves.png"), dpi=150)
    plt.close()

    # meaning_bars_phases.png
    labels = ["W1", "Attack", "Recovery"]
    Ms = [M_w1_mean, M_att_mean, M_rec_mean]
    lows = [M_w1_lo, M_att_lo, M_rec_lo]
    highs = [M_w1_hi, M_att_hi, M_rec_hi]
    errs = np.vstack([np.array(Ms) - np.array(lows), np.array(highs) - np.array(Ms)])
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(Ms)), Ms, yerr=errs, capsize=4)
    plt.xticks(range(len(Ms)), labels)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("M (Meaning Index)")
    plt.title("Meaning by phase")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "meaning_bars_phases.png"), dpi=150)
    plt.close()

    # group_summary.csv
    group_row = {
        "M_W1": M_w1_mean,
        "M_W1_lo": M_w1_lo,
        "M_W1_hi": M_w1_hi,
        "M_Attack": M_att_mean,
        "M_Attack_lo": M_att_lo,
        "M_Attack_hi": M_att_hi,
        "M_Recovery": M_rec_mean,
        "M_Recovery_lo": M_rec_lo,
        "M_Recovery_hi": M_rec_hi,
        "EM": EM,
        "A_M": A_M,
        "R_M": R_M,
        "disp_recovery_mean": disp_recovery_mean,
        "PASS": bool(pass_cond),
    }
    pd.DataFrame([group_row]).to_csv(
        os.path.join(outdir, "group_summary.csv"), index=False
    )

    params = {
        "seed": seed,
        "N": N,
        "topology": topology,
        "trials": TRIALS,
        "cycles": CYCLES,
        "theta": theta,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "beta": beta,
        "gamma": gamma,
        "frac": frac,
        "T_stop": T_stop,
        "wT": wT,
        "wD": wD,
        "wV": wV,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # README
    readme = []
    readme.append("# Meaning Function — Phase 5")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {seed} | N: {N} | topology: {topology}")
    readme.append("")
    readme.append(f"M_W1 = {M_w1_mean:.3f} [{M_w1_lo:.3f}, {M_w1_hi:.3f}]")
    readme.append(f"M_Attack = {M_att_mean:.3f} [{M_att_lo:.3f}, {M_att_hi:.3f}]")
    readme.append(f"M_Recovery = {M_rec_mean:.3f} [{M_rec_lo:.3f}, {M_rec_hi:.3f}]")
    readme.append("")
    readme.append(f"Elastic Meaning (EM) = {EM:.3f}")
    readme.append(f"Meaning Loss Area (A_M) = {A_M:.4f}")
    readme.append(f"Meaning Retention (R_M) = {R_M:.3f}")
    readme.append("")
    readme.append(f'PASS: {"YES" if pass_cond else "NO"}')
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    # console summary
    print("\n=== Meaning Function Phase 5 Summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    print(
        f"W1: M = {M_w1_mean:.3f}  Attack: {M_att_mean:.3f}  Recovery: {M_rec_mean:.3f}"
    )
    print(
        f'Elastic Meaning (EM): {EM:.3f}  Meaning Retention (R_M): {R_M:.3f}  PASS: {"YES" if pass_cond else "NO"}'
    )
    print("=======================================\n")


if __name__ == "__main__":
    main()

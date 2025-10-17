#!/usr/bin/env python3
"""
Meaning Phase 5.3 — Anneal mutation + curiosity floor

Runs baseline (fixed values) and an annealed+floor evolution mode. Writes summaries, comparisons, plots, README, params.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Bootstrap helper
def bootstrap_mean_ci(data, n_boot=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data = np.array(data)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = rng.choice(data, size=data.size, replace=True)
        means.append(s.mean())
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(data.mean()), float(lo), float(hi)


# Design matrix
def design_matrix(A, B, C):
    ones = np.ones_like(A, dtype=float)
    AB = (A * B).astype(float)
    BC = (B * C).astype(float)
    return np.vstack(
        [ones, A.astype(float), B.astype(float), C.astype(float), AB, BC]
    ).T


# Topology builder
def build_topology(n, topology, rng):
    adj = np.zeros((n, n), dtype=float)
    if topology == "ring":
        for i in range(n):
            adj[i, (i - 1) % n] = 1
            adj[i, (i + 1) % n] = 1
    elif topology == "ws":
        k = 4
        beta = 0.2
        for i in range(n):
            for j in range(1, k // 2 + 1):
                nbr = (i + j) % n
                adj[i, nbr] = 1
                adj[nbr, i] = 1
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if rng.random() < beta:
                    old = (i + j) % n
                    possible = list(range(n))
                    possible.remove(i)
                    new = int(rng.choice(possible))
                    adj[i, old] = 0
                    adj[old, i] = 0
                    adj[i, new] = 1
                    adj[new, i] = 1
    elif topology == "er":
        p = 0.15
        rand = rng.random((n, n))
        adj = (rand < p).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj) + np.triu(adj).T
    else:
        raise ValueError("unknown topology")
    np.fill_diagonal(adj, 0)
    return adj


# Score functions
def simulate_W1(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0, 0.08, size=n)
    S = 0.55 * A + 0.30 * B - 0.35 * C + eps
    S = np.clip(S, 0, 1)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


def simulate_W2(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0, 0.08, size=n)
    S = 0.15 * A + 0.50 * B - 0.10 * C + 0.20 * (A * B) - 0.25 * (B * C) + eps
    S = np.clip(S, 0, 1)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


# Single-run engine (returns summary and diagnostics)
def run_mode(
    seed,
    N,
    topology,
    trials,
    cycles,
    theta,
    alpha_min,
    alpha_max,
    beta,
    gamma,
    frac,
    tstop,
    evolve_mode,
    evolve_settings,
    wT,
    wD,
    wV,
    simplex=False,
):
    rng = np.random.default_rng(int(seed))
    per_cycle = max(1, trials // cycles)
    base = np.linspace(0.2, 0.8, cycles)
    cycle_ps = []
    for i in range(cycles):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    adj = build_topology(N, topology, rng)
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    d = 6
    Ws = np.zeros((N, d), dtype=float)
    u = rng.uniform(0.2, 0.8, size=(N, 3))

    alphas = np.full(N, (alpha_min + alpha_max) / 2.0)

    sigma_opt = 0.07

    m = int(round(frac * N))
    misinfo_set = []

    # evolve settings
    q_frac = evolve_settings.get("q_frac", 0.2)
    curiosity_floor = float(evolve_settings.get("curiosity_floor", 0.25))
    eta_base = float(evolve_settings.get("eta_base", 0.18))
    eta_bump = float(evolve_settings.get("eta_bump", 0.22))
    bump_start = int(evolve_settings.get("bump_start", 8))
    bump_end = int(evolve_settings.get("bump_end", 11))
    sigma_hi = float(evolve_settings.get("sigma_hi", 0.06))
    sigma_lo = float(evolve_settings.get("sigma_lo", 0.03))
    sigma_hold_until = int(evolve_settings.get("sigma_hold_until", 10))

    # storage
    T_arr = np.zeros(cycles)
    D_arr = np.zeros(cycles)
    V_arr = np.zeros(cycles)
    M_arr = np.zeros(cycles)
    accs_store = []

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < 6:
            A, B, C, S, Y = simulate_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_W2(per_cycle, pA, pB, pC, rng)
        X = design_matrix(A, B, C)
        per_agent_acc = np.zeros(N)
        p_mat = np.zeros((per_cycle, N))
        for i in range(N):
            p = 1.0 / (1.0 + np.exp(-X.dot(Ws[i])))
            p_mat[:, i] = p
            yhat = (p >= 0.5).astype(int)
            per_agent_acc[i] = float((yhat == Y).mean())
            grad = X.T.dot(p - Y) / X.shape[0]
            Ws[i] = Ws[i] - 0.3 * grad

        accs_store.append(per_agent_acc.copy())

        # misinfo injection during attack
        if ci == 6:
            misinfo_set = (
                list(rng.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )
        if (ci >= 6) and (ci <= (tstop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # adaptive alpha
        d_pred = np.zeros(N)
        d_param = np.zeros(N)
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                d_pred[i] = 0
                d_param[i] = 0
            else:
                diffs = [np.mean(np.abs(p_mat[:, i] - p_mat[:, j])) for j in nbrs]
                d_pred[i] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
                mean_reg = Ws[nbrs].mean(axis=0)
                d_param[i] = float(np.linalg.norm(Ws[i] - mean_reg))

        def safe_z(x):
            m = x.mean()
            s = x.std()
            return np.zeros_like(x) if s == 0 else (x - m) / s

        z_pred = safe_z(d_pred)
        z_param = safe_z(d_param)
        D_sig = 0.5 * z_pred + 0.5 * z_param
        alphas = np.clip(
            alphas * np.exp(-1.2 * D_sig) + (0.05 * (alpha_max - alphas) * (D_sig < 0)),
            alpha_min,
            alpha_max,
        )

        # consensus update with misinfo weighting
        W_new = Ws.copy()
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                weighted = Ws[i]
            else:
                nbrs = np.array(nbrs)
                mis_nbrs = (
                    np.intersect1d(nbrs, misinfo_set)
                    if len(misinfo_set) > 0
                    else np.array([], dtype=int)
                )
                reg_nbrs = np.setdiff1d(nbrs, mis_nbrs)
                mean_reg = (
                    Ws[reg_nbrs].mean(axis=0) if reg_nbrs.size > 0 else np.zeros(d)
                )
                mean_mis = (
                    Ws[mis_nbrs].mean(axis=0) if mis_nbrs.size > 0 else np.zeros(d)
                )
                if (ci >= 6) and (ci <= (tstop - 1)) and (mis_nbrs.size > 0):
                    if reg_nbrs.size > 0:
                        weighted = (mean_reg + gamma * mean_mis) / (1.0 + gamma)
                    else:
                        weighted = mean_mis
                else:
                    weighted = (
                        mean_reg
                        if reg_nbrs.size > 0
                        else (mean_mis if mis_nbrs.size > 0 else Ws[i])
                    )
            alpha_i = float(alphas[i])
            W_new[i] = (1 - alpha_i) * Ws[i] + alpha_i * weighted
        Ws = W_new

        # value signals
        c_i = p_mat.var(axis=0)
        p_i = np.zeros(N)
        r_i = np.zeros(N)
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            nbrs = neighbors[i]
            if len(nbrs) == 0 or wi_norm == 0:
                p_i[i] = 0.0
            else:
                mean_nb = Ws[nbrs].mean(axis=0)
                p_i[i] = 1.0 - (np.linalg.norm(wi - mean_nb) / (wi_norm + 1e-9))
            r_i[i] = 1.0 - abs(per_agent_acc.mean() - theta)
        V_i = np.clip((u * np.vstack([c_i, p_i, r_i]).T).sum(axis=1) / 3.0, 0.0, 1.0)

        D_t = float(
            np.clip(1.0 - (abs(per_agent_acc.std() - sigma_opt) / sigma_opt), 0.0, 1.0)
        )
        T_t = float(per_agent_acc.mean())
        V_t = float(np.mean(V_i))
        M_t = wT * T_t + wD * D_t + wV * V_t

        T_arr[ci] = T_t
        D_arr[ci] = D_t
        V_arr[ci] = V_t
        M_arr[ci] = M_t

        # evolution step if enabled
        if evolve_mode:
            # compute schedule eta and sigma for this cycle index (1-based t)
            t = ci + 1
            eta_t = eta_bump if (bump_start <= t <= bump_end) else eta_base
            if t <= sigma_hold_until:
                sigma_t = sigma_hi
            else:
                sigma_t = sigma_hi - (
                    (t - sigma_hold_until)
                    * (sigma_hi - sigma_lo)
                    / (cycles - sigma_hold_until)
                )
            # selection
            pay = V_i.copy()
            q = int(np.ceil(q_frac * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            u = (1 - eta_t) * u + eta_t * u_success[None, :]
            # mutation (annealed)
            u = u + rng.normal(0.0, sigma_t, size=u.shape)
            # curiosity floor
            u[:, 0] = np.maximum(u[:, 0], curiosity_floor)
            u = np.clip(u, 0.0, 1.0)
            if simplex:
                s = u.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                u = u / s

    # phases indices
    w1_idx = list(range(0, 6))
    att_idx = list(range(6, tstop))
    rec_idx = list(range(tstop, cycles))
    rng_bs = np.random.default_rng(int(seed) + 12345)
    M_w1_mean, M_w1_lo, M_w1_hi = bootstrap_mean_ci(
        M_arr[w1_idx], n_boot=500, rng=rng_bs
    )
    M_att_mean, M_att_lo, M_att_hi = bootstrap_mean_ci(
        M_arr[att_idx], n_boot=500, rng=rng_bs
    )
    M_rec_mean, M_rec_lo, M_rec_hi = bootstrap_mean_ci(
        M_arr[rec_idx], n_boot=500, rng=rng_bs
    )

    DeltaM = M_rec_mean - M_w1_mean
    A_M = (
        float(np.sum(np.maximum(0.0, M_w1_mean - M_arr[att_idx])))
        if len(att_idx) > 0
        else 0.0
    )
    EM = float((M_rec_mean - M_att_mean) / (M_w1_mean if M_w1_mean != 0 else 1.0))
    R_M = float(M_rec_mean / (M_w1_mean if M_w1_mean != 0 else 1.0))
    if len(rec_idx) > 0:
        rec_accs = [accs_store[i] for i in rec_idx]
        T_rec = float(np.mean([np.mean(a) for a in rec_accs]))
        disp_rec = float(np.mean([np.std(a) for a in rec_accs]))
    else:
        T_rec = 0.0
        disp_rec = 0.0

    summary = {
        "M_w1": M_w1_mean,
        "M_w1_lo": M_w1_lo,
        "M_w1_hi": M_w1_hi,
        "M_att": M_att_mean,
        "M_att_lo": M_att_lo,
        "M_att_hi": M_att_hi,
        "M_rec": M_rec_mean,
        "M_rec_lo": M_rec_lo,
        "M_rec_hi": M_rec_hi,
        "DeltaM": DeltaM,
        "A_M": A_M,
        "EM": EM,
        "R_M": R_M,
        "T_rec": T_rec,
        "disp_rec": disp_rec,
    }
    diagnostics = {
        "T_arr": T_arr.tolist(),
        "D_arr": D_arr.tolist(),
        "V_arr": V_arr.tolist(),
        "M_arr": M_arr.tolist(),
        "accs_store": [a.tolist() for a in accs_store],
    }
    return summary, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--topology", type=str, default="ws")
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--fast", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.40)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--frac", type=float, default=0.11)
    parser.add_argument("--tstop", type=int, default=10)
    parser.add_argument("--wT", type=float, default=0.4)
    parser.add_argument("--wD", type=float, default=0.3)
    parser.add_argument("--wV", type=float, default=0.3)
    parser.add_argument("--eta_base", type=float, default=0.18)
    parser.add_argument("--eta_bump", type=float, default=0.22)
    parser.add_argument("--bump_start", type=int, default=8)
    parser.add_argument("--bump_end", type=int, default=11)
    parser.add_argument("--sigma_hi", type=float, default=0.06)
    parser.add_argument("--sigma_lo", type=float, default=0.03)
    parser.add_argument("--sigma_hold_until", type=int, default=10)
    parser.add_argument("--q_frac", type=float, default=0.20)
    parser.add_argument("--curiosity_floor", type=float, default=0.25)
    parser.add_argument("--simplex", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    N = args.agents
    topology = args.topology
    cycles = args.cycles
    trials = args.trials
    theta = args.theta
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    beta = args.beta
    gamma = args.gamma
    frac = args.frac
    tstop = args.tstop
    wT = args.wT
    wD = args.wD
    wV = args.wV
    eta_base = args.eta_base
    eta_bump = args.eta_bump
    bump_start = args.bump_start
    bump_end = args.bump_end
    sigma_hi = args.sigma_hi
    sigma_lo = args.sigma_lo
    sigma_hold_until = args.sigma_hold_until
    q_frac = args.q_frac
    curiosity_floor = args.curiosity_floor
    simplex = bool(args.simplex)

    trials_run = trials
    if args.fast == 1:
        trials_run = max(1000, trials // 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"meaning_phase5_3_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # baseline
    print("Running baseline (fixed values)...")
    baseline_summary, baseline_diag = run_mode(
        seed,
        N,
        topology,
        trials_run,
        cycles,
        theta,
        alpha_min,
        alpha_max,
        beta,
        gamma,
        frac,
        tstop,
        False,
        {},
        wT,
        wD,
        wV,
        simplex=simplex,
    )
    pd.DataFrame([baseline_summary]).to_csv(
        os.path.join(outdir, "summary_baseline.csv"), index=False
    )

    # annealed+floor
    evolve_settings = {
        "q_frac": q_frac,
        "curiosity_floor": curiosity_floor,
        "eta_base": eta_base,
        "eta_bump": eta_bump,
        "bump_start": bump_start,
        "bump_end": bump_end,
        "sigma_hi": sigma_hi,
        "sigma_lo": sigma_lo,
        "sigma_hold_until": sigma_hold_until,
    }
    print("Running annealed+floor evolution mode...")
    anneal_summary, anneal_diag = run_mode(
        seed + 1,
        N,
        topology,
        trials_run,
        cycles,
        theta,
        alpha_min,
        alpha_max,
        beta,
        gamma,
        frac,
        tstop,
        True,
        evolve_settings,
        wT,
        wD,
        wV,
        simplex=simplex,
    )
    pd.DataFrame([anneal_summary]).to_csv(
        os.path.join(outdir, "summary_anneal.csv"), index=False
    )

    # comparison
    compare = {"baseline": baseline_summary, "anneal": anneal_summary}
    with open(os.path.join(outdir, "compare_summary.csv"), "w") as f:
        f.write("mode,")
        f.write(",".join(baseline_summary.keys()) + "\n")
        f.write(
            "baseline," + ",".join([str(x) for x in baseline_summary.values()]) + "\n"
        )
        f.write("anneal," + ",".join([str(x) for x in anneal_summary.values()]) + "\n")

    # feasibility & win
    feasible = True
    if anneal_summary["T_rec"] < (baseline_summary["T_rec"] - 0.03):
        feasible = False
    if not (0.02 <= anneal_summary["disp_rec"] <= 0.12):
        feasible = False
    if anneal_summary["A_M"] > baseline_summary["A_M"]:
        feasible = False
    win = feasible and (anneal_summary["DeltaM"] >= baseline_summary["DeltaM"] + 0.03)

    # plots
    cycles_idx = np.arange(1, cycles + 1)
    # curves: both modes
    plt.figure(figsize=(9, 5))
    plt.plot(cycles_idx, baseline_diag["T_arr"], marker="o", label="T_baseline")
    plt.plot(cycles_idx, baseline_diag["D_arr"], marker="o", label="D_baseline")
    plt.plot(cycles_idx, baseline_diag["V_arr"], marker="o", label="V_baseline")
    plt.plot(cycles_idx, baseline_diag["M_arr"], marker="o", label="M_baseline")
    plt.plot(
        cycles_idx, anneal_diag["T_arr"], marker="x", linestyle="--", label="T_anneal"
    )
    plt.plot(
        cycles_idx, anneal_diag["D_arr"], marker="x", linestyle="--", label="D_anneal"
    )
    plt.plot(
        cycles_idx, anneal_diag["V_arr"], marker="x", linestyle="--", label="V_anneal"
    )
    plt.plot(
        cycles_idx, anneal_diag["M_arr"], marker="x", linestyle="--", label="M_anneal"
    )
    plt.axvline(x=6.5, color="k", linestyle=":")
    plt.axvline(x=tstop + 0.0, color="k", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_truth_div_val_meaning.png"), dpi=150)
    plt.close()

    # values trajectory for anneal: compute per-cycle u mean ±1sd by re-running anneal with instrumentation
    # We can reuse anneal_diag (which currently lacks per-cycle u); to keep it simple and light, re-run a single instrumented anneal with same seed
    def run_anneal_instrumented():
        s = seed + 2
        rng = np.random.default_rng(int(s))
        per_cycle = max(1, trials_run // cycles)
        base = np.linspace(0.2, 0.8, cycles)
        cycle_ps = []
        for i in range(cycles):
            pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
            pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
            pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
            cycle_ps.append((pA, pB, pC))
        adj = build_topology(N, topology, rng)
        neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]
        d = 6
        Ws = np.zeros((N, d))
        u = np.random.default_rng(int(s + 1)).uniform(0.2, 0.8, size=(N, 3))
        u_means = []
        u_sds = []
        for ci, (pA, pB, pC) in enumerate(cycle_ps):
            if ci < 6:
                A, B, C, S, Y = simulate_W1(per_cycle, pA, pB, pC, rng)
            else:
                A, B, C, S, Y = simulate_W2(per_cycle, pA, pB, pC, rng)
            X = design_matrix(A, B, C)
            p_mat = np.zeros((per_cycle, N))
            per_agent_acc = np.zeros(N)
            for i in range(N):
                p = 1.0 / (1.0 + np.exp(-X.dot(Ws[i])))
                p_mat[:, i] = p
                yhat = (p >= 0.5).astype(int)
                per_agent_acc[i] = float((yhat == Y).mean())
                grad = X.T.dot(p - Y) / X.shape[0]
                Ws[i] = Ws[i] - 0.3 * grad
            # misinfo + consensus omitted for instrumented trajectory (approximate)
            # evolve step with schedules
            t = ci + 1
            eta_t = eta_bump if (bump_start <= t <= bump_end) else eta_base
            if t <= sigma_hold_until:
                sigma_t = sigma_hi
            else:
                sigma_t = sigma_hi - (
                    (t - sigma_hold_until)
                    * (sigma_hi - sigma_lo)
                    / (cycles - sigma_hold_until)
                )
            pay = np.zeros(N)
            c_i = p_mat.var(axis=0)
            p_i = np.zeros(N)
            r_i = np.zeros(N)
            for i in range(N):
                wi = Ws[i]
                wi_norm = np.linalg.norm(wi)
                if len(neighbors[i]) == 0 or wi_norm == 0:
                    p_i[i] = 0
                else:
                    mean_nb = Ws[neighbors[i]].mean(axis=0)
                    p_i[i] = 1.0 - (np.linalg.norm(wi - mean_nb) / (wi_norm + 1e-9))
                r_i[i] = 1.0 - abs(per_agent_acc.mean() - theta)
            V_i = np.clip((u * np.vstack([c_i, p_i, r_i]).T).sum(axis=1) / 3.0, 0, 1)
            pay = V_i
            q = int(np.ceil(q_frac * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            u = (1 - eta_t) * u + eta_t * u_success[None, :]
            u = u + np.random.default_rng(int(s + ci)).normal(
                0.0, sigma_t, size=u.shape
            )
            u[:, 0] = np.maximum(u[:, 0], curiosity_floor)
            u = np.clip(u, 0, 1)
            if simplex:
                ssum = u.sum(axis=1, keepdims=True)
                ssum[ssum == 0] = 1.0
                u = u / ssum
            u_means.append(u.mean(axis=0))
            u_sds.append(u.std(axis=0))
        return np.array(u_means), np.array(u_sds)

    u_means, u_sds = run_anneal_instrumented()
    # plot values trajectory
    plt.figure(figsize=(8, 4))
    x = np.arange(1, cycles + 1)
    plt.plot(x, u_means[:, 0], label="curiosity_mean")
    plt.fill_between(
        x, u_means[:, 0] - u_sds[:, 0], u_means[:, 0] + u_sds[:, 0], alpha=0.2
    )
    plt.plot(x, u_means[:, 1], label="cooperation_mean")
    plt.fill_between(
        x, u_means[:, 1] - u_sds[:, 1], u_means[:, 1] + u_sds[:, 1], alpha=0.2
    )
    plt.plot(x, u_means[:, 2], label="coherence_mean")
    plt.fill_between(
        x, u_means[:, 2] - u_sds[:, 2], u_means[:, 2] + u_sds[:, 2], alpha=0.2
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "values_trajectory_anneal.png"), dpi=150)
    plt.close()

    # schedule plot
    x = np.arange(1, cycles + 1)
    eta_sched = [eta_bump if (bump_start <= t <= bump_end) else eta_base for t in x]
    sigma_sched = [
        (
            sigma_hi
            if t <= sigma_hold_until
            else (
                sigma_hi
                - (
                    (t - sigma_hold_until)
                    * (sigma_hi - sigma_lo)
                    / (cycles - sigma_hold_until)
                )
            )
        )
        for t in x
    ]
    plt.figure(figsize=(6, 3))
    plt.plot(x, eta_sched, label="eta_sel(t)")
    plt.plot(x, sigma_sched, label="sigma_mut(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "schedule_eta_sigma.png"), dpi=150)
    plt.close()

    # bars by phase
    def phase_mean_ci(arr):
        rngp = np.random.default_rng(2025)
        m, lo, hi = bootstrap_mean_ci(arr, n_boot=500, rng=rngp)
        return m, lo, hi

    baseline_w1 = np.array(baseline_diag["M_arr"][:6])
    baseline_att = np.array(baseline_diag["M_arr"][6:tstop])
    baseline_rec = np.array(baseline_diag["M_arr"][tstop:])
    anneal_w1 = np.array(anneal_diag["M_arr"][:6])
    anneal_att = np.array(anneal_diag["M_arr"][6:tstop])
    anneal_rec = np.array(anneal_diag["M_arr"][tstop:])
    b_m = [
        float(np.mean(baseline_w1)),
        float(np.mean(baseline_att)) if baseline_att.size > 0 else 0.0,
        float(np.mean(baseline_rec)) if baseline_rec.size > 0 else 0.0,
    ]
    a_m = [
        float(np.mean(anneal_w1)),
        float(np.mean(anneal_att)) if anneal_att.size > 0 else 0.0,
        float(np.mean(anneal_rec)) if anneal_rec.size > 0 else 0.0,
    ]
    # simple CIs using bootstrap
    b_cis = [
        bootstrap_mean_ci(baseline_w1)[1:],
        bootstrap_mean_ci(baseline_att)[1:],
        bootstrap_mean_ci(baseline_rec)[1:],
    ]
    a_cis = [
        bootstrap_mean_ci(anneal_w1)[1:],
        bootstrap_mean_ci(anneal_att)[1:],
        bootstrap_mean_ci(anneal_rec)[1:],
    ]
    labels = ["W1", "Attack", "Recovery"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    # prepare symmetric yerr arrays: shape (2, n)
    b_lower = np.array([bm - lo for (lo, hi), bm in zip(b_cis, b_m)])
    b_upper = np.array([hi - bm for (lo, hi), bm in zip(b_cis, b_m)])
    a_lower = np.array([am - lo for (lo, hi), am in zip(a_cis, a_m)])
    a_upper = np.array([hi - am for (lo, hi), am in zip(a_cis, a_m)])
    b_yerr = np.vstack([b_lower, b_upper])
    a_yerr = np.vstack([a_lower, a_upper])
    plt.bar(x - width / 2, b_m, width, yerr=b_yerr, label="baseline")
    plt.bar(x + width / 2, a_m, width, yerr=a_yerr, label="anneal")
    plt.xticks(x, labels)
    plt.ylabel("M mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bars_meaning_by_phase.png"), dpi=150)
    plt.close()

    # dispersion compare: plot mean std of accs per cycle for both
    base_disp = [np.std(a) for a in baseline_diag["accs_store"]]
    ann_disp = [np.std(a) for a in anneal_diag["accs_store"]]
    x = np.arange(1, cycles + 1)
    plt.figure(figsize=(8, 3))
    plt.plot(x, base_disp, label="disp_baseline")
    plt.plot(x, ann_disp, label="disp_anneal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dispersion_consensus_compare.png"), dpi=150)
    plt.close()

    # README + params
    readme = []
    readme.append("# Meaning Phase5.3 — Anneal + Curiosity Floor")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(
        f'Baseline DeltaM: {baseline_summary["DeltaM"]:.4f} | Anneal DeltaM: {anneal_summary["DeltaM"]:.4f}'
    )
    readme.append(f"Feasible: {feasible} | Win (ΔM ≥ baseline+0.03 & feasible): {win}")
    readme.append(
        "Files: summary_baseline.csv, summary_anneal.csv, compare_summary.csv, plots..."
    )
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "seed": seed,
        "N": N,
        "topology": topology,
        "cycles": cycles,
        "trials": trials_run,
        "fast": args.fast,
        "eta_base": eta_base,
        "eta_bump": eta_bump,
        "bump_start": bump_start,
        "bump_end": bump_end,
        "sigma_hi": sigma_hi,
        "sigma_lo": sigma_lo,
        "sigma_hold_until": sigma_hold_until,
        "q_frac": q_frac,
        "curiosity_floor": curiosity_floor,
        "wT": wT,
        "wD": wD,
        "wV": wV,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # write compare CSV
    df = pd.DataFrame(
        [{"mode": "baseline", **baseline_summary}, {"mode": "anneal", **anneal_summary}]
    )
    df.to_csv(os.path.join(outdir, "compare_summary.csv"), index=False)

    # console summary
    print(
        f"Baseline: M W1/Atk/Rec = {baseline_summary['M_w1']:.3f}/{baseline_summary['M_att']:.3f}/{baseline_summary['M_rec']:.3f} | ΔM={baseline_summary['DeltaM']:.4f} | A_M={baseline_summary['A_M']:.4f}"
    )
    print(
        f"Annealed: M W1/Atk/Rec = {anneal_summary['M_w1']:.3f}/{anneal_summary['M_att']:.3f}/{anneal_summary['M_rec']:.3f} | ΔM={anneal_summary['DeltaM']:.4f} | A_M={anneal_summary['A_M']:.4f} | feasible={feasible} | win={win}"
    )
    print(f"Results folder: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()

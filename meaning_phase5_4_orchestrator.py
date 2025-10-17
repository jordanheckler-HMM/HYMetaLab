#!/usr/bin/env python3
"""
Phase 5.4 Orchestrator — compare baseline, anneal_v1 (5.3), anneal_v2 (new schedules)
Saves results to discovery_results/meaning_phase5_4_<stamp> and copies to /mnt/data if available.
"""
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Lightweight bootstrap
def bootstrap_mean_ci(data, n_boot=300, rng=None):
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


# design matrix and scores (copied)
def design_matrix(A, B, C):
    ones = np.ones_like(A, dtype=float)
    AB = (A * B).astype(float)
    BC = (B * C).astype(float)
    return np.vstack(
        [ones, A.astype(float), B.astype(float), C.astype(float), AB, BC]
    ).T


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


# Core single-run engine (adapted from 5.3) with variant selection
def run_variant(
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
    mode,
    variant,
    params,
):
    """
    mode: 'baseline' (no evolution), 'anneal_v1', 'anneal_v2'
    variant-specific schedules applied inside.
    Returns summary dict and diagnostics (per-cycle arrays, accs_store, u trajectory if instrumented)
    """
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

    # default evolve params (from 5.3)
    eta_base = params.get("eta_base", 0.18)
    eta_bump = params.get("eta_bump", 0.22)
    bump_start = params.get("bump_start", 8)
    bump_end = params.get("bump_end", 11)
    sigma_hi = params.get("sigma_hi", 0.06)
    sigma_lo = params.get("sigma_lo", 0.03)
    sigma_hold_until = params.get("sigma_hold_until", 10)
    q_frac = params.get("q_frac", 0.20)
    curiosity_floor = params.get("curiosity_floor", 0.25)
    simplex = bool(params.get("simplex", False))

    # for v2 override schedules
    def eta_t_for_variant(t):
        # t is 1-based
        if variant == "anneal_v2":
            if 1 <= t <= 7:
                return 0.18
            elif 8 <= t <= 11:
                return 0.22
            elif 12 <= t <= 14:
                return 0.20
            else:
                return 0.18
        else:
            # anneal_v1 or baseline
            return eta_bump if (bump_start <= t <= bump_end) else eta_base

    def sigma_t_for_variant(t):
        if variant == "anneal_v2":
            if 1 <= t <= 10:
                return 0.06
            elif 11 <= t <= 12:
                # linear anneal from 0.06 to 0.04 over t=11..12
                return 0.06 - ((t - 11) * (0.06 - 0.04) / (2 - 1))
            else:
                return 0.04
        else:
            # anneal_v1 schedule from 5.3
            if t <= sigma_hold_until:
                return sigma_hi
            else:
                return sigma_hi - (
                    (t - sigma_hold_until)
                    * (sigma_hi - sigma_lo)
                    / (cycles - sigma_hold_until)
                )

    # storage
    T_arr = np.zeros(cycles)
    D_arr = np.zeros(cycles)
    V_arr = np.zeros(cycles)
    M_arr = np.zeros(cycles)
    accs_store = []
    u_store = []

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

        # misinfo
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

        # consensus update
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
        M_t = (
            params.get("wT", 0.4) * T_t
            + params.get("wD", 0.3) * D_t
            + params.get("wV", 0.3) * V_t
        )

        T_arr[ci] = T_t
        D_arr[ci] = D_t
        V_arr[ci] = V_t
        M_arr[ci] = M_t

        # evolution step depending on variant
        if mode != "baseline":
            t = ci + 1
            if variant == "anneal_v1":
                eta_t = eta_bump if (bump_start <= t <= bump_end) else eta_base
                if t <= sigma_hold_until:
                    sigma_t = sigma_hi
                else:
                    sigma_t = sigma_hi - (
                        (t - sigma_hold_until)
                        * (sigma_hi - sigma_lo)
                        / (cycles - sigma_hold_until)
                    )
                cur_floor = curiosity_floor
            elif variant == "anneal_v2":
                eta_t = eta_t_for_variant(t)
                sigma_t = sigma_t_for_variant(t)
                cur_floor = params.get("curiosity_floor_v2", 0.28)
            else:
                eta_t = eta_base
                sigma_t = sigma_hi
                cur_floor = curiosity_floor

            pay = V_i.copy()
            q = int(np.ceil(q_frac * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            u = (1 - eta_t) * u + eta_t * u_success[None, :]
            u = u + rng.normal(0.0, sigma_t, size=u.shape)
            # curiosity floor enforcement
            u[:, 0] = np.maximum(u[:, 0], cur_floor)
            u = np.clip(u, 0.0, 1.0)
            if simplex:
                s = u.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                u = u / s
        # store u means for plotting
        u_store.append(u.mean(axis=0))

    # compute phase stats
    w1_idx = list(range(0, 6))
    att_idx = list(range(6, tstop))
    rec_idx = list(range(tstop, cycles))
    rng_bs = np.random.default_rng(int(seed) + 2025)
    M_w1_mean, M_w1_lo, M_w1_hi = bootstrap_mean_ci(
        M_arr[w1_idx], n_boot=300, rng=rng_bs
    )
    M_att_mean, M_att_lo, M_att_hi = bootstrap_mean_ci(
        M_arr[att_idx], n_boot=300, rng=rng_bs
    )
    M_rec_mean, M_rec_lo, M_rec_hi = bootstrap_mean_ci(
        M_arr[rec_idx], n_boot=300, rng=rng_bs
    )

    DeltaM = M_rec_mean - M_w1_mean
    A_M = (
        float(np.sum(np.maximum(0.0, M_w1_mean - M_arr[att_idx])))
        if len(att_idx) > 0
        else 0.0
    )
    EM = float((M_rec_mean - M_att_mean) / (M_w1_mean if M_w1_mean != 0 else 1.0))
    R_M = float(M_rec_mean / (M_w1_mean if M_w1_mean != 0 else 1.0))
    T_rec = (
        float(np.mean([np.mean(a) for a in [accs_store[i] for i in rec_idx]]))
        if len(rec_idx) > 0
        else 0.0
    )
    disp_rec_mean = (
        float(np.mean([np.std(a) for a in [accs_store[i] for i in rec_idx]]))
        if len(rec_idx) > 0
        else 0.0
    )
    disp_rec_max = (
        float(np.max([np.std(a) for a in [accs_store[i] for i in rec_idx]]))
        if len(rec_idx) > 0
        else 0.0
    )

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
        "disp_rec_mean": disp_rec_mean,
        "disp_rec_max": disp_rec_max,
    }
    diagnostics = {
        "T_arr": T_arr.tolist(),
        "D_arr": D_arr.tolist(),
        "V_arr": V_arr.tolist(),
        "M_arr": M_arr.tolist(),
        "accs_store": [a.tolist() for a in accs_store],
        "u_store": np.array(u_store).tolist(),
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
    parser.add_argument("--simplex", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    N = args.agents
    topology = args.topology
    cycles = args.cycles
    trials = args.trials
    fast = args.fast
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
    simplex = bool(args.simplex)

    trials_run = trials
    if fast == 1:
        trials_run = max(1000, trials // 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"meaning_phase5_4_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    mnt_out = "/mnt/data"
    if not os.path.exists(mnt_out):
        try:
            os.makedirs(mnt_out, exist_ok=True)
        except Exception:
            mnt_out = None

    # params defaults (v1 = phase5.3 defaults)
    params = {
        "eta_base": 0.18,
        "eta_bump": 0.22,
        "bump_start": 8,
        "bump_end": 11,
        "sigma_hi": 0.06,
        "sigma_lo": 0.03,
        "sigma_hold_until": 10,
        "q_frac": 0.20,
        "curiosity_floor": 0.25,
        "wT": wT,
        "wD": wD,
        "wV": wV,
        "simplex": simplex,
    }

    # run baseline
    print("Running baseline...")
    base_summary, base_diag = run_variant(
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
        "baseline",
        "baseline",
        params,
    )
    pd.DataFrame([base_summary]).to_csv(
        os.path.join(outdir, "summary_baseline_v54.csv"), index=False
    )

    # run anneal_v1 (same as 5.3)
    print("Running anneal_v1 (v5.3)...")
    v1_summary, v1_diag = run_variant(
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
        "anneal",
        "anneal_v1",
        params,
    )
    pd.DataFrame([v1_summary]).to_csv(
        os.path.join(outdir, "summary_anneal_v1.csv"), index=False
    )

    # run anneal_v2 with overrides
    params_v2 = params.copy()
    params_v2["curiosity_floor_v2"] = 0.28
    print("Running anneal_v2 (new schedules)...")
    v2_summary, v2_diag = run_variant(
        seed + 2,
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
        "anneal",
        "anneal_v2",
        params_v2,
    )
    pd.DataFrame([v2_summary]).to_csv(
        os.path.join(outdir, "summary_anneal_v2.csv"), index=False
    )

    # aggregate summaries into CSV
    df = pd.DataFrame(
        [
            {"variant": "baseline_fixed", **base_summary},
            {"variant": "anneal_v1", **v1_summary},
            {"variant": "anneal_v2", **v2_summary},
        ]
    )
    df.to_csv(os.path.join(outdir, "summary_aggregate_v54.csv"), index=False)

    # compute guardrail checks and PASS/FAIL
    # Guardrails: T_rec_v >= T_rec_baseline AND disp_rec_mean in [0.02,0.12]
    def check_guards(summ, base):
        t_ok = (summ["T_rec"] - base["T_rec"]) >= 0.0
        disp_ok = 0.02 <= summ["disp_rec_mean"] <= 0.12
        return t_ok and disp_ok

    v2_guards = check_guards(v2_summary, base_summary)
    v2_deltaM_ok = (v2_summary["DeltaM"] - base_summary["DeltaM"]) >= 0.03
    pass_v2 = v2_guards and v2_deltaM_ok

    # produce plots requested
    cycles_idx = np.arange(1, cycles + 1)
    # bars_meaning_by_phase_v2
    labels = ["W1", "Attack", "Recovery"]
    base_vals = [
        np.mean(base_diag["M_arr"][:6]),
        np.mean(base_diag["M_arr"][6:tstop]),
        np.mean(base_diag["M_arr"][tstop:]),
    ]
    v1_vals = [
        np.mean(v1_diag["M_arr"][:6]),
        np.mean(v1_diag["M_arr"][6:tstop]),
        np.mean(v1_diag["M_arr"][tstop:]),
    ]
    v2_vals = [
        np.mean(v2_diag["M_arr"][:6]),
        np.mean(v2_diag["M_arr"][6:tstop]),
        np.mean(v2_diag["M_arr"][tstop:]),
    ]
    # CIs
    base_cis = [
        bootstrap_mean_ci(base_diag["M_arr"][:6])[1:],
        bootstrap_mean_ci(base_diag["M_arr"][6:tstop])[1:],
        bootstrap_mean_ci(base_diag["M_arr"][tstop:])[1:],
    ]
    v1_cis = [
        bootstrap_mean_ci(v1_diag["M_arr"][:6])[1:],
        bootstrap_mean_ci(v1_diag["M_arr"][6:tstop])[1:],
        bootstrap_mean_ci(v1_diag["M_arr"][tstop:])[1:],
    ]
    v2_cis = [
        bootstrap_mean_ci(v2_diag["M_arr"][:6])[1:],
        bootstrap_mean_ci(v2_diag["M_arr"][6:tstop])[1:],
        bootstrap_mean_ci(v2_diag["M_arr"][tstop:])[1:],
    ]

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 4))

    def cis_to_yerr(cis, means):
        lowers = np.array([m - lo for (lo, hi), m in zip(cis, means)])
        uppers = np.array([hi - m for (lo, hi), m in zip(cis, means)])
        return np.vstack([lowers, uppers])

    base_yerr = cis_to_yerr(base_cis, base_vals)
    v1_yerr = cis_to_yerr(v1_cis, v1_vals)
    v2_yerr = cis_to_yerr(v2_cis, v2_vals)
    plt.bar(x - width, base_vals, width, yerr=base_yerr, label="baseline")
    plt.bar(x, v1_vals, width, yerr=v1_yerr, label="anneal_v1")
    plt.bar(x + width, v2_vals, width, yerr=v2_yerr, label="anneal_v2")
    plt.xticks(x, labels)
    plt.ylabel("M mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bars_meaning_by_phase_v2.png"), dpi=150)
    plt.close()

    # curves plot (T,D,V,M)
    plt.figure(figsize=(10, 5))
    plt.plot(cycles_idx, base_diag["T_arr"], label="T_baseline", color="C0")
    plt.plot(cycles_idx, base_diag["D_arr"], label="D_baseline", color="C1")
    plt.plot(cycles_idx, base_diag["V_arr"], label="V_baseline", color="C2")
    plt.plot(cycles_idx, base_diag["M_arr"], label="M_baseline", color="C3")
    # v2 dashed
    plt.plot(cycles_idx, v2_diag["T_arr"], label="T_v2", linestyle="--", color="C0")
    plt.plot(cycles_idx, v2_diag["D_arr"], label="D_v2", linestyle="--", color="C1")
    plt.plot(cycles_idx, v2_diag["V_arr"], label="V_v2", linestyle="--", color="C2")
    plt.plot(cycles_idx, v2_diag["M_arr"], label="M_v2", linestyle="--", color="C3")
    plt.axvline(x=6.5, color="k", linestyle=":")
    plt.axvline(x=tstop + 0.0, color="k", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_truth_div_val_meaning_v2.png"), dpi=150)
    plt.close()

    # dispersion compare v2
    base_disp = [np.std(a) for a in base_diag["accs_store"]]
    v2_disp = [np.std(a) for a in v2_diag["accs_store"]]
    plt.figure(figsize=(8, 3))
    plt.plot(cycles_idx, base_disp, label="disp_baseline")
    plt.plot(cycles_idx, v2_disp, label="disp_v2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dispersion_consensus_compare_v2.png"), dpi=150)
    plt.close()

    # values trajectory anneal_v2
    u_means = np.array(v2_diag["u_store"])
    u_sds = np.std(u_means, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(cycles_idx, u_means[:, 0], label="curiosity_mean")
    plt.fill_between(
        cycles_idx,
        u_means[:, 0] - np.std(u_means[:, 0]),
        u_means[:, 0] + np.std(u_means[:, 0]),
        alpha=0.2,
    )
    plt.plot(cycles_idx, u_means[:, 1], label="cooperation_mean")
    plt.fill_between(
        cycles_idx,
        u_means[:, 1] - np.std(u_means[:, 1]),
        u_means[:, 1] + np.std(u_means[:, 1]),
        alpha=0.2,
    )
    plt.plot(cycles_idx, u_means[:, 2], label="coherence_mean")
    plt.fill_between(
        cycles_idx,
        u_means[:, 2] - np.std(u_means[:, 2]),
        u_means[:, 2] + np.std(u_means[:, 2]),
        alpha=0.2,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "values_trajectory_anneal_v2.png"), dpi=150)
    plt.close()

    # schedule plot for v2
    x = cycles_idx
    eta_sched = [
        (
            0.18
            if 1 <= t <= 7
            else (0.22 if 8 <= t <= 11 else (0.20 if 12 <= t <= 14 else 0.18))
        )
        for t in x
    ]
    sigma_sched = [
        (
            0.06
            if 1 <= t <= 10
            else (
                0.06 - ((t - 11) * (0.06 - 0.04) / (2 - 1)) if 11 <= t <= 12 else 0.04
            )
        )
        for t in x
    ]
    plt.figure(figsize=(6, 3))
    plt.plot(x, eta_sched, label="eta_sel(t) v2")
    plt.plot(x, sigma_sched, label="sigma_mut(t) v2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "schedule_eta_sigma_v2.png"), dpi=150)
    plt.close()

    # write summary CSV for anneal_v2
    pd.DataFrame([v2_summary]).to_csv(
        os.path.join(outdir, "summary_anneal_v2.csv"), index=False
    )

    # README line and decision
    deltaM_gap = v2_summary["DeltaM"] - base_summary["DeltaM"]
    trec_delta = v2_summary["T_rec"] - base_summary["T_rec"]
    readme = f"PASS={pass_v2} | DeltaM_gap={deltaM_gap:.4f} | T_rec_delta={trec_delta:.4f} | Disp_rec_mean={v2_summary['disp_rec_mean']:.4f} | Disp_rec_max={v2_summary['disp_rec_max']:.4f}"
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(readme + "\n")

    # copy artifacts to /mnt/data if available
    if mnt_out:
        import shutil

        dest = os.path.join(mnt_out, os.path.basename(outdir))
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(outdir, dest)

    # console table
    def short_row(name, s):
        return f"{name:14s} {s['M_w1']:.3f}   {s['M_att']:.3f}     {s['M_rec']:.3f}   {s['DeltaM']:.3f}   {s['T_rec']:.3f}   {s['disp_rec_mean']:.3f}/{s['disp_rec_max']:.3f}"

    print(
        "\nVariant         M_W1   M_Attack   M_Recovery   ΔM   T_rec   Disp(Rec mean/max)"
    )
    print(short_row("baseline_fixed", base_summary))
    print(short_row("anneal_v1", v1_summary))
    print(short_row("anneal_v2", v2_summary))

    decision = "PASS" if pass_v2 else "FAIL"
    reason = ""
    if not v2_guards:
        reason = "Guardrails failed (T_rec or dispersion out of range)"
    elif not v2_deltaM_ok:
        reason = "ΔM improvement < 0.03"
    print(f"\nDecision: {decision}. Reason: {reason}")
    if not pass_v2:
        # suggest tweak
        if not v2_deltaM_ok:
            suggest = "Consider reducing sigma_floor (e.g. to 0.03) or increasing eta_tail by +0.01"
        else:
            suggest = "Consider lowering curiosity_floor slightly (e.g. -0.01) to increase T_rec"
        print("Next tweak suggestion:", suggest)

    print("Results folder:", os.path.abspath(outdir))
    if mnt_out:
        print(
            "Also copied to /mnt/data at",
            os.path.join(mnt_out, os.path.basename(outdir)),
        )


if __name__ == "__main__":
    main()

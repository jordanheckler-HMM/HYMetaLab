#!/usr/bin/env python3
"""
Meaning Phase 5.2 — Sweep η_sel, σ_mut, q_frac
Runs a baseline fixed-values run, then sweeps evolution hyperparameters and compares results.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Lightweight bootstrap CI for mean
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


# World and topology helpers (copied/adapted)
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


# simulate worlds
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


# single-run engine reusing Phase5.1 logic but minimal
def run_single(
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
    evolve_params,
    wT,
    wD,
    wV,
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
    # initialize u
    u = rng.uniform(0.2, 0.8, size=(N, 3))

    # adaptive alpha
    alphas = np.full(N, (alpha_min + alpha_max) / 2.0)
    tau = 0.0
    k_up = 1.2
    k_down = 0.05
    lambda_mix = 0.5

    sigma_opt = 0.07

    m = int(round(frac * N))
    misinfo_set = []

    # storage
    T_t = np.zeros(cycles, dtype=float)
    D_t = np.zeros(cycles, dtype=float)
    V_t = np.zeros(cycles, dtype=float)
    M_t = np.zeros(cycles, dtype=float)
    group_mean_acc = np.zeros(cycles, dtype=float)
    sigma_acc = np.zeros(cycles, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)

    # evolve params if provided
    eta_sel = evolve_params.get("eta_sel", 0.15)
    sigma_mut = evolve_params.get("sigma_mut", 0.04)
    q_frac = evolve_params.get("q_frac", 0.25)
    evolve_mode = evolve_params.get("mode", False)
    simplex = evolve_params.get("simplex", False)

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < 6:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)
        X = design_matrix(A, B, C)
        logits = np.zeros((per_cycle, N), dtype=float)
        accs = np.zeros(N, dtype=float)
        for i in range(N):
            p = 1.0 / (1.0 + np.exp(-X.dot(Ws[i])))
            logits[:, i] = p
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            accs[i] = acc
            grad = X.T.dot(p - Y) / X.shape[0]
            Ws[i] = Ws[i] - 0.3 * grad
        group_mean_acc[ci] = accs.mean()
        sigma_acc[ci] = accs.std()

        if ci == 6:
            misinfo_set = (
                list(rng.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )
        if (ci >= 6) and (ci <= (tstop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # adaptive alpha controller
        p_mat = 1.0 / (1.0 + np.exp(-X.dot(Ws.T)))
        d_pred = np.zeros(N)
        d_param = np.zeros(N)
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

        def safe_z(x):
            m = x.mean()
            s = x.std()
            if s == 0:
                return np.zeros_like(x)
            return (x - m) / s

        z_pred = safe_z(d_pred)
        z_param = safe_z(d_param)
        D_sig = 0.5 * z_pred + 0.5 * z_param
        grow_mask = D_sig < 0.0
        new_alphas = alphas * np.exp(-1.2 * D_sig)
        new_alphas = new_alphas + (0.05 * (alpha_max - alphas) * grow_mask)
        alphas = np.clip(new_alphas, alpha_min, alpha_max)

        # consensus
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
                if (ci >= 6) and (ci <= (tstop - 1)) and (mis_nbrs.size > 0):
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
            r_i[i] = 1.0 - abs(accs.mean() - theta)
        V_i = np.clip((u * np.vstack([c_i, p_i, r_i]).T).sum(axis=1) / 3.0, 0.0, 1.0)
        V_t = float(np.mean(V_i))
        D_t = float(
            np.clip(1.0 - (abs(sigma_acc[ci] - sigma_opt) / sigma_opt), 0.0, 1.0)
        )
        T_t = float(group_mean_acc[ci])
        M_t = wT * T_t + wD * D_t + wV * V_t

        # store per-cycle in arrays
        # allocate arrays lazily
        if ci == 0:
            T_arr = np.zeros(cycles)
            D_arr = np.zeros(cycles)
            V_arr = np.zeros(cycles)
            M_arr = np.zeros(cycles)
        T_arr[ci] = T_t
        D_arr[ci] = D_t
        V_arr[ci] = V_t
        M_arr[ci] = M_t

        # evolution step
        if evolve_mode:
            pay = V_i.copy()
            q = int(np.ceil(q_frac * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            u = (1 - eta_sel) * u + eta_sel * u_success[None, :]
            u = u + rng.normal(0.0, sigma_mut, size=u.shape)
            u = np.clip(u, 0.0, 1.0)
            if simplex:
                s = u.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                u = u / s

    # phases
    w1_idx = list(range(0, 6))
    att_idx = list(range(6, tstop))
    rec_idx = list(range(tstop, cycles))
    rng_bs = np.random.default_rng(int(seed) + 2025)
    M_w1_mean, M_w1_lo, M_w1_hi = bootstrap_mean_ci(
        T_arr[w1_idx] * 0 + M_arr[w1_idx], n_boot=300, rng=rng_bs
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
    T_rec = float(np.mean(group_mean_acc[rec_idx])) if len(rec_idx) > 0 else 0.0
    disp_rec = float(np.mean(sigma_acc[rec_idx])) if len(rec_idx) > 0 else 0.0

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
    }
    return summary, diagnostics


def parse_list(s, cast=float):
    return [cast(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--topology", type=str, default="ws")
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--trials_sweep", type=int, default=3000)
    parser.add_argument("--fast", type=int, default=1)
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
    parser.add_argument("--grid_eta", type=str, default="0.08,0.12,0.18,0.25")
    parser.add_argument("--grid_sigma", type=str, default="0.02,0.04,0.06")
    parser.add_argument("--grid_q", type=str, default="0.20,0.30,0.40")
    parser.add_argument("--simplex", type=int, default=0)
    args = parser.parse_args()

    seed = int(args.seed)
    N = int(args.agents)
    topology = args.topology
    cycles = int(args.cycles)
    trials = int(args.trials)
    trials_sweep = int(args.trials_sweep) if args.fast == 1 else trials
    theta = float(args.theta)
    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    beta = float(args.beta)
    gamma = float(args.gamma)
    frac = float(args.frac)
    tstop = int(args.tstop)
    wT = float(args.wT)
    wD = float(args.wD)
    wV = float(args.wV)

    etas = parse_list(args.grid_eta, float)
    sigmas = parse_list(args.grid_sigma, float)
    qs = parse_list(args.grid_q, float)
    simplex = bool(int(args.simplex))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"meaning_phase5_2_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # baseline fixed run (longer trials)
    print("Running baseline fixed-values run...")
    baseline_summary, baseline_diag = run_single(
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
        {"mode": False},
        wT,
        wD,
        wV,
    )
    pd.DataFrame([baseline_summary]).to_csv(
        os.path.join(outdir, "baseline_fixed.csv"), index=False
    )

    # sweep
    records = []
    total = len(etas) * len(sigmas) * len(qs)
    print(f"Running sweep over {total} configs (trials per config={trials_sweep})")
    cnt = 0
    for eta in etas:
        for sigma in sigmas:
            for q in qs:
                cnt += 1
                evo_params = {
                    "mode": True,
                    "eta_sel": eta,
                    "sigma_mut": sigma,
                    "q_frac": q,
                    "simplex": simplex,
                }
                summary, diag = run_single(
                    seed + cnt,
                    N,
                    topology,
                    trials_sweep,
                    cycles,
                    theta,
                    alpha_min,
                    alpha_max,
                    beta,
                    gamma,
                    frac,
                    tstop,
                    evo_params,
                    wT,
                    wD,
                    wV,
                )
                feasible = True
                # feasibility constraints
                if summary["T_rec"] < (baseline_summary["T_rec"] - 0.03):
                    feasible = False
                if not (0.02 <= summary["disp_rec"] <= 0.12):
                    feasible = False
                if summary["A_M"] > baseline_summary["A_M"]:
                    feasible = False
                row = {"eta_sel": eta, "sigma_mut": sigma, "q_frac": q}
                row.update(summary)
                row["feasible"] = feasible
                records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir, "sweep_results.csv"), index=False)

    # frontier: top 8 feasible by DeltaM desc then A_M asc
    feas = df[df["feasible"] == True].copy()
    if feas.shape[0] > 0:
        front = feas.sort_values(by=["DeltaM", "A_M"], ascending=[False, True]).head(8)
    else:
        front = pd.DataFrame()
    front.to_csv(os.path.join(outdir, "frontier_table.csv"), index=False)

    # plots
    # heatmaps: slice at q_frac closest to 0.30
    target_q = 0.30
    q_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - target_q))
    sel_q = qs[q_idx]
    heat_Delta = np.full((len(etas), len(sigmas)), np.nan)
    heat_feas = np.zeros((len(etas), len(sigmas)))
    for i, eta in enumerate(etas):
        for j, sig in enumerate(sigmas):
            sel = df[
                (df["eta_sel"] == eta)
                & (df["sigma_mut"] == sig)
                & (df["q_frac"] == sel_q)
            ]
            if sel.shape[0] > 0:
                heat_Delta[i, j] = sel["DeltaM"].values[0]
                heat_feas[i, j] = 1 if sel["feasible"].values[0] else 0
    plt.figure(figsize=(6, 4))
    plt.imshow(heat_Delta, aspect="auto", origin="lower")
    plt.xticks(range(len(sigmas)), [str(s) for s in sigmas])
    plt.yticks(range(len(etas)), [str(e) for e in etas])
    plt.xlabel("sigma_mut")
    plt.ylabel("eta_sel")
    plt.title(f"DeltaM (q={sel_q})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heat_deltaM_eta_sigma.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(heat_feas, aspect="auto", origin="lower")
    plt.xticks(range(len(sigmas)), [str(s) for s in sigmas])
    plt.yticks(range(len(etas)), [str(e) for e in etas])
    plt.xlabel("sigma_mut")
    plt.ylabel("eta_sel")
    plt.title(f"Feasible (q={sel_q})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heat_feasible_eta_sigma.png"), dpi=150)
    plt.close()

    # pareto A_M vs DeltaM
    plt.figure(figsize=(7, 5))
    sizes = (df["T_rec"] - df["T_rec"].min() + 1e-3) * 200
    plt.scatter(df["A_M"], df["DeltaM"], s=sizes)
    plt.xlabel("A_M")
    plt.ylabel("DeltaM")
    plt.title("Pareto: A_M vs DeltaM")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_AM_vs_deltaM.png"), dpi=150)
    plt.close()

    # best feasible curves
    if front.shape[0] > 0:
        best = front.iloc[0]
        evo_params = {
            "mode": True,
            "eta_sel": best["eta_sel"],
            "sigma_mut": best["sigma_mut"],
            "q_frac": best["q_frac"],
            "simplex": simplex,
        }
        summary_best, diag_best = run_single(
            seed + 999,
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
            evo_params,
            wT,
            wD,
            wV,
        )
        T_arr = np.array(diag_best["T_arr"])
        D_arr = np.array(diag_best["D_arr"])
        V_arr = np.array(diag_best["V_arr"])
        M_arr = np.array(diag_best["M_arr"])
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, cycles + 1), T_arr, marker="o", label="T")
        plt.plot(range(1, cycles + 1), D_arr, marker="o", label="D")
        plt.plot(range(1, cycles + 1), V_arr, marker="o", label="V")
        plt.plot(range(1, cycles + 1), M_arr, marker="o", label="M")
        plt.axvline(x=6.5, color="k", linestyle="--")
        plt.axvline(x=tstop + 0.0, color="k", linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "best_curves.png"), dpi=150)
        plt.close()
        # values trajectory best: reuse final u stored? we don't return u; skip detailed per-cycle u and just write placeholder
        # for simplicity plot M_arr only
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, cycles + 1), M_arr, marker="o")
        plt.title("M_t for best feasible")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "values_trajectory_best.png"), dpi=150)
        plt.close()

    # README and params
    readme = []
    readme.append("# Meaning Phase5.2 Sweep")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {seed} | N: {N} | topology: {topology}")
    readme.append("")
    readme.append("Baseline:")
    readme.append(pd.DataFrame([baseline_summary]).to_markdown(index=False))
    readme.append("")
    readme.append(
        f"Sweep grid: eta={etas}, sigma={sigmas}, q={qs} | trials_sweep={trials_sweep}"
    )
    readme.append("")
    if front.shape[0] > 0:
        best = front.iloc[0]
        readme.append(
            f"Best feasible: eta={best['eta_sel']}, sigma={best['sigma_mut']}, q={best['q_frac']} | DeltaM={best['DeltaM']:.4f} | A_M={best['A_M']:.4f}"
        )
    else:
        readme.append("No feasible configs found")
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params_out = {
        "seed": seed,
        "N": N,
        "topology": topology,
        "cycles": cycles,
        "trials_baseline": trials,
        "trials_sweep": trials_sweep,
        "grids": {"eta": etas, "sigma": sigmas, "q": qs},
        "wT": wT,
        "wD": wD,
        "wV": wV,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params_out, f, indent=2)

    # console summary
    feasible_count = int(df["feasible"].sum())
    if front.shape[0] > 0:
        b = front.iloc[0]
        print(
            f"Baseline (fixed): M_w1={baseline_summary['M_w1']:.3f} M_att={baseline_summary['M_att']:.3f} M_rec={baseline_summary['M_rec']:.3f} A_M={baseline_summary['A_M']:.4f}"
        )
        print(
            f"Best feasible (evo): eta={b['eta_sel']} sigma={b['sigma_mut']} q={b['q_frac']} | DeltaM={b['DeltaM']:.4f} | A_M={b['A_M']:.4f}"
        )
    else:
        print("No feasible configs found in sweep")
    print(f"Feasible configs: {feasible_count} / {total}")
    print(f"Results folder: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()

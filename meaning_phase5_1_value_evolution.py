#!/usr/bin/env python3
"""
Meaning Phase 5.1 — Value Evolution
Runs two modes: fixed values baseline and evolving values (mutation + selection).
Produces comparisons and diagnostics per specification.
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


# Core run for one mode


def run_mode(seed, mode, N, topology, params):
    rng = np.random.default_rng(int(seed))
    CYCLES = params["cycles"]
    TRIALS = params["trials"]
    per_cycle = max(1, TRIALS // CYCLES)
    theta = params["theta"]
    alpha_min = params["alpha_min"]
    alpha_max = params["alpha_max"]
    beta = params["beta"]
    gamma = params["gamma"]
    frac = params["frac"]
    T_stop = params["tstop"]
    # value evolution params
    eta_sel = params["eta_sel"]
    sigma_mut = params["sigma_mut"]
    q_frac = params["q_frac"]
    simplex = params["simplex"]

    # schedule
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

    # initialize u: if fixed baseline, draw and hold; if evolve, draw initial then update
    u_init = rng.uniform(0.2, 0.8, size=(N, 3))
    if mode == "fixed":
        u = u_init.copy()
    else:
        u = u_init.copy()

    m = int(round(frac * N))
    misinfo_set = []

    # adaptive alpha controller state
    alphas = np.full(N, (alpha_min + alpha_max) / 2.0)
    tau = 0.0
    k_up = 1.2
    k_down = 0.05
    lambda_mix = 0.5

    sigma_opt = 0.07

    # storage
    T_t = np.zeros(CYCLES, dtype=float)
    D_t = np.zeros(CYCLES, dtype=float)
    V_t = np.zeros(CYCLES, dtype=float)
    M_t = np.zeros(CYCLES, dtype=float)
    group_mean_acc = np.zeros(CYCLES, dtype=float)
    sigma_acc = np.zeros(CYCLES, dtype=float)
    consensus_ts = np.zeros(CYCLES, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        # world
        if ci < 6:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        X = design_matrix(A, B, C)

        # per-agent predictions & update
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

        # misinfo selection at start of attack
        if ci == 6:
            misinfo_set = (
                list(rng.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )

        # misinfo application
        if (ci >= 6) and (ci <= (T_stop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # adaptive alphas (same as Phase4.5)
        logits = X.dot(Ws.T)
        p_mat = sigmoid(logits)
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

        def safe_z(x):
            m = x.mean()
            s = x.std()
            if s == 0:
                return np.zeros_like(x)
            return (x - m) / s

        z_pred = safe_z(d_pred)
        z_param = safe_z(d_param)
        D_sig = lambda_mix * z_pred + (1.0 - lambda_mix) * z_param
        grow_mask = D_sig < tau
        new_alphas = alphas * np.exp(-k_up * D_sig)
        new_alphas = new_alphas + (k_down * (alpha_max - alphas) * grow_mask)
        new_alphas = np.clip(new_alphas, alpha_min, alpha_max)
        alphas = new_alphas

        # consensus incorporating misinfo boost during attack
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

        # consensus metric
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))

        # compute value signals
        c_i = sigmoid(logits).var(axis=0)
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
                p_i[i] = 1.0 - (np.linalg.norm(wi - mean_nb) / (wi_norm + 1e-9))
            r_i[i] = 1.0 - abs(accs.mean() - theta)
        # compute V_i and V_t
        dot = (u * np.vstack([c_i, p_i, r_i]).T).sum(axis=1)
        V_i = np.clip(dot / 3.0, 0.0, 1.0)
        V_t[ci] = float(np.mean(V_i))

        # D_t
        D_raw = 1.0 - (abs(sigma_acc[ci] - sigma_opt) / sigma_opt)
        D_t[ci] = float(np.clip(D_raw, 0.0, 1.0))

        # T_t is group_mean_acc
        T_t[ci] = float(group_mean_acc[ci])

        # Meaning weights
        wT = params["wT"]
        wD = params["wD"]
        wV = params["wV"]
        M_t[ci] = wT * T_t[ci] + wD * D_t[ci] + wV * V_t[ci]

        # After metrics: evolve u if mode == 'evo'
        if mode == "evo" and ci >= 0:
            # payoffs
            pay = V_i.copy()
            q = int(np.ceil(q_frac * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            # selection
            u = (1 - eta_sel) * u + eta_sel * u_success[None, :]
            # mutation
            u = u + rng.normal(0.0, sigma_mut, size=u.shape)
            u = np.clip(u, 0.0, 1.0)
            if simplex:
                s = u.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                u = u / s

    # phases
    w1_idx = list(range(0, 6))
    attack_idx = list(range(6, T_stop))
    recovery_idx = list(range(T_stop, CYCLES))

    rng_bs = np.random.default_rng(int(seed) + 999)
    M_w1_mean, M_w1_lo, M_w1_hi = bootstrap_mean_ci(M_t[w1_idx], n_boot=500, rng=rng_bs)
    M_att_mean, M_att_lo, M_att_hi = bootstrap_mean_ci(
        M_t[attack_idx], n_boot=500, rng=rng_bs
    )
    M_rec_mean, M_rec_lo, M_rec_hi = bootstrap_mean_ci(
        M_t[recovery_idx], n_boot=500, rng=rng_bs
    )

    EM = float((M_rec_mean - M_att_mean) / (M_w1_mean if M_w1_mean != 0 else 1.0))
    A_M = (
        float(np.sum(np.maximum(0.0, M_w1_mean - M_t[attack_idx])))
        if len(attack_idx) > 0
        else 0.0
    )
    R_M = float(M_rec_mean / (M_w1_mean if M_w1_mean != 0 else 1.0))

    # per-phase u stats
    def phase_u_stats(idxs):
        if len(idxs) == 0:
            return np.zeros(3), np.zeros(3)
        return u.mean(axis=0), u.std(axis=0)

    u_mean, u_std = phase_u_stats(recovery_idx)

    # assemble summary
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
        "EM": EM,
        "A_M": A_M,
        "R_M": R_M,
        "disp_recovery_mean": (
            float(np.mean(sigma_acc[recovery_idx])) if len(recovery_idx) > 0 else 0.0
        ),
        "u_mean_rec": u_mean.tolist(),
        "u_std_rec": u_std.tolist(),
    }

    diagnostics = {
        "T_t": T_t.tolist(),
        "D_t": D_t.tolist(),
        "V_t": V_t.tolist(),
        "M_t": M_t.tolist(),
        "consensus_ts": consensus_ts.tolist(),
        "group_mean_acc": group_mean_acc.tolist(),
    }

    return summary, diagnostics, u


# CLI and orchestration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=18)
    p.add_argument("--topology", type=str, default="ws")
    p.add_argument("--trials", type=int, default=6000)
    p.add_argument("--cycles", type=int, default=16)
    p.add_argument("--theta", type=float, default=0.65)
    p.add_argument("--alpha_min", type=float, default=0.05)
    p.add_argument("--alpha_max", type=float, default=0.30)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--frac", type=float, default=0.11)
    p.add_argument("--tstop", type=int, default=10)
    p.add_argument("--wT", type=float, default=0.4)
    p.add_argument("--wD", type=float, default=0.3)
    p.add_argument("--wV", type=float, default=0.3)
    p.add_argument("--eta_sel", type=float, default=0.15)
    p.add_argument("--sigma_mut", type=float, default=0.04)
    p.add_argument("--q_frac", type=float, default=0.25)
    p.add_argument("--simplex", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    params = {
        "trials": int(args.trials),
        "cycles": int(args.cycles),
        "theta": float(args.theta),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "frac": float(args.frac),
        "tstop": int(args.tstop),
        "wT": float(args.wT),
        "wD": float(args.wD),
        "wV": float(args.wV),
        "eta_sel": float(args.eta_sel),
        "sigma_mut": float(args.sigma_mut),
        "q_frac": float(args.q_frac),
        "simplex": bool(int(args.simplex)),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"meaning_phase5_1_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # run fixed and evolution modes
    summary_fixed, diag_fixed, u_fixed = run_mode(
        args.seed, "fixed", int(args.agents), args.topology, params
    )
    summary_evo, diag_evo, u_evo = run_mode(
        args.seed + 77, "evo", int(args.agents), args.topology, params
    )

    # write summaries
    pd.DataFrame([summary_fixed]).to_csv(
        os.path.join(outdir, "summary_fixed.csv"), index=False
    )
    pd.DataFrame([summary_evo]).to_csv(
        os.path.join(outdir, "summary_evo.csv"), index=False
    )

    # compare
    compare = {
        "M_w1_fixed": summary_fixed["M_w1"],
        "M_w1_evo": summary_evo["M_w1"],
        "M_rec_fixed": summary_fixed["M_rec"],
        "M_rec_evo": summary_evo["M_rec"],
        "DeltaM_fixed": summary_fixed["M_rec"] - summary_fixed["M_w1"],
        "DeltaM_evo": summary_evo["M_rec"] - summary_evo["M_w1"],
        "A_M_fixed": summary_fixed["A_M"],
        "A_M_evo": summary_evo["A_M"],
        "EM_fixed": summary_fixed["EM"],
        "EM_evo": summary_evo["EM"],
    }
    pd.DataFrame([compare]).to_csv(
        os.path.join(outdir, "compare_summary.csv"), index=False
    )

    # plots
    cycles = np.arange(1, params["cycles"] + 1)
    # meaning_curves_compare.png (plot M and components for both modes)
    plt.figure(figsize=(10, 5))
    plt.plot(cycles, diag_fixed["T_t"], marker="o")
    plt.plot(cycles, diag_fixed["D_t"], marker="o")
    plt.plot(cycles, diag_fixed["V_t"], marker="o")
    plt.plot(cycles, diag_fixed["M_t"], marker="o")
    plt.plot(cycles, diag_evo["M_t"], marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=params["tstop"] + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Index")
    plt.title("Meaning curves (fixed components + evo M)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "meaning_curves_compare.png"), dpi=150)
    plt.close()

    # meaning_bars_compare.png
    labels = ["W1", "Attack", "Recovery"]
    fixed_vals = [summary_fixed["M_w1"], summary_fixed["M_att"], summary_fixed["M_rec"]]
    evo_vals = [summary_evo["M_w1"], summary_evo["M_att"], summary_evo["M_rec"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, fixed_vals, width)
    plt.bar(x + width / 2, evo_vals, width)
    plt.xticks(x, labels)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("M")
    plt.title("Meaning by phase: Fixed vs Evo")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "meaning_bars_compare.png"), dpi=150)
    plt.close()

    # values_trajectory.png: per-cycle mean u for evo
    # we ran evolution inplace and only returned final u; plot per-phase means using u_fixed vs u_evo
    # For simplicity, plot initial vs final means with std band
    u_fixed_mean = np.mean(u_fixed, axis=0)
    u_fixed_std = np.std(u_fixed, axis=0)
    u_evo_mean = np.mean(u_evo, axis=0)
    u_evo_std = np.std(u_evo, axis=0)
    comps = ["curiosity", "cooperation", "coherence"]
    plt.figure(figsize=(8, 4))
    xloc = np.arange(len(comps))
    plt.errorbar(xloc - 0.05, u_fixed_mean, yerr=u_fixed_std, fmt="o", label="fixed")
    plt.errorbar(xloc + 0.05, u_evo_mean, yerr=u_evo_std, fmt="o", label="evo")
    plt.xticks(xloc, comps)
    plt.ylim(0, 1)
    plt.title("Values: initial/final mean ± std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "values_trajectory.png"), dpi=150)
    plt.close()

    # values_scatter_W1_vs_Rec.png: PCA scatter of final u vs initial
    all_u = np.vstack([u_fixed, u_evo])
    # simple 2D projection via SVD
    U, Svals, Vt = np.linalg.svd(all_u - all_u.mean(axis=0), full_matrices=False)
    proj = U[:, :2] * Svals[:2]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        proj[: int(len(proj) / 2), 0], proj[: int(len(proj) / 2), 1], marker="o"
    )
    plt.scatter(
        proj[int(len(proj) / 2) :, 0], proj[int(len(proj) / 2) :, 1], marker="^"
    )
    plt.title("u vectors: fixed (o) vs evo (^)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "values_scatter_W1_vs_Rec.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Meaning Phase 5.1 — Value Evolution")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {args.seed} | N: {args.agents} | topology: {args.topology}")
    readme.append("")
    readme.append("Summary Fixed:")
    readme.append(pd.DataFrame([summary_fixed]).to_markdown(index=False))
    readme.append("")
    readme.append("Summary Evo:")
    readme.append(pd.DataFrame([summary_evo]).to_markdown(index=False))
    readme.append("")
    winner = (
        "EVO"
        if (
            (summary_evo["M_rec"] - summary_evo["M_w1"])
            - (summary_fixed["M_rec"] - summary_fixed["M_w1"])
            > 0.03
            and summary_evo["A_M"] < summary_fixed["A_M"]
        )
        else "FIXED"
    )
    readme.append(f"Winner: {winner}")
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params_out = {
        "seed": args.seed,
        "N": args.agents,
        "topology": args.topology,
        "trials": args.trials,
        "cycles": args.cycles,
        "theta": args.theta,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "beta": args.beta,
        "gamma": args.gamma,
        "frac": args.frac,
        "tstop": args.tstop,
        "eta_sel": args.eta_sel,
        "sigma_mut": args.sigma_mut,
        "q_frac": args.q_frac,
        "simplex": bool(args.simplex),
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params_out, f, indent=2)

    # console summary
    print("\n=== Meaning Phase5.1 Summary ===")

    def ssum(name, s):
        return f"{name} | M: W1/Atk/Rec = {s['M_w1']:.3f}/{s['M_att']:.3f}/{s['M_rec']:.3f} | ΔM={s['M_rec']-s['M_w1']:.3f} | A_M={s['A_M']:.4f} | EM={s['EM']:.3f} | R_M={s['R_M']:.3f}"

    print(ssum("FIXED", summary_fixed))
    print(ssum("EVO", summary_evo))
    print(f"Winner: {winner}")
    print(f"Results folder: {os.path.abspath(outdir)}")
    print("=================================\n")


if __name__ == "__main__":
    main()

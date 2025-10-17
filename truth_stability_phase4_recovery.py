#!/usr/bin/env python3
"""
Truth Stability — Phase 4.4 (Recovery After Misinformation)
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


def run_recovery(
    seed, N, topology, alpha, beta, gamma, frac, TRIALS, CYCLES, theta, T_stop
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

    agent_accs = np.zeros((CYCLES, N), dtype=float)
    agent_briers = np.zeros((CYCLES, N), dtype=float)
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)
    drift_to_false = np.zeros(CYCLES, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    m = int(round(frac * N))
    misinfo_set = []

    # cycle numbering: 1..CYCLES (indices 0..CYCLES-1)
    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        # world selection
        if ci < 6:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        X = design_matrix(A, B, C)
        # per-agent update
        for i in range(N):
            w = Ws[i]
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            brier = float(((p - Y) ** 2).mean())
            agent_accs[ci, i] = acc
            agent_briers[ci, i] = brier
            grad = X.T.dot(p - Y) / X.shape[0]
            w = w - 0.3 * grad
            Ws[i] = w

        # determine misinfo nodes at start of attack (cycle 7 => index 6)
        if ci == 6:
            if m > 0:
                all_indices = list(range(N))
                misinfo_set = list(rng.choice(all_indices, size=m, replace=False))
            else:
                misinfo_set = []

        # apply misinfo during attack cycles (7..T_stop inclusive => indices 6..T_stop-1)
        if (ci >= 6) and (ci <= (T_stop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # consensus with misinfo broadcast boost during attack only
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
                # only apply broadcast boost during attack
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
                mean_nbr = weighted
            W_new[i] = (1 - alpha) * Ws[i] + alpha * mean_nbr
        Ws = W_new

        # record consensus/dispersion/drift
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))
        sims = []
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            if wi_norm == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(wi, v) / (wi_norm * v_norm)))
        drift_to_false[ci] = float(np.mean(sims))

    # phases indices
    w1_idx = list(range(0, 6))
    attack_idx = list(range(6, T_stop))
    recovery_idx = list(range(T_stop, CYCLES))

    w1_mean_acc = float(agent_accs[w1_idx, :].mean())
    attack_mean_acc = (
        float(agent_accs[attack_idx, :].mean()) if len(attack_idx) > 0 else float("nan")
    )
    recovery_mean_acc = (
        float(agent_accs[recovery_idx, :].mean())
        if len(recovery_idx) > 0
        else float("nan")
    )

    # T_group for phases: fraction of cycles in phase where group mean >= theta
    group_mean_acc = agent_accs.mean(axis=1)

    def T_for_idx(idxs):
        if len(idxs) == 0:
            return 0.0, 0.0, 0.0
        passes = int((group_mean_acc[idxs] >= theta).sum())
        p, lo, hi = wilson_ci(passes, len(idxs))
        return p, lo, hi

    T_w1, lo1, hi1 = T_for_idx(w1_idx)
    T_attack, lao, hai = T_for_idx(attack_idx)
    T_recovery, lrec_lo, lrec_hi = T_for_idx(recovery_idx)

    # Recovery latency Lr: first recovery cycle τ >= T_stop+1 (index >= T_stop) where group mean acc >= (W1_mean_acc - 0.03)
    Lr = float("inf")
    target = w1_mean_acc - 0.03
    for offset, idx in enumerate(recovery_idx, start=1):
        if group_mean_acc[idx] >= target:
            Lr = (
                offset - 1
            )  # cycles after T_stop; but spec expects cycles count since T_stop? we'll present as cycles after T_stop
            # We'll count τ as number of cycles after T_stop (1 means immediate next cycle).
            Lr = offset
            break
    if Lr == float("inf"):
        Lr_val = None
    else:
        Lr_val = int(Lr)

    # Residual error R = (W1_mean_acc - last_cycle_acc) clipped to [0,1]
    last_cycle_acc = float(group_mean_acc[-1])
    R = float(np.clip(w1_mean_acc - last_cycle_acc, 0.0, 1.0))

    # Overshoot O: max(mean_acc(t) - W1_mean_acc) for t >= T_stop+1 (index >= T_stop)
    if len(recovery_idx) > 0:
        O = float(max(0.0, np.max(group_mean_acc[recovery_idx] - w1_mean_acc)))
    else:
        O = 0.0

    # consensus_min_post and dispersion mean in recovery
    consensus_min_post = (
        float(np.min(consensus_ts[recovery_idx])) if len(recovery_idx) > 0 else 0.0
    )
    disp_recovery_mean = (
        float(np.mean(dispersion[recovery_idx])) if len(recovery_idx) > 0 else 0.0
    )

    # null baseline
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)
    null_passes = int((group_mean_acc > tau).sum())
    p_null, nl, nh = wilson_ci(null_passes, CYCLES)

    # PASS conditions
    truth_cond = (T_recovery >= 0.75) and not (lrec_lo <= p_null <= lrec_hi)
    Lr_cond = (Lr != float("inf")) and (Lr <= 2)
    R_cond = R <= 0.05
    disp_cond = (disp_recovery_mean >= 0.02) and (disp_recovery_mean <= 0.12)
    experiment_pass = truth_cond and Lr_cond and R_cond and disp_cond

    # per-agent summaries
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
        b_w1 = float(agent_briers[w1_idx, i].mean())
        b_attack = (
            float(agent_briers[attack_idx, i].mean())
            if len(attack_idx) > 0
            else float("nan")
        )
        b_rec = (
            float(agent_briers[recovery_idx, i].mean())
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
                "brier_w1": round(b_w1, 3),
                "brier_attack": round(b_attack, 3) if not np.isnan(b_attack) else "",
                "brier_recovery": round(b_rec, 3) if not np.isnan(b_rec) else "",
            }
        )

    group_row = {
        "T_group_w1": round(T_w1, 3),
        "ci_w1_low": round(lo1, 3),
        "ci_w1_high": round(hi1, 3),
        "T_group_attack": round(T_attack, 3),
        "ci_attack_low": round(lao, 3),
        "ci_attack_high": round(hai, 3),
        "T_group_recovery": round(T_recovery, 3),
        "ci_rec_low": round(lrec_lo, 3),
        "ci_rec_high": round(lrec_hi, 3),
        "Lr": (Lr_val if Lr_val is not None else "inf"),
        "Residual_R": round(R, 4),
        "Overshoot_O": round(O, 4),
        "consensus_min_post": round(consensus_min_post, 6),
        "dispersion_mean_recovery": round(disp_recovery_mean, 6),
    }

    return {
        "agent_rows": agent_rows,
        "group_row": group_row,
        "consensus_ts": consensus_ts.tolist(),
        "dispersion": dispersion.tolist(),
        "drift_to_false": drift_to_false.tolist(),
        "group_mean_acc": group_mean_acc.tolist(),
        "experiment_pass": bool(experiment_pass),
        "misinfo_set": misinfo_set,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=18)
    p.add_argument("--topology", type=str, default="ws")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--theta", type=float, default=0.65)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--frac", type=float, default=0.11)
    p.add_argument("--trials", type=int, default=6000)
    p.add_argument("--cycles", type=int, default=16)
    p.add_argument("--tstop", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    RAND = int(args.seed)
    N = int(args.agents)
    topology = args.topology
    alpha = float(args.alpha)
    theta = float(args.theta)
    beta = float(args.beta)
    gamma = float(args.gamma)
    frac = float(args.frac)
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    T_stop = int(args.tstop)

    res = run_recovery(
        RAND, N, topology, alpha, beta, gamma, frac, TRIALS, CYCLES, theta, T_stop
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_recovery_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    df_agents = pd.DataFrame(res["agent_rows"])
    df_agents.to_csv(os.path.join(outdir, "agents_summary.csv"), index=False)
    pd.DataFrame([res["group_row"]]).to_csv(
        os.path.join(outdir, "group_summary.csv"), index=False
    )

    group_mean_acc = np.array(res["group_mean_acc"])
    consensus_ts = np.array(res["consensus_ts"])
    dispersion = np.array(res["dispersion"])
    drift = np.array(res["drift_to_false"])

    cycles = np.arange(1, CYCLES + 1)
    # curves_truth.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, group_mean_acc, marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Group mean accuracy")
    plt.title("Group mean accuracy over cycles (recovery)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_truth.png"), dpi=150)
    plt.close()

    # curves_consensus.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, consensus_ts, marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Consensus (mean ||w_i - w_bar||)")
    plt.title("Consensus over cycles (recovery)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_consensus.png"), dpi=150)
    plt.close()

    # curves_dispersion.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, dispersion, marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy std across agents")
    plt.title("Dispersion over cycles (recovery)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_dispersion.png"), dpi=150)
    plt.close()

    # drift_to_false.png
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, drift, marker="o")
    plt.axvline(x=6.5, color="k", linestyle="--")
    plt.axvline(x=T_stop + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Drift to false (mean cosine sim)")
    plt.title("Drift to false over cycles (recovery)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drift_to_false.png"), dpi=150)
    plt.close()

    # truth_bars_phases.png
    labels = ["W1", "Attack", "Recovery"]
    Tvals = [
        res["group_row"]["T_group_w1"],
        res["group_row"]["T_group_attack"],
        res["group_row"]["T_group_recovery"],
    ]
    lows = [
        res["group_row"]["ci_w1_low"],
        res["group_row"]["ci_attack_low"],
        res["group_row"]["ci_rec_low"],
    ]
    highs = [
        res["group_row"]["ci_w1_high"],
        res["group_row"]["ci_attack_high"],
        res["group_row"]["ci_rec_high"],
    ]
    errs = np.vstack(
        [np.array(Tvals) - np.array(lows), np.array(highs) - np.array(Tvals)]
    )
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(Tvals)), Tvals, yerr=errs, capsize=4)
    plt.xticks(range(len(Tvals)), labels)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("T_group")
    plt.title("Group Truth Stability by Phase (recovery)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_phases.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Truth Stability — Phase 4.4 (Recovery After Misinformation)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(
        f"Seed: {RAND} | N: {N} | Topology: {topology} | alpha: {alpha} | theta: {theta}"
    )
    readme.append(
        f"Attack beta: {beta} | gamma: {gamma} | frac: {frac} | T_stop: {T_stop}"
    )
    readme.append("")
    readme.append(pd.DataFrame([res["group_row"]]).to_markdown(index=False))
    readme.append("")
    readme.append(
        f'Misinfo nodes (count {len(res["misinfo_set"])}): {res["misinfo_set"]}'
    )
    readme.append("")
    readme.append(f'Experiment PASS: {"YES" if res["experiment_pass"] else "NO"}')
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RAND,
        "N": N,
        "topology": topology,
        "alpha": alpha,
        "theta": theta,
        "beta": beta,
        "gamma": gamma,
        "frac": frac,
        "trials": TRIALS,
        "cycles": CYCLES,
        "T_stop": T_stop,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\n=== Recovery experiment summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    print(f'PASS?: {"YES" if res["experiment_pass"] else "NO"}')
    print(
        f'T_group_recovery = {res["group_row"]["T_group_recovery"]} CI=[{res["group_row"]["ci_rec_low"]},{res["group_row"]["ci_rec_high"]}]'
    )
    print(
        f'Lr = {res["group_row"]["Lr"]}  Residual R = {res["group_row"]["Residual_R"]}  Overshoot O = {res["group_row"]["Overshoot_O"]}'
    )
    print(
        f'dispersion_recovery = {res["group_row"]["dispersion_mean_recovery"]}  consensus_min_post = {res["group_row"]["consensus_min_post"]}'
    )
    print(f'Misinfo nodes ({len(res["misinfo_set"])}) : {res["misinfo_set"]}')
    print("===================================\n")


if __name__ == "__main__":
    main()

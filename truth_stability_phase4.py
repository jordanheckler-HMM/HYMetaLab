#!/usr/bin/env python3
"""
Truth Stability — Phase 4 (Networked Agents & Collective Convergence)
Deterministic, vectorized, matplotlib-only plots.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Utilities ----------


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


# ---------- Topology builders (adjacency matrices) ----------


def build_topology(n_agents, topology, rng):
    adj = np.zeros((n_agents, n_agents), dtype=float)
    if topology == "ring":
        for i in range(n_agents):
            adj[i, (i - 1) % n_agents] = 1
            adj[i, (i + 1) % n_agents] = 1
    elif topology == "ws":
        # simple WS: start with ring k=4 then rewire with beta
        k = 4
        beta = 0.2
        for i in range(n_agents):
            for j in range(1, k // 2 + 1):
                nbr = (i + j) % n_agents
                adj[i, nbr] = 1
                adj[nbr, i] = 1
        # rewire
        for i in range(n_agents):
            for j in range(1, k // 2 + 1):
                if rng.random() < beta:
                    # rewire edge i-(i+j) to i - random non-self
                    possible = list(range(n_agents))
                    possible.remove(i)
                    new = rng.choice(possible)
                    adj[i, (i + j) % n_agents] = 0
                    adj[(i + j) % n_agents, i] = 0
                    adj[i, new] = 1
                    adj[new, i] = 1
    elif topology == "er":
        p = 0.15
        rand = rng.random((n_agents, n_agents))
        adj = (rand < p).astype(float)
        np.fill_diagonal(adj, 0)
        # make symmetric
        adj = np.triu(adj) + np.triu(adj).T
    else:
        raise ValueError("Unknown topology")
    # ensure no self loops
    np.fill_diagonal(adj, 0)
    return adj


# ---------- Simulation W1/W2 ----------


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


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--topology", type=str, default="ws")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()

    RANDOM_SEED = int(args.seed)
    N = int(args.agents)
    topology = args.topology
    alpha = float(args.alpha)
    theta = float(args.theta)
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    per_cycle = max(1, TRIALS // CYCLES)

    rng = np.random.default_rng(RANDOM_SEED)

    # per-cycle probabilities (deterministic)
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    cycles_split = CYCLES // 2

    # build topology
    adj = build_topology(N, topology, rng)
    # adjacency neighbor lists
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    # initialize agents' weights
    d = 6
    Ws = np.zeros((N, d), dtype=float)

    # stats per cycle
    agent_accs = np.zeros((CYCLES, N), dtype=float)
    agent_briers = np.zeros((CYCLES, N), dtype=float)
    agent_claims = np.zeros((CYCLES, N), dtype=int)  # pass/fail
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)
    wbar_history = np.zeros((CYCLES, d), dtype=float)

    # tau for null baseline (deterministic)
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    start_time = datetime.now()

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        X = design_matrix(A, B, C)
        # per-agent local update
        for i in range(N):
            w = Ws[i]
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            brier = float(((p - Y) ** 2).mean())
            agent_accs[ci, i] = acc
            agent_briers[ci, i] = brier
            agent_claims[ci, i] = 1 if acc >= theta else 0
            # gradient step k=1
            grad = X.T.dot(p - Y) / X.shape[0]
            w = w - 0.3 * grad
            Ws[i] = w
        # consensus step: each agent averages neighbors' weights
        W_new = Ws.copy()
        Wbar = Ws.mean(axis=0)
        for i in range(N):
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                mean_nbr = Ws[i]
            else:
                mean_nbr = Ws[nbrs].mean(axis=0)
            W_new[i] = (1 - alpha) * Ws[i] + alpha * mean_nbr
        Ws = W_new
        # record network stats
        wbar = Ws.mean(axis=0)
        wbar_history[ci] = wbar
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))

    # aggregate agent summaries
    agent_rows = []
    for i in range(N):
        accs = agent_accs[:, i]
        passes_w1 = int(agent_claims[:cycles_split, i].sum())
        passes_w2 = int(agent_claims[cycles_split:, i].sum())
        N_w1 = cycles_split
        N_w2 = CYCLES - cycles_split
        p1, l1, h1 = wilson_ci(passes_w1, N_w1)
        p2, l2, h2 = wilson_ci(passes_w2, N_w2)
        agent_rows.append(
            {
                "agent": i,
                "acc_w1_mean": round(float(accs[:cycles_split].mean()), 3),
                "acc_w2_mean": round(float(accs[cycles_split:].mean()), 3),
                "T_w1": round(p1, 3),
                "ci_w1_low": round(l1, 3),
                "ci_w1_high": round(h1, 3),
                "T_w2": round(p2, 3),
                "ci_w2_low": round(l2, 3),
                "ci_w2_high": round(h2, 3),
                "passes_w1": passes_w1,
                "passes_w2": passes_w2,
            }
        )
    df_agents = pd.DataFrame(agent_rows)

    # group claim per cycle: mean accuracy of network >= theta
    group_passes = (agent_accs.mean(axis=1) >= theta).astype(int)
    passes_group_w1 = int(group_passes[:cycles_split].sum())
    passes_group_w2 = int(group_passes[cycles_split:].sum())
    p_g1, gl1, gh1 = wilson_ci(passes_group_w1, cycles_split)
    p_g2, gl2, gh2 = wilson_ci(passes_group_w2, CYCLES - cycles_split)
    passes_group_over = int(group_passes.sum())
    p_go, glo, gho = wilson_ci(passes_group_over, CYCLES)

    # convergence metrics
    pre_mean_consensus = float(np.mean(consensus_ts[:cycles_split]))
    post_mean_consensus = float(np.mean(consensus_ts[cycles_split:]))
    # Lc: first cycle after switch where consensus_t <= (pre_mean_consensus * (1 - 0.3))
    threshold = pre_mean_consensus * (1 - 0.3)
    Lc = None
    for idx in range(cycles_split, CYCLES):
        if consensus_ts[idx] <= threshold:
            Lc = int(idx - cycles_split)
            break
    if Lc is None:
        Lc = CYCLES
    consensus_min_post = float(np.min(consensus_ts[cycles_split:]))

    # dispersion pre/post
    disp_pre_mean = float(np.mean(dispersion[:cycles_split]))
    disp_post_mean = float(np.mean(dispersion[cycles_split:]))

    # Experiment PASS gate
    null_passes = 0
    # define null as overall mean accuracy > tau per cycle
    overall_mean_acc_per_cycle = agent_accs.mean(axis=1)
    null_passes = int((overall_mean_acc_per_cycle > tau).sum())
    p_null, nl, nh = wilson_ci(null_passes, CYCLES)

    truth_cond = (
        (p_g1 >= 0.7)
        and (p_g2 >= 0.7)
        and not ((gl1 <= p_null <= gh1) or (gl2 <= p_null <= gh2))
    )
    convergence_cond = Lc <= 2
    no_collapse = disp_post_mean > 0.02
    experiment_pass = truth_cond and convergence_cond and no_collapse

    # write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    df_agents.to_csv(os.path.join(outdir, "agents_summary.csv"), index=False)

    group_row = {
        "T_group_w1": round(p_g1, 3),
        "ci_w1_low": round(gl1, 3),
        "ci_w1_high": round(gh1, 3),
        "T_group_w2": round(p_g2, 3),
        "ci_w2_low": round(gl2, 3),
        "ci_w2_high": round(gh2, 3),
        "T_group_overall": round(p_go, 3),
        "ci_over_low": round(glo, 3),
        "ci_over_high": round(gho, 3),
        "Lc": int(Lc),
        "consensus_min_post": round(consensus_min_post, 4),
        "disp_pre_mean": round(disp_pre_mean, 4),
        "disp_post_mean": round(disp_post_mean, 4),
    }
    pd.DataFrame([group_row]).to_csv(
        os.path.join(outdir, "group_summary.csv"), index=False
    )

    # consensus_curve.png
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, CYCLES + 1), consensus_ts, marker="o")
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Consensus (mean ||w_i - w_bar||)")
    plt.title("Consensus over cycles")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consensus_curve.png"), dpi=150)
    plt.close()

    # dispersion_curve.png
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, CYCLES + 1), dispersion, marker="o")
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy std across agents")
    plt.title("Opinion dispersion over cycles")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dispersion_curve.png"), dpi=150)
    plt.close()

    # truth_bars_group.png
    labels = ["W1", "W2", "Overall"]
    Ts = [p_g1, p_g2, p_go]
    lows = [gl1, gl2, glo]
    highs = [gh1, gh2, gho]
    errs = np.vstack([np.array(Ts) - np.array(lows), np.array(highs) - np.array(Ts)])
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(Ts)), Ts, yerr=errs, capsize=4)
    plt.xticks(range(len(Ts)), labels)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("T_group")
    plt.title("Group Truth Stability")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_group.png"), dpi=150)
    plt.close()

    # network_snapshot.png (circular layout)
    plt.figure(figsize=(6, 6))
    theta_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xs = np.cos(theta_angles)
    ys = np.sin(theta_angles)
    last_accs = agent_accs[-1]
    # draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > 0:
                plt.plot([xs[i], xs[j]], [ys[i], ys[j]], alpha=0.4)
    # nodes colored by last accuracy using default colormap
    sc = plt.scatter(xs, ys, s=120, c=last_accs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x * 1.08, y * 1.08, str(i), fontsize=7, ha="center", va="center")
    plt.title("Network snapshot (node color = last-cycle accuracy)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "network_snapshot.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Truth Stability — Phase 4 (Networked Agents)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(
        f"Seed: {RANDOM_SEED} | Agents: {N} | Topology: {topology} | alpha: {alpha} | theta: {theta}"
    )
    readme.append("")
    readme.append(pd.DataFrame([group_row]).to_markdown(index=False))
    readme.append("")
    readme.append(
        "Notes: consensus measured as mean parameter distance to network mean; dispersion is std of accuracies."
    )
    readme.append("")
    readme.append(f'Experiment PASS: {"YES" if experiment_pass else "NO"}')
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RANDOM_SEED,
        "agents": N,
        "topology": topology,
        "alpha": alpha,
        "theta": theta,
        "trials": TRIALS,
        "cycles": CYCLES,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\n=== Truth Stability Phase 4 Summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    print(
        f"T_group_w1/T_group_w2 = {round(p_g1,3)}/{round(p_g2,3)} CIs W1[{round(gl1,3)},{round(gh1,3)}] W2[{round(gl2,3)},{round(gh2,3)}]"
    )
    print(f'PASS?: {"YES" if experiment_pass else "NO"}')
    print(
        f"Lc (cycles after switch): {Lc}  min_consensus_post: {consensus_min_post}  disp_pre_mean: {disp_pre_mean} disp_post_mean: {disp_post_mean}"
    )
    print("========================================\n")


if __name__ == "__main__":
    main()

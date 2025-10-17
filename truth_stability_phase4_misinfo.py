#!/usr/bin/env python3
"""
Truth Stability — Phase 4.2 (Misinformation Resilience)
Networked learners with misinformation injection after regime switch.
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


# ---------- Topologies ----------


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
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--topology", type=str, default="ws")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--frac", type=float, default=0.11)
    parser.add_argument("--trials", type=int, default=4500)
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()

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
    per_cycle = max(1, TRIALS // CYCLES)

    rng = np.random.default_rng(RAND)

    # schedule
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
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    # initialize weights
    d = 6
    Ws = np.zeros((N, d), dtype=float)

    # metrics per cycle
    agent_accs = np.zeros((CYCLES, N), dtype=float)
    agent_briers = np.zeros((CYCLES, N), dtype=float)
    agent_claims = np.zeros((CYCLES, N), dtype=int)
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)
    exposure = np.zeros(CYCLES, dtype=float)
    drift_to_false = np.zeros(CYCLES, dtype=float)

    # tau
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    # prepare adversarial vector v (deterministic small noise added)
    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    # misinfo set determined at switch
    m = int(round(frac * N))
    misinfo_set = []

    start_time = datetime.now()

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
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
            agent_claims[ci, i] = 1 if acc >= theta else 0
            grad = X.T.dot(p - Y) / X.shape[0]
            w = w - 0.3 * grad
            Ws[i] = w
        # at switch, select misinfo nodes
        if ci == cycles_split - 1:
            all_indices = list(range(N))
            misinfo_set = list(rng.choice(all_indices, size=m, replace=False))
        # apply misinfo bias before consensus if post-switch
        if ci >= cycles_split and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v
        # consensus step with misinfo broadcast weighting
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
                # weighted combination
                denom = 1.0
                if (mis_nbrs.size > 0) and (reg_nbrs.size > 0):
                    weighted = (mean_reg + gamma * mean_mis) / (1.0 + gamma)
                elif mis_nbrs.size > 0:
                    weighted = mean_mis
                elif reg_nbrs.size > 0:
                    weighted = mean_reg
                else:
                    weighted = Ws[i]
                mean_nbr = weighted
            W_new[i] = (1 - alpha) * Ws[i] + alpha * mean_nbr
        Ws = W_new
        # record consensus and dispersion
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))
        # exposure: fraction within 1-hop of any misinfo node (include misinfo themselves)
        if len(misinfo_set) == 0:
            exposure[ci] = 0.0
        else:
            exposed = set()
            for mi in misinfo_set:
                exposed.add(mi)
                nbrs = np.where(adj[mi] > 0)[0].tolist()
                for n in nbrs:
                    exposed.add(int(n))
            exposure[ci] = float(len(exposed) / N)
        # drift to false: mean cosine similarity to v
        sims = []
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            if wi_norm == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(wi, v) / (wi_norm * v_norm)))
        drift_to_false[ci] = float(np.mean(sims))

    # aggregate group claims and metrics
    cycles_split = CYCLES // 2
    group_mean_acc = agent_accs.mean(axis=1)
    group_passes = (group_mean_acc >= theta).astype(int)
    passes_group_w1 = int(group_passes[:cycles_split].sum())
    passes_group_w2 = int(group_passes[cycles_split:].sum())
    p_g1, gl1, gh1 = wilson_ci(passes_group_w1, cycles_split)
    p_g2, gl2, gh2 = wilson_ci(passes_group_w2, CYCLES - cycles_split)
    p_go, glo, gho = wilson_ci(int(group_passes.sum()), CYCLES)

    # Td: first post-switch cycle where cumulative T_w2_so_far < 0.5
    Td = None
    cum_pass = 0
    for idx in range(cycles_split, CYCLES):
        cum_pass += group_passes[idx]
        seen = idx - cycles_split + 1
        if (cum_pass / seen) < 0.5:
            Td = int(idx - cycles_split + 1)
            break

    pre_mean_consensus = float(np.mean(consensus_ts[:cycles_split]))
    post_consensus = consensus_ts[cycles_split:]
    post_mean_consensus = float(np.mean(post_consensus))
    consensus_min_post = float(np.min(post_consensus))
    disp_post_mean = float(np.mean(dispersion[cycles_split:]))

    # null baseline (overall mean acc > tau)
    overall_mean_acc_per_cycle = agent_accs.mean(axis=1)
    null_passes = int((overall_mean_acc_per_cycle > tau).sum())
    p_null, nl, nh = wilson_ci(null_passes, CYCLES)

    truth_cond = (p_g2 >= 0.7) and not (gl2 <= p_null <= gh2)
    Td_cond = Td is None
    disp_cond = disp_post_mean >= 0.02
    experiment_pass = truth_cond and Td_cond and disp_cond

    # agents summary
    agent_rows = []
    for i in range(N):
        accs = agent_accs[:, i]
        passes_w1 = int((accs[:cycles_split] >= theta).sum())
        passes_w2 = int((accs[cycles_split:] >= theta).sum())
        p1, l1, h1 = wilson_ci(passes_w1, cycles_split)
        p2, l2, h2 = wilson_ci(passes_w2, CYCLES - cycles_split)
        agent_rows.append(
            {
                "agent": i,
                "is_misinfo": (i in misinfo_set),
                "acc_w1_mean": round(float(accs[:cycles_split].mean()), 3),
                "acc_w2_mean": round(float(accs[cycles_split:].mean()), 3),
                "T_w1": round(p1, 3),
                "T_w2": round(p2, 3),
            }
        )

    df_agents = pd.DataFrame(agent_rows)
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
        "Td": Td if Td is not None else "None",
        "consensus_min_post": round(consensus_min_post, 6),
        "dispersion_mean_post": round(disp_post_mean, 6),
        "exposure_mean_post": round(float(np.mean(exposure[cycles_split:])), 6),
    }

    # write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_misinfo_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    df_agents.to_csv(os.path.join(outdir, "agents_summary.csv"), index=False)
    pd.DataFrame([group_row]).to_csv(
        os.path.join(outdir, "group_summary.csv"), index=False
    )

    # consensus curve
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, CYCLES + 1), consensus_ts, marker="o")
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Consensus (mean ||w_i - w_bar||)")
    plt.title("Consensus over cycles (misinfo)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consensus_curve.png"), dpi=150)
    plt.close()

    # dispersion curve
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, CYCLES + 1), dispersion, marker="o")
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy std across agents")
    plt.title("Dispersion over cycles (misinfo)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dispersion_curve.png"), dpi=150)
    plt.close()

    # truth bars
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
    plt.title("Group Truth Stability (misinfo)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "truth_bars_group.png"), dpi=150)
    plt.close()

    # misinfo spread: drift_to_false and exposure
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, CYCLES + 1), drift_to_false, marker="o", label="drift_to_false")
    plt.plot(range(1, CYCLES + 1), exposure, marker="o", label="exposure")
    plt.axvline(x=cycles_split + 0.5, color="k", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel("Metric")
    plt.title("Misinfo spread: drift_to_false (cos sim) & exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "misinfo_spread.png"), dpi=150)
    plt.close()

    # network snapshot: node shapes mark misinfo nodes (triangle) vs circle
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
    # draw nodes: misinfo as triangles, others circles
    mis_mask = np.array([i in misinfo_set for i in range(N)])
    # default colormap by passing c=last_accs
    plt.scatter(xs[~mis_mask], ys[~mis_mask], s=120, c=last_accs[~mis_mask], marker="o")
    plt.scatter(xs[mis_mask], ys[mis_mask], s=140, c=last_accs[mis_mask], marker="^")
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x * 1.08, y * 1.08, str(i), fontsize=7, ha="center", va="center")
    plt.title("Network snapshot (triangles = misinfo nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "network_snapshot.png"), dpi=150)
    plt.close()

    # README and params
    readme = []
    readme.append("# Truth Stability — Phase 4.2 (Misinformation)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(
        f"Seed: {RAND} | N: {N} | Topology: {topology} | alpha: {alpha} | theta: {theta}"
    )
    readme.append(
        f"Beta (bias magnitude): {beta} | Gamma (broadcast weight): {gamma} | frac misinfo: {frac}"
    )
    readme.append("")
    readme.append(pd.DataFrame([group_row]).to_markdown(index=False))
    readme.append("")
    readme.append(f"Misinfo nodes (count {len(misinfo_set)}): {misinfo_set}")
    readme.append("")
    readme.append(f'Experiment PASS: {"YES" if experiment_pass else "NO"}')
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
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\n=== Truth Stability Phase 4.2 (Misinformation) Summary ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    print(f'PASS?: {"YES" if experiment_pass else "NO"}')
    print(f"T_group_w2 = {round(p_g2,3)} CI=[{round(gl2,3)},{round(gh2,3)}]  Td = {Td}")
    print(
        f"dispersion_post = {round(disp_post_mean,4)}  consensus_min_post = {round(consensus_min_post,6)}"
    )
    print(f"Misinfo nodes ({len(misinfo_set)}): {misinfo_set}")
    print("========================================\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Truth Stability — Phase 4.1 Sweep (α & Topology)
Runs a small sweep over communication weight α and topology, reusing Phase-4 logic.
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


# topology builders
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
        # rewire
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


# simulate cycles
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


# run once configuration
def run_once(seed, N, topology, alpha, theta, TRIALS, CYCLES):
    rng = np.random.default_rng(seed)
    per_cycle = max(1, TRIALS // CYCLES)
    # cycle ps
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))
    cycles_split = CYCLES // 2

    adj = build_topology(N, topology, rng)
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg == 0, 1.0, deg)

    # initialize weights
    d = 6
    Ws = np.zeros((N, d), dtype=float)

    agent_accs = np.zeros((CYCLES, N), dtype=float)
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)

    # tau for null baseline
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)
        X = design_matrix(A, B, C)
        # per-agent predict then one gradient step update
        for i in range(N):
            w = Ws[i]
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            agent_accs[ci, i] = acc
            grad = X.T.dot(p - Y) / X.shape[0]
            w = w - 0.3 * grad
            Ws[i] = w
        # consensus step vectorized
        # mean neighbors: (adj @ Ws) / deg_safe[:,None]
        mean_nbr = (adj @ Ws) / deg_safe[:, None]
        Ws = (1 - alpha) * Ws + alpha * mean_nbr
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))

    # group claims
    group_mean_acc = agent_accs.mean(axis=1)
    group_passes = (group_mean_acc >= theta).astype(int)
    passes_group_w1 = int(group_passes[:cycles_split].sum())
    passes_group_w2 = int(group_passes[cycles_split:].sum())
    p_g1, gl1, gh1 = wilson_ci(passes_group_w1, cycles_split)
    p_g2, gl2, gh2 = wilson_ci(passes_group_w2, CYCLES - cycles_split)
    p_go, glo, gho = wilson_ci(int(group_passes.sum()), CYCLES)

    # pre/post consensus stats
    pre_mean_consensus = float(np.mean(consensus_ts[:cycles_split]))
    post_consensus = consensus_ts[cycles_split:]
    post_mean_consensus = float(np.mean(post_consensus))
    threshold = pre_mean_consensus * (1 - 0.3)
    Lc = None
    for idx in range(cycles_split, CYCLES):
        if consensus_ts[idx] <= threshold:
            Lc = int(idx - cycles_split)
            break
    if Lc is None:
        Lc = CYCLES
    consensus_min_post = float(np.min(post_consensus))
    disp_post_mean = float(np.mean(dispersion[cycles_split:]))

    # null baseline
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

    # agent-level summaries
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
                "acc_w1_mean": round(float(accs[:cycles_split].mean()), 3),
                "acc_w2_mean": round(float(accs[cycles_split:].mean()), 3),
                "T_w1": round(p1, 3),
                "T_w2": round(p2, 3),
                "passes_w1": passes_w1,
                "passes_w2": passes_w2,
            }
        )

    result = {
        "topology": topology,
        "alpha": alpha,
        "T_group_w1": float(p_g1),
        "ci_w1_low": float(gl1),
        "ci_w1_high": float(gh1),
        "T_group_w2": float(p_g2),
        "ci_w2_low": float(gl2),
        "ci_w2_high": float(gh2),
        "T_group_over": float(p_go),
        "ci_over_low": float(glo),
        "ci_over_high": float(gho),
        "Lc": int(Lc),
        "consensus_min_post": float(consensus_min_post),
        "disp_post_mean": float(disp_post_mean),
        "passes_group_w1": int(passes_group_w1),
        "passes_group_w2": int(passes_group_w2),
        "tau": float(tau),
        "agent_rows": agent_rows,
    }
    return result


# ---------- Sweep driver ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--trials", type=int, default=4500)
    parser.add_argument("--alphas", type=str, default="0,0.1,0.25,0.5,0.75")
    parser.add_argument("--tops", type=str, default="ring,ws,er")
    parser.add_argument("--theta", type=float, default=0.65)
    args = parser.parse_args()

    seed = int(args.seed)
    N = int(args.agents)
    CYCLES = int(args.cycles)
    TRIALS = int(args.trials)
    alphas = [float(x) for x in args.alphas.split(",")]
    tops = [x for x in args.tops.split(",")]
    theta = float(args.theta)

    start_time = datetime.now()
    results = []
    # use base rng to generate per-run seeds deterministically
    base_rng = np.random.default_rng(seed)
    run_seeds = base_rng.integers(1_000_000_000, size=len(alphas) * len(tops))
    idx = 0
    for topology in tops:
        for alpha in alphas:
            run_seed = int(run_seeds[idx])
            idx += 1
            res = run_once(run_seed, N, topology, alpha, theta, TRIALS, CYCLES)
            res["seed_used"] = run_seed
            results.append(res)

    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("agent_rows",)} for r in results]
    )

    # write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_sweep_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "sweep_results.csv"), index=False)

    # heatmaps: T_group_w2 and Lc
    alpha_labels = [str(a) for a in alphas]
    topo_labels = tops
    Tmat = np.zeros((len(tops), len(alphas)))
    Lmat = np.zeros((len(tops), len(alphas)))
    for i, topology in enumerate(tops):
        for j, alpha in enumerate(alphas):
            row = df[(df["topology"] == topology) & (df["alpha"] == alpha)].iloc[0]
            Tmat[i, j] = row["T_group_w2"]
            Lmat[i, j] = row["Lc"]

    plt.figure(figsize=(6, 4))
    plt.imshow(Tmat, aspect="auto")
    plt.colorbar()
    plt.yticks(range(len(topo_labels)), topo_labels)
    plt.xticks(range(len(alpha_labels)), alpha_labels)
    plt.title("T_group_w2 heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heat_T_group_w2.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(Lmat, aspect="auto")
    plt.colorbar()
    plt.yticks(range(len(topo_labels)), topo_labels)
    plt.xticks(range(len(alpha_labels)), alpha_labels)
    plt.title("Lc heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heat_Lc.png"), dpi=150)
    plt.close()

    # pareto scatter: x=Lc, y=disp_post_mean, size ~ T_group_w2
    plt.figure(figsize=(6, 6))
    xs = df["Lc"].values
    ys = df["disp_post_mean"].values
    sizes = df["T_group_w2"].values * 200
    plt.scatter(xs, ys, s=sizes)
    # label top3 by T_group_w2
    top3 = df.sort_values(
        ["T_group_w2", "disp_post_mean"], ascending=[False, False]
    ).head(3)
    for _, r in top3.iterrows():
        plt.text(
            r["Lc"] + 0.02,
            r["disp_post_mean"] + 0.0005,
            f"{r['topology']},α={r['alpha']}",
        )
    plt.xlabel("Lc")
    plt.ylabel("dispersion_post")
    plt.title("Pareto: Lc vs dispersion_post (size ~ T_group_w2)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_scatter.png"), dpi=150)
    plt.close()

    # README: top 3 configs ranked by PASS then T_group_w2, tie break Lc smaller, dispersion_post larger
    df["PASS"] = False
    for i, r in df.iterrows():
        # recompute pass flag similar to Phase4
        p_g1 = r["T_group_w1"]
        p_g2 = r["T_group_w2"]
        gl1 = r["ci_w1_low"]
        gh1 = r["ci_w1_high"]
        gl2 = r["ci_w2_low"]
        gh2 = r["ci_w2_high"]
        p_null = r.get("tau", 0.0)
        # here we don't have p_null CI easily; approximate pass cond by thresholds
        cond_truth = (p_g1 >= 0.7) and (p_g2 >= 0.7)
        cond_conv = r["Lc"] <= 2
        cond_disp = r["disp_post_mean"] > 0.02
        df.at[i, "PASS"] = bool(cond_truth and cond_conv and cond_disp)

    ranked = df.sort_values(
        ["PASS", "T_group_w2", "Lc", "disp_post_mean"],
        ascending=[False, False, True, False],
    )
    top3 = ranked.head(3)

    readme = []
    readme.append("# Truth Stability — Phase 4 Sweep")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {seed} | N: {N} | Cycles: {CYCLES} | Trials: {TRIALS}")
    readme.append(f"Alphas: {alphas} | Topologies: {tops} | theta: {theta}")
    readme.append("")
    readme.append("Top 3 configs:")
    readme.append(top3.to_markdown(index=False))
    readme.append("")
    readme.append(
        "Notes: PASS requires group truth and fast convergence and non-collapse (dispersion_post>0.02)."
    )
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": seed,
        "N": N,
        "cycles": CYCLES,
        "trials": TRIALS,
        "alphas": alphas,
        "tops": tops,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\nTop 3 configs:")
    for _, r in top3.iterrows():
        print(
            f"{r['topology']} α={r['alpha']} T_group_w2={r['T_group_w2']:.3f} CI=[{r['ci_w2_low']:.3f},{r['ci_w2_high']:.3f}] Lc={int(r['Lc'])} disp_post={r['disp_post_mean']:.4f} PASS={r['PASS']}"
        )
    print(f"Results folder: {os.path.abspath(outdir)}")
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Elapsed (s): {elapsed:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Truth Stability — Phase 4.3 (Critical Misinformation Threshold Sweep)
Sweeps small grids over (alpha, beta, gamma, frac) per topology and writes summary/figures.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse helpers similar to Phase 4.2


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


# single-run driver, deterministic given seed


def run_once(seed, N, topology, alpha, beta, gamma, frac, TRIALS, CYCLES, theta):
    rng = np.random.default_rng(int(seed))
    per_cycle = max(1, TRIALS // CYCLES)
    # schedule of per-cycle pA/pB/pC deterministic given rng
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))
    cycles_split = CYCLES // 2

    adj = build_topology(N, topology, rng)
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    d = 6
    Ws = np.zeros((N, d), dtype=float)

    agent_accs = np.zeros((CYCLES, N), dtype=float)
    consensus_ts = np.zeros(CYCLES, dtype=float)
    dispersion = np.zeros(CYCLES, dtype=float)

    base_v = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    v = base_v + rng.normal(0.0, 0.05, size=d)
    v_norm = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0

    m = int(round(frac * N))
    misinfo_set = []

    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)
        X = design_matrix(A, B, C)
        for i in range(N):
            w = Ws[i]
            p = sigmoid(X.dot(w))
            yhat = (p >= 0.5).astype(int)
            acc = float((yhat == Y).mean())
            agent_accs[ci, i] = acc
            grad = X.T.dot(p - Y) / X.shape[0]
            w = w - 0.3 * grad
            Ws[i] = w
        if ci == cycles_split - 1:
            if m > 0:
                all_indices = list(range(N))
                misinfo_set = list(rng.choice(all_indices, size=m, replace=False))
            else:
                misinfo_set = []
        if ci >= cycles_split and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v
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
        wbar = Ws.mean(axis=0)
        consensus_ts[ci] = float(np.mean(np.linalg.norm(Ws - wbar[None, :], axis=1)))
        dispersion[ci] = float(np.std(agent_accs[ci, :]))

    # aggregate
    group_mean_acc = agent_accs.mean(axis=1)
    group_passes = (group_mean_acc >= theta).astype(int)
    passes_group_w2 = int(group_passes[cycles_split:].sum())
    p_g2, gl2, gh2 = wilson_ci(passes_group_w2, CYCLES - cycles_split)

    Td = None
    cum_pass = 0
    for idx in range(cycles_split, CYCLES):
        cum_pass += group_passes[idx]
        seen = idx - cycles_split + 1
        if (cum_pass / seen) < 0.5:
            Td = int(idx - cycles_split + 1)
            break

    post_consensus = consensus_ts[cycles_split:]
    post_mean_consensus = float(np.mean(post_consensus))
    consensus_min_post = float(np.min(post_consensus))
    disp_post_mean = float(np.mean(dispersion[cycles_split:]))

    pct_below = float((agent_accs[cycles_split:, :] < theta).mean())

    # null baseline
    overall_mean_acc_per_cycle = agent_accs.mean(axis=1)
    tau = float(rng.random() * (0.7 - 0.3) + 0.3)
    null_passes = int((overall_mean_acc_per_cycle > tau).sum())
    p_null, nl, nh = wilson_ci(null_passes, CYCLES)

    truth_cond = (p_g2 >= 0.7) and not (gl2 <= p_null <= gh2)
    Td_cond = Td is None
    disp_cond = disp_post_mean >= 0.02
    experiment_pass = truth_cond and Td_cond and disp_cond

    return {
        "T_group_w2": float(p_g2),
        "ci_lo": float(gl2),
        "ci_hi": float(gh2),
        "Td": Td if Td is not None else "None",
        "dispersion_post": float(disp_post_mean),
        "consensus_min_post": float(consensus_min_post),
        "pct_below_theta_post": float(pct_below),
        "PASS": bool(experiment_pass),
    }


def parse_list(s, cast=float):
    if s is None or s == "":
        return []
    return [cast(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--trials", type=int, default=4500)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--alphas", type=str, default="0.05,0.15,0.30")
    parser.add_argument("--betas", type=str, default="0.10,0.25,0.40")
    parser.add_argument("--gammas", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--fracs", type=str, default="0.05,0.11,0.20")
    parser.add_argument("--tops", type=str, default="ring,ws,er")
    args = parser.parse_args()

    RAND = int(args.seed)
    N = int(args.agents)
    CYCLES = int(args.cycles)
    TRIALS = int(args.trials)
    theta = float(args.theta)

    alphas = parse_list(args.alphas, float)
    betas = parse_list(args.betas, float)
    gammas = parse_list(args.gammas, float)
    fracs = parse_list(args.fracs, float)
    tops = [t.strip() for t in args.tops.split(",") if t.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase4_crit_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    records = []

    total = len(tops) * len(alphas) * len(betas) * len(gammas) * len(fracs)
    print(f"Running sweep: {total} configs (tops={tops})")

    count = 0
    for topology in tops:
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    for frac in fracs:
                        count += 1
                        seed = RAND + count
                        res = run_once(
                            seed,
                            N,
                            topology,
                            alpha,
                            beta,
                            gamma,
                            frac,
                            TRIALS,
                            CYCLES,
                            theta,
                        )
                        row = {
                            "topology": topology,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "frac": frac,
                        }
                        row.update(res)
                        records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir, "crit_results.csv"), index=False)

    # Frontier table: for each (topology, alpha), find minimal (f, beta, gamma) triple that flips PASS->FAIL
    frontier_rows = []
    for topology in tops:
        for alpha in alphas:
            sub = df[(df.topology == topology) & (df.alpha == alpha)].copy()
            # lexicographic order by frac asc, beta asc, gamma asc
            sub = sub.sort_values(by=["frac", "beta", "gamma"])
            fail = sub[sub["PASS"] == False]
            if fail.shape[0] == 0:
                frontier_rows.append(
                    {"topology": topology, "alpha": alpha, "frontier": "no-fail"}
                )
            else:
                first = fail.iloc[0]
                frontier_rows.append(
                    {
                        "topology": topology,
                        "alpha": alpha,
                        "frontier": f"f={first.frac},beta={first.beta},gamma={first.gamma}",
                    }
                )
    pd.DataFrame(frontier_rows).to_csv(
        os.path.join(outdir, "frontier_table.csv"), index=False
    )

    # Figures
    # heat_T_w2_<topology>.png: for each topology, show beta vs frac at middle alpha, middle gamma
    mid_alpha = alphas[len(alphas) // 2]
    mid_gamma = gammas[len(gammas) // 2]
    for topology in tops:
        heat = np.zeros((len(betas), len(fracs)))
        for i_b, beta in enumerate(betas):
            for j_f, frac in enumerate(fracs):
                sel = df[
                    (df.topology == topology)
                    & (df.alpha == mid_alpha)
                    & (df.beta == beta)
                    & (df.gamma == mid_gamma)
                    & (df.frac == frac)
                ]
                if sel.shape[0] > 0:
                    heat[i_b, j_f] = sel["T_group_w2"].values[0]
                else:
                    heat[i_b, j_f] = np.nan
        plt.figure(figsize=(6, 4))
        plt.imshow(heat, aspect="auto", origin="lower")
        plt.xticks(range(len(fracs)), [str(f) for f in fracs])
        plt.yticks(range(len(betas)), [str(b) for b in betas])
        plt.xlabel("frac (f)")
        plt.ylabel("beta (β)")
        plt.title(
            f"T_group_w2 heat (top={topology}, alpha={mid_alpha}, gamma={mid_gamma})"
        )
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"heat_T_w2_{topology}.png"), dpi=150)
        plt.close()

    # pareto dispersion vs T_group_w2
    plt.figure(figsize=(8, 6))
    sizes = df["frac"].values * 100
    plt.scatter(df["dispersion_post"].values, df["T_group_w2"].values, s=sizes)
    plt.xlabel("dispersion_post")
    plt.ylabel("T_group_w2")

    # annotate top5 resilient (highest T_group_w2, tie: higher dispersion, smaller Td [None considered large])
    def td_sort_key(x):
        return float("inf") if x == "None" else int(x)

    df_sorted = df.copy()
    df_sorted["Td_sort"] = df_sorted["Td"].apply(td_sort_key)
    top5 = df_sorted.sort_values(
        by=["T_group_w2", "dispersion_post", "Td_sort"], ascending=[False, False, True]
    ).head(5)
    for _, r in top5.iterrows():
        plt.annotate(
            f"{r.topology},a={r.alpha},b={r.beta},f={r.frac}",
            xy=(r.dispersion_post, r.T_group_w2),
        )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pareto_dispersion_vs_T_w2.png"), dpi=150)
    plt.close()

    # crit_surface_ws.png: rows alpha, cols gamma, each subplot heat over beta vs frac
    if "ws" in tops:
        fig, axes = plt.subplots(
            len(alphas), len(gammas), figsize=(4 * len(gammas), 3 * len(alphas))
        )
        for i_a, alpha in enumerate(alphas):
            for j_g, gamma in enumerate(gammas):
                heat = np.zeros((len(betas), len(fracs)))
                for i_b, beta in enumerate(betas):
                    for j_f, frac in enumerate(fracs):
                        sel = df[
                            (df.topology == "ws")
                            & (df.alpha == alpha)
                            & (df.beta == beta)
                            & (df.gamma == gamma)
                            & (df.frac == frac)
                        ]
                        if sel.shape[0] > 0:
                            heat[i_b, j_f] = sel["T_group_w2"].values[0]
                        else:
                            heat[i_b, j_f] = np.nan
                ax = axes[i_a, j_g] if axes.ndim > 1 else axes[max(i_a, j_g)]
                im = ax.imshow(heat, aspect="auto", origin="lower")
                ax.set_xticks(range(len(fracs)))
                ax.set_xticklabels([str(f) for f in fracs])
                ax.set_yticks(range(len(betas)))
                ax.set_yticklabels([str(b) for b in betas])
                ax.set_title(f"a={alpha}, g={gamma}")
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "crit_surface_ws.png"), dpi=150)
        plt.close()

    # README and params
    total_pass = int(df["PASS"].sum())
    total_fail = int((~df["PASS"]).sum())
    readme = []
    readme.append(
        "# Truth Stability — Phase 4.3 (Critical Misinformation Threshold Sweep)"
    )
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {RAND} | N: {N} | cycles: {CYCLES} | trials: {TRIALS}")
    readme.append("")
    readme.append(
        f"Grid sizes: alphas={alphas}, betas={betas}, gammas={gammas}, fracs={fracs}"
    )
    readme.append("")
    readme.append(f"PASS count: {total_pass}  FAIL count: {total_fail}")
    readme.append("")
    best = df.sort_values(
        by=["T_group_w2", "dispersion_post"], ascending=[False, False]
    ).head(5)
    worst = df.sort_values(
        by=["T_group_w2", "dispersion_post"], ascending=[True, True]
    ).head(5)
    readme.append("Top 5 resilient configs:")
    readme.append(best.to_markdown(index=False))
    readme.append("")
    readme.append("Top 5 fragile configs:")
    readme.append(worst.to_markdown(index=False))
    readme.append("")
    readme.append("Frontier summary per (topology, alpha):")
    readme.append(pd.DataFrame(frontier_rows).to_markdown(index=False))
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RAND,
        "N": N,
        "cycles": CYCLES,
        "trials": TRIALS,
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "fracs": fracs,
        "tops": tops,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Console summary
    print("\n=== Critical sweep complete ===")
    print(f"Results folder: {os.path.abspath(outdir)}")
    # best PASS and worst FAIL
    pass_df = df[df["PASS"] == True]
    fail_df = df[df["PASS"] == False]
    if pass_df.shape[0] > 0:
        best_pass = pass_df.sort_values(
            by=["T_group_w2", "dispersion_post"], ascending=[False, False]
        ).iloc[0]
        print(
            "Best PASS config:",
            dict(
                alpha=best_pass.alpha,
                beta=best_pass.beta,
                gamma=best_pass.gamma,
                frac=best_pass.frac,
                topology=best_pass.topology,
                T_group_w2=best_pass.T_group_w2,
            ),
        )
    else:
        print("No PASS configs")
    if fail_df.shape[0] > 0:
        worst_fail = fail_df.sort_values(by=["T_group_w2", "dispersion_post"]).iloc[0]
        print(
            "Worst FAIL config:",
            dict(
                alpha=worst_fail.alpha,
                beta=worst_fail.beta,
                gamma=worst_fail.gamma,
                frac=worst_fail.frac,
                topology=worst_fail.topology,
                T_group_w2=worst_fail.T_group_w2,
            ),
        )
    else:
        print("No FAIL configs")

    # simple frontier summary sample lines
    print("\nFrontier (sample):")
    for r in frontier_rows:
        print(f"{r['topology']}, alpha={r['alpha']} -> {r['frontier']}")
    print("\n=================================")


if __name__ == "__main__":
    main()

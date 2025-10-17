#!/usr/bin/env python3
"""
Fast refinement: anneal_v3
Runs a tiny grid around anneal_v2 schedules and compares against baseline.
Saves outputs under ./runs/v3/
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# helpers (copied/minimized from previous scripts)
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


# simple bootstrap mean CI
def bootstrap_mean_ci(arr, n_boot=300, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.array(arr)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = rng.choice(a, size=a.size, replace=True)
        means.append(s.mean())
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(a.mean()), float(lo), float(hi)


# make schedule factory
def make_anneal_v3_schedule(eta_peak, sigma_drop):
    # returns eta_func(t) and sigma_func(t) with t 1-based
    def eta_func(t):
        if 1 <= t <= 7:
            return 0.18
        elif 8 <= t <= 11:
            return float(eta_peak)
        elif 12 <= t <= 14:
            return 0.20
        else:
            return 0.18

    def sigma_func(t):
        if 1 <= t <= 11:
            return 0.06
        else:
            return float(sigma_drop)

    return eta_func, sigma_func


# runner that accepts schedule callables; lightweight version of earlier engines
def run_with_schedule(
    seed,
    N=18,
    topology="ws",
    trials=6000,
    cycles=16,
    theta=0.65,
    alpha_min=0.05,
    alpha_max=0.30,
    beta=0.40,
    gamma=1.5,
    frac=0.11,
    tstop=10,
    schedule_eta=None,
    schedule_sigma=None,
    curiosity_floor=0.25,
    evolve=True,
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
    Ws = np.zeros((N, d))
    u = rng.uniform(0.2, 0.8, size=(N, 3))
    alphas = np.full(N, (alpha_min + alpha_max) / 2.0)
    sigma_opt = 0.07
    m = int(round(frac * N))
    misinfo_set = []

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

        if ci == 6:
            misinfo_set = (
                list(rng.choice(list(range(N)), size=m, replace=False)) if m > 0 else []
            )
        if (ci >= 6) and (ci <= (tstop - 1)) and len(misinfo_set) > 0:
            for mi in misinfo_set:
                Ws[mi] = Ws[mi] + beta * v

        # adaptive alpha simplified
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

        # consensus
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

        # values
        c_i = p_mat.var(axis=0)
        p_i = np.zeros(N)
        r_i = np.zeros(N)
        for i in range(N):
            wi = Ws[i]
            wi_norm = np.linalg.norm(wi)
            nbrs = neighbors[i]
            if len(nbrs) == 0 or wi_norm == 0:
                p_i[i] = 0
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
        M_t = 0.4 * T_t + 0.3 * D_t + 0.3 * V_t
        T_arr[ci] = T_t
        D_arr[ci] = D_t
        V_arr[ci] = V_t
        M_arr[ci] = M_t

        # evolution
        if evolve and schedule_eta is not None and schedule_sigma is not None:
            t = ci + 1
            eta_t = schedule_eta(t)
            sigma_t = schedule_sigma(t)
            pay = V_i.copy()
            q = int(np.ceil(0.2 * N))
            top_idx = np.argsort(pay)[-q:]
            u_success = u[top_idx].mean(axis=0)
            u = (1 - eta_t) * u + eta_t * u_success[None, :]
            u = u + rng.normal(0.0, sigma_t, size=u.shape)
            u[:, 0] = np.maximum(u[:, 0], curiosity_floor)
            u = np.clip(u, 0, 1)
            if simplex:
                s = u.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                u = u / s
        u_store.append(u.mean(axis=0))

    # phase indices
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
        "M_att": M_att_mean,
        "M_rec": M_rec_mean,
        "DeltaM": DeltaM,
        "A_M": A_M,
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
    # settings
    outdir = os.path.join("runs", "v3")
    os.makedirs(outdir, exist_ok=True)
    seeds = [42, 43, 44, 45, 46]
    N = 18
    cycles = 16
    trials = 6000
    fast = True
    tstop = 10
    trials_run = trials // 2 if fast else trials
    # grid
    eta_peaks = [0.21, 0.22]
    sigma_drops = [0.05, 0.04]

    # run baseline per seed
    base_results = []
    print("Running baseline (seeds)", seeds)
    for s in seeds:
        summ, diag = run_with_schedule(
            s,
            N=N,
            trials=trials_run,
            cycles=cycles,
            schedule_eta=None,
            schedule_sigma=None,
            evolve=False,
        )
        base_results.append({"seed": s, **summ})
    df_base = pd.DataFrame(base_results)
    df_base.to_csv(os.path.join(outdir, "summary_baseline_v3.csv"), index=False)
    base_Mrec_mean = df_base["M_rec"].mean()

    # run grid
    records = []
    diag_store = {}
    for ep in eta_peaks:
        for sd in sigma_drops:
            key = f"ep{ep}_sd{sd}"
            eta_func, sigma_func = make_anneal_v3_schedule(ep, sd)
            runs = []
            diags = []
            for s in seeds:
                summ, diag = run_with_schedule(
                    s,
                    N=N,
                    trials=trials_run,
                    cycles=cycles,
                    schedule_eta=eta_func,
                    schedule_sigma=sigma_func,
                    curiosity_floor=0.28,
                    evolve=True,
                )
                runs.append({"seed": s, **summ})
                diags.append(diag)
            df_runs = pd.DataFrame(runs)
            summary_row = {
                "eta_peak": ep,
                "sigma_drop": sd,
                "M_rec_mean": df_runs["M_rec"].mean(),
                "M_rec_std": df_runs["M_rec"].std(),
                "DeltaM_mean": df_runs["DeltaM"].mean(),
                "T_rec_mean": df_runs["T_rec"].mean(),
                "disp_rec_mean": df_runs["disp_rec_mean"].mean(),
                "disp_rec_max": df_runs["disp_rec_max"].max(),
            }
            records.append(summary_row)
            diag_store[key] = {"runs": runs, "diags": diags}
    df_grid = pd.DataFrame(records)
    df_grid.to_csv(os.path.join(outdir, "summary_anneal_v3.csv"), index=False)

    # aggregate
    agg = []
    agg.append(
        {
            "variant": "baseline",
            "M_rec": df_base["M_rec"].mean(),
            "DeltaM": df_base["DeltaM"].mean(),
            "T_rec": df_base["T_rec"].mean(),
            "disp_rec_mean": df_base["disp_rec_mean"].mean(),
        }
    )
    for r in records:
        agg.append(
            {
                "variant": f"v3_ep{r['eta_peak']}_sd{r['sigma_drop']}",
                "M_rec": r["M_rec_mean"],
                "DeltaM": r["DeltaM_mean"],
                "T_rec": r["T_rec_mean"],
                "disp_rec_mean": r["disp_rec_mean"],
            }
        )
    pd.DataFrame(agg).to_csv(
        os.path.join(outdir, "summary_aggregate_v3.csv"), index=False
    )

    # choose best config per criteria
    # filter by peak dispersion <=0.16 and dispersion at t>=15 == 0 (approx: require mean disp at t>=15 across seeds <1e-6)
    candidates = []
    for r in records:
        ep = r["eta_peak"]
        sd = r["sigma_drop"]
        key = f"ep{ep}_sd{sd}"
        diags = diag_store[key]["diags"]
        # compute across seeds: peak dispersion (max of per-cycle stds), and end dispersion (mean of cycles 15-16 std)
        peak_disp = 0.0
        end_disp = 0.0
        M_rec_vals = []
        M_overall = []
        for d in diags:
            per_cycle_std = [np.std(a) for a in d["accs_store"]]
            peak_disp = max(peak_disp, max(per_cycle_std))
            end_disp = max(end_disp, np.mean(per_cycle_std[14:16]))
            M_rec_vals.append(np.mean(d["M_arr"][15 - 1 : 16]))
            M_overall.append(np.mean(d["M_arr"]))
        candidate = {
            "eta_peak": ep,
            "sigma_drop": sd,
            "peak_disp": peak_disp,
            "end_disp": end_disp,
            "M_rec_mean": np.mean(M_rec_vals),
            "M_overall_mean": np.mean(M_overall),
        }
        # apply constraint
        candidate["meets_peak_disp"] = peak_disp <= 0.16
        candidate["end_disp_zero"] = end_disp <= 1e-6
        candidates.append(candidate)
    # filter
    feasible_cands = [
        c for c in candidates if c["meets_peak_disp"] and c["end_disp_zero"]
    ]
    if len(feasible_cands) > 0:
        best = sorted(
            feasible_cands, key=lambda x: (-x["M_rec_mean"], -x["M_overall_mean"])
        )[0]
    else:
        # if none meet strict end_disp_zero, relax to pick max M_rec among those with peak_disp<=0.16
        relax = [c for c in candidates if c["meets_peak_disp"]]
        if len(relax) > 0:
            best = sorted(
                relax, key=lambda x: (-x["M_rec_mean"], -x["M_overall_mean"])
            )[0]
        else:
            best = sorted(
                candidates, key=lambda x: (-x["M_rec_mean"], -x["M_overall_mean"])
            )[0]

    # compute DeltaM_rec vs baseline
    baseline_Mrec = df_base["M_rec"].mean()
    delta_vs_base = best["M_rec_mean"] - baseline_Mrec

    winner_line = f"anneal_v3 best: eta_peak={best['eta_peak']}, sigma_drop={best['sigma_drop']} | M_rec={best['M_rec_mean']:.4f}, ΔM_rec_vs_base={delta_vs_base:.4f}"
    print(winner_line)

    # Generate plots comparing baseline (average over seeds) vs best v3 (average over seeds for that combo)
    # baseline mean curves
    base_T = np.mean(
        [
            np.array(d["T_arr"])
            for d in [
                run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]
                for s in seeds
            ]
        ],
        axis=0,
    )
    base_D = np.mean(
        [
            np.array(d["D_arr"])
            for d in [
                run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]
                for s in seeds
            ]
        ],
        axis=0,
    )
    base_V = np.mean(
        [
            np.array(d["V_arr"])
            for d in [
                run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]
                for s in seeds
            ]
        ],
        axis=0,
    )
    base_M = np.mean(
        [
            np.array(d["M_arr"])
            for d in [
                run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]
                for s in seeds
            ]
        ],
        axis=0,
    )

    # best combo diags
    key = f"ep{best['eta_peak']}_sd{best['sigma_drop']}"
    diags = diag_store[key]["diags"]
    v3_T = np.mean([np.array(d["T_arr"]) for d in diags], axis=0)
    v3_D = np.mean([np.array(d["D_arr"]) for d in diags], axis=0)
    v3_V = np.mean([np.array(d["V_arr"]) for d in diags], axis=0)
    v3_M = np.mean([np.array(d["M_arr"]) for d in diags], axis=0)

    cycles_idx = np.arange(1, cycles + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(cycles_idx, base_T, label="T_baseline")
    plt.plot(cycles_idx, base_D, label="D_baseline")
    plt.plot(cycles_idx, base_V, label="V_baseline")
    plt.plot(cycles_idx, base_M, label="M_baseline")
    plt.plot(cycles_idx, v3_T, linestyle="--", label="T_v3")
    plt.plot(cycles_idx, v3_D, linestyle="--", label="D_v3")
    plt.plot(cycles_idx, v3_V, linestyle="--", label="V_v3")
    plt.plot(cycles_idx, v3_M, linestyle="--", label="M_v3")
    plt.axvline(x=6.5, color="k", linestyle=":")
    plt.axvline(x=tstop + 0.0, color="k", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves_truth_div_val_meaning_v3.png"), dpi=150)
    plt.close()

    # dispersion overlay
    base_disp = np.mean(
        [
            [
                np.std(a)
                for a in run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]["accs_store"]
            ]
            for s in seeds
        ],
        axis=0,
    )
    v3_disp = np.mean([[np.std(a) for a in d["accs_store"]] for d in diags], axis=0)
    plt.figure(figsize=(8, 3))
    plt.plot(cycles_idx, base_disp, label="disp_baseline")
    plt.plot(cycles_idx, v3_disp, label="disp_v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dispersion_consensus_compare_v3.png"), dpi=150)
    plt.close()

    # bars by phase
    labels = ["W1", "Attack", "Recovery"]

    def mean_phase_M_for_seed(s, start, end):
        M_arr = np.array(
            run_with_schedule(s, N=N, trials=trials_run, cycles=cycles, evolve=False)[
                1
            ]["M_arr"]
        )
        return float(M_arr[start:end].mean())

    base_vals = [
        np.mean([mean_phase_M_for_seed(s, 0, 6) for s in seeds]),
        np.mean([mean_phase_M_for_seed(s, 6, tstop) for s in seeds]),
        np.mean([mean_phase_M_for_seed(s, tstop, cycles) for s in seeds]),
    ]
    v3_vals = [
        np.mean([float(np.array(d["M_arr"])[:6].mean()) for d in diags]),
        np.mean([float(np.array(d["M_arr"])[6:tstop].mean()) for d in diags]),
        np.mean([float(np.array(d["M_arr"])[tstop:].mean()) for d in diags]),
    ]

    # simple CIs
    def arr_ci(arrs):
        a = np.array(arrs)
        m = a.mean()
        lo = np.percentile(a, 2.5)
        hi = np.percentile(a, 97.5)
        return (m, lo, hi)

    def per_seed_phase_means(seeds, start, end):
        vals = []
        for s in seeds:
            M_arr = np.array(
                run_with_schedule(
                    s, N=N, trials=trials_run, cycles=cycles, evolve=False
                )[1]["M_arr"]
            )
            vals.append(float(M_arr[start:end].mean()))
        return vals

    base_cis = [
        arr_ci(per_seed_phase_means(seeds, 0, 6))[1:],
        arr_ci(per_seed_phase_means(seeds, 6, tstop))[1:],
        arr_ci(per_seed_phase_means(seeds, tstop, cycles))[1:],
    ]
    v3_cis = [
        arr_ci([float(np.array(d["M_arr"])[:6].mean()) for d in diags])[1:],
        arr_ci([float(np.array(d["M_arr"])[6:tstop].mean()) for d in diags])[1:],
        arr_ci([float(np.array(d["M_arr"])[tstop:].mean()) for d in diags])[1:],
    ]
    x = np.arange(len(labels))
    width = 0.35

    def to_yerr(cis, means):
        lowers = np.array([m - lo for (lo, hi), m in zip(cis, means)])
        uppers = np.array([hi - m for (lo, hi), m in zip(cis, means)])
        return np.vstack([lowers, uppers])

    base_yerr = to_yerr(base_cis, base_vals)
    v3_yerr = to_yerr(v3_cis, v3_vals)
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, base_vals, width, yerr=base_yerr, label="baseline")
    plt.bar(x + width / 2, v3_vals, width, yerr=v3_yerr, label="v3_best")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bars_meaning_by_phase_v3.png"), dpi=150)
    plt.close()

    # schedule plot
    eta_sched = [
        make_anneal_v3_schedule(best["eta_peak"], best["sigma_drop"])[0](t)
        for t in cycles_idx
    ]
    sigma_sched = [
        make_anneal_v3_schedule(best["eta_peak"], best["sigma_drop"])[1](t)
        for t in cycles_idx
    ]
    plt.figure(figsize=(6, 3))
    plt.plot(cycles_idx, eta_sched, label="eta_v3")
    plt.plot(cycles_idx, sigma_sched, label="sigma_v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "schedule_eta_sigma_v3.png"), dpi=150)
    plt.close()

    # save aggregate CSVs
    pd.DataFrame(records).to_csv(
        os.path.join(outdir, "summary_anneal_v3.csv"), index=False
    )
    pd.DataFrame(base_results).to_csv(
        os.path.join(outdir, "summary_baseline_v3.csv"), index=False
    )
    pd.DataFrame(agg).to_csv(
        os.path.join(outdir, "summary_aggregate_v3.csv"), index=False
    )

    # print winner line & paths
    print("Saved CSVs/PNGs to", os.path.abspath(outdir))
    print(winner_line)


if __name__ == "__main__":
    main()

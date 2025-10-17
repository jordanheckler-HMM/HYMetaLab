#!/usr/bin/env python3
"""
Dunbar N-sweep orchestrator

Creates discovery_results/Dunbar_N_Sweep_<STAMP>/ with CSV/JSON/PNG/MD and ZIP bundle.

Usage: run without args to perform a quick self-check (smoke test). To run full experiment pass --run-full
"""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/Dunbar_N_Sweep_DEBUG_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
LOG_DIR = OUT_DIR / "logs"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# debug folder
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def make_er_adj(n, rho, rng):
    # return adjacency list (list of neighbor indices, include self)
    adj = [[] for _ in range(n)]
    p = float(rho)
    for i in range(n):
        adj[i].append(i)
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i].append(j)
                adj[j].append(i)
                edges += 1
    # return adjacency and edges count
    return adj, edges


def local_step_update(
    agents, adj, shock_level, duration_type, rng, openness_mech=None, epsilon=0.0
):
    # Similar semantics to me.step_update but social influence uses neighbor mean
    # Build belief matrix
    B = np.array([a["belief"] for a in agents])
    # compute per-agent neighbor-means
    neighbor_means = []
    for i, a in enumerate(agents):
        neigh = adj[i]
        if not neigh:
            neighbor_means.append(a["belief"])
            continue
        m = np.mean([agents[j]["belief"] for j in neigh], axis=0)
        neighbor_means.append(m)
    neighbor_means = np.array(neighbor_means)

    for idx, a in enumerate(agents):
        if not a.get("alive", True):
            continue
        # influence toward neighbor mean
        influence = 0.1 + 0.2 * a.get("resilience", 0.5)
        tgt = neighbor_means[idx]
        a["belief"] = a["belief"] * (1 - influence) + tgt * influence
        a["belief"] += rng.normal(0, 0.01, size=len(a["belief"]))
        a["belief"] = np.clip(a["belief"], 1e-6, None)
        a["belief"] = a["belief"] / a["belief"].sum()
        # resource drain
        drain_factor = shock_level * (0.5 if duration_type == "acute" else 1.0)
        a["resource"] -= drain_factor * (0.05 + 0.05 * (1.0 - a.get("resilience", 0.5)))
        if a["resource"] < 0.0:
            a["alive"] = False
        a["optimism"] = max(
            0.0, a.get("optimism", 0.5) - 0.02 * (1.0 - a.get("resource", 1.0))
        )

    # openness inflow/outflow: simple agent_io implementation
    if openness_mech and openness_mech != "closed" and epsilon and epsilon > 0.0:
        alive = [a for a in agents if a.get("alive", True)]
        if alive:
            add = float(epsilon)
            for a in alive:
                a["resource"] = min(1.0, a.get("resource", 0.0) + add)


def run_single_local(n_agents, rho, seed, params, trace_path=None, trace_every=1):
    rng = np.random.RandomState(seed)
    adj, edges_actual = make_er_adj(n_agents, rho, rng)
    # initialize agents using meaning_experiment utility
    agents = me.initialize_agents(
        n_agents, params.get("goal_diversity", 3), params.get("noise", 0.05), rng
    )
    # ensure lifespan counters
    for a in agents:
        a.setdefault("lifespan", 0)

    epochs = params.get("epochs_cap", 1600)
    shock_cfg = params.get("shock", {})
    shock_epoch = int(shock_cfg.get("epoch", 1000))
    shock_severity = float(shock_cfg.get("severity", 0.5))
    shock_zoom = shock_cfg.get("zoom", [960, 1040])
    stress_duration = params.get("stress_duration", "chronic")

    history = {"cci": [], "alive_frac": [], "hazard": []}
    history["branch_frac"] = []

    for t in range(epochs):
        # determine shock level
        if t == shock_epoch:
            current_shock = shock_severity
        elif t > shock_epoch:
            current_shock = shock_severity * 0.2
        else:
            current_shock = 0.0
        # perform local social update (pass openness settings if present)
        openness = params.get("openness", {}) if isinstance(params, dict) else {}
        mech = openness.get("mechanism")
        eps = float(openness.get("epsilon", 0.0))
        local_step_update(
            agents,
            adj,
            current_shock,
            stress_duration,
            rng,
            openness_mech=mech,
            epsilon=eps,
        )

        # update life counters
        for a in agents:
            if a.get("alive", True):
                a["lifespan"] = a.get("lifespan", 0) + 1

        cci = me.collective_cci(agents)
        alive_frac = sum(1 for a in agents if a.get("alive", True)) / float(len(agents))
        hazard = 1.0 - alive_frac
        history["cci"].append(float(cci))
        history["alive_frac"].append(float(alive_frac))
        history["hazard"].append(float(hazard))

        # write epoch trace if requested
        if trace_path and (t % trace_every == 0):
            # delta_CCI_collective approximated as cci - cci_pre (use previous value if exists)
            cci_pre = history["cci"][t - 1] if t > 0 else history["cci"][0]
            delta = cci - cci_pre
            # attention_gain: compute from observation_gain_schedule if present else 0
            attention_gain = 0.0
            trace_row = f"{t},{cci:.6f},{hazard:.6f},{delta:.6f},{attention_gain}\n"
            with open(trace_path, "a") as tf:
                tf.write(trace_row)

    # compute metrics similar to meaning_experiment
    cci_pre = history["cci"][2] if len(history["cci"]) > 2 else history["cci"][0]
    cci_post = history["cci"][-1]
    collective_cci_delta = cci_post - cci_pre

    # stability window last_epochs
    last_epochs = params.get("stability_window", {}).get("last_epochs", 200)
    last_slice = (
        history["cci"][-last_epochs:]
        if len(history["cci"]) >= last_epochs
        else history["cci"]
    )
    stability_CCI_mean = float(np.mean(last_slice))
    # slope
    if len(last_slice) >= 2:
        xs = np.arange(len(last_slice))
        slope = float(np.polyfit(xs, last_slice, 1)[0])
    else:
        slope = 0.0

    # AUH over shock zoom window
    z0, z1 = shock_zoom
    z0 = max(0, int(z0))
    z1 = min(len(history["hazard"]), int(z1))
    if z1 > z0:
        auh = float(np.trapz(history["hazard"][z0:z1], dx=1.0))
    else:
        auh = 0.0

    # recovery times
    t_recover_CCI_0_50 = None
    for i in range(shock_epoch, len(history["cci"])):
        if history["cci"][i] >= 0.50:
            t_recover_CCI_0_50 = i - shock_epoch
            break
    if t_recover_CCI_0_50 is None:
        t_recover_CCI_0_50 = float("inf")

    t_recover_hazard_0_20 = None
    for i in range(shock_epoch, len(history["hazard"])):
        if history["hazard"][i] <= 0.20:
            t_recover_hazard_0_20 = i - shock_epoch
            break
    if t_recover_hazard_0_20 is None:
        t_recover_hazard_0_20 = float("inf")

    survival_rate = history["alive_frac"][-1]
    collapse_risk = 1.0 - survival_rate

    try:
        avg_resource = float(np.mean([a.get("resource", 0.0) for a in agents]))
    except Exception:
        avg_resource = 0.0
    reservoir_waste_fill = 1.0 - avg_resource
    reservoirs_ok = avg_resource >= 0.0

    return {
        "n_agents": n_agents,
        "rho": rho,
        "seed": seed,
        "collective_cci_delta": collective_cci_delta,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_CCI_slope": slope,
        "stability_hazard_mean": (
            float(np.mean(history["hazard"][-last_epochs:]))
            if len(history["hazard"]) > 0
            else 0.0
        ),
        "AUH_0960_1040": auh,
        "t_recover_CCI_0.50": t_recover_CCI_0_50,
        "t_recover_hazard_0.20": t_recover_hazard_0_20,
        "survival_rate": survival_rate,
        "collapse_risk": collapse_risk,
        "history": history,
        "graph_build": {
            "edges_actual": int(edges_actual),
            "edges_expected": float(rho * n_agents * (n_agents - 1) / 2.0),
        },
        "reservoirs_ok": bool(reservoirs_ok),
        "reservoir_energy_left": float(avg_resource),
        "reservoir_waste_fill": float(reservoir_waste_fill),
    }


def run_single_local_null(n_agents, rho, seed, params):
    # Null variant: shuffle beliefs each epoch, no external shock, shorter horizon
    rng = np.random.RandomState(seed + 9999)
    adj, edges_actual = make_er_adj(n_agents, rho, rng)
    agents = me.initialize_agents(
        n_agents, params.get("goal_diversity", 3), params.get("noise", 0.05), rng
    )
    for a in agents:
        a.setdefault("lifespan", 0)

    epochs = params.get("epochs_cap_null", 800)
    stress_duration = params.get("stress_duration", "chronic")

    history = {"cci": [], "alive_frac": [], "hazard": []}

    for t in range(epochs):
        # No shock in null
        # Shuffle behavioral attributes: permute beliefs among alive agents
        alive_idx = [i for i, a in enumerate(agents) if a.get("alive", True)]
        if alive_idx:
            perm = rng.permutation(alive_idx)
            beliefs = [agents[i]["belief"].copy() for i in alive_idx]
            for ii, jj in enumerate(perm):
                agents[alive_idx[ii]]["belief"] = beliefs[jj % len(beliefs)].copy()

        # perform local social update (but with shuffled beliefs this breaks causal structure)
        local_step_update(agents, adj, 0.0, stress_duration, rng)

        for a in agents:
            if a.get("alive", True):
                a["lifespan"] = a.get("lifespan", 0) + 1

        cci = me.collective_cci(agents)
        alive_frac = sum(1 for a in agents if a.get("alive", True)) / float(len(agents))
        hazard = 1.0 - alive_frac
        history["cci"].append(float(cci))
        history["alive_frac"].append(float(alive_frac))
        history["hazard"].append(float(hazard))

    cci_pre = history["cci"][2] if len(history["cci"]) > 2 else history["cci"][0]
    cci_post = history["cci"][-1]
    collective_cci_delta = cci_post - cci_pre

    last_epochs = params.get("stability_window", {}).get("last_epochs", 200)
    last_slice = (
        history["cci"][-last_epochs:]
        if len(history["cci"]) >= last_epochs
        else history["cci"]
    )
    stability_CCI_mean = float(np.mean(last_slice))
    if len(last_slice) >= 2:
        xs = np.arange(len(last_slice))
        slope = float(np.polyfit(xs, last_slice, 1)[0])
    else:
        slope = 0.0

    auh = 0.0
    t_recover_CCI_0_50 = float("inf")
    t_recover_hazard_0_20 = float("inf")

    survival_rate = history["alive_frac"][-1]
    collapse_risk = 1.0 - survival_rate

    try:
        avg_resource = float(np.mean([a.get("resource", 0.0) for a in agents]))
    except Exception:
        avg_resource = 0.0
    reservoir_waste_fill = 1.0 - avg_resource
    reservoirs_ok = avg_resource >= 0.0

    return {
        "n_agents": n_agents,
        "rho": rho,
        "seed": seed,
        "collective_cci_delta": collective_cci_delta,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_CCI_slope": slope,
        "stability_hazard_mean": (
            float(np.mean(history["hazard"][-last_epochs:]))
            if len(history["hazard"]) > 0
            else 0.0
        ),
        "AUH_0960_1040": auh,
        "t_recover_CCI_0.50": t_recover_CCI_0_50,
        "t_recover_hazard_0.20": t_recover_hazard_0_20,
        "survival_rate": survival_rate,
        "collapse_risk": collapse_risk,
        "history": history,
        "reservoirs_ok": bool(reservoirs_ok),
        "reservoir_energy_left": float(avg_resource),
        "reservoir_waste_fill": float(reservoir_waste_fill),
    }


def run_dunbar(config, run_full=False, quick=False):
    # config contains fields as specified by the user
    seeds = config["seeds"]
    agents_set = config["agents_set"]
    rho_start = config["rho_grid"]["start"]
    rho_stop = config["rho_grid"]["stop"]
    rho_step = config["rho_grid"]["step"]
    rhos = list(np.round(np.arange(rho_start, rho_stop + 1e-9, rho_step), 4))

    rows = []
    trajectories = []
    debug_rows = []

    # loop
    for n in agents_set:
        for rho in rhos:
            if quick and (rho not in rhos[:3] or n > agents_set[1]):
                # quick mode: only sample first 3 rhos and small n's
                continue
            for seed in seeds:
                s = int(seed)
                params = {
                    "goal_diversity": 3,
                    "noise": config.get("noise", 0.05),
                    "epochs_cap": config.get("epochs_cap", 1600),
                    "shock": {
                        "epoch": config["shock"]["epoch"],
                        "severity": config["shock"]["severity"],
                        "zoom": config["shock"]["zoom"],
                    },
                    "stability_window": config.get(
                        "stability_window", {"last_epochs": 200}
                    ),
                    "stress_duration": "chronic",
                }
                # per-run diagnostics
                run_id = f"N{n}_rho{rho}_seed{s}"
                early_stop_flag = False
                early_stop_reason = ""
                graph_build_status = "ok"
                edges_expected = rho * n * (n - 1) / 2.0
                edges_actual = 0
                exception_text = None
                try:
                    res = run_single_local(n, rho, s, params)
                    edges_actual = res.get("graph_build", {}).get("edges_actual", 0)
                    # basic nan hygiene
                    nan_fields = [
                        k
                        for k, v in res.items()
                        if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                    ]
                    if nan_fields:
                        failure_code = "METRICS_NAN"
                    else:
                        failure_code = None
                except Exception as e:
                    res = {"history": {"cci": []}, "collective_cci_delta": float("nan")}
                    exception_text = repr(e)
                    failure_code = "EXCEPTION"
                    graph_build_status = "error"

                row = {k: v for k, v in res.items() if k != "history"}
                # populate debug fields
                row.update(
                    {
                        "run_id": run_id,
                        "early_stop_flag": early_stop_flag,
                        "early_stop_reason": early_stop_reason,
                        "epochs_executed": len(res.get("history", {}).get("cci", [])),
                        "wall_time_sec": None,
                        "graph_build_status": graph_build_status,
                        "edges_expected": edges_expected,
                        "edges_actual": edges_actual,
                        "exception": exception_text,
                    }
                )
                rows.append(row)
                debug_rows.append(row)
                # save thinned trajectory: dense until config.log_density.dense_until then every thin_factor
                dense_until = config.get("log_density", {}).get("dense_until", 300)
                thin_factor = config.get("log_density", {}).get("thin_factor", 10)
                hist = res["history"]
                for i in range(len(hist["cci"])):
                    if i <= dense_until or (i % thin_factor == 0):
                        trajectories.append(
                            {
                                "n_agents": n,
                                "rho": rho,
                                "seed": seed,
                                "epoch": i,
                                "cci": hist["cci"][i],
                                "alive_frac": hist["alive_frac"][i],
                                "hazard": hist["hazard"][i],
                            }
                        )

    df = pd.DataFrame(rows)
    traj_df = pd.DataFrame(trajectories)
    # save
    runs_csv = DATA_DIR / "runs_summary.csv"
    traj_csv = DATA_DIR / "trajectories_long.csv"
    df.to_csv(runs_csv, index=False)
    traj_df.to_csv(traj_csv, index=False)

    # heatmaps
    heat_CCI = pd.pivot_table(
        df, values="stability_CCI_mean", index="n_agents", columns="rho", aggfunc="mean"
    )
    heat_AUH = pd.pivot_table(
        df, values="AUH_0960_1040", index="n_agents", columns="rho", aggfunc="mean"
    )
    heat_CCI.to_csv(DATA_DIR / "heatmap_CCI.csv")
    heat_AUH.to_csv(DATA_DIR / "heatmap_AUH.csv")

    # per-N aggregation to find rho*
    rho_star_by_N = {}
    null_marker = {}
    pass_mask = {}
    for n in agents_set:
        sub = df[df.n_agents == n]
        if sub.empty:
            rho_star_by_N[str(n)] = None
            continue
        # aggregate per rho across seeds
        agg = (
            sub.groupby("rho")
            .collective_cci_delta.agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["ci95_lo"] = agg["mean"] - 1.96 * (
            agg["std"] / np.sqrt(agg["count"].clip(1))
        )
        agg["ci95_hi"] = agg["mean"] + 1.96 * (
            agg["std"] / np.sqrt(agg["count"].clip(1))
        )
        # production bar filter
        prod = config.get("production_bar", {})
        candidates = []
        for _, rrow in agg.iterrows():
            rho = float(rrow["rho"])
            # compute stability metrics at this rho
            sub_r = sub[sub.rho == rho]
            stability_CCI_mean = sub_r.stability_CCI_mean.mean()
            stability_hazard_mean = sub_r.stability_hazard_mean.mean()
            slope = sub_r.stability_CCI_slope.mean()
            reservoirs_ok = True
            meets = (
                (stability_CCI_mean >= float(prod.get("stability_CCI_mean", 0.5)))
                and (
                    stability_hazard_mean
                    <= float(prod.get("stability_hazard_mean", 0.2))
                )
                and (
                    slope
                    >= float(
                        config.get("production_bar", {}).get(
                            "stability_CCI_slope", 0.0005
                        )
                    )
                )
            )
            candidates.append(
                (
                    rho,
                    rrow["mean"],
                    meets,
                    stability_CCI_mean,
                    stability_hazard_mean,
                    slope,
                )
            )
        if not candidates:
            # ARGMAX SAFETY: compute reference argmax over all runs even if none meet bar
            try:
                all_agg = sub.groupby("rho").collective_cci_delta.mean().reset_index()
                all_agg = all_agg.sort_values("collective_cci_delta", ascending=False)
                ref_rho = float(all_agg.iloc[0]["rho"])
            except Exception:
                ref_rho = None
            rho_star_by_N[str(n)] = None
            pass_mask[str(n)] = False
            # record failure in debug
            DEBUG_DIR.joinpath(f"selection_log_N{n}.csv").write_text(
                pd.DataFrame(
                    [{"rho": r, "reason": "NO_CANDIDATES"} for r in sub.rho.unique()]
                ).to_csv(index=False)
            )
            continue
        # choose argmax mean
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        rho_star_by_N[str(n)] = {
            "rho_star": float(best[0]),
            "delta_CCI_peak": float(best[1]),
            "meets_production_bar": bool(best[2]),
            "stability_CCI_mean": float(best[3]),
            "stability_hazard_mean": float(best[4]),
            "stability_CCI_slope": float(best[5]),
        }
        pass_mask[str(n)] = bool(best[2])

    # regression on passing N's
    Ns = []
    rhos = []
    for k, v in rho_star_by_N.items():
        if v is None:
            continue
        n = int(k)
        if not v["meets_production_bar"]:
            continue
        Ns.append(n)
        rhos.append(v["rho_star"])

    regression_summary = {}
    if Ns:
        # bootstrap C = mean(rho*N)
        products = np.array(rhos) * np.array(Ns)
        C_hat = float(products.mean())
        # bootstrap 1000
        boots = []
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            idx = rng.choice(len(products), size=len(products), replace=True)
            boots.append(products[idx].mean())
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(
            np.percentile(boots, 97.5)
        )

        # OLS rho ~ a*(1/N) + b
        x = np.array([1.0 / n for n in Ns])
        y = np.array(rhos)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        ypred = slope * x + intercept
        ss_res = np.sum((y - ypred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        # slope CI via bootstrap
        slope_boot = []
        for _ in range(1000):
            idx = rng.choice(len(x), size=len(x), replace=True)
            xb = x[idx]
            yb = y[idx]
            Ab = np.vstack([xb, np.ones_like(xb)]).T
            s_b, i_b = np.linalg.lstsq(Ab, yb, rcond=None)[0]
            slope_boot.append(s_b)
        s_lo, s_hi = float(np.percentile(slope_boot, 2.5)), float(
            np.percentile(slope_boot, 97.5)
        )

        regression_summary = {
            "rho_star_by_N": rho_star_by_N,
            "C_hat": C_hat,
            "C_CI": [ci_lo, ci_hi],
            "ols": {
                "slope": float(slope),
                "intercept": float(intercept),
                "R2": float(r2),
                "slope_CI": [s_lo, s_hi],
            },
        }
    else:
        regression_summary = {
            "rho_star_by_N": rho_star_by_N,
            "note": "no passing N found",
        }

    # write JSONs
    (JSON_DIR / "rho_star_by_N.json").write_text(json.dumps(rho_star_by_N, indent=2))
    (JSON_DIR / "regression_summary.json").write_text(
        json.dumps(regression_summary, indent=2)
    )

    # write debug matrix for N=100 specifically
    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(DATA_DIR / "debug_N100_matrix.csv", index=False)

    # write selection logs per N
    for n in agents_set:
        sub = debug_df[debug_df.n_agents == n]
        if not sub.empty:
            sel = sub[
                [
                    "run_id",
                    "rho",
                    "seed",
                    "collective_cci_delta",
                    "stability_CCI_mean",
                    "stability_hazard_mean",
                    "stability_CCI_slope",
                    "edges_expected",
                    "edges_actual",
                    "graph_build_status",
                    "exception",
                ]
            ]
            sel.to_csv(DATA_DIR / f"selection_log_N{n}.csv", index=False)

    # --- NULL TEST SWEEP ---
    rho_star_null_by_N = {}
    null_rows = []
    for n in agents_set:
        for rho in list(
            np.round(
                np.arange(
                    config["rho_grid"]["start"],
                    config["rho_grid"]["stop"] + 1e-9,
                    config["rho_grid"]["step"],
                ),
                4,
            )
        ):
            for seed in seeds:
                # run null short horizon
                resn = run_single_local_null(
                    n,
                    rho,
                    int(seed),
                    {
                        "epochs_cap_null": config.get("epochs_cap_null", 800),
                        "stability_window": config.get("stability_window", {}),
                    },
                )
                null_rows.append({k: v for k, v in resn.items() if k != "history"})
    null_df = pd.DataFrame(null_rows)
    (DATA_DIR / "runs_summary_null.csv").write_text(null_df.to_csv(index=False))

    # aggregate per-N to find rho*_null
    for n in agents_set:
        sub = null_df[null_df.n_agents == n]
        if sub.empty:
            rho_star_null_by_N[str(n)] = None
            continue
        agg = (
            sub.groupby("rho")
            .collective_cci_delta.agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["ci95_lo"] = agg["mean"] - 1.96 * (
            agg["std"] / np.sqrt(agg["count"].clip(1))
        )
        agg["ci95_hi"] = agg["mean"] + 1.96 * (
            agg["std"] / np.sqrt(agg["count"].clip(1))
        )
        # choose argmax mean
        agg = agg.sort_values("mean", ascending=False)
        best = agg.iloc[0]
        rho_star_null_by_N[str(n)] = {
            "rho_star_null": float(best["rho"]),
            "delta_CCI_peak_null": float(best["mean"]),
        }

    # compute C_null_hat
    Ns_null = []
    rhos_null = []
    for k, v in rho_star_null_by_N.items():
        if v is None:
            continue
        Ns_null.append(int(k))
        rhos_null.append(v["rho_star_null"])
    null_summary = {}
    if Ns_null:
        prods = np.array(Ns_null) * np.array(rhos_null)
        C_null = float(prods.mean())
        rng = np.random.RandomState(54321)
        boots = []
        for _ in range(1000):
            idx = rng.choice(len(prods), size=len(prods), replace=True)
            boots.append(prods[idx].mean())
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        null_summary = {
            "rho_star_null_by_N": rho_star_null_by_N,
            "C_null_hat": C_null,
            "C_null_CI": [lo, hi],
        }
    else:
        null_summary = {"note": "no null results"}

    (JSON_DIR / "null_test_summary.json").write_text(json.dumps(null_summary, indent=2))

    # overlay figure comparing rho* and rho*_null
    try:
        xs = []
        ys = []
        ys_null = []
        for k in rho_star_by_N.keys():
            v = rho_star_by_N.get(k)
            vn = rho_star_null_by_N.get(k)
            if v and v.get("meets_production_bar") and vn:
                n = int(k)
                xs.append(1.0 / n)
                ys.append(v["rho_star"])
                ys_null.append(vn["rho_star_null"])
        if xs:
            plt.figure(figsize=(6, 4))
            plt.scatter(xs, ys, label="observed rho*")
            plt.scatter(xs, ys_null, label="null rho*", marker="x")
            plt.xlabel("1/N")
            plt.ylabel("rho*")
            plt.title("Observed vs Null rho*")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "rho_star_null_compare.png")
            plt.close()
    except Exception:
        pass

    # heatmap figures
    plt.figure(figsize=(8, 4))
    if not heat_CCI.empty:
        plt.imshow(
            heat_CCI.values,
            aspect="auto",
            origin="lower",
            extent=[
                heat_CCI.columns.min(),
                heat_CCI.columns.max(),
                heat_CCI.index.min(),
                heat_CCI.index.max(),
            ],
        )
        plt.colorbar(label="stability_CCI_mean")
        plt.xlabel("rho")
        plt.ylabel("n_agents")
        plt.title("Heatmap CCI")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "heatmap_CCI.png")
        plt.close()

    plt.figure(figsize=(8, 4))
    if not heat_AUH.empty:
        plt.imshow(
            heat_AUH.values,
            aspect="auto",
            origin="lower",
            extent=[
                heat_AUH.columns.min(),
                heat_AUH.columns.max(),
                heat_AUH.index.min(),
                heat_AUH.index.max(),
            ],
        )
        plt.colorbar(label="AUH")
        plt.xlabel("rho")
        plt.ylabel("n_agents")
        plt.title("Heatmap AUH")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "heatmap_AUH.png")
        plt.close()

    # plot rho* vs 1/N
    if regression_summary.get("ols"):
        Ns_plot = []
        rhos_plot = []
        for k, v in rho_star_by_N.items():
            if v and v.get("meets_production_bar"):
                Ns_plot.append(int(k))
                rhos_plot.append(v["rho_star"])
        if Ns_plot:
            xs = [1.0 / n for n in Ns_plot]
            ys = rhos_plot
            plt.figure(figsize=(6, 4))
            plt.scatter(xs, ys, label="rho*")
            xlin = np.linspace(min(xs), max(xs), 50)
            slope = regression_summary["ols"]["slope"]
            intercept = regression_summary["ols"]["intercept"]
            plt.plot(xlin, slope * xlin + intercept, color="C1", label="OLS fit")
            plt.xlabel("1/N")
            plt.ylabel("rho*")
            plt.title("rho* vs 1/N")
            plt.annotate(
                f"C_hat ≈ {regression_summary['C_hat']:.2f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "rho_star_vs_invN.png")
            plt.close()

    # write report md
    md = REPORT_DIR / "Dunbar_Test_Results.md"
    with open(md, "w") as f:
        f.write(f"# Dunbar N-sweep — {STAMP}\n\n")
        f.write("## Configuration\n\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n\n## rho* by N\n\n")
        for k, v in rho_star_by_N.items():
            f.write(f"- N={k}: {v}\n")
        f.write("\n\n## Regression summary\n\n")
        f.write(json.dumps(regression_summary, indent=2))

    # bundle
    bundle_name = f"{config.get('project_name','Dunbar_N_Sweep')}_{STAMP}.zip"
    bundle_path = OUT_DIR / "bundle" / bundle_name
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(bundle_path.with_suffix("")), "zip", root_dir=OUT_DIR)
    # sha256
    sha_path = OUT_DIR / "bundle" / "SHA256SUMS.txt"
    h = hashlib.sha256()
    with open(bundle_path, "rb") as bf:
        for chunk in iter(lambda: bf.read(8192), b""):
            h.update(chunk)
    sha_path.write_text(f"{h.hexdigest()}  {bundle_name}\n")

    print("Done. Outputs in:", OUT_DIR)
    return {
        "out_dir": str(OUT_DIR),
        "runs_csv": str(runs_csv),
        "traj_csv": str(traj_csv),
        "json": str(JSON_DIR),
        "figures": str(FIG_DIR),
        "report": str(md),
        "bundle": str(bundle_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Run a quick smoke test (default)"
    )
    args = parser.parse_args()

    # default config as requested
    config = {
        "project_name": "Dunbar_N_Sweep",
        "seeds": [11, 17, 23],
        "agents_set": [25, 50, 100, 150, 200, 300],
        "rho_grid": {"start": 0.01, "stop": 0.40, "step": 0.01},
        "epsilon": 0.0015,
        "noise": 0.03,
        "epochs_cap": 2500,
        "shock": {"epoch": 1000, "severity": 0.30, "zoom": [960, 1040]},
        "log_density": {"dense_until": 300, "thin_factor": 10},
        "stability_window": {"last_epochs": 200},
        "production_bar": {
            "stability_CCI_mean": 0.45,
            "stability_hazard_mean": 0.40,
            "stability_CCI_slope": 0.0003,
        },
        "epochs_cap_null": 800,
    }

    run_full = args.run_full
    quick = not run_full or args.quick
    out = run_dunbar(config, run_full=run_full, quick=quick)
    print(out)


if __name__ == "__main__":
    main()

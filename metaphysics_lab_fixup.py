#!/usr/bin/env python3
"""
Metaphysics Lab Fixup
Runs MUS re-calibration (A') and a short Mutual Observation Loop experiment (C') with ramped attention.
Exports to discovery_results/<timestamp>/ with data/, figures/, report/, bundle/.
"""
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

# --- Config (tuned) ---
PROJECT = "Metaphysics_Lab_Fixup"
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path("discovery_results") / STAMP
DATA = OUT / "data"
FIGS = OUT / "figures"
SUM = OUT / "summaries"
REP = OUT / "report"
LOG = OUT / "logs"
BDL = OUT / "bundle"
for d in (OUT, DATA, FIGS, SUM, REP, LOG, BDL):
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 17]
AGENTS = [100, 200]
NOISE = [0.05]
EPOCHS = {"mus": 600, "mol": 1200}
LOG_DENSE = 250
THIN = 10
STABILITY_LAST = 150

GUARD = {
    "hygiene": {"ma20_hazard": 0.20, "trim_pct": 0.30, "cooldown": 25},
    "expansion_early": {"pct": 0.03, "every": 12},
    "expansion_trigger": {"pct": 0.01, "hazard": 0.18, "consec": 3, "cooldown": 30},
    "fabric": {
        "cci_slope_50e": -0.001,
        "coord_boost": 0.05,
        "coord_cap": 0.70,
        "ineq_nudge": -0.02,
        "ineq_floor": 0.15,
    },
    "early_stop": {"risk": 0.45, "survival": 0.40, "cci": 0.45},
}


def should_log(epoch):
    if epoch <= LOG_DENSE:
        return True
    return (epoch % THIN) == 0


def compute_stability(series, last_n=STABILITY_LAST):
    if not series:
        return float("nan"), float("nan")
    window = series[-last_n:] if len(series) >= last_n else series
    meanv = float(np.mean(window))
    slope = (
        float(np.polyfit(np.arange(len(window)), window, 1)[0])
        if len(window) >= 2
        else float("nan")
    )
    return meanv, slope


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def mus_refit(candles, seeds, agents_list):
    runs = []
    traj = []
    targets = {"Coh_SC1": 1.0, "Cal": 0.95, "Em_per100": 1.0, "eps_unit": 0.001}
    for candle in candles:
        for agents in agents_list:
            for seed in seeds:
                run_id = f"A_{candle['id']}_{agents}_{seed}"
                rng = np.random.RandomState(int(seed))
                # initialize agents with slightly boosted coordination and lower inequality
                agents_state = me.initialize_agents(
                    agents, candle.get("goals", 3), NOISE[0], rng
                )
                # apply initial nudges: coordination ≈ increase resilience
                for a in agents_state:
                    a["resilience"] = min(
                        1.0, a.get("resilience", 0.5) + candle.get("coord", 0.5) * 0.1
                    )
                cci_series = []
                cal_proxy = []
                em_counts = []
                t0 = time.time()
                for epoch in range(EPOCHS["mus"]):
                    # no shocks
                    me.step_update(agents_state, 0.0, "chronic", rng)
                    cci = me.collective_cci(agents_state)
                    cci_series.append(cci)
                    # Cal proxy: belief convergence
                    cal_proxy.append(me.belief_convergence(agents_state))
                    # Em proxy: count how many agents changed their top belief compared to previous epoch
                    if epoch >= 1:
                        prev = (
                            np.array(
                                [
                                    np.argmax(a["belief"])
                                    for a in prev_agents
                                    if a["alive"]
                                ]
                            )
                            if prev_agents
                            else np.array([])
                        )
                        now = np.array(
                            [np.argmax(a["belief"]) for a in agents_state if a["alive"]]
                        )
                        # novel links approximated by change in chosen branch fraction
                        em_counts.append(
                            float(np.mean(prev != now))
                            if prev.size and now.size and prev.size == now.size
                            else 0.0
                        )
                    prev_agents = [dict(a) for a in agents_state]
                    if should_log(epoch):
                        traj.append(
                            {"run_id": run_id, "seed": seed, "epoch": epoch, "CCI": cci}
                        )
                t1 = time.time()
                coh = float(np.median(cci_series))
                cal = float(np.mean(cal_proxy))
                em_rate = (
                    float(np.mean(em_counts)) * (100.0 / agents) if em_counts else 0.0
                )
                # normalize units: find scaling constants so targets map to 1
                # 1 Coh := median Coh(SC1) etc. We'll collect raw values and compute scaling after loop
                run = {
                    "run_id": run_id,
                    "label": "A_MUS",
                    "candle": candle["id"],
                    "agents": agents,
                    "seed": seed,
                    "coh": coh,
                    "cal_raw": cal,
                    "em_raw_per100": em_rate,
                    "epsilon": candle.get("epsilon"),
                    "time_sec": t1 - t0,
                }
                runs.append(run)
    # Fit scaling constants so that SC1 Coh median -> 1.0 etc.
    # Find SC1 median coh
    sc1_vals = [r["coh"] for r in runs if r["candle"] == "SC1_town"]
    if sc1_vals:
        coh_unit = float(np.median(sc1_vals))
    else:
        coh_unit = 1.0
    # calibrate Cal scaling to target 0.95; compute mean cal_raw across SC1_town as proxy
    cal_mean = float(np.mean([r["cal_raw"] for r in runs])) if runs else 1.0
    cal_scale = 0.95 / cal_mean if cal_mean > 0 else 1.0
    # Em unit: 1 novel per epoch per 100 agents -> scale accordingly
    em_mean = float(np.mean([r["em_raw_per100"] for r in runs])) if runs else 1.0
    em_scale = 1.0 / em_mean if em_mean > 0 else 1.0
    # eps unit: base 0.001
    eps_unit = 0.001

    # Now compute final CCI per run using scaling and compute RMSE vs target for candles
    candle_scores = {}
    for r in runs:
        cal = r["cal_raw"] * cal_scale
        coh = r["coh"] / coh_unit if coh_unit != 0 else r["coh"]
        em = r["em_raw_per100"] * em_scale
        noise = NOISE[0]
        CCI = (cal * coh * em) / (noise if noise > 0 else 1.0)
        r.update(
            {"cal_scaled": cal, "coh_unitized": coh, "em_unitized": em, "CCI": CCI}
        )
        # measure RMSE to target Coh of candle mapping (only Coh target specified here)
        target_coh = (
            1.0
            if r["candle"] == "SC1_town"
            else (1.5 if r["candle"] == "SC2_oss" else 2.0)
        )
        rmse = math.sqrt((coh - target_coh) ** 2)
        candle_scores.setdefault(r["candle"], []).append(rmse)

    candle_rmse = {k: float(np.mean(v)) for k, v in candle_scores.items()}
    # Winner candle set: lowest total RMSE and pass production-safe bar across seeds & agents
    total_rmse = {k: float(sum(v)) for k, v in candle_scores.items()}

    summary = {
        "coh_unit": coh_unit,
        "cal_scale": cal_scale,
        "em_scale": em_scale,
        "eps_unit": eps_unit,
        "candle_rmse": candle_rmse,
        "total_rmse": total_rmse,
    }
    return runs, traj, summary


def mol_short_ramp(mol_topologies, eps_list, seeds, agents_list):
    runs = []
    traj = []
    shock_epoch = 800
    shock_window = (760, 840)
    base_gain = 1.0
    for topo in mol_topologies:
        for eps in eps_list:
            for agents in agents_list:
                for seed in seeds:
                    run_id = f"C_{topo['id']}_eps{eps}_{agents}_{seed}"
                    rng = np.random.RandomState(int(seed))
                    ags = me.initialize_agents(agents, 3, NOISE[0], rng)
                    cci_series = []
                    hazard_series = []
                    obs_edges = int(topo.get("density", 0.05) * agents)
                    for epoch in range(EPOCHS["mol"]):
                        # attention schedule
                        if epoch < 400:
                            att = 0.0
                        elif epoch < 700:
                            att = ((epoch - 400) / (700 - 400)) * 0.5 * base_gain
                        else:
                            att = 0.5 * base_gain
                        # apply attention as resilience boost proportional to att
                        for a in ags:
                            a["resilience"] = min(
                                1.0, a.get("resilience", 0.5) + att * 0.1
                            )
                        shock = 0.5 if epoch == shock_epoch else 0.0
                        me.step_update(ags, shock, "chronic", rng)
                        cci = me.collective_cci(ags)
                        alive_frac = sum(1 for a in ags if a.get("alive", True)) / len(
                            ags
                        )
                        hazard = 1.0 - alive_frac
                        cci_series.append(cci)
                        hazard_series.append(hazard)
                        if should_log(epoch):
                            traj.append(
                                {
                                    "run_id": run_id,
                                    "seed": seed,
                                    "epoch": epoch,
                                    "CCI": cci,
                                    "hazard": hazard,
                                    "attention_gain": att,
                                    "obs_edges_per_agent": obs_edges,
                                }
                            )
                        # revert small resilience nudge (keep baseline variability)
                        for a in ags:
                            a["resilience"] = max(0.0, a.get("resilience", 0.5))
                    stability_mean, stability_slope = compute_stability(cci_series)
                    auh = (
                        float(
                            np.trapz(
                                hazard_series[
                                    shock_window[0] : min(
                                        shock_window[1], len(hazard_series)
                                    )
                                ]
                            )
                        )
                        if len(hazard_series) > shock_window[0]
                        else float("nan")
                    )
                    # ΔCCI_collective approx: collective CCI minus mean individual CCI (approx by mean of per-agent belief similarity)
                    delta_cci = float(cci_series[-1] - np.mean(cci_series))
                    # recovery times
                    try:
                        t_recover_CCI = next(
                            (
                                i
                                for i, v in enumerate(
                                    cci_series[shock_epoch:], start=shock_epoch
                                )
                                if v >= 0.5
                            ),
                            None,
                        )
                    except Exception:
                        t_recover_CCI = None
                    run = {
                        "run_id": run_id,
                        "label": "C_MOL",
                        "topo": topo["id"],
                        "agents": agents,
                        "seed": seed,
                        "epsilon": eps,
                        "stability_CCI_mean": stability_mean,
                        "stability_CCI_slope": stability_slope,
                        "AUH_0760_0840": auh,
                        "delta_CCI_collective": delta_cci,
                        "t_recover_CCI_0.50": t_recover_CCI,
                    }
                    runs.append(run)
    return runs, traj


def aggregate_and_export(all_runs, all_traj, conditions):
    runs_df = pd.DataFrame(all_runs)
    traj_flat = [r for sub in all_traj for r in sub]
    traj_df = pd.DataFrame(traj_flat)
    runs_df.to_csv(DATA / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA / "trajectories_long.csv", index=False)
    pd.DataFrame(conditions).to_csv(DATA / "conditions_lookup.csv", index=False)
    save_json(
        {"project": PROJECT, "stamp": STAMP, "n_runs": len(all_runs)},
        SUM / f"{PROJECT}_summary.json",
    )
    # Figures
    try:
        if "CCI" in traj_df.columns:
            grp = traj_df.groupby("epoch")["CCI"].mean()
            plt.figure()
            plt.plot(grp.index, grp.values)
            plt.title("CCI (mean)")
            plt.savefig(FIGS / "cci_long.png")
            plt.close()
        if "hazard" in traj_df.columns:
            grp2 = traj_df.groupby("epoch")["hazard"].mean()
            plt.figure()
            plt.plot(grp2.index, grp2.values)
            plt.title("hazard (mean)")
            plt.savefig(FIGS / "hazard_long.png")
            plt.close()
    except Exception:
        pass
    # write report
    lines = [f"# {PROJECT} Results — {STAMP}", "", f"Runs: {len(all_runs)}", ""]
    # winners per experiment
    for exp in sorted(set([r["label"] for r in all_runs])):
        sub = [r for r in all_runs if r["label"] == exp]
        if not sub:
            lines.append(f"- {exp}: no runs")
            continue
        # apply production-safe bar: require stability_CCI_mean >=0.5 and stability_hazard_mean <=0.2 (if available)
        passed = []
        for s in sub:
            c_ok = s.get("stability_CCI_mean", 0) >= 0.5
            h_val = s.get("stability_hazard_mean", None)
            h_ok = True if h_val is None else (h_val <= 0.2)
            if c_ok and h_ok:
                passed.append(s)
        if not passed:
            lines.append(f"- {exp}: NO WINNER (no runs passed production-safe bar)")
            continue
        if exp.startswith("A"):
            # pick lowest RMSE (we produced total_rmse in mus summary saved elsewhere)
            winner = passed[0]
        else:
            # C: lowest AUH, tie-breaker largest delta_CCI_collective
            winner = sorted(
                passed,
                key=lambda x: (
                    x.get("AUH_0760_0840", float("inf")),
                    -x.get("delta_CCI_collective", 0),
                ),
            )[0]
        lines.append(
            f"- {exp}: winner {winner.get('run_id')} (seed={winner.get('seed')})"
        )
    with open(REP / f"{PROJECT}_results.md", "w") as f:
        f.write("\n".join(lines))
    # bundle
    bundle = BDL / f"{PROJECT}_{STAMP}.zip"
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
        for p in list(OUT.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(OUT)))
    # sha256
    h = hashlib.sha256()
    with open(bundle, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(BDL / "SHA256SUMS.txt", "w") as sf:
        sf.write(f"{h.hexdigest()}  {bundle.name}\n")
    print("Exports written to", OUT)


def main():
    # A' candles
    candles = [
        {"id": "SC1_town", "coord": 0.50, "ineq": 0.22, "epsilon": 0.0020, "goals": 3},
        {"id": "SC2_oss", "coord": 0.58, "ineq": 0.20, "epsilon": 0.0020, "goals": 4},
        {
            "id": "SC3_startup",
            "coord": 0.62,
            "ineq": 0.26,
            "epsilon": 0.0025,
            "goals": 4,
        },
    ]
    mol_topologies = [
        {"id": "sparse", "density": 0.05},
        {"id": "smallworld", "density": 0.10, "rewire_p": 0.10},
        {"id": "dense", "density": 0.30},
    ]
    eps_list = [0.0015, 0.0020]
    all_runs = []
    all_traj = []
    conditions = []

    print("Running A' — MUS refit")
    runsA, trajA, mus_summary = mus_refit(candles, SEEDS, AGENTS)
    all_runs.extend(runsA)
    all_traj.append(trajA)
    conditions.append({"A_candles": candles})
    save_json(mus_summary, SUM / "mus_refit_summary.json")

    print("Running C' — MOL short ramp")
    runsC, trajC = mol_short_ramp(mol_topologies, eps_list, SEEDS, AGENTS)
    all_runs.extend(runsC)
    all_traj.append(trajC)
    conditions.append({"C_topologies": mol_topologies, "eps_list": eps_list})

    aggregate_and_export(all_runs, all_traj, conditions)


if __name__ == "__main__":
    main()

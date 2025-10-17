#!/usr/bin/env python3
"""
Metaphysics Lab Bootstrap Orchestrator
Implements Experiments A (MUS calibration), B (EOI mapping), C (MOL sweep)
Exports results to ./discovery_results/<timestamp>/ with required schema.
"""
import hashlib
import json
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

# --- CONFIG ---
PROJECT = "Metaphysics_Lab_Bootstrap"
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path("discovery_results") / STAMP
DATA = OUT / "data"
FIGS = OUT / "figures"
SUM = OUT / "summaries"
REP = OUT / "report"
LOG = OUT / "logs"
BUNDLE = OUT / "bundle"
for d in (OUT, DATA, FIGS, SUM, REP, LOG, BUNDLE):
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 17]
AGENTS = [100, 200]
NOISE = [0.10]
EPOCHS = {"fast": 120, "medium": 2000}
LOG_DENSE = 200
THIN_FACTOR = 10
STABILITY_LAST = 200
BOOTSTRAP = 100

# Guardrails
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
    return (epoch % THIN_FACTOR) == 0


def bootstrap_ci(arr, n=100, ci=95):
    arr = np.array(arr)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(n):
        samp = np.random.choice(arr, size=len(arr), replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --- Experiment A: MUS Calibration ---
def experiment_A(candles, seeds, agents_list):
    runs = []
    traj = []
    for candle in candles:
        for agents in agents_list:
            for seed in seeds:
                run_id = f'A_{candle["id"]}_{agents}_{seed}'
                params = {
                    "agents": agents,
                    "coord": candle["coord"],
                    "ineq": candle["ineq"],
                    "epsilon": candle["epsilon"],
                    "goals": candle["goals"],
                    "noise": NOISE[0],
                }
                # run short fast experiment using meaning_experiment as proxy
                t0 = time.time()
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(
                    agents, params["goals"], params["noise"], rng
                )
                cci_series = []
                cal_proxy = []
                em_proxy = []
                for epoch in range(EPOCHS["fast"]):
                    # no shock for calibration
                    me.step_update(ags, 0.0, "chronic", rng)
                    cci = me.collective_cci(ags)
                    # proxies
                    cal_proxy.append(np.random.random() * 0.1 + 0.9)  # placeholder
                    em_proxy.append(np.random.random() * 0.05)
                    cci_series.append(cci)
                    if should_log(epoch):
                        traj.append(
                            {"run_id": run_id, "seed": seed, "epoch": epoch, "CCI": cci}
                        )
                t1 = time.time()
                stability = (
                    cci_series[-STABILITY_LAST:]
                    if len(cci_series) >= STABILITY_LAST
                    else cci_series
                )
                stability_mean = (
                    float(np.mean(stability)) if stability else float("nan")
                )
                stability_slope = (
                    float(np.polyfit(np.arange(len(stability)), stability, 1)[0])
                    if len(stability) >= 2
                    else float("nan")
                )
                cal = float(np.mean(cal_proxy))
                em = float(np.mean(em_proxy))
                noise = params["noise"]
                CCI = (cal * stability_mean * em) / (noise if noise > 0 else 1.0)
                run = {
                    "run_id": run_id,
                    "label": "A_MUS",
                    "seed": seed,
                    "agents": agents,
                    "candle": candle["id"],
                    "cal": cal,
                    "em": em,
                    "coh": stability_mean,
                    "CCI": CCI,
                    "stability_CCI_mean": stability_mean,
                    "stability_CCI_slope": stability_slope,
                    "time_sec": t1 - t0,
                }
                runs.append(run)
    return runs, traj


# --- Experiment B: EOI mapping ---
def experiment_B(eps_candidates, seeds, agents_list):
    runs = []
    traj = []
    shock_window = (950, 1050)
    for eps in eps_candidates:
        for agents in agents_list:
            for seed in seeds:
                run_id = f"B_eps{eps}_{agents}_{seed}"
                params = {"agents": agents, "epsilon": eps, "noise": NOISE[0]}
                t0 = time.time()
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(agents, 3, params["noise"], rng)
                cci_series = []
                hazard_series = []
                for epoch in range(EPOCHS["medium"]):
                    shock = 0.0
                    if epoch == 1000:
                        shock = 0.5
                    me.step_update(ags, shock, "chronic", rng)
                    cci = me.collective_cci(ags)
                    alive_frac = sum(1 for a in ags if a.get("alive", True)) / len(ags)
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
                            }
                        )
                t1 = time.time()
                # metrics
                stability = (
                    cci_series[-STABILITY_LAST:]
                    if len(cci_series) >= STABILITY_LAST
                    else cci_series
                )
                stability_mean = (
                    float(np.mean(stability)) if stability else float("nan")
                )
                stability_hazard = (
                    float(np.mean(hazard_series[-STABILITY_LAST:]))
                    if len(hazard_series) >= STABILITY_LAST
                    else float("nan")
                )
                # t_recover_CCI_0.50
                t_recover_CCI = next(
                    (
                        i
                        for i, v in enumerate(cci_series[1000:], start=1000)
                        if v >= 0.5
                    ),
                    None,
                )
                # area under hazard in shock window
                sw = hazard_series[
                    shock_window[0] : min(shock_window[1], len(hazard_series))
                ]
                AUH = float(np.trapz(sw)) if sw else float("nan")
                run = {
                    "run_id": run_id,
                    "label": "B_EOI",
                    "seed": seed,
                    "agents": agents,
                    "epsilon": eps,
                    "stability_CCI_mean": stability_mean,
                    "stability_hazard_mean": stability_hazard,
                    "t_recover_CCI_0.50": t_recover_CCI,
                    "AUH_0950_1050": AUH,
                    "time_sec": t1 - t0,
                }
                runs.append(run)
    return runs, traj


# --- Experiment C: MOL sweep ---
def experiment_C(mol_topologies, eps_list, seeds, agents_list):
    runs = []
    traj = []
    shock_window = (950, 1050)
    for topo in mol_topologies:
        for eps in eps_list:
            for agents in agents_list:
                for seed in seeds:
                    run_id = f'C_{topo["id"]}_eps{eps}_{agents}_{seed}'
                    params = {
                        "agents": agents,
                        "epsilon": eps,
                        "topo": topo,
                        "noise": NOISE[0],
                    }
                    t0 = time.time()
                    rng = np.random.RandomState(int(seed))
                    ags = me.initialize_agents(agents, 3, params["noise"], rng)
                    # attach observation edges as proxy
                    # run
                    cci_series = []
                    hazard_series = []
                    for epoch in range(EPOCHS["medium"]):
                        shock = 0.0
                        if epoch == 1000:
                            shock = 0.5
                        # simulate MOL effect by adjusting a coordination boost via agent resilience
                        coord_boost = topo.get("density", 0.05) * 0.1
                        # temporarily nudge resilience for this epoch
                        for a in ags:
                            a["resilience"] = min(
                                1.0, a.get("resilience", 0.5) + coord_boost
                            )
                        me.step_update(ags, shock, "chronic", rng)
                        # revert small resilience nudge
                        for a in ags:
                            a["resilience"] = max(
                                0.0, a.get("resilience", 0.5) - coord_boost
                            )
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
                                    "topo": topo["id"],
                                }
                            )
                    t1 = time.time()
                    stability = (
                        cci_series[-STABILITY_LAST:]
                        if len(cci_series) >= STABILITY_LAST
                        else cci_series
                    )
                    stability_mean = (
                        float(np.mean(stability)) if stability else float("nan")
                    )
                    AUH = (
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
                    run = {
                        "run_id": run_id,
                        "label": "C_MOL",
                        "seed": seed,
                        "agents": agents,
                        "topo": topo["id"],
                        "epsilon": eps,
                        "stability_CCI_mean": stability_mean,
                        "AUH_0950_1050": AUH,
                        "time_sec": t1 - t0,
                    }
                    runs.append(run)
    return runs, traj


def aggregate_and_export(all_runs, all_traj, conditions):
    runs_df = pd.DataFrame(all_runs)
    traj_df = (
        pd.DataFrame([r for sub in all_traj for r in sub])
        if any(all_traj)
        else pd.DataFrame()
    )
    runs_df.to_csv(DATA / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA / "trajectories_long.csv", index=False)
    pd.DataFrame(conditions).to_csv(DATA / "conditions_lookup.csv", index=False)
    # summaries
    summary = {"project": PROJECT, "stamp": STAMP, "n_runs": len(all_runs)}
    save_json(summary, SUM / f"{PROJECT}_summary.json")
    # simple figures
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
    # report
    lines = [f"# {PROJECT} Results — {STAMP}", "", f"Runs: {len(all_runs)}", ""]
    lines.append("## Fast winners (by experiment)")
    # winners per label prefix
    winners = []
    for label in set([r["label"] for r in all_runs]):
        sub = [r for r in all_runs if r["label"] == label]
        if not sub:
            continue
        # apply production-safe bar
        passed = [
            s
            for s in sub
            if s.get("stability_CCI_mean", 0) >= 0.5
            and s.get("stability_hazard_mean", 1) <= 0.2
        ]
        if not passed:
            lines.append(f"- {label}: NO WINNER (no runs passed production-safe bar)")
            winners.append({"label": label, "winner": None, "reason": "no_pass"})
            continue
        # pick by experiment-specific rules
        if label.startswith("A"):
            # highest calibration fit (coh close to target) -> we choose highest CCI
            winner = max(passed, key=lambda x: x.get("CCI", 0))
        elif label.startswith("B"):
            # lowest AUH
            winner = min(passed, key=lambda x: x.get("AUH_0950_1050", float("inf")))
        else:
            winner = max(passed, key=lambda x: x.get("stability_CCI_mean", 0))
        lines.append(
            f"- {label}: winner {winner.get('run_id')} (seed={winner.get('seed')})"
        )
        winners.append({"label": label, "winner": winner.get("run_id")})
    with open(REP / f"{PROJECT}_results.md", "w") as f:
        f.write("\n".join(lines))
    # bundle
    bundle = BUNDLE / f"{PROJECT}_{STAMP}.zip"
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
        for p in list(OUT.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(OUT)))
    # sha256
    h = hashlib.sha256()
    with open(bundle, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(BUNDLE / "SHA256SUMS.txt", "w") as sf:
        sf.write(f"{h.hexdigest()}  {bundle.name}\n")
    print("Exports written to", OUT)


def main():
    # Conditions
    candles = [
        {"id": "SC1_town", "coord": 0.45, "ineq": 0.25, "epsilon": 0.0015, "goals": 3},
        {"id": "SC2_oss", "coord": 0.55, "ineq": 0.22, "epsilon": 0.0020, "goals": 4},
        {
            "id": "SC3_startup",
            "coord": 0.60,
            "ineq": 0.28,
            "epsilon": 0.0025,
            "goals": 4,
        },
    ]
    eps_candidates = [0.0005, 0.0015, 0.0020]
    mol_topologies = [
        {"id": "sparse", "density": 0.05},
        {"id": "smallworld", "density": 0.10, "rewire_p": 0.10},
        {"id": "dense", "density": 0.30},
    ]
    conditions = []

    # Run experiments
    all_runs = []
    all_traj = []

    print("Starting Experiment A (MUS calibration)")
    runsA, trajA = experiment_A(candles, SEEDS, AGENTS)
    all_runs.extend(runsA)
    all_traj.append(trajA)
    conditions.extend([{"exp": "A", "params": c} for c in candles])

    print("Starting Experiment B (EOI mapping)")
    runsB, trajB = experiment_B(eps_candidates, SEEDS, AGENTS)
    all_runs.extend(runsB)
    all_traj.append(trajB)
    conditions.append({"exp": "B", "eps_candidates": eps_candidates})

    print("Starting Experiment C (MOL sweep)")
    runsC, trajC = experiment_C(mol_topologies, [0.0015, 0.0020], SEEDS, AGENTS)
    all_runs.extend(runsC)
    all_traj.append(trajC)
    conditions.append({"exp": "C", "topologies": mol_topologies})

    # export
    aggregate_and_export(all_runs, all_traj, conditions)


if __name__ == "__main__":
    main()

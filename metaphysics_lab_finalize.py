#!/usr/bin/env python3
"""
Metaphysics Lab Finalize
Runs A" (MUS final calibration), B" (Observation Density Law), C" (Energy vs Information resilience)
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

# --- Config ---
PROJECT = "Metaphysics_Lab_Finalize"
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

SEEDS = [11, 17, 23]
AGENTS = [100, 200]
NOISE = [0.05]
EPOCHS = {"mus": 800, "law": 1600, "evi": 2000}
LOG_DENSE = 300
THIN = 10
STABILITY_LAST = 200

GUARD = {"early_stop": {"risk": 0.45, "survival": 0.40, "cci": 0.45}}


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


# ---------------- A" MUS final calibration ----------------
def run_mus_final(candles, seeds, agents_list):
    runs = []
    traj = []
    for candle in candles:
        for agents in agents_list:
            for seed in seeds:
                run_id = f'A_{candle["id"]}_{agents}_{seed}'
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(
                    agents, candle.get("goals", 3), NOISE[0], rng
                )
                # apply coordination nudges
                for a in ags:
                    a["resilience"] = min(
                        1.0, a.get("resilience", 0.5) + candle.get("coord", 0.55) * 0.12
                    )
                cci_series = []
                cal_proxy = []
                em_counts = []
                prev_choice = None
                t0 = time.time()
                for epoch in range(EPOCHS["mus"]):
                    me.step_update(ags, 0.0, "chronic", rng)
                    cci = me.collective_cci(ags)
                    cci_series.append(cci)
                    cal_proxy.append(me.belief_convergence(ags))
                    # em proxy: fraction of alive agents whose top-belief changed since last epoch
                    choices = [
                        int(np.argmax(a["belief"])) for a in ags if a.get("alive", True)
                    ]
                    if prev_choice is not None and len(choices) == len(prev_choice):
                        changes = np.mean(
                            [1 if a != b else 0 for a, b in zip(choices, prev_choice)]
                        )
                        em_counts.append(changes)
                    prev_choice = list(choices)
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
                runs.append(
                    {
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
                )
    # Fit scaling
    sc1_vals = [r["coh"] for r in runs if r["candle"] == "SC1_town"]
    coh_unit = float(np.median(sc1_vals)) if sc1_vals else 1.0
    cal_mean = float(np.mean([r["cal_raw"] for r in runs])) if runs else 1.0
    cal_scale = 0.95 / cal_mean if cal_mean > 0 else 1.0
    em_mean = float(np.mean([r["em_raw_per100"] for r in runs])) if runs else 1.0
    em_scale = 1.0 / em_mean if em_mean > 0 else 1.0
    eps_unit = 0.001
    # finalize run metrics and RMSE
    candle_rmse = {}
    for r in runs:
        cal = r["cal_raw"] * cal_scale
        coh = r["coh"] / coh_unit if coh_unit != 0 else r["coh"]
        em = r["em_raw_per100"] * em_scale
        noise = NOISE[0]
        CCI = (cal * coh * em) / (noise if noise > 0 else 1.0)
        r.update(
            {"cal_scaled": cal, "coh_unitized": coh, "em_unitized": em, "CCI": CCI}
        )
        target_coh = (
            1.0
            if r["candle"] == "SC1_town"
            else (1.5 if r["candle"] == "SC2_oss" else 2.0)
        )
        rmse = math.sqrt((coh - target_coh) ** 2)
        candle_rmse.setdefault(r["candle"], []).append(rmse)
    candle_rmse_avg = {k: float(np.mean(v)) for k, v in candle_rmse.items()}
    summary = {
        "coh_unit": coh_unit,
        "cal_scale": cal_scale,
        "em_scale": em_scale,
        "eps_unit": eps_unit,
        "candle_rmse": candle_rmse_avg,
    }
    return runs, traj, summary


# ---------------- B" Observation Density Law ----------------
def run_observer_law(densities, seeds, agents_list, eps_list):
    runs = []
    traj = []
    shock_epoch = 1000
    shock_window = (960, 1040)
    for density in densities:
        for eps in eps_list:
            for agents in agents_list:
                for seed in seeds:
                    run_id = f"B_den{density}_eps{eps}_{agents}_{seed}"
                    rng = np.random.RandomState(int(seed))
                    ags = me.initialize_agents(agents, 3, NOISE[0], rng)
                    cci_series = []
                    hazard_series = []
                    for epoch in range(EPOCHS["law"]):
                        # attention schedule
                        if epoch < 400:
                            att = 0.0
                        elif epoch < 800:
                            att = ((epoch - 400) / (800 - 400)) * 0.5
                        else:
                            att = 0.5
                        # apply observation-induced resilience boost
                        for a in ags:
                            a["resilience"] = min(
                                1.0, a.get("resilience", 0.5) + att * density * 0.2
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
                                    "obs_edges_per_agent": density * agents,
                                }
                            )
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
                    delta_cci = float(cci_series[-1] - np.mean(cci_series))
                    # t_recover
                    t_rec = next(
                        (
                            i
                            for i, v in enumerate(
                                cci_series[shock_epoch:], start=shock_epoch
                            )
                            if v >= 0.5
                        ),
                        None,
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "label": "B_law",
                            "density": density,
                            "epsilon": eps,
                            "agents": agents,
                            "seed": seed,
                            "stability_CCI_mean": stability_mean,
                            "stability_CCI_slope": stability_slope,
                            "AUH_0960_1040": auh,
                            "delta_CCI": delta_cci,
                            "t_recover_CCI_0.50": t_rec,
                        }
                    )
    # Fit piecewise critical density estimate by looking at second derivative of delta_CCI vs density
    # Aggregate per-density mean delta_CCI
    df = pd.DataFrame(runs)
    agg = df.groupby("density").delta_CCI.mean().reset_index()
    dens = agg["density"].values
    vals = agg["delta_CCI"].values
    # finite second derivative
    if len(vals) >= 3:
        second = np.diff(vals, n=2)
        idx = int(np.argmax(np.abs(second))) + 1
        critical_density = float(dens[idx])
    else:
        critical_density = float(dens[np.argmax(vals)]) if len(vals) else 0.0
    return (
        runs,
        traj,
        {"critical_density": critical_density, "agg_delta": agg.to_dict(orient="list")},
    )


# ---------------- C" Energy vs Information resilience ----------------
def run_evi(eps_list, densities, seeds, agents_list):
    runs = []
    traj = []
    shock_epoch = 1000
    shock_window = (960, 1040)
    baseline_eps = 0.0015
    baseline_density = 0.05
    for arm in ["E_lim", "I_lim"]:
        if arm == "E_lim":
            param_list = eps_list
        else:
            param_list = densities
        for p in param_list:
            for agents in agents_list:
                for seed in seeds:
                    run_id = f"C_{arm}_{p}_{agents}_{seed}"
                    rng = np.random.RandomState(int(seed))
                    ags = me.initialize_agents(agents, 3, NOISE[0], rng)
                    cci_series = []
                    hazard_series = []
                    survival_series = []
                    for epoch in range(EPOCHS["evi"]):
                        # attention baseline density for E_lim; for I_lim, eps fixed
                        if arm == "E_lim":
                            eps = p
                            density = baseline_density
                        else:
                            eps = baseline_eps
                            density = p
                        # apply small resilience based on density
                        if epoch < 400:
                            att = 0.0
                        elif epoch < 800:
                            att = ((epoch - 400) / (800 - 400)) * 0.5
                        else:
                            att = 0.5
                        for a in ags:
                            a["resilience"] = min(
                                1.0, a.get("resilience", 0.5) + att * density * 0.2
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
                        survival_series.append(alive_frac)
                        if should_log(epoch):
                            traj.append(
                                {
                                    "run_id": run_id,
                                    "seed": seed,
                                    "epoch": epoch,
                                    "CCI": cci,
                                    "hazard": hazard,
                                    "survival": alive_frac,
                                    "epsilon": eps,
                                    "density": density,
                                }
                            )
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
                    # recovery times
                    t_rec = next(
                        (
                            i
                            for i, v in enumerate(
                                cci_series[shock_epoch:], start=shock_epoch
                            )
                            if v >= 0.5
                        ),
                        None,
                    )
                    survival_end = float(survival_series[-1])
                    runs.append(
                        {
                            "run_id": run_id,
                            "label": "C_evi",
                            "arm": arm,
                            "param": p,
                            "agents": agents,
                            "seed": seed,
                            "stability_CCI_mean": stability_mean,
                            "AUH_0960_1040": auh,
                            "t_recover_CCI_0.50": t_rec,
                            "survival_end": survival_end,
                        }
                    )
    # compute elasticities
    df = pd.DataFrame(runs)
    # baseline survival for eps=baseline_eps,density=baseline_density
    baseline = df[
        ((df["arm"] == "E_lim") & (df["param"] == baseline_eps))
        | ((df["arm"] == "I_lim") & (df["param"] == baseline_density))
    ]
    # We compute elasticity approx comparing param one-step change relative to baseline
    # For simplicity compute percent changes relative to baseline mean survival_end
    try:
        base_surv = float(baseline.survival_end.mean())
    except Exception:
        base_surv = None
    # compute elasticity tables
    elasticities = {}
    if base_surv is not None and base_surv > 0:
        # E_survival_eps: percent change survival per percent change eps
        # Compare param values to baseline_eps
        e_rows = df[df["label"] == "C_evi"]
        # group by arm and param
        for arm in ["E_lim", "I_lim"]:
            grp = (
                e_rows[e_rows["arm"] == arm]
                .groupby("param")
                .survival_end.mean()
                .to_dict()
            )
            elasticities[arm] = grp
    return runs, traj, {"elasticities": elasticities}


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
    # reports: simple winners by experiment
    lines = [f"# {PROJECT} Results — {STAMP}", "", f"Runs: {len(all_runs)}", ""]
    for exp in sorted(set([r["label"] for r in all_runs])):
        sub = [r for r in all_runs if r["label"] == exp]
        if not sub:
            lines.append(f"- {exp}: no runs")
            continue
        # production-safe bar: stability_CCI_mean>=0.5 and (stability_hazard_mean if present)<=0.2
        passed = []
        for s in sub:
            c_ok = s.get("stability_CCI_mean", 0) >= 0.5
            h_ok = True
            if "stability_hazard_mean" in s:
                h_ok = s.get("stability_hazard_mean", 1) <= 0.2
            if c_ok and h_ok:
                passed.append(s)
        if not passed:
            lines.append(f"- {exp}: NO WINNER (no runs passed production-safe bar)")
            continue
        # choose winner by experiment
        if exp == "A_MUS":
            # lowest total RMSE across candles - already in mus summary; pick first passed
            winner = passed[0]
        elif exp == "B_law":
            winner = sorted(
                passed,
                key=lambda x: (
                    x.get("AUH_0960_1040", float("inf")),
                    -x.get("delta_CCI", 0),
                ),
            )[0]
        else:
            winner = sorted(
                passed,
                key=lambda x: (
                    x.get("AUH_0960_1040", float("inf")),
                    x.get("t_recover_CCI_0.50", float("inf")),
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
    h = hashlib.sha256()
    with open(bundle, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(BDL / "SHA256SUMS.txt", "w") as sf:
        sf.write(f"{h.hexdigest()}  {bundle.name}\n")
    print("Exports written to", OUT)


def main():
    candles = [
        {"id": "SC1_town", "coord": 0.55, "ineq": 0.20, "epsilon": 0.0020, "goals": 3},
        {"id": "SC2_oss", "coord": 0.63, "ineq": 0.18, "epsilon": 0.0020, "goals": 4},
        {
            "id": "SC3_startup",
            "coord": 0.67,
            "ineq": 0.24,
            "epsilon": 0.0025,
            "goals": 4,
        },
    ]
    densities = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    smallworld = {"id": "smallworld", "density": 0.10, "rewire_p": 0.10}
    eps_list = [0.0015, 0.0020]
    all_runs = []
    all_traj = []
    conditions = []

    print('Running A" — MUS final calibration')
    runsA, trajA, mus_summary = run_mus_final(candles, SEEDS, AGENTS)
    all_runs.extend(runsA)
    all_traj.append(trajA)
    conditions.append({"A_candles": candles})
    save_json(mus_summary, SUM / "mus_final_summary.json")

    print('Running B" — Observation Density Law')
    runsB, trajB, law_summary = run_observer_law(densities, SEEDS, AGENTS, eps_list)
    all_runs.extend(runsB)
    all_traj.append(trajB)
    conditions.append({"B_densities": densities, "eps_list": eps_list})
    save_json(law_summary, SUM / "observer_law_summary.json")

    print('Running C" — Energy vs Information resilience')
    e_eps_list = [0.0005, 0.0015, 0.0020]
    e_densities = [0.00, 0.05, 0.10, 0.20]
    runsC, trajC, evi_summary = run_evi(e_eps_list, e_densities, SEEDS, AGENTS)
    all_runs.extend(runsC)
    all_traj.append(trajC)
    conditions.append({"C_evi_eps": e_eps_list, "C_evi_dens": e_densities})
    save_json(evi_summary, SUM / "evi_summary.json")

    aggregate_and_export(all_runs, all_traj, conditions)


if __name__ == "__main__":
    main()

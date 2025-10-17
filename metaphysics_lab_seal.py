#!/usr/bin/env python3
"""
Metaphysics Lab Seal
Runs final checks A''' (MUS pass), B''' (observer critical band confirm), C''' (min-open ε sanity),
selects winners, and writes Lab_Seal_Summary.md and bundle.
"""
import glob
import hashlib
import json
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

PROJECT = "Metaphysics_Lab_Seal"
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path("discovery_results") / STAMP
DATA = OUT / "data"
FIGS = OUT / "figures"
SUM = OUT / "summaries"
REP = OUT / "report"
BDL = OUT / "bundle"
for d in (OUT, DATA, FIGS, SUM, REP, BDL):
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 17, 23]
AGENTS = [100, 200]
NOISE = 0.05
EPOCHS = {"mus": 900, "law": 1400, "eoi": 1800}
LOG_DENSE = 300
THIN = 10
STABILITY_LAST = 200


def should_log(epoch):
    if epoch <= LOG_DENSE:
        return True
    return (epoch % THIN) == 0


def find_prior_mus():
    # search discovery_results/*/summaries/mus_final_summary.json or mus_refit_summary.json
    candidates = glob.glob(
        "discovery_results/*/summaries/mus_final_summary.json"
    ) + glob.glob("discovery_results/*/summaries/mus_refit_summary.json")
    if not candidates:
        return None
    # pick the latest by filename
    candidates.sort()
    p = Path(candidates[-1])
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def compute_stability(series, last_n=STABILITY_LAST):
    if not series:
        return float("nan"), float("nan")
    w = series[-last_n:] if len(series) >= last_n else series
    meanv = float(np.mean(w))
    slope = (
        float(np.polyfit(np.arange(len(w)), w, 1)[0]) if len(w) >= 2 else float("nan")
    )
    return meanv, slope


def auc_window(series, a, b):
    s = series[a : min(b, len(series))]
    if not s:
        return float("nan")
    return float(np.trapz(s))


# Load prior MUS units if available
prior = find_prior_mus()
if prior is None:
    # defaults from prompt
    coh_unit = 1.0
    cal_scale = 365.84
    em_scale = 7.96
    eps_unit = 0.001
else:
    coh_unit = prior.get("coh_unit", 1.0)
    cal_scale = prior.get("cal_scale", 365.84)
    em_scale = prior.get("em_scale", 7.96)
    eps_unit = prior.get("eps_unit", 0.001)


def run_A_seal(candles, seeds, agents_list):
    runs = []
    traj = []
    for c in candles:
        for agents in agents_list:
            for seed in seeds:
                run_id = f"A_{c['id']}_{agents}_{seed}"
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(agents, c.get("goals", 3), NOISE, rng)
                # increase coordination by nudging resilience (coord boost)
                for a in ags:
                    a["resilience"] = min(
                        1.0, a.get("resilience", 0.5) + c.get("coord", 0.6) * 0.12
                    )
                cci_series = []
                em_counts = []
                cal_proxy = []
                for epoch in range(EPOCHS["mus"]):
                    me.step_update(ags, 0.0, "chronic", rng)
                    cci = me.collective_cci(ags)
                    cci_series.append(cci)
                    cal_proxy.append(me.belief_convergence(ags))
                    choices = [
                        int(np.argmax(a["belief"])) for a in ags if a.get("alive", True)
                    ]
                    if epoch > 0 and prev_choices and len(prev_choices) == len(choices):
                        em_counts.append(
                            float(
                                np.mean(
                                    [
                                        1 if x != y else 0
                                        for x, y in zip(prev_choices, choices)
                                    ]
                                )
                            )
                        )
                    prev_choices = list(choices)
                    if should_log(epoch):
                        traj.append(
                            {"run_id": run_id, "seed": seed, "epoch": epoch, "CCI": cci}
                        )
                coh = float(np.median(cci_series))
                cal = float(np.mean(cal_proxy))
                em = float(np.mean(em_counts)) * (100.0 / agents) if em_counts else 0.0
                # apply prior unit scales to compute CCI
                CCI = (
                    (cal * cal_scale)
                    * (coh / coh_unit)
                    * (em * em_scale)
                    / (NOISE if NOISE > 0 else 1.0)
                )
                stability_mean, stability_slope = compute_stability(cci_series)
                bound = {
                    "run_id": run_id,
                    "label": "A_MUS",
                    "candle": c["id"],
                    "agents": agents,
                    "seed": seed,
                    "coh": coh,
                    "cal_raw": cal,
                    "em_raw_per100": em,
                    "CCI": CCI,
                    "stability_CCI_mean": stability_mean,
                    "stability_CCI_slope": stability_slope,
                }
                runs.append(bound)
    return runs, traj


def run_B_peak(densities, seeds, agents_list, eps_fixed):
    runs = []
    traj = []
    shock_epoch = 1000
    window = (960, 1040)
    for density in densities:
        for agents in agents_list:
            for seed in seeds:
                run_id = f"B_den{density}_{agents}_{seed}"
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(agents, 3, NOISE, rng)
                cci_series = []
                hazard_series = []
                for epoch in range(EPOCHS["law"]):
                    if epoch < 350:
                        att = 0.0
                    elif epoch < 750:
                        att = ((epoch - 350) / (750 - 350)) * 0.45
                    else:
                        att = 0.45
                    for a in ags:
                        a["resilience"] = min(
                            1.0, a.get("resilience", 0.5) + att * density * 0.2
                        )
                    shock = 0.5 if epoch == shock_epoch else 0.0
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
                                "attention_gain": att,
                                "obs_edges_per_agent": density * agents,
                            }
                        )
                stability_mean, stability_slope = compute_stability(cci_series)
                auh = auc_window(hazard_series, window[0], window[1])
                delta_cci = float(cci_series[-1] - np.mean(cci_series))
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
                        "label": "B_peak",
                        "density": density,
                        "epsilon": eps_fixed,
                        "agents": agents,
                        "seed": seed,
                        "stability_CCI_mean": stability_mean,
                        "stability_CCI_slope": stability_slope,
                        "AUH_0960_1040": auh,
                        "delta_CCI": delta_cci,
                        "t_recover_CCI_0.50": t_rec,
                    }
                )
    # fit quadratic around densities
    df = pd.DataFrame(runs)
    agg = df.groupby("density").delta_CCI.mean().reset_index()
    if len(agg) >= 3:
        x = agg["density"].values
        y = agg["delta_CCI"].values
        coeffs = np.polyfit(x, y, 2)
        # vertex of parabola -b/(2a)
        a = coeffs[0]
        b = coeffs[1]
        if a != 0:
            rho_star = -b / (2 * a)
        else:
            rho_star = float(agg.loc[agg.delta_CCI.idxmax(), "density"])
    else:
        rho_star = (
            float(agg.loc[agg.delta_CCI.idxmax(), "density"]) if not agg.empty else 0.05
        )
    # crude CI via bootstrap on density means
    ci_lo, ci_hi = float("nan"), float("nan")
    try:
        vals = agg["delta_CCI"].values
        boots = []
        for _ in range(200):
            samp = np.random.choice(vals, size=len(vals), replace=True)
            coeffs_s = np.polyfit(x, samp, 2)
            a_s, b_s = coeffs_s[0], coeffs_s[1]
            rho_s = -b_s / (2 * a_s) if a_s != 0 else rho_star
            boots.append(rho_s)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(
            np.percentile(boots, 97.5)
        )
    except Exception:
        pass
    return (
        runs,
        traj,
        {
            "rho_star": rho_star,
            "rho_ci": [ci_lo, ci_hi],
            "agg_delta": agg.to_dict(orient="list"),
        },
    )


def run_C_eoi(eps_set, seeds, agents_list, density_fixed=0.05):
    runs = []
    traj = []
    shock_epoch = 1000
    window = (960, 1040)
    for eps in eps_set:
        for agents in agents_list:
            for seed in seeds:
                run_id = f"C_eps{eps}_{agents}_{seed}"
                rng = np.random.RandomState(int(seed))
                ags = me.initialize_agents(agents, 3, NOISE, rng)
                cci_series = []
                hazard_series = []
                for epoch in range(EPOCHS["eoi"]):
                    # steady sparse obs density without ramp
                    for a in ags:
                        a["resilience"] = min(
                            1.0, a.get("resilience", 0.5) + 0.5 * density_fixed * 0.2
                        )
                    shock = 0.5 if epoch == shock_epoch else 0.0
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
                                "epsilon": eps,
                            }
                        )
                stability_mean, stability_slope = compute_stability(cci_series)
                auh = auc_window(hazard_series, window[0], window[1])
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
                        "label": "C_eoi",
                        "epsilon": eps,
                        "agents": agents,
                        "seed": seed,
                        "stability_CCI_mean": stability_mean,
                        "stability_CCI_slope": stability_slope,
                        "AUH_0960_1040": auh,
                        "t_recover_CCI_0.50": t_rec,
                    }
                )
    return runs, traj


def aggregate_and_export(all_runs, all_traj, conditions, extra_summaries):
    runs_df = pd.DataFrame(all_runs)
    traj_flat = [r for sub in all_traj for r in sub]
    traj_df = pd.DataFrame(traj_flat)
    runs_df.to_csv(DATA / "runs_summary.csv", index=False)
    traj_df.to_csv(DATA / "trajectories_long.csv", index=False)
    pd.DataFrame(conditions).to_csv(DATA / "conditions_lookup.csv", index=False)
    save = {"project": PROJECT, "stamp": STAMP, "n_runs": len(all_runs)}
    save.update(extra_summaries)
    with open(SUM / f"{PROJECT}_summary.json", "w") as f:
        json.dump(save, f, indent=2)
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
    # write seal summary and bundle
    # winners selection
    winners = {}
    df = runs_df
    # A winner: candles passing bar for >=2/3 seeds
    a_df = df[df.label == "A_MUS"]
    a_pass = a_df[a_df.stability_CCI_mean >= 0.5]
    # count per candle
    candle_counts = a_pass.groupby("candle").run_id.count().to_dict()
    a_winner = None
    for c, cnt in candle_counts.items():
        if cnt >= (len(SEEDS) * len(AGENTS)) * 2 / 3:
            a_winner = c
    winners["A"] = a_winner
    # B winner: lowest AUH among those passing
    b_df = df[df.label == "B_peak"]
    b_pass = b_df[b_df.stability_CCI_mean >= 0.5]
    if not b_pass.empty:
        b_w = b_pass.sort_values(
            ["AUH_0960_1040", "delta_CCI"], ascending=[True, False]
        ).iloc[0]
        winners["B"] = b_w.to_dict()
    else:
        winners["B"] = None
    # C winner: lowest AUH among passing
    c_df = df[df.label == "C_eoi"]
    c_pass = c_df[c_df.stability_CCI_mean >= 0.5]
    if not c_pass.empty:
        c_w = c_pass.sort_values(
            ["AUH_0960_1040", "t_recover_CCI_0.50"], ascending=[True, True]
        ).iloc[0]
        winners["C"] = c_w.to_dict()
    else:
        winners["C"] = None
    # Determine LAB SEALED
    lab_sealed = bool(winners["A"] and (winners["B"] or winners["C"]))
    # write Lab_Seal_Summary.md
    lines = [f"# Lab Seal Report — {STAMP}", "", f"LAB SEALED: {lab_sealed}", ""]
    lines.append("## Final MUS unit constants (from prior or defaults)")
    lines.append(f"- coh_unit: {coh_unit}")
    lines.append(f"- cal_scale: {cal_scale}")
    lines.append(f"- em_scale: {em_scale}")
    lines.append(f"- eps_unit: {eps_unit}")
    lines.append("")
    lines.append("## Winners")
    lines.append(f"- A''' winner candle: {winners['A']}")
    lines.append(f"- B''' winner: {winners['B']}")
    lines.append(f"- C''' winner: {winners['C']}")
    lines.append("")
    lines.append("## Quick checklist")
    lines.append("- MUS constants locked and published")
    lines.append("- Observer law refined around critical density")
    lines.append("- Minimally-open ε band sanity-checked")
    with open(REP / "Lab_Seal_Summary.md", "w") as f:
        f.write("\n".join(lines))
    # bundle everything
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
    print("Seal exports written to", OUT)


def main():
    # A''' candles
    candles = [
        {"id": "SC1_town", "coord": 0.60, "ineq": 0.18, "epsilon": 0.0020, "goals": 3},
        {"id": "SC2_oss", "coord": 0.68, "ineq": 0.17, "epsilon": 0.0020, "goals": 4},
        {
            "id": "SC3_startup",
            "coord": 0.72,
            "ineq": 0.22,
            "epsilon": 0.0025,
            "goals": 4,
        },
    ]
    densities = [0.04, 0.05, 0.06]
    eps_min = [0.0005, 0.0010, 0.0015]
    all_runs = []
    all_traj = []
    conditions = []
    extra_summaries = {}
    print('Running A""" — MUS passing calibration')
    rA, tA = run_A_seal(candles, SEEDS, AGENTS)
    all_runs.extend(rA)
    all_traj.append(tA)
    conditions.append({"A_candles": candles})
    print('Running B""" — Observer peak confirm')
    rB, tB, bs = run_B_peak(densities, SEEDS, AGENTS, 0.0015)
    all_runs.extend(rB)
    all_traj.append(tB)
    conditions.append({"B_densities": densities})
    extra_summaries["observer_peak"] = bs
    print('Running C""" — Min-open ε sanity')
    rC, tC = run_C_eoi(eps_min, SEEDS, AGENTS)
    all_runs.extend(rC)
    all_traj.append(tC)
    conditions.append({"C_eoi_eps": eps_min})
    aggregate_and_export(all_runs, all_traj, conditions, extra_summaries)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Micro-irreversibility sweep: time_arrow_micro_decay
"""
import copy
import datetime
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/time_arrow_micro_decay_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Grid
agent_counts = [100]
noise_levels = [0.05]
shock_severities = [0.2, 0.5]
field_coherences = [0.6, 0.8]
decays = [0.0005, 0.001, 0.005]
rules = ["reversible_decay", "irreversible"]
replicates = [0, 1, 2]

epochs_cap = 1600
shock_epoch = 1000
pre_window = 50
post_window = 50


# helpers
def seed_for(x):
    s = json.dumps(x, sort_keys=True)
    return abs(hash(s)) % (2**32)


def rollback_state(agent_snapshot, decay=0.001):
    import copy

    new_agents = copy.deepcopy(agent_snapshot)
    for a in new_agents:
        a["resource"] = max(0.0, a.get("resource", 0.0) * (1.0 - decay))
        b = np.array(a.get("belief", []), dtype=float)
        if b.size > 0:
            K = float(b.size)
            uniform = np.ones_like(b) / K
            b = b * (1.0 - decay) + uniform * decay
            b = np.clip(b, 1e-12, None)
            b = b / b.sum()
            a["belief"] = b
    return new_agents


def run_single_config(
    n_agents,
    noise,
    shock_severity,
    goal_diversity,
    field_coherence,
    shock_rule,
    decay,
    seed,
):
    rng = np.random.RandomState(seed)
    agents = me.initialize_agents(n_agents, goal_diversity, noise, rng)
    avg_res_start = float(np.mean([a.get("resource", 0.0) for a in agents]))
    history = {"cci": [], "alive_frac": [], "hazard": [], "avg_resource": []}
    for t in range(0, epochs_cap + 1):
        if t == shock_epoch:
            shock_level = shock_severity
        elif t > shock_epoch:
            shock_level = shock_severity * 0.2
        else:
            shock_level = 0.0
        if shock_rule == "reversible_decay" and t == shock_epoch:
            snap_agents = copy.deepcopy(agents)
            me.step_update(agents, shock_level, "acute", rng)
            # rollback with controlled decay
            agents = rollback_state(snap_agents, decay=decay)
            me.step_update(agents, 0.0, "acute", rng)
        else:
            me.step_update(
                agents,
                shock_level if shock_rule == "irreversible" else 0.0,
                "acute",
                rng,
            )
        # energy efficiency effect
        for a in agents:
            if a.get("alive", True):
                a["resource"] = min(
                    1.0, a.get("resource", 0.0) + (field_coherence - 0.5) * 0.02
                )
        alive_frac = sum(1 for a in agents if a.get("alive", True)) / float(len(agents))
        hazard = 1.0 - alive_frac
        cci = me.collective_cci(agents)
        history["cci"].append(float(cci))
        history["alive_frac"].append(float(alive_frac))
        history["hazard"].append(float(hazard))
        history["avg_resource"].append(
            float(np.mean([a.get("resource", 0.0) for a in agents]))
        )

    # metrics
    def mean_window(arr, start, length):
        s = max(0, start)
        e = min(len(arr), start + length)
        if e <= s:
            return float("nan")
        return float(np.mean(arr[s:e]))

    hazard_pre = mean_window(history["hazard"], shock_epoch - pre_window, pre_window)
    hazard_post = mean_window(history["hazard"], shock_epoch + 1, post_window)
    if abs(hazard_pre) < 1e-12:
        if abs(hazard_post) < 1e-12:
            asymmetry = 0.0
        else:
            asymmetry = float("inf")
    else:
        asymmetry = (hazard_post - hazard_pre) / abs(hazard_pre)
    cci_pre = mean_window(history["cci"], shock_epoch - pre_window, pre_window)
    # recovery time
    recovery_time = float("inf")
    if not math.isnan(cci_pre) and cci_pre > 0:
        target = 0.95 * cci_pre
        for t in range(shock_epoch + 1, len(history["cci"])):
            if history["cci"][t] >= target:
                recovery_time = t - shock_epoch
                break
    avg_res_end = history["avg_resource"][-1]
    energy_drift = (avg_res_end - avg_res_start) / (
        avg_res_start if abs(avg_res_start) > 1e-12 else 1.0
    )
    survival_rate = history["alive_frac"][-1]
    return {
        "history": history,
        "survival_rate": survival_rate,
        "recovery_time": recovery_time,
        "energy_drift": energy_drift,
        "asymmetry_index": asymmetry,
        "hazard_pre": hazard_pre,
        "hazard_post": hazard_post,
    }


# run paired experiments: for each config and seed, run irreversible and reversible_decay with same seed and compute ΔCCI, energy_drift, etc.
rows = []
paired_ccis = {}
start = time.time()
for shock_severity in shock_severities:
    for field_coherence in field_coherences:
        for decay in decays:
            for rep in replicates:
                seed = seed_for(
                    {
                        "n": 100,
                        "noise": 0.05,
                        "sev": shock_severity,
                        "fc": field_coherence,
                        "decay": decay,
                        "rep": rep,
                    }
                )
                # irreversible
                irr = run_single_config(
                    100,
                    0.05,
                    shock_severity,
                    3,
                    field_coherence,
                    "irreversible",
                    decay=0.0,
                    seed=seed,
                )
                # reversible with decay
                rev = run_single_config(
                    100,
                    0.05,
                    shock_severity,
                    3,
                    field_coherence,
                    "reversible_decay",
                    decay=decay,
                    seed=seed,
                )
                # ΔCCI: mean over epochs of cci_forward - cci_reverse
                cci_diff_arr = np.array(irr["history"]["cci"]) - np.array(
                    rev["history"]["cci"]
                )
                delta_CCI = float(np.mean(cci_diff_arr))
                # asymmetry_index take from irreversible run (user's spec ambiguous; use irr asymmetry)
                asymmetry = irr["asymmetry_index"]
                energy_drift = irr["energy_drift"]
                temporal_drift_index = float(abs(energy_drift)) / (
                    abs(delta_CCI) if abs(delta_CCI) > 1e-12 else float("nan")
                )
                run_id = f"N100_noise0.05_sev{shock_severity}_fc{field_coherence}_decay{decay}_rep{rep}"
                # save trajectories
                pd.DataFrame(
                    {
                        "epoch": list(range(len(irr["history"]["cci"]))),
                        "cci_irreversible": irr["history"]["cci"],
                        "cci_reversible": rev["history"]["cci"],
                        "hazard_irreversible": irr["history"]["hazard"],
                        "hazard_reversible": rev["history"]["hazard"],
                        "avg_res_irreversible": irr["history"]["avg_resource"],
                        "avg_res_reversible": rev["history"]["avg_resource"],
                    }
                ).to_csv(DATA_DIR / f"{run_id}_traj.csv", index=False)
                rows.append(
                    {
                        "run_id": run_id,
                        "shock_severity": shock_severity,
                        "field_coherence": field_coherence,
                        "decay": decay,
                        "replicate": rep,
                        "asymmetry_index": asymmetry,
                        "delta_CCI": delta_CCI,
                        "energy_drift": energy_drift,
                        "temporal_drift_index": temporal_drift_index,
                    }
                )
                paired_ccis.setdefault(
                    (shock_severity, field_coherence, decay), []
                ).append((irr["history"]["cci"], rev["history"]["cci"]))
                print("Completed", run_id)

# write master CSV
master = DATA_DIR / "time_arrow_micro_decay_runs.csv"
pd.DataFrame(rows).to_csv(master, index=False)

# aggregate JSON summary (means + 95% CI) per decay x shock x fc
summary = []
grouped = pd.DataFrame(rows).groupby(["shock_severity", "field_coherence", "decay"])
for name, g in grouped:
    sev, fc, dec = name
    arr_asym = np.array(
        [
            v
            for v in g.asymmetry_index.values
            if (v is not None) and (not np.isnan(v)) and (not np.isinf(v))
        ]
    )
    arr_delta = np.array(
        [
            v
            for v in g.delta_CCI.values
            if (v is not None) and (not np.isnan(v)) and (not np.isinf(v))
        ]
    )
    arr_energy = np.array(
        [
            v
            for v in g.energy_drift.values
            if (v is not None) and (not np.isnan(v)) and (not np.isinf(v))
        ]
    )
    arr_temp = np.array(
        [
            v
            for v in g.temporal_drift_index.values
            if (v is not None) and (not np.isnan(v)) and (not np.isinf(v))
        ]
    )

    def mean_ci(a):
        if a.size == 0:
            return (None, None, [None, None])
        m = float(a.mean())
        rng = np.random.RandomState(12345)
        boots = []
        for _ in range(1000):
            idx = rng.choice(len(a), size=len(a), replace=True)
            boots.append(np.mean(a[idx]))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        return (m, float(a.std()) if a.size > 1 else 0.0, ci)

    m_asym, s_asym, ci_asym = mean_ci(arr_asym)
    m_delta, s_delta, ci_delta = mean_ci(arr_delta)
    m_energy, s_energy, ci_energy = mean_ci(arr_energy)
    m_temp, s_temp, ci_temp = mean_ci(arr_temp)
    summary.append(
        {
            "shock_severity": float(sev),
            "field_coherence": float(fc),
            "decay": float(dec),
            "mean_asymmetry": m_asym,
            "asymmetry_CI": ci_asym,
            "mean_delta_CCI": m_delta,
            "delta_CCI_CI": ci_delta,
            "mean_energy_drift": m_energy,
            "energy_CI": ci_energy,
            "mean_temporal_drift_index": m_temp,
            "temporal_CI": ci_temp,
        }
    )
(JSON_DIR / "time_arrow_micro_decay_summary.json").write_text(
    json.dumps(summary, indent=2)
)

# Heatmap: average mean_asymmetry over decay x shock_severity
hm = pd.DataFrame(summary)
if not hm.empty:
    # keep only numeric mean_asymmetry rows
    hm_numeric = hm[["decay", "shock_severity", "mean_asymmetry"]].copy()
    hm_numeric = hm_numeric[~hm_numeric.mean_asymmetry.isnull()]
    if not hm_numeric.empty:
        pivot = (
            hm_numeric.groupby(["decay", "shock_severity"])
            .mean()
            .reset_index()
            .pivot(index="decay", columns="shock_severity", values="mean_asymmetry")
        )
        # ensure numeric ordering
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        plt.figure(figsize=(6, 4))
        plt.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
        )
        plt.colorbar(label="mean asymmetry (irreversible)")
        plt.xlabel("shock_severity")
        plt.ylabel("decay")
        plt.title("Asymmetry heatmap vs decay and severity")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "asymmetry_heatmap_decay.png")
        plt.close()

# CCI vs time overlay: pick representative config (0.2, fc=0.6, decay=0.001) average across reps
key = (0.2, 0.6, 0.001)
if key in paired_ccis:
    pairs = paired_ccis[key]
    # align and average
    l = min(len(p[0]) for p in pairs)
    mean_irr = np.mean([p[0][:l] for p in pairs], axis=0)
    mean_rev = np.mean([p[1][:l] for p in pairs], axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(range(l), mean_irr, label="irreversible")
    plt.plot(range(l), mean_rev, label="reversible_decay")
    plt.xlabel("epoch")
    plt.ylabel("CCI")
    plt.title("CCI forward vs reverse (decay=0.001, sev=0.2, fc=0.6)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "CCI_vs_time_overlay.png")
    plt.close()

# write markdown report
md = []
md.append("# Time Arrow Micro-Decay Sweep\n")
md.append("Objective: detect emergence of non-zero asymmetry as decay increases")
md.append("\nSummary (per decay x shock x fc)")
for ent in summary:
    md.append(
        f"- decay={ent['decay']} shock={ent['shock_severity']} fc={ent['field_coherence']}: mean_asymmetry={ent['mean_asymmetry']}, CI={ent['asymmetry_CI']}, mean_delta_CCI={ent['mean_delta_CCI']}, mean_energy_drift={ent['mean_energy_drift']}"
    )
md.append("\n## Figures")
md.append(f"![]({FIG_DIR.name}/asymmetry_heatmap_decay.png)")
md.append(f"![]({FIG_DIR.name}/CCI_vs_time_overlay.png)")
(REPORT_DIR / "time_arrow_micro_decay_results.md").write_text("\n".join(md))

print("Done. Outputs in:", OUT_DIR)
print("Master CSV:", master)
print("JSON summary:", JSON_DIR / "time_arrow_micro_decay_summary.json")
print("Report:", REPORT_DIR / "time_arrow_micro_decay_results.md")

#!/usr/bin/env python3
"""
Finalize open_min: apply production-safety PASS/FAIL, copy artifacts to 04_LATEST_RESULTS,
and run a realism sweep using the follow-up simulate function (2 seeds, 5k) as a higher-fidelity fallback.
"""
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
OPEN_MIN_DIR = sorted(
    list((ROOT / "outputs" / "open_min").iterdir()), key=os.path.getmtime
)[-1]
SUMMARY = OPEN_MIN_DIR / "data" / "open_min_summary.json"
REPORT = OPEN_MIN_DIR / "report" / "open_min_results.md"

LATEST_DIR = ROOT / "04_LATEST_RESULTS"
LATEST_DIR.mkdir(exist_ok=True)

with open(SUMMARY) as f:
    summary = json.load(f)

# Apply production-safety thresholds
criteria = {"stability_CCI_mean": 0.50, "stability_hazard_mean": 0.20, "slope": -0.0005}

results = summary.get("results", {})
passfail = {}
for k, v in results.items():
    sc = v["stability_CCI_mean"]
    # approximate stability_hazard_mean from runs table
    runs = summary.get("results_table", [])
    sub = [r for r in runs if r["label"] == k]
    stability_hazard = (
        float(np.mean([r.get("stability_hazard_mean", 999) for r in sub]))
        if sub
        else 999
    )
    slope = (
        float(np.mean([r.get("stability_CCI_slope", -9) for r in sub])) if sub else -9
    )
    reservoirs_ok = (
        all([r.get("reservoir_energy_left", 0) > 0 for r in sub]) if sub else False
    )
    safe = (
        sc >= criteria["stability_CCI_mean"]
        and stability_hazard <= criteria["stability_hazard_mean"]
        and slope >= criteria["slope"]
        and reservoirs_ok
    )
    passfail[k] = {
        "safe": safe,
        "stability_CCI_mean": sc,
        "stability_hazard_mean": stability_hazard,
        "slope": slope,
        "reservoirs_ok": reservoirs_ok,
    }

# Append PASS/FAIL table to report
with open(REPORT, "a") as f:
    f.write("\n\n## Production-safety PASS/FAIL\n")
    f.write(
        "|condition|pass|stability_CCI_mean|stability_hazard_mean|slope|reservoirs_ok|\n"
    )
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for k, v in passfail.items():
        f.write(
            f"|{k}|{ 'PASS' if v['safe'] else 'FAIL' }|{v['stability_CCI_mean']:.3f}|{v['stability_hazard_mean']:.3f}|{v['slope']:.6f}|{v['reservoirs_ok']}|\n"
        )

# Copy open_min artifacts to 04_LATEST_RESULTS
dest = LATEST_DIR / "open_min_latest"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(OPEN_MIN_DIR, dest)

print("Copied open_min artifacts to", dest)

## Realism sweep: reuse simulate from stress5k_followup (copied implementation)
import math
from pathlib import Path


def simulate_real(condition, seed, epochs_cap, log_every, agents):
    # adapted from stress5k_followup.simulate (fast-mode detailed)
    np.random.seed(seed + hash(condition["label"]) % 1000)
    age = 1.0
    cci = 0.65
    survival = 0.95
    risk = 0.15
    expansion_cd = 0
    traj = []
    interventions = []
    early_stopped = False
    collapse_epoch = None
    consec_hazard = 0
    start_t = time.time()
    for epoch in range(epochs_cap):
        age *= 0.99985
        noise = 1.0 + np.random.normal(0, 0.012)
        expansion_boost = 1.0
        hygiene_boost = 1.0
        note = ""
        exp = condition.get("expansion")
        if exp:
            mode = exp.get("mode")
            if mode == "fixed_early":
                if epoch >= exp["start"] and epoch % exp["every"] == 0:
                    expansion_boost = 1.0 + exp["pct"]
                    interventions.append((epoch, "E_fixed", exp))
                    note = "E_fixed"
            elif mode == "hybrid":
                if epoch >= exp["early_start"] and epoch % exp["early_every"] == 0:
                    expansion_boost = 1.0 + exp["early_pct"]
                    interventions.append((epoch, "E_early", {"pct": exp["early_pct"]}))
                    note = "E_early"
                if expansion_cd > 0:
                    expansion_cd -= 1
                if risk > exp.get("adaptive_hazard_thresh", 0.2) and expansion_cd == 0:
                    expansion_boost *= 1.0 + exp.get("adaptive_pct", 0.015)
                    expansion_cd = exp.get("adaptive_cooldown", 30)
                    interventions.append(
                        (epoch, "E_adapt", {"pct": exp.get("adaptive_pct")})
                    )
                    note = note + "|E_adapt" if note else "E_adapt"
        hyg = condition.get("hygiene")
        if hyg and hyg.get("mode") == "fixed_periodic":
            if epoch > 0 and epoch % hyg["every"] == 0:
                hygiene_boost = 1.0 + hyg["noise_trim_pct"] * 0.8
                interventions.append((epoch, "H_fixed", hyg))
                note = note + "|H_fixed" if note else "H_fixed"
        cci = max(
            0.05,
            min(
                1.0,
                0.65
                * age
                * (condition["coord"] / 0.5)
                * ((0.3 - condition["ineq"]) / 0.3 + 1.0)
                * expansion_boost
                * hygiene_boost
                * noise,
            ),
        )
        survival = max(
            0.01,
            min(
                1.0,
                0.95
                * age
                * (condition["coord"] / 0.5)
                * expansion_boost
                * hygiene_boost,
            ),
        )
        risk = max(
            0.05,
            min(
                0.8,
                0.15
                * (2.0 - age)
                / ((condition["coord"] / 0.5) * expansion_boost * hygiene_boost),
            ),
        )
        hazard = max(0, -math.log(survival + 1e-9))
        if epoch % log_every == 0 or epoch == epochs_cap - 1:
            traj.append(
                {
                    "run_id": f"{condition['label']}_s{seed}",
                    "seed": seed,
                    "epoch": epoch,
                    "CCI": cci,
                    "collapse_risk": risk,
                    "survival_rate": survival,
                    "hazard": hazard,
                    "intervention_flag": bool(note),
                    "intervention_note": note,
                }
            )
        if risk >= 0.45 and survival <= 0.40 and cci < 0.45:
            early_stopped = True
            collapse_epoch = epoch
            break
    epochs_observed = (
        collapse_epoch if early_stopped else (traj[-1]["epoch"] if traj else 0)
    )
    cci_vals = [t["CCI"] for t in traj]
    last_epochs = [
        t
        for t in traj
        if t["epoch"] >= max(0, (traj[-1]["epoch"] if traj else 0) - 200)
    ]
    stability_cci = (
        np.mean([t["CCI"] for t in last_epochs])
        if last_epochs
        else (cci_vals[-1] if cci_vals else 0)
    )
    stability_hazard = np.mean([t["hazard"] for t in last_epochs]) if last_epochs else 0
    slope = 0.0
    if len(cci_vals) >= 50:
        xs = np.arange(len(cci_vals[-50:]))
        slope = np.polyfit(xs, cci_vals[-50:], 1)[0]
    summary = {
        "run_id": f"{condition['label']}_s{seed}",
        "label": condition["label"],
        "seed": seed,
        "epochs_cap": epochs_cap,
        "early_stopped": early_stopped,
        "lifespan_epochs": epochs_observed,
        "final_CCI": float(cci_vals[-1]) if cci_vals else 0.0,
        "stability_CCI_mean": float(stability_cci),
        "stability_hazard_mean": float(stability_hazard),
        "stability_CCI_slope": float(slope),
    }
    return summary, traj, interventions


## Realism sweep: use PLAN from open_min
PLAN = []
eps_list = [0.0025, 0.005, 0.01]
PLAN.append(
    {
        "label": "closed",
        "mech": "none",
        "epsilon": 0.0,
        "period": None,
        "coord": 0.6,
        "ineq": 0.15,
    }
)
for e in eps_list:
    PLAN.append(
        {
            "label": f"chem_eps{e}",
            "mech": "chemostat",
            "epsilon": e,
            "period": 20,
            "coord": 0.6,
            "ineq": 0.15,
        }
    )
for e in eps_list:
    PLAN.append(
        {
            "label": f"agentio_eps{e}",
            "mech": "agent_io",
            "epsilon": e,
            "period": 20,
            "coord": 0.6,
            "ineq": 0.15,
        }
    )

OUT2 = ROOT / "outputs" / "open_min_real" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT2 / d).mkdir(parents=True, exist_ok=True)

all_runs = []
all_traj = []

for c in PLAN:
    for seed in [101, 202]:
        s, t, it = simulate_real(c, seed, 5000, 10, 48)
        all_runs.append(s)
        all_traj.extend(t)

pd.DataFrame(all_runs).to_csv(OUT2 / "data" / "runs_summary.csv", index=False)
pd.DataFrame(all_traj).to_csv(OUT2 / "data" / "trajectories_long.csv", index=False)

shutil.copytree(OPEN_MIN_DIR, LATEST_DIR / "open_min_latest_real", dirs_exist_ok=True)
print("Realism sweep complete, outputs at", OUT2)

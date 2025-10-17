#!/usr/bin/env python3
"""
Focused follow-up runner (20-30 min):
- Re-run F1_EH_early_periodic to 5000 epochs
- Test aggressive hygiene (every 25 epochs, trim=0.30)
- Test multi-mode expansion: early + adaptive
Exports: CSV/JSON/PNGs/MD/ZIP in outputs/stress5k_followup/<timestamp>
"""
import hashlib
import json
import os
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Config
TIME_BUDGET_SEC = 1800  # 30 minutes guard for this follow-up
EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 5000
LOG_EVERY_PHASE1 = 1
LOG_EVERY_PHASE2 = 10
AGENTS_PHASE1 = 64
AGENTS_PHASE2 = 48
SEEDS_FINAL = 2

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = Path(f"outputs/stress5k_followup/{timestamp}")
for d in ["data", "figures", "report", "logs", "bundle"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg):
    t = time.time() - start_time
    s = f"[{datetime.now().strftime('%H:%M:%S')}] ({t:.1f}s) {msg}"
    print(s)
    log_lines.append(s)


# Conditions (F1 anchor only as requested)
base = {"family": "F1", "label": "F1", "coord": 0.60, "ineq": 0.15}

conditions = []
# 1) Re-run F1_EH_early_periodic to 5000
c1 = dict(base)
c1.update(
    {
        "label": "F1_EH_early_periodic_5000",
        "expansion": {"mode": "fixed_early", "pct": 0.03, "every": 12, "start": 12},
        "hygiene": {
            "mode": "fixed_periodic",
            "every": 40,
            "noise_trim_pct": 0.12,
            "recalibrate": True,
        },
    }
)
conditions.append(c1)

# 2) Aggressive hygiene
c2 = dict(base)
c2.update(
    {
        "label": "F1_H_aggressive",
        "hygiene": {
            "mode": "fixed_periodic",
            "every": 25,
            "noise_trim_pct": 0.30,
            "recalibrate": True,
        },
    }
)
conditions.append(c2)

# 3) Multi-mode expansion: early + adaptive
c3 = dict(base)
c3.update(
    {
        "label": "F1_E_early_adapt",
        "expansion": {
            "mode": "hybrid",
            "early_pct": 0.03,
            "early_every": 12,
            "early_start": 12,
            "adaptive_pct": 0.015,
            "adaptive_hazard_thresh": 0.20,
            "adaptive_cooldown": 30,
        },
    }
)
conditions.append(c3)

start_time = time.time()


# Simple simulation function adapted from prior orchestrators
def simulate(condition, seed, epochs_cap, log_every, agents):
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

    for epoch in range(epochs_cap):
        # time guard
        if time.time() - start_time > TIME_BUDGET_SEC - 5:
            # stop gracefully if near budget
            break
        age *= 0.9998
        noise = 1.0 + np.random.normal(0, 0.01)
        expansion_boost = 1.0
        hygiene_boost = 1.0
        note = ""

        # expansion behaviors
        exp = condition.get("expansion")
        if exp:
            mode = exp.get("mode")
            if mode == "fixed_early":
                if epoch >= exp["start"] and epoch % exp["every"] == 0:
                    expansion_boost = 1.0 + exp["pct"]
                    interventions.append((epoch, "E_fixed", exp))
                    note = "E_fixed"
            elif mode == "triggered":
                # triggered requires tracking recent hazard; approximate by risk
                if epoch > 0 and risk > exp.get("hazard_thresh", 0.2):
                    consec_hazard += 1
                else:
                    consec_hazard = 0
                if consec_hazard >= exp.get("consec", 5) and expansion_cd <= 0:
                    expansion_boost = 1.0 + exp["pct"]
                    expansion_cd = exp.get("cooldown", 40)
                    interventions.append((epoch, "E_trig", exp))
                    note = "E_trig"
            elif mode == "hybrid":
                # apply early pulses
                if epoch >= exp["early_start"] and epoch % exp["early_every"] == 0:
                    expansion_boost = 1.0 + exp["early_pct"]
                    interventions.append((epoch, "E_early", {"pct": exp["early_pct"]}))
                    note = "E_early"
                # adaptive
                if expansion_cd > 0:
                    expansion_cd -= 1
                if risk > exp.get("adaptive_hazard_thresh", 0.2) and expansion_cd == 0:
                    expansion_boost *= 1.0 + exp.get("adaptive_pct", 0.015)
                    expansion_cd = exp.get("adaptive_cooldown", 30)
                    interventions.append(
                        (epoch, "E_adapt", {"pct": exp.get("adaptive_pct")})
                    )
                    note = note + "|E_adapt" if note else "E_adapt"

        # hygiene behaviors
        hyg = condition.get("hygiene")
        if hyg:
            mode = hyg.get("mode")
            if mode == "fixed_periodic":
                if epoch > 0 and epoch % hyg["every"] == 0:
                    hygiene_boost = 1.0 + hyg["noise_trim_pct"] * 0.8
                    interventions.append((epoch, "H_fixed", hyg))
                    note = note + "|H_fixed" if note else "H_fixed"
            elif mode == "triggered":
                # detect CCI drop in window
                # simplistic: track last 20 cci values
                pass

        # compute metrics
        cci = max(
            0.1,
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
            0.1,
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
        hazard = max(0, -np.log(survival + 1e-9))

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

        # early stop rule
        if risk >= 0.45 and survival <= 0.40 and cci < 0.45:
            early_stopped = True
            collapse_epoch = epoch
            break

    # summary
    epochs_observed = (
        collapse_epoch
        if early_stopped
        else min(epochs_cap - 1, traj[-1]["epoch"] if traj else 0)
    )
    cci_vals = [t["CCI"] for t in traj]
    hazard_vals = [t["hazard"] for t in traj if t["hazard"] > 0]
    peak_cci = max(cci_vals) if cci_vals else 0
    final_cci = cci_vals[-1] if cci_vals else 0
    hazard_peak = max(hazard_vals) if hazard_vals else 0

    # stability window last 200 epochs approximate
    last_epochs = [
        t
        for t in traj
        if t["epoch"] >= max(0, (traj[-1]["epoch"] if traj else 0) - 200)
    ]
    stability_cci = (
        np.mean([t["CCI"] for t in last_epochs]) if last_epochs else final_cci
    )
    stability_hazard = (
        np.mean([t["hazard"] for t in last_epochs]) if last_epochs else hazard_peak
    )
    # slope of last 50 logged points
    slope = 0
    if len(cci_vals) >= 50:
        xs = np.arange(len(cci_vals[-50:]))
        slope = np.polyfit(xs, cci_vals[-50:], 1)[0]

    summary = {
        "run_id": f"{condition['label']}_s{seed}",
        "family": condition["family"],
        "label": condition["label"],
        "seed": seed,
        "epochs_cap": epochs_cap,
        "agents": agents,
        "fast_mode": True,
        "early_stopped": early_stopped,
        "coordination_strength": condition["coord"],
        "goal_inequality": condition["ineq"],
        "expansion_mode": (
            condition.get("expansion", {}).get("mode")
            if condition.get("expansion")
            else None
        ),
        "expansion_params": json.dumps(condition.get("expansion", {})),
        "hygiene_mode": (
            condition.get("hygiene", {}).get("mode")
            if condition.get("hygiene")
            else None
        ),
        "hygiene_params": json.dumps(condition.get("hygiene", {})),
        "lifespan_epochs": epochs_observed,
        "collapse_flag": early_stopped,
        "peak_CCI": float(peak_cci),
        "final_CCI": float(final_cci),
        "hazard_peak": float(hazard_peak),
        "stability_CCI_mean": float(stability_cci),
        "stability_hazard_mean": float(stability_hazard),
        "stability_CCI_slope": float(slope),
        "divergence_epoch": None,
        "time_sec": time.time() - start_time,
    }

    return summary, traj, interventions


# Run experiments
all_runs = []
all_traj = []
trigger_counts = {}

log(f"Starting follow-up run with {len(conditions)} conditions")
for cond in conditions:
    log(f"Running condition {cond['label']}")
    # Phase1 quick run to 600
    s1, t1, int1 = simulate(cond, 1, EPOCHS_PHASE1, LOG_EVERY_PHASE1, AGENTS_PHASE1)
    # Phase2 extended
    s2, t2, int2 = simulate(cond, 1, EPOCHS_PHASE2, LOG_EVERY_PHASE2, AGENTS_PHASE2)
    # merge summaries (prefer phase2 values)
    run_summary = s2.copy()
    run_summary["label"] = cond["label"]
    run_summary["family"] = cond["family"]
    all_runs.append(run_summary)
    # combine trajectories but thin in phase2 already
    combined_traj = t1 + t2
    all_traj.extend(combined_traj)
    # tally triggers
    for it in int1 + int2:
        kind = it[1]
        trigger_counts[kind] = trigger_counts.get(kind, 0) + 1
    log(
        f"Completed {cond['label']}: final_CCI={run_summary['final_CCI']:.3f}, stability_CCI={run_summary['stability_CCI_mean']:.3f}"
    )

# Aggregate and exports
runs_df = pd.DataFrame(all_runs)
traj_df = pd.DataFrame(all_traj)

# conditions lookup
conds = [{"run_id": c["label"], "params": json.dumps(c)} for c in conditions]
conds_df = pd.DataFrame(conds)

runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
conds_df.to_csv(OUT / "data" / "conditions_lookup.csv", index=False)
traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)

summary_json = {
    "timestamp": timestamp,
    "config": {
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "agents_phase1": AGENTS_PHASE1,
        "agents_phase2": AGENTS_PHASE2,
        "log_every_phase1": LOG_EVERY_PHASE1,
        "log_every_phase2": LOG_EVERY_PHASE2,
        "seeds_final": SEEDS_FINAL,
        "time_budget_sec": TIME_BUDGET_SEC,
    },
    "results": runs_df.to_dict(orient="index"),
    "trigger_counts": trigger_counts,
}
with open(OUT / "data" / "stress5k_followup_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

# Simple figures
if not traj_df.empty:
    plt.figure(figsize=(10, 6))
    for r in runs_df["label"]:
        sub = traj_df[traj_df["run_id"].str.startswith(r)]
        if not sub.empty:
            stats = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(stats["epoch"], stats["CCI"], label=r)
    plt.legend()
    plt.title("CCI follow-up")
    plt.savefig(OUT / "figures" / "cci_followup.png")
    plt.close()

# Write report
report = f"Follow-up stress5k run at {timestamp}\n\nConditions: {[c['label'] for c in conditions]}\n\nTrigger counts: {trigger_counts}\n\nRuns summary:\n{runs_df[['label','final_CCI','stability_CCI_mean','stability_hazard_mean','lifespan_epochs']].to_string(index=False)}\n"
with open(OUT / "report" / "stress5k_followup_results.md", "w") as f:
    f.write(report)

# bundle
# checksums
checks = {}
for root, dirs, files in os.walk(OUT):
    for fn in files:
        if fn.endswith(".zip") or fn == "SHA256SUMS.txt":
            continue
        p = Path(root) / fn
        rel = p.relative_to(OUT)
        checks[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
with open(OUT / "SHA256SUMS.txt", "w") as f:
    for k, v in sorted(checks.items()):
        f.write(f"{v}  {k}\n")

bundle_name = f"stress5k_followup_{timestamp}.zip"
with zipfile.ZipFile(OUT / "bundle" / bundle_name, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(OUT):
        for fn in files:
            if fn == bundle_name:
                continue
            z.write(
                os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), OUT)
            )

# save logs
with open(OUT / "logs" / "runner.log", "w") as f:
    f.write("\n".join(log_lines))

# final checklist
log("Follow-up run complete")
print("\nFINAL CHECKLIST:")
print("Outputs:", OUT)
print("Runs CSV:", OUT / "data" / "runs_summary.csv")
print("Trajectories CSV:", OUT / "data" / "trajectories_long.csv")
print("Summary JSON:", OUT / "data" / "stress5k_followup_summary.json")
print("Bundle:", OUT / "bundle" / bundle_name)

print("\nDone.")

#!/usr/bin/env python3
"""
Minimal openness mechanisms sweep and exporter (open_min)

Implements:
- Chemostat throughput (system-level) hook
- Per-agent I/O (need-based) hook
- Guardrails: finite reservoirs, fairness/coord adjustments, hygiene & expansion triggers
- Sweep: closed, chemostat eps in {0.0025,0.005,0.01}, agent_io same eps
- Seeds=2, Horizon=5000, Agents 64->48, log cadence 1->10
- Exports: CSVs, JSON, PNGs, MD report, ZIP bundle, SHA256

This is a fast-mode synthetic runner for screening and reporting.
"""
import hashlib
import json
import math
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = Path(f"outputs/open_min/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

## CONFIG
SEEDS = [101, 202]
EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 5000
AGENTS_P1 = 64
AGENTS_P2 = 48
LOG_P1 = 1
LOG_P2 = 10
TIME_START = time.time()

PLAN = []
eps_list = [0.0025, 0.005, 0.01]
PLAN.append({"label": "closed", "mech": "none", "epsilon": 0.0, "period": None})
for e in eps_list:
    PLAN.append(
        {"label": f"chem_eps{e}", "mech": "chemostat", "epsilon": e, "period": 20}
    )
for e in eps_list:
    PLAN.append(
        {"label": f"agentio_eps{e}", "mech": "agent_io", "epsilon": e, "period": 20}
    )

DEFAULTS = {
    "seeds": SEEDS,
    "epochs_phase1": EPOCHS_PHASE1,
    "epochs_phase2": EPOCHS_PHASE2,
    "agents_p1": AGENTS_P1,
    "agents_p2": AGENTS_P2,
    "log_p1": LOG_P1,
    "log_p2": LOG_P2,
}

events = []
runs = []
traj = []


def apply_chemostat(state, cond, epoch, reservoir):
    # only every cond['period'] epochs
    if cond["period"] is None or epoch % cond["period"] != 0:
        return 0.0, 0.0
    eps = cond["epsilon"]
    imp = min(eps * state["energy"], reservoir["energy_budget"])
    # mass_high_noise approximated as fraction of mass
    exp = min(eps * state["mass_high_noise"], reservoir["waste_capacity"])
    # apply
    reservoir["energy_budget"] -= imp
    reservoir["waste_capacity"] -= exp
    state["energy"] += imp
    state["mass_high_noise"] -= exp
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "mechanism": "chemostat",
            "event_type": "chemostat_cycle",
            "exergy_in": imp,
            "waste_out": exp,
            "reservoir_energy_left": reservoir["energy_budget"],
            "reservoir_waste_fill": reservoir["waste_capacity"],
        }
    )
    return imp, exp


def apply_agent_io(state, cond, epoch, reservoir):
    # each epoch: distribute small intake to low-CCI agents from reservoir up to eps*energy/period
    eps = cond["epsilon"]
    period = cond["period"]
    if period is None:
        return 0.0, 0.0
    budget = min(eps * state["energy"] / period, reservoir["energy_budget"])
    if budget <= 0:
        return 0.0, 0.0
    # approximate number low-CCI agents fraction
    low_frac = max(0.05, 0.5 * (1.0 - state["avg_CCI"]))
    intake = budget * low_frac
    intake = min(intake, reservoir["energy_budget"])
    # export waste from top-noise 30% agents
    cap_waste = min(eps * state["mass"] / period, reservoir["waste_capacity"])
    waste = cap_waste * 0.30
    reservoir["energy_budget"] -= intake
    reservoir["waste_capacity"] -= waste
    state["energy"] += intake
    state["mass_high_noise"] -= waste
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "mechanism": "agent_io",
            "event_type": "agent_io_cycle",
            "exergy_in": intake,
            "waste_out": waste,
            "reservoir_energy_left": reservoir["energy_budget"],
            "reservoir_waste_fill": reservoir["waste_capacity"],
        }
    )
    return intake, waste


def hazard_from_survival(prev_s, s):
    if prev_s <= 0 or s <= 0:
        return 0.0
    return max(0.0, math.log(prev_s) - math.log(s))


def run_condition(cond):
    for seed in DEFAULTS["seeds"]:
        run_id = f"{cond['label']}_s{seed}"
        # initialize system state (synthetic)
        state = {
            "run_id": run_id,
            "seed": seed,
            "energy": 1.0,
            "mass": 1.0,
            "mass_high_noise": 0.2,
            "avg_CCI": 0.65,
            "coord": 0.60,
            "ineq": 0.15,
        }
        # reservoirs (finite)
        reservoir = {
            "energy_budget": 0.25 * state["energy"],
            "waste_capacity": 0.25 * state["mass"],
        }

        # tracking
        start_t = time.time()
        early_stopped = False
        collapsed = False
        traj_local = []
        prev_survival = 0.95
        hazard_window = []
        hygiene_cd = 0
        expansion_cd = 0
        coord_temp_timer = 0

        # run phase1
        for epoch in range(DEFAULTS["epochs_phase1"]):
            # logging cadence
            log_every = DEFAULTS["log_p1"]
            # per-epoch agent dynamics (fast-mode)
            # compute survival and CCI
            noise = np.random.normal(1.0, 0.01)
            expansion_boost = 1.0
            hygiene_boost = 1.0

            # apply hooks
            if cond["mech"] == "chemostat":
                apply_chemostat(state, cond, epoch, reservoir)
            elif cond["mech"] == "agent_io":
                apply_agent_io(state, cond, epoch, reservoir)

            # expansion: early +3% every 12
            if epoch >= 12 and epoch % 12 == 0:
                expansion_boost *= 1.03
            # triggered expansion if hazard high for 3 epochs
            if (
                len(hazard_window) >= 3
                and all(h > 0.18 for h in hazard_window[-3:])
                and expansion_cd == 0
            ):
                expansion_boost *= 1.01
                expansion_cd = 30

            # update survival and CCI
            state["avg_CCI"] = max(
                0.05,
                min(
                    1.0,
                    state["avg_CCI"] * 0.9999 * expansion_boost * hygiene_boost * noise,
                ),
            )
            survival = max(
                0.01, min(1.0, prev_survival * 0.9999 * expansion_boost * hygiene_boost)
            )
            hazard = hazard_from_survival(prev_survival, survival)
            hazard_window.append(hazard)

            # hygiene trigger: 20-epoch MA >0.20 -> trim worst-noise 30%, recalibrate; cooldown 25
            if (
                len(hazard_window) >= 20
                and np.mean(hazard_window[-20:]) > 0.20
                and hygiene_cd == 0
            ):
                # trim
                state["mass_high_noise"] *= 0.7
                state["avg_CCI"] = min(1.0, state["avg_CCI"] * 1.02)
                hygiene_cd = 25
                events.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "epoch": epoch,
                        "mechanism": "hygiene",
                        "event_type": "triggered_trim",
                        "exergy_in": 0.0,
                        "waste_out": 0.3 * state["mass_high_noise"],
                        "reservoir_energy_left": reservoir["energy_budget"],
                        "reservoir_waste_fill": reservoir["waste_capacity"],
                    }
                )

            if hygiene_cd > 0:
                hygiene_cd -= 1
            if expansion_cd > 0:
                expansion_cd -= 1

            # fairness/coordness adjustment if slope50 < -0.001
            last50 = (
                [t["CCI"] for t in traj_local[-50:]] if len(traj_local) >= 50 else None
            )
            if last50 and len(last50) >= 50:
                xs = np.arange(len(last50))
                slope = np.polyfit(xs, last50, 1)[0]
                if slope < -0.001 and coord_temp_timer == 0:
                    # temporarily modify
                    old_coord = state["coord"]
                    old_ineq = state["ineq"]
                    state["coord"] = min(0.70, state["coord"] + 0.05)
                    state["ineq"] = max(0.15, state["ineq"] - 0.02)
                    coord_temp_timer = 30
            if coord_temp_timer > 0:
                coord_temp_timer -= 1
                if coord_temp_timer == 0:
                    # reset to defaults
                    state["coord"] = 0.60
                    state["ineq"] = 0.15

            # compute collapse risk simple proxy
            collapse_risk = max(0.0, 0.2 * (1.0 - state["avg_CCI"]))

            if epoch % log_every == 0 or epoch == DEFAULTS["epochs_phase1"] - 1:
                traj_local.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "epoch": epoch,
                        "CCI": state["avg_CCI"],
                        "collapse_risk": collapse_risk,
                        "survival_rate": survival,
                        "hazard": hazard,
                        "intervention_flag": False,
                        "intervention_note": "",
                    }
                )

            prev_survival = survival

            # early-stop
            if collapse_risk >= 0.45 and survival <= 0.40 and state["avg_CCI"] < 0.45:
                early_stopped = True
                break

        # Phase2: extended
        if not early_stopped:
            for epoch in range(DEFAULTS["epochs_phase1"], DEFAULTS["epochs_phase2"]):
                log_every = DEFAULTS["log_p2"]
                noise = np.random.normal(1.0, 0.01)
                expansion_boost = 1.0
                hygiene_boost = 1.0

                # hooks
                if cond["mech"] == "chemostat":
                    apply_chemostat(state, cond, epoch, reservoir)
                elif cond["mech"] == "agent_io":
                    apply_agent_io(state, cond, epoch, reservoir)

                if epoch >= 12 and epoch % 12 == 0:
                    expansion_boost *= 1.03
                if (
                    len(hazard_window) >= 3
                    and all(h > 0.18 for h in hazard_window[-3:])
                    and expansion_cd == 0
                ):
                    expansion_boost *= 1.01
                    expansion_cd = 30

                state["avg_CCI"] = max(
                    0.01,
                    min(
                        1.0,
                        state["avg_CCI"]
                        * 0.99995
                        * expansion_boost
                        * hygiene_boost
                        * noise,
                    ),
                )
                survival = max(
                    0.001,
                    min(1.0, prev_survival * 0.99995 * expansion_boost * hygiene_boost),
                )
                hazard = hazard_from_survival(prev_survival, survival)
                hazard_window.append(hazard)

                # hygiene trigger
                if (
                    len(hazard_window) >= 20
                    and np.mean(hazard_window[-20:]) > 0.20
                    and hygiene_cd == 0
                ):
                    state["mass_high_noise"] *= 0.7
                    state["avg_CCI"] = min(1.0, state["avg_CCI"] * 1.02)
                    hygiene_cd = 25
                    events.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "epoch": epoch,
                            "mechanism": "hygiene",
                            "event_type": "triggered_trim",
                            "exergy_in": 0.0,
                            "waste_out": 0.3 * state["mass_high_noise"],
                            "reservoir_energy_left": reservoir["energy_budget"],
                            "reservoir_waste_fill": reservoir["waste_capacity"],
                        }
                    )

                if hygiene_cd > 0:
                    hygiene_cd -= 1
                if expansion_cd > 0:
                    expansion_cd -= 1

                last50 = (
                    [t["CCI"] for t in traj_local[-50:]]
                    if len(traj_local) >= 50
                    else None
                )
                if last50 and len(last50) >= 50:
                    xs = np.arange(len(last50))
                    slope = np.polyfit(xs, last50, 1)[0]
                    if slope < -0.001 and coord_temp_timer == 0:
                        state["coord"] = min(0.70, state["coord"] + 0.05)
                        state["ineq"] = max(0.15, state["ineq"] - 0.02)
                        coord_temp_timer = 30
                if coord_temp_timer > 0:
                    coord_temp_timer -= 1
                    if coord_temp_timer == 0:
                        state["coord"] = 0.60
                        state["ineq"] = 0.15

                collapse_risk = max(0.0, 0.2 * (1.0 - state["avg_CCI"]))
                if epoch % log_every == 0 or epoch == DEFAULTS["epochs_phase2"] - 1:
                    traj_local.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "epoch": epoch,
                            "CCI": state["avg_CCI"],
                            "collapse_risk": collapse_risk,
                            "survival_rate": survival,
                            "hazard": hazard,
                            "intervention_flag": False,
                            "intervention_note": "",
                        }
                    )

                prev_survival = survival

                if (
                    collapse_risk >= 0.45
                    and survival <= 0.40
                    and state["avg_CCI"] < 0.45
                ):
                    early_stopped = True
                    break

        # finalize run summary
        cci_vals = [r["CCI"] for r in traj_local]
        hazard_vals = [r["hazard"] for r in traj_local]
        final_cci = float(cci_vals[-1]) if cci_vals else 0.0
        peak_cci = float(max(cci_vals)) if cci_vals else 0.0
        stability_window = [
            r["CCI"]
            for r in traj_local
            if r["epoch"]
            >= max(0, (traj_local[-1]["epoch"] - 200) if traj_local else 0)
        ]
        stability_mean = (
            float(np.mean(stability_window)) if stability_window else final_cci
        )
        stability_hazard = (
            float(
                np.mean(
                    [
                        r["hazard"]
                        for r in traj_local
                        if r["epoch"]
                        >= max(0, (traj_local[-1]["epoch"] - 200) if traj_local else 0)
                    ]
                )
            )
            if traj_local
            else 0.0
        )
        slope = 0.0
        if len(traj_local) >= 50:
            last50 = [r["CCI"] for r in traj_local[-50:]]
            slope = float(np.polyfit(np.arange(len(last50)), last50, 1)[0])

        run_summary = {
            "run_id": run_id,
            "family": "F1",
            "label": cond["label"],
            "seed": seed,
            "epochs_cap": DEFAULTS["epochs_phase2"],
            "agents": (DEFAULTS["agents_p2"]),
            "early_stopped": early_stopped,
            "epsilon": cond["epsilon"],
            "period": cond["period"],
            "mechanism": cond["mech"],
            "coord": state["coord"],
            "inequality": state["ineq"],
            "lifespan_epochs": traj_local[-1]["epoch"] if traj_local else 0,
            "collapse_flag": early_stopped,
            "peak_CCI": peak_cci,
            "final_CCI": final_cci,
            "stability_CCI_mean": stability_mean,
            "stability_hazard_mean": stability_hazard,
            "stability_CCI_slope": slope,
            "hazard_peak": max(hazard_vals) if hazard_vals else 0.0,
            "reservoir_energy_left": reservoir["energy_budget"],
            "reservoir_waste_fill": reservoir["waste_capacity"],
            "time_sec": time.time() - start_t,
        }
        runs.append(run_summary)
        for r in traj_local:
            traj.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "epoch": r["epoch"],
                    "CCI": r["CCI"],
                    "collapse_risk": r["collapse_risk"],
                    "survival_rate": r["survival_rate"],
                    "hazard": r["hazard"],
                    "intervention_flag": r["intervention_flag"],
                    "intervention_note": r["intervention_note"],
                }
            )

        # record reservoir usage event
        events.append(
            {
                "run_id": run_id,
                "seed": seed,
                "epoch": DEFAULTS["epochs_phase2"],
                "mechanism": "reservoir_summary",
                "event_type": "final",
                "exergy_in": 0.0,
                "waste_out": 0.0,
                "reservoir_energy_left": reservoir["energy_budget"],
                "reservoir_waste_fill": reservoir["waste_capacity"],
            }
        )

        # continue to next seed


def analyze_and_export():
    runs_df = pd.DataFrame(runs)
    traj_df = pd.DataFrame(traj)
    events_df = pd.DataFrame(events)

    # conditions lookup
    conds = []
    for c in PLAN:
        conds.append({"run_id": c["label"], "params": json.dumps(c)})
    conds_df = pd.DataFrame(conds)

    # bootstrap per-condition means (<=100 resamples)
    summary = {
        "timestamp": timestamp,
        "config": DEFAULTS,
        "results": {},
        "trigger_counts": {},
    }
    for c in PLAN:
        sub = runs_df[runs_df["label"] == c["label"]]
        if sub.empty:
            continue
        finals = sub["final_CCI"].values
        stabs = sub["stability_CCI_mean"].values

        def ci(vals):
            if len(vals) == 0:
                return [None, None]
            res = []
            for _ in range(100):
                samp = np.random.choice(vals, size=len(vals), replace=True)
                res.append(np.mean(samp))
            return [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]

        summary["results"][c["label"]] = {
            "n_runs": len(sub),
            "final_CCI_mean": float(np.mean(finals)),
            "final_CCI_ci": ci(finals),
            "stability_CCI_mean": float(np.mean(stabs)),
            "stability_CCI_ci": ci(stabs),
        }

    # winner selection
    winner = None
    # production-safe criteria
    safe_candidates = []
    for k, v in summary["results"].items():
        sc = v["stability_CCI_mean"]
        sh = v["final_CCI_mean"]
        # use stability_hazard placeholder: approximate from runs
        sub = runs_df[runs_df["label"] == k]
        stability_hazard_mean = (
            float(np.mean(sub["stability_hazard_mean"])) if not sub.empty else 999
        )
        slope = float(np.mean(sub["stability_CCI_slope"])) if not sub.empty else -9
        if (
            sc >= 0.50
            and stability_hazard_mean <= 0.20
            and slope >= -0.0005
            and all(sub["reservoir_energy_left"] > 0)
        ):
            safe_candidates.append((k, v))
    if safe_candidates:
        winner = safe_candidates[0][0]
    else:
        # pick best stability_CCI_mean
        best = None
        best_val = -9
        for k, v in summary["results"].items():
            if v["stability_CCI_mean"] > best_val:
                best = k
                best_val = v["stability_CCI_mean"]
        winner = best

    summary["winner"] = winner
    summary["results_table"] = runs_df.to_dict(orient="records")

    # write exports
    runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
    traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)
    events_df.to_csv(OUT / "data" / "openness_events.csv", index=False)
    conds_df.to_csv(OUT / "data" / "conditions_lookup.csv", index=False)
    with open(OUT / "data" / "open_min_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures
    if not traj_df.empty:
        plt.figure(figsize=(8, 4))
        for label in runs_df["label"].unique():
            sub = traj_df[traj_df["run_id"].str.contains(label)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(grp["epoch"], grp["CCI"], label=label)
        plt.legend(fontsize=8)
        plt.title("CCI open_min")
        plt.savefig(OUT / "figures" / "cci_long.png")
        plt.close()

    # report
    md = f"Open_min sweep results ({timestamp})\n\nPLAN: {[p['label'] for p in PLAN]}\n\nWinner: {winner}\n\nSummary results:\n"
    for k, v in summary["results"].items():
        md += f"- {k}: stability_mean={v['stability_CCI_mean']:.3f}, ci={v['stability_CCI_ci']}\n"
    with open(OUT / "report" / "open_min_results.md", "w") as f:
        f.write(md)

    # bundle and sha
    checks = {}
    for root, dirs, files in os.walk(OUT):
        for fn in files:
            if fn == ".DS_Store":
                continue
            p = Path(root) / fn
            rel = p.relative_to(OUT)
            checks[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with open(OUT / "SHA256SUMS.txt", "w") as f:
        for k, v in sorted(checks.items()):
            f.write(f"{v}  {k}\n")
    bundle_name = f"open_min_{timestamp}.zip"
    with zipfile.ZipFile(OUT / "bundle" / bundle_name, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(OUT):
            for fn in files:
                if fn == bundle_name:
                    continue
                z.write(
                    os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), OUT)
                )

    print("Exports written to", OUT)


def main():
    for c in PLAN:
        run_condition(c)
    analyze_and_export()


if __name__ == "__main__":
    main()

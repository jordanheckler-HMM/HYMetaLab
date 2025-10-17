#!/usr/bin/env python3
"""
Chemostat budget retry sweep

 ε ∈ {0.00125,0.0025,0.005}, period=40
 closed baseline + chemostat conditions
 Seeds=2, epochs=5000, agents 64->48, logging 1->10
 Guardrails/risk controls same as open_min_runner
 Exports under outputs/open_min_chem_retry/<timestamp>/
"""
import hashlib
import json
import math
import os
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = Path(f"outputs/open_min_chem_retry/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# Config
SEEDS = [101, 202]
EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 5000
AGENTS_P1 = 64
AGENTS_P2 = 48
LOG_P1 = 1
LOG_P2 = 10

PLAN = []
PLAN.append({"label": "closed", "mech": "none", "epsilon": 0.0, "period": None})
for e in [0.00125, 0.0025, 0.005]:
    PLAN.append(
        {
            "label": f"chemostat_eps{e}_period40",
            "mech": "chemostat",
            "epsilon": e,
            "period": 40.0,
        }
    )

events = []
runs = []
traj = []


def hazard_from_survival(prev_s, s):
    if prev_s <= 0 or s <= 0:
        return 0.0
    return max(0.0, math.log(prev_s) - math.log(s))


def apply_chemostat_hook(state, cond, epoch, reservoir):
    # fires when epoch % period == 0
    if cond["period"] is None:
        return 0.0, 0.0
    if epoch % int(cond["period"]) != 0:
        return 0.0, 0.0
    eps = cond["epsilon"]
    imp = min(eps * state["energy"], reservoir["energy_budget"])
    exp = min(eps * state["mass_high_noise"], reservoir["waste_capacity"])
    # apply
    reservoir["energy_budget"] -= imp
    reservoir["waste_capacity"] -= exp
    state["energy"] += imp
    state["mass_high_noise"] = max(0.0, state["mass_high_noise"] - exp)
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "event_type": "chemostat_cycle",
            "exergy_in": imp,
            "waste_out": exp,
            "reservoir_energy_left": reservoir["energy_budget"],
            "reservoir_waste_fill": reservoir["waste_capacity"],
        }
    )
    return imp, exp


def run_one_condition(cond):
    for seed in SEEDS:
        run_id = f"{cond['label']}_s{seed}"
        state = {
            "run_id": run_id,
            "seed": seed,
            "energy": 1.0,
            "mass": 1.0,
            "mass_high_noise": 0.20,
            "avg_CCI": 0.65,
            "coord": 0.60,
            "ineq": 0.15,
        }
        reservoir = {
            "energy_budget": 0.25 * state["energy"],
            "waste_capacity": 0.25 * state["mass"],
        }

        prev_survival = 0.95
        hazard_window = []
        hygiene_cd = 0
        expansion_cd = 0
        coord_temp_timer = 0
        traj_local = []
        early_stopped = False

        # phase1
        for epoch in range(EPOCHS_PHASE1):
            log_every = LOG_P1
            # hook
            if cond["mech"] == "chemostat":
                apply_chemostat_hook(state, cond, epoch, reservoir)

            # expansion early every 12
            expansion_boost = 1.0
            if epoch >= 12 and epoch % 12 == 0:
                expansion_boost *= 1.03
            # triggered expansion
            if (
                len(hazard_window) >= 3
                and all(h > 0.18 for h in hazard_window[-3:])
                and expansion_cd == 0
            ):
                expansion_boost *= 1.01
                expansion_cd = 30

            # update metrics
            noise = np.random.normal(1.0, 0.01)
            state["avg_CCI"] = max(
                0.01, min(1.0, state["avg_CCI"] * 0.9999 * expansion_boost * noise)
            )
            survival = max(0.01, min(1.0, prev_survival * 0.9999 * expansion_boost))
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
                        "event_type": "hygiene_trim",
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

            # fabric guards
            if len(traj_local) >= 50:
                last50 = [t["CCI"] for t in traj_local[-50:]]
                slope = np.polyfit(np.arange(len(last50)), last50, 1)[0]
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
            if epoch % log_every == 0 or epoch == EPOCHS_PHASE1 - 1:
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

            if collapse_risk >= 0.45 and survival <= 0.40 and state["avg_CCI"] < 0.45:
                early_stopped = True
                break

        # phase2
        if not early_stopped:
            for epoch in range(EPOCHS_PHASE1, EPOCHS_PHASE2):
                log_every = LOG_P2
                if cond["mech"] == "chemostat":
                    apply_chemostat_hook(state, cond, epoch, reservoir)
                expansion_boost = 1.0
                if epoch >= 12 and epoch % 12 == 0:
                    expansion_boost *= 1.03
                if (
                    len(hazard_window) >= 3
                    and all(h > 0.18 for h in hazard_window[-3:])
                    and expansion_cd == 0
                ):
                    expansion_boost *= 1.01
                    expansion_cd = 30
                noise = np.random.normal(1.0, 0.01)
                state["avg_CCI"] = max(
                    0.01, min(1.0, state["avg_CCI"] * 0.99995 * expansion_boost * noise)
                )
                survival = max(
                    0.001, min(1.0, prev_survival * 0.99995 * expansion_boost)
                )
                hazard = hazard_from_survival(prev_survival, survival)
                hazard_window.append(hazard)

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
                            "event_type": "hygiene_trim",
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

                if len(traj_local) >= 50:
                    last50 = [t["CCI"] for t in traj_local[-50:]]
                    slope = np.polyfit(np.arange(len(last50)), last50, 1)[0]
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
                if epoch % log_every == 0 or epoch == EPOCHS_PHASE2 - 1:
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

        # finalize
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
        if len(cci_vals) >= 50:
            last50 = [r["CCI"] for r in traj_local[-50:]]
            slope = float(np.polyfit(np.arange(len(last50)), last50, 1)[0])

        reservoirs_ok = (
            reservoir["energy_budget"] > 0 and reservoir["waste_capacity"] > 0
        )

        run_summary = {
            "run_id": run_id,
            "label": cond["label"],
            "seed": seed,
            "epsilon": cond["epsilon"],
            "period": cond["period"],
            "epochs_cap": EPOCHS_PHASE2,
            "agents": AGENTS_P2,
            "early_stopped": early_stopped,
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
            "reservoirs_ok": reservoirs_ok,
            "time_sec": 0.0,
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
        events.append(
            {
                "run_id": run_id,
                "seed": seed,
                "epoch": EPOCHS_PHASE2,
                "event_type": "reservoir_final",
                "exergy_in": 0.0,
                "waste_out": 0.0,
                "reservoir_energy_left": reservoir["energy_budget"],
                "reservoir_waste_fill": reservoir["waste_capacity"],
            }
        )


def analyze_and_export():
    runs_df = pd.DataFrame(runs)
    traj_df = pd.DataFrame(traj)
    events_df = pd.DataFrame(events)

    conds = [{"run_id": c["label"], "params": json.dumps(c)} for c in PLAN]
    conds_df = pd.DataFrame(conds)

    # bootstrap <=100
    summary = {
        "timestamp": timestamp,
        "config": {"eps_list": [0.00125, 0.0025, 0.005], "period": 40, "seeds": SEEDS},
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

    # winner criteria
    winner = None
    safe_candidates = []
    for k, v in summary["results"].items():
        sc = v["stability_CCI_mean"]
        sub = runs_df[runs_df["label"] == k]
        stability_hazard_mean = (
            float(np.mean(sub["stability_hazard_mean"])) if not sub.empty else 999
        )
        slope = float(np.mean(sub["stability_CCI_slope"])) if not sub.empty else -9
        reservoirs_ok = all(sub["reservoirs_ok"]) if not sub.empty else False
        if (
            sc >= 0.50
            and stability_hazard_mean <= 0.20
            and slope >= -0.0005
            and reservoirs_ok
        ):
            safe_candidates.append(k)
    if safe_candidates:
        winner = safe_candidates[0]
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
    with open(OUT / "data" / "open_min_chem_retry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures: CCI long
    if not traj_df.empty:
        plt.figure(figsize=(8, 4))
        for lbl in runs_df["label"].unique():
            sub = traj_df[traj_df["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(grp["epoch"], grp["CCI"], label=lbl)
        plt.legend(fontsize=8)
        plt.title("CCI open_min_chem_retry")
        plt.savefig(OUT / "figures" / "cci_long.png")
        plt.close()

    # markdown report
    md = f"Open_min Chem Retry results ({timestamp})\n\nPLAN: {[p['label'] for p in PLAN]}\n\nWinner: {winner}\n\nResults summary:\n"
    for k, v in summary["results"].items():
        md += f"- {k}: stability_mean={v['stability_CCI_mean']:.3f}, ci={v['stability_CCI_ci']}\n"
    # pass/fail reasons
    md += "\nPASS/FAIL and reasons:\n"
    md += "|condition|PASS/FAIL|reason|\n|---|---:|---|\n"
    for k, v in summary["results"].items():
        sub = runs_df[runs_df["label"] == k]
        reservoirs_ok = all(sub["reservoirs_ok"]) if not sub.empty else False
        reason = "OK" if reservoirs_ok else "reservoirs exhausted"
        passfail = (
            "PASS" if (v["stability_CCI_mean"] >= 0.50 and reservoirs_ok) else "FAIL"
        )
        md += f"|{k}|{passfail}|{reason}|\n"
    with open(OUT / "report" / "open_min_chem_retry_results.md", "w") as f:
        f.write(md)

    # bundle + sha
    checks = {}
    for root, dirs, files in os.walk(OUT):
        for fn in files:
            p = Path(root) / fn
            rel = p.relative_to(OUT)
            checks[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with open(OUT / "SHA256SUMS.txt", "w") as f:
        for k, v in sorted(checks.items()):
            f.write(f"{v}  {k}\n")
    bundle_name = f"open_min_chem_retry_{timestamp}.zip"
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
        run_one_condition(c)
    analyze_and_export()


if __name__ == "__main__":
    main()

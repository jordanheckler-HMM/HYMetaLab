#!/usr/bin/env python3
"""
10k head-to-head open-min sweep

Implements minimal chemostat, chemostat_ctrl, agent_io hooks, guardrails and risk controls.
Exports CSV/JSON/PNGs/MD + ZIP under outputs/open_min_h2h/<timestamp>/
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
OUT = Path(f"outputs/open_min_h2h/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# CONFIG
SEEDS = [101, 202, 303, 404, 505, 606]
EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 10000
AGENTS_P1 = 64
AGENTS_P2 = 48
LOG_P1 = 1
LOG_P2 = 10
SNAPSHOT = 200

PLAN = [
    {"label": "closed", "mech": "none", "epsilon": 0.0, "period": None},
    {
        "label": "chemostat_eps0.0020_period40",
        "mech": "chemostat",
        "epsilon": 0.0020,
        "period": 40,
    },
    {
        "label": "chemostatCTRL_eps0.0025_period40",
        "mech": "chemostat_ctrl",
        "epsilon": 0.0025,
        "period": 40,
    },
    {
        "label": "agentio_eps0.0015_period40",
        "mech": "agent_io",
        "epsilon": 0.0015,
        "period": 40,
    },
    {
        "label": "agentio_eps0.0020_period40",
        "mech": "agent_io",
        "epsilon": 0.0020,
        "period": 40,
    },
]

DEFAULTS = {
    "epochs": EPOCHS_PHASE2,
    "agents_p1": AGENTS_P1,
    "agents_p2": AGENTS_P2,
    "log1": LOG_P1,
    "log2": LOG_P2,
    "snapshot": SNAPSHOT,
    "seeds": SEEDS,
    "reservoir_energy_frac": 0.25,
    "reservoir_waste_frac": 0.25,
    "hygiene": {"ma20_hazard": 0.20, "trim_pct": 0.30, "cooldown": 25},
    "expansion": {
        "early": {"pct": 0.03, "every": 12},
        "trig": {"pct": 0.01, "hazard": 0.18, "consec": 3, "cooldown": 30},
    },
    "fabric": {
        "cci_slope_50e": -0.001,
        "coord_boost": 0.05,
        "coord_cap": 0.70,
        "ineq_nudge": -0.02,
        "ineq_floor": 0.15,
    },
    "early_stop": {"risk": 0.45, "survival": 0.40, "cci": 0.45},
}

events = []
runs = []
traj = []


def hazard_from_survival(prev_s, s):
    if prev_s <= 0 or s <= 0:
        return 0.0
    return max(0.0, math.log(prev_s) - math.log(s))


def hook_chemostat(state, cond, epoch, reservoir):
    if cond["period"] is None or epoch % int(cond["period"]) != 0:
        return 0.0, 0.0
    eps = cond["epsilon"]
    imp = min(eps * state["energy"], reservoir["energy_left"])
    exp = min(eps * state["mass_high_noise"], reservoir["waste_left"])
    reservoir["energy_left"] -= imp
    reservoir["waste_left"] -= exp
    state["energy"] += imp
    state["mass_high_noise"] = max(0.0, state["mass_high_noise"] - exp)
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "mechanism": "chemostat",
            "event_type": "chemostat_cycle",
            "exergy_in": imp,
            "waste_out": exp,
            "reservoir_energy_left": reservoir["energy_left"],
            "reservoir_waste_fill": reservoir["waste_left"],
        }
    )
    return imp, exp


def hook_chemostat_ctrl(state, cond, epoch, reservoir):
    if cond["period"] is None or epoch % int(cond["period"]) != 0:
        return 0.0, 0.0
    eps = cond["epsilon"]
    remaining = max(1, math.ceil((DEFAULTS["epochs"] - epoch) / cond["period"]))
    target_imp = eps * state["energy"]
    cap_imp = reservoir["energy_left"] / remaining
    imp = min(target_imp, cap_imp)
    target_exp = eps * state["mass_high_noise"]
    cap_exp = reservoir["waste_left"] / remaining
    exp = min(target_exp, cap_exp)
    reservoir["energy_left"] -= imp
    reservoir["waste_left"] -= exp
    state["energy"] += imp
    state["mass_high_noise"] = max(0.0, state["mass_high_noise"] - exp)
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "mechanism": "chemostat_ctrl",
            "event_type": "chemostat_ctrl_cycle",
            "exergy_in": imp,
            "waste_out": exp,
            "reservoir_energy_left": reservoir["energy_left"],
            "reservoir_waste_fill": reservoir["waste_left"],
        }
    )
    return imp, exp


def hook_agent_io(state, cond, epoch, reservoir):
    eps = cond["epsilon"]
    budget = min((eps * state["energy"]) / 20.0, reservoir["energy_left"])
    if budget <= 0:
        return 0.0, 0.0
    low_frac = max(0.05, 0.5 * (1.0 - state["avg_CCI"]))
    intake = min(budget, budget * low_frac)
    intake = min(intake, reservoir["energy_left"])
    exp_cap = min((eps * state["mass"]) / 20.0, reservoir["waste_left"])
    exp_used = min(exp_cap, exp_cap * 0.30)
    reservoir["energy_left"] -= intake
    reservoir["waste_left"] -= exp_used
    state["energy"] += intake
    state["mass_high_noise"] = max(0.0, state["mass_high_noise"] - exp_used)
    events.append(
        {
            "run_id": state["run_id"],
            "seed": state["seed"],
            "epoch": epoch,
            "mechanism": "agent_io",
            "event_type": "agent_io_cycle",
            "exergy_in": intake,
            "waste_out": exp_used,
            "reservoir_energy_left": reservoir["energy_left"],
            "reservoir_waste_fill": reservoir["waste_left"],
        }
    )
    return intake, exp_used


HOOKS = {
    "chemostat": hook_chemostat,
    "chemostat_ctrl": hook_chemostat_ctrl,
    "agent_io": hook_agent_io,
}


def run_condition(cond):
    for seed in DEFAULTS["seeds"]:
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
        np.random.seed(int(seed))
        reservoir = {
            "energy_left": DEFAULTS["reservoir_energy_frac"] * state["energy"],
            "waste_left": DEFAULTS["reservoir_waste_frac"] * state["mass"],
        }
        prev_survival = 0.95
        hazard_window = []
        hygiene_cd = 0
        expansion_cd = 0
        coord_temp_timer = 0
        traj_local = []
        early_stopped = False
        start_t = time.time()

        # phase1
        for epoch in range(EPOCHS_PHASE1):
            log_every = LOG_P1
            # hooks
            if cond["mech"] == "chemostat":
                HOOKS["chemostat"](state, cond, epoch, reservoir)
            elif cond["mech"] == "chemostat_ctrl":
                HOOKS["chemostat_ctrl"](state, cond, epoch, reservoir)
            elif cond["mech"] == "agent_io":
                HOOKS["agent_io"](state, cond, epoch, reservoir)

            expansion_boost = 1.0
            if (
                epoch >= DEFAULTS["expansion"]["early"]["every"]
                and epoch % DEFAULTS["expansion"]["early"]["every"] == 0
            ):
                expansion_boost *= 1.0 + DEFAULTS["expansion"]["early"]["pct"]
            if (
                len(hazard_window) >= DEFAULTS["expansion"]["trig"]["consec"]
                and all(
                    h > DEFAULTS["expansion"]["trig"]["hazard"]
                    for h in hazard_window[-DEFAULTS["expansion"]["trig"]["consec"] :]
                )
                and expansion_cd == 0
            ):
                expansion_boost *= 1.0 + DEFAULTS["expansion"]["trig"]["pct"]
                expansion_cd = DEFAULTS["expansion"]["trig"]["cooldown"]

            noise = np.random.normal(1.0, 0.01)
            state["avg_CCI"] = max(
                0.01, min(1.0, state["avg_CCI"] * 0.9999 * expansion_boost * noise)
            )
            survival = max(0.01, min(1.0, prev_survival * 0.9999 * expansion_boost))
            hazard = hazard_from_survival(prev_survival, survival)
            hazard_window.append(hazard)

            if (
                len(hazard_window) >= 20
                and np.mean(hazard_window[-20:]) > DEFAULTS["hygiene"]["ma20_hazard"]
                and hygiene_cd == 0
            ):
                state["mass_high_noise"] *= 1.0 - DEFAULTS["hygiene"]["trim_pct"]
                state["avg_CCI"] = min(1.0, state["avg_CCI"] * 1.02)
                hygiene_cd = DEFAULTS["hygiene"]["cooldown"]
                events.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "epoch": epoch,
                        "mechanism": "hygiene",
                        "event_type": "triggered_trim",
                        "exergy_in": 0.0,
                        "waste_out": DEFAULTS["hygiene"]["trim_pct"]
                        * state["mass_high_noise"],
                        "reservoir_energy_left": reservoir["energy_left"],
                        "reservoir_waste_fill": reservoir["waste_left"],
                    }
                )
            if hygiene_cd > 0:
                hygiene_cd -= 1
            if expansion_cd > 0:
                expansion_cd -= 1

            if len(traj_local) >= 50:
                last50 = [t["CCI"] for t in traj_local[-50:]]
                slope = np.polyfit(np.arange(len(last50)), last50, 1)[0]
                if (
                    slope < DEFAULTS["fabric"]["cci_slope_50e"]
                    and coord_temp_timer == 0
                ):
                    state["coord"] = min(
                        DEFAULTS["fabric"]["coord_cap"],
                        state["coord"] + DEFAULTS["fabric"]["coord_boost"],
                    )
                    state["ineq"] = max(
                        DEFAULTS["fabric"]["ineq_floor"],
                        state["ineq"] + DEFAULTS["fabric"]["ineq_nudge"],
                    )
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

            if (
                collapse_risk >= DEFAULTS["early_stop"]["risk"]
                and survival <= DEFAULTS["early_stop"]["survival"]
                and state["avg_CCI"] < DEFAULTS["early_stop"]["cci"]
            ):
                early_stopped = True
                break

        # phase2
        if not early_stopped:
            for epoch in range(EPOCHS_PHASE1, EPOCHS_PHASE2):
                log_every = LOG_P2
                if cond["mech"] == "chemostat":
                    HOOKS["chemostat"](state, cond, epoch, reservoir)
                elif cond["mech"] == "chemostat_ctrl":
                    HOOKS["chemostat_ctrl"](state, cond, epoch, reservoir)
                elif cond["mech"] == "agent_io":
                    HOOKS["agent_io"](state, cond, epoch, reservoir)

                expansion_boost = 1.0
                if (
                    epoch >= DEFAULTS["expansion"]["early"]["every"]
                    and epoch % DEFAULTS["expansion"]["early"]["every"] == 0
                ):
                    expansion_boost *= 1.0 + DEFAULTS["expansion"]["early"]["pct"]
                if (
                    len(hazard_window) >= DEFAULTS["expansion"]["trig"]["consec"]
                    and all(
                        h > DEFAULTS["expansion"]["trig"]["hazard"]
                        for h in hazard_window[
                            -DEFAULTS["expansion"]["trig"]["consec"] :
                        ]
                    )
                    and expansion_cd == 0
                ):
                    expansion_boost *= 1.0 + DEFAULTS["expansion"]["trig"]["pct"]
                    expansion_cd = DEFAULTS["expansion"]["trig"]["cooldown"]

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
                    and np.mean(hazard_window[-20:])
                    > DEFAULTS["hygiene"]["ma20_hazard"]
                    and hygiene_cd == 0
                ):
                    state["mass_high_noise"] *= 1.0 - DEFAULTS["hygiene"]["trim_pct"]
                    state["avg_CCI"] = min(1.0, state["avg_CCI"] * 1.02)
                    hygiene_cd = DEFAULTS["hygiene"]["cooldown"]
                    events.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "epoch": epoch,
                            "mechanism": "hygiene",
                            "event_type": "triggered_trim",
                            "exergy_in": 0.0,
                            "waste_out": DEFAULTS["hygiene"]["trim_pct"]
                            * state["mass_high_noise"],
                            "reservoir_energy_left": reservoir["energy_left"],
                            "reservoir_waste_fill": reservoir["waste_left"],
                        }
                    )
                if hygiene_cd > 0:
                    hygiene_cd -= 1
                if expansion_cd > 0:
                    expansion_cd -= 1

                if len(traj_local) >= 50:
                    last50 = [t["CCI"] for t in traj_local[-50:]]
                    slope = np.polyfit(np.arange(len(last50)), last50, 1)[0]
                    if (
                        slope < DEFAULTS["fabric"]["cci_slope_50e"]
                        and coord_temp_timer == 0
                    ):
                        state["coord"] = min(
                            DEFAULTS["fabric"]["coord_cap"],
                            state["coord"] + DEFAULTS["fabric"]["coord_boost"],
                        )
                        state["ineq"] = max(
                            DEFAULTS["fabric"]["ineq_floor"],
                            state["ineq"] + DEFAULTS["fabric"]["ineq_nudge"],
                        )
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
                    collapse_risk >= DEFAULTS["early_stop"]["risk"]
                    and survival <= DEFAULTS["early_stop"]["survival"]
                    and state["avg_CCI"] < DEFAULTS["early_stop"]["cci"]
                ):
                    early_stopped = True
                    break

        # finalize
        cci_vals = [r["CCI"] for r in traj_local]
        hazard_vals = [r["hazard"] for r in traj_local]
        final_cci = float(cci_vals[-1]) if cci_vals else 0.0
        peak_cci = float(max(cci_vals)) if cci_vals else 0.0
        last_epoch = traj_local[-1]["epoch"] if traj_local else 0
        stability_window = [
            r["CCI"]
            for r in traj_local
            if r["epoch"] >= max(0, (last_epoch - 200) if traj_local else 0)
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
                        if r["epoch"] >= max(0, (last_epoch - 200) if traj_local else 0)
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

        reservoirs_ok = reservoir["energy_left"] > 0 and reservoir["waste_left"] > 0
        run_summary = {
            "run_id": run_id,
            "label": cond["label"],
            "seed": seed,
            "epsilon": cond["epsilon"],
            "period": cond["period"],
            "mechanism": cond["mech"],
            "epochs_cap": EPOCHS_PHASE2,
            "agents": AGENTS_P2,
            "early_stopped": early_stopped,
            "lifespan_epochs": last_epoch,
            "collapse_flag": early_stopped,
            "peak_CCI": peak_cci,
            "final_CCI": final_cci,
            "stability_CCI_mean": stability_mean,
            "stability_hazard_mean": stability_hazard,
            "stability_CCI_slope": slope,
            "hazard_peak": max(hazard_vals) if hazard_vals else 0.0,
            "reservoir_energy_left": reservoir["energy_left"],
            "reservoir_waste_fill": reservoir["waste_left"],
            "reservoirs_ok": reservoirs_ok,
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
        events.append(
            {
                "run_id": run_id,
                "seed": seed,
                "epoch": EPOCHS_PHASE2,
                "mechanism": "reservoir_final",
                "event_type": "final",
                "exergy_in": 0.0,
                "waste_out": 0.0,
                "reservoir_energy_left": reservoir["energy_left"],
                "reservoir_waste_fill": reservoir["waste_left"],
            }
        )


def analyze_and_export():
    runs_df = pd.DataFrame(runs)
    traj_df = pd.DataFrame(traj)
    events_df = pd.DataFrame(events)
    conds = [{"run_id": c["label"], "params": json.dumps(c)} for c in PLAN]
    conds_df = pd.DataFrame(conds)

    summary = {"timestamp": timestamp, "config": DEFAULTS, "results": {}}
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
            "reservoirs_ok_count": int(sub["reservoirs_ok"].sum()),
        }

    # winner selection per your logic
    safe_candidates = []
    for k, v in summary["results"].items():
        sub = runs_df[runs_df["label"] == k]
        if sub.empty:
            continue
        stability_hazard_mean = float(np.mean(sub["stability_hazard_mean"]))
        slope = float(np.mean(sub["stability_CCI_slope"]))
        reservoirs_ok = all(sub["reservoirs_ok"])
        if (
            v["stability_CCI_mean"] >= 0.50
            and stability_hazard_mean <= 0.20
            and slope >= -0.0005
            and reservoirs_ok
        ):
            safe_candidates.append(
                {
                    "label": k,
                    "stability_hazard_mean": stability_hazard_mean,
                    "stability_CCI_mean": v["stability_CCI_mean"],
                }
            )

    winner_block = {"winner": None, "production_safe": False, "reason": None}
    if safe_candidates:
        # choose lowest stability_hazard_mean then highest stability_CCI_mean
        safe_candidates = sorted(
            safe_candidates,
            key=lambda x: (x["stability_hazard_mean"], -x["stability_CCI_mean"]),
        )
        winner_block["winner"] = safe_candidates[0]["label"]
        winner_block["production_safe"] = True
    else:
        # pick highest stability_CCI_mean
        best_label = None
        best_val = -9
        for k, v in summary["results"].items():
            if v["stability_CCI_mean"] > best_val:
                best_val = v["stability_CCI_mean"]
                best_label = k
        winner_block["winner"] = best_label
        winner_block["production_safe"] = False
        # limiting metrics
        sub = runs_df[runs_df["label"] == best_label]
        winner_block["limiting"] = {
            "stability_hazard_mean": float(np.mean(sub["stability_hazard_mean"])),
            "slope": float(np.mean(sub["stability_CCI_slope"])),
            "reservoirs_ok_all": all(sub["reservoirs_ok"]),
        }

    summary["winner_block"] = winner_block
    summary["results_table"] = runs_df.to_dict(orient="records")

    runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
    traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)
    events_df.to_csv(OUT / "data" / "openness_events.csv", index=False)
    conds_df.to_csv(OUT / "data" / "conditions_lookup.csv", index=False)
    with open(OUT / "data" / "open_min_h2h_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures: CCI long, hazard long, stability window
    if not traj_df.empty:
        plt.figure(figsize=(10, 6))
        for lbl in runs_df["label"].unique():
            sub = traj_df[traj_df["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(grp["epoch"], grp["CCI"], label=lbl)
        plt.legend(fontsize=8)
        plt.title("CCI open_min_h2h")
        plt.savefig(OUT / "figures" / "cci_long.png")
        plt.close()

    # hazard plot
    if not traj_df.empty:
        plt.figure(figsize=(10, 6))
        for lbl in runs_df["label"].unique():
            sub = traj_df[traj_df["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["hazard"].mean().reset_index()
            plt.plot(grp["epoch"], grp["hazard"], label=lbl)
        plt.legend(fontsize=8)
        plt.title("hazard open_min_h2h")
        plt.savefig(OUT / "figures" / "hazard_long.png")
        plt.close()

    # report
    md = f"Open_min H2H results ({timestamp})\n\nPLAN: {[p['label'] for p in PLAN]}\n\nWinner: {winner_block['winner']} (production_safe={winner_block['production_safe']})\n\nResults:\n"
    for k, v in summary["results"].items():
        md += f"- {k}: stability_mean={v['stability_CCI_mean']:.3f}, ci={v['stability_CCI_ci']} (n={v['n_runs']})\n"
    md += "\nPASS/FAIL:\n|condition|PASS/FAIL|reason|\n|---|---:|---|\n"
    for k, v in summary["results"].items():
        sub = runs_df[runs_df["label"] == k]
        reservoirs_ok = all(sub["reservoirs_ok"]) if not sub.empty else False
        passfail = (
            "PASS"
            if (
                v["stability_CCI_mean"] >= 0.50
                and float(np.mean(sub["stability_hazard_mean"])) <= 0.20
                and float(np.mean(sub["stability_CCI_slope"])) >= -0.0005
                and reservoirs_ok
            )
            else "FAIL"
        )
        reason = (
            "OK"
            if passfail == "PASS"
            else (
                "reservoirs exhausted"
                if not reservoirs_ok
                else "fails stability/hazard/slope"
            )
        )
        md += f"|{k}|{passfail}|{reason}|\n"
    with open(OUT / "report" / "open_min_h2h_results.md", "w") as f:
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
    bundle_name = f"open_min_h2h_{timestamp}.zip"
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

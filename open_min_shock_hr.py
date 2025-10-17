#!/usr/bin/env python3
"""
High-resolution Shock-Recovery runner

Conditions: closed; agentio_eps0.0015_period40; agentio_eps0.0020_period40
Seeds: [101,202]; Horizon: 10000; high-res logging 6900-7100 every-1
Shock: epochs 7000-7004 (5 epochs), each epoch affect top-70% by noise, +30% noise

Exports under outputs/open_min_shock_hr/<timestamp>/
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
OUT = Path(f"outputs/open_min_shock_hr/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# CONFIG
SEEDS = [101, 202]
HORIZON = 10000
EPOCHS_PHASE1 = 600
AGENTS_P1 = 64
AGENTS_P2 = 48
LOG_DEFAULT = 10
LOG_HIGH = 1
HIGH_LOG_START = 6900
HIGH_LOG_END = 7100

PLAN = [
    {"label": "closed", "mech": "none", "epsilon": 0.0, "period": None},
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
    "epochs": HORIZON,
    "agents_p1": AGENTS_P1,
    "agents_p2": AGENTS_P2,
    "log_default": LOG_DEFAULT,
    "log_high": LOG_HIGH,
    "high_start": HIGH_LOG_START,
    "high_end": HIGH_LOG_END,
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
shock_events = []


def hazard_from_survival(prev_s, s):
    if prev_s <= 0 or s <= 0:
        return 0.0
    return max(0.0, math.log(prev_s) - math.log(s))


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


def run_hr():
    for cond in PLAN:
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
            hazard_hist = []
            per_epoch_hazard = [None] * (HORIZON + 1)
            per_epoch_cci = [None] * (HORIZON + 1)
            hygiene_cd = 0
            expansion_cd = 0
            coord_temp_timer = 0
            start_t = time.time()

            for epoch in range(HORIZON + 1):
                # hook
                if cond["mech"] == "agent_io":
                    hook_agent_io(state, cond, epoch, reservoir)

                # shock window 7000-7004
                if 7000 <= epoch <= 7004:
                    # re-evaluate top-70% by noise each epoch
                    affected_pct = 0.70
                    noise_delta = 0.30
                    pre_mean_noise = state["mass_high_noise"]
                    # approximate effect: increase mass_high_noise by noise_delta * affected_pct fraction
                    delta = pre_mean_noise * noise_delta * affected_pct
                    state["mass_high_noise"] += delta
                    post_mean_noise = state["mass_high_noise"]
                    shock_events.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "epoch": epoch,
                            "affected_pct": affected_pct,
                            "noise_delta": noise_delta,
                            "pre_mean_noise": pre_mean_noise,
                            "post_mean_noise": post_mean_noise,
                        }
                    )

                # expansion
                expansion_boost = 1.0
                if (
                    epoch >= DEFAULTS["expansion"]["early"]["every"]
                    and epoch % DEFAULTS["expansion"]["early"]["every"] == 0
                    and epoch > 0
                ):
                    expansion_boost *= 1.0 + DEFAULTS["expansion"]["early"]["pct"]
                if (
                    len(hazard_hist) >= DEFAULTS["expansion"]["trig"]["consec"]
                    and all(
                        h > DEFAULTS["expansion"]["trig"]["hazard"]
                        for h in hazard_hist[-DEFAULTS["expansion"]["trig"]["consec"] :]
                    )
                    and expansion_cd == 0
                ):
                    expansion_boost *= 1.0 + DEFAULTS["expansion"]["trig"]["pct"]
                    expansion_cd = DEFAULTS["expansion"]["trig"]["cooldown"]

                noise = np.random.normal(1.0, 0.01)
                state["avg_CCI"] = max(
                    0.0, min(1.0, state["avg_CCI"] * 0.99995 * expansion_boost * noise)
                )
                survival = max(0.0, min(1.0, prev_survival * 0.99995 * expansion_boost))
                hazard = hazard_from_survival(prev_survival, survival)
                hazard_hist.append(hazard)
                per_epoch_hazard[epoch] = hazard
                per_epoch_cci[epoch] = state["avg_CCI"]

                # hygiene
                if (
                    len(hazard_hist) >= 20
                    and np.mean(hazard_hist[-20:]) > DEFAULTS["hygiene"]["ma20_hazard"]
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

                # fabric guards
                if epoch >= 50:
                    start_idx = max(0, epoch - 49)
                    cci_vals = [
                        per_epoch_cci[i]
                        for i in range(start_idx, epoch + 1)
                        if per_epoch_cci[i] is not None
                    ]
                    if len(cci_vals) >= 10:
                        slope = np.polyfit(np.arange(len(cci_vals)), cci_vals, 1)[0]
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

                # logging: high-res window 6900-7100
                if HIGH_LOG_START <= epoch <= HIGH_LOG_END:
                    log_every = LOG_HIGH
                else:
                    log_every = LOG_DEFAULT
                if epoch % log_every == 0 or epoch == HORIZON:
                    traj.append(
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

            # post-run metrics
            # t_recover_CCI/hazard
            t_recover_cci = None
            t_recover_hazard = None
            for e in range(7000, HORIZON + 1):
                if (
                    per_epoch_cci[e] is not None
                    and per_epoch_cci[e] >= 0.50
                    and t_recover_cci is None
                ):
                    t_recover_cci = e
                if (
                    per_epoch_hazard[e] is not None
                    and per_epoch_hazard[e] <= 0.20
                    and t_recover_hazard is None
                ):
                    t_recover_hazard = e
                if t_recover_cci is not None and t_recover_hazard is not None:
                    break

            # AUH windows
            auh_6950_7050 = float(
                np.nansum([per_epoch_hazard[e] for e in range(6950, 7051)])
            )
            auh_7000_8000 = float(
                np.nansum([per_epoch_hazard[e] for e in range(7000, 8001)])
            )
            min_cci_6950_7050 = float(
                np.nanmin(
                    [
                        per_epoch_cci[e]
                        for e in range(6950, 7051)
                        if per_epoch_cci[e] is not None
                    ]
                )
            )
            peak_hazard_6950_7050 = float(
                np.nanmax(
                    [
                        per_epoch_hazard[e]
                        for e in range(6950, 7051)
                        if per_epoch_hazard[e] is not None
                    ]
                )
            )

            reservoirs_ok = reservoir["energy_left"] > 0 and reservoir["waste_left"] > 0

            run_summary = {
                "run_id": run_id,
                "label": cond["label"],
                "seed": seed,
                "epsilon": cond["epsilon"],
                "period": cond["period"],
                "mechanism": cond["mech"],
                "epochs_cap": HORIZON,
                "agents": AGENTS_P2,
                "t_recover_CCI": t_recover_cci,
                "t_recover_hazard": t_recover_hazard,
                "AUH_6950_7050": auh_6950_7050,
                "AUH_7000_8000": auh_7000_8000,
                "Min_CCI_6950_7050": min_cci_6950_7050,
                "Peak_hazard_6950_7050": peak_hazard_6950_7050,
                "reservoir_energy_left": reservoir["energy_left"],
                "reservoir_waste_fill": reservoir["waste_left"],
                "budgets_ok": reservoirs_ok,
                "time_sec": time.time() - start_t,
            }
            runs.append(run_summary)

    # write outputs
    runs_df = pd.DataFrame(runs)
    traj_df = pd.DataFrame(traj)
    shock_df = pd.DataFrame(shock_events)
    runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
    traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)
    shock_df.to_csv(OUT / "data" / "shock_events.csv", index=False)

    # summary per-condition AUH and tie-break
    summary = {"timestamp": timestamp, "config": DEFAULTS, "results": {}}
    for c in PLAN:
        sub = runs_df[runs_df["label"] == c["label"]]
        if sub.empty:
            continue
        vals = sub["AUH_6950_7050"].values

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
            "AUH_6950_7050_mean": float(np.mean(vals)),
            "AUH_6950_7050_ci": ci(vals),
            "budgets_ok_count": int(sub["budgets_ok"].sum()),
            "t_recover_CCI": list(sub["t_recover_CCI"]),
            "t_recover_hazard": list(sub["t_recover_hazard"]),
        }

    # winner selection: lowest AUH_6950_7050, tie-break min(max(t_recover_CCI,t_recover_hazard)), must have budgets_ok
    candidates = []
    for k, v in summary["results"].items():
        sub = runs_df[runs_df["label"] == k]
        if v["budgets_ok_count"] != len(sub):
            continue
        candidates.append((k, v["AUH_6950_7050_mean"]))
    winner = None
    winner_safe = False
    if candidates:
        candidates = sorted(candidates, key=lambda x: x[1])
        best_val = candidates[0][1]
        ties = [c[0] for c in candidates if abs(c[1] - best_val) < 1e-12]
        if len(ties) == 1:
            winner = ties[0]
            winner_safe = True
        else:
            best = None
            best_rec = 10**9
            for lbl in ties:
                sub = runs_df[runs_df["label"] == lbl]
                recs = []
                for a, b in zip(sub["t_recover_CCI"], sub["t_recover_hazard"]):
                    ra = a if a is not None else HORIZON + 1
                    rb = b if b is not None else HORIZON + 1
                    recs.append(max(ra, rb))
                worst = max(recs)
                if worst < best_rec:
                    best_rec = worst
                    best = lbl
            winner = best
            winner_safe = True
    else:
        # no budget-safe candidate; pick lowest AUH_6950_7050 ignoring budgets and flag not safe
        all_cands = [
            (k, v["AUH_6950_7050_mean"]) for k, v in summary["results"].items()
        ]
        if all_cands:
            winner = sorted(all_cands, key=lambda x: x[1])[0][0]
            winner_safe = False

    summary["winner"] = {"winner": winner, "production_safe": winner_safe}
    with open(OUT / "data" / "stress_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures: zoomed CCI and hazard 6900-7100
    if not traj_df.empty:
        zoom = traj_df[(traj_df["epoch"] >= 6900) & (traj_df["epoch"] <= 7100)]
        plt.figure(figsize=(10, 6))
        for lbl in runs_df["label"].unique():
            sub = zoom[zoom["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(grp["epoch"], grp["CCI"], label=lbl)
        plt.axvspan(7000, 7004, color="k", alpha=0.08, label="shock")
        plt.legend(fontsize=8)
        plt.title("CCI zoom 6900-7100")
        plt.savefig(OUT / "figures" / "cci_zoom_6900_7100.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for lbl in runs_df["label"].unique():
            sub = zoom[zoom["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["hazard"].mean().reset_index()
            plt.plot(grp["epoch"], grp["hazard"], label=lbl)
        plt.axvspan(7000, 7004, color="k", alpha=0.08, label="shock")
        plt.legend(fontsize=8)
        plt.title("hazard zoom 6900-7100")
        plt.savefig(OUT / "figures" / "hazard_zoom_6900_7100.png")
        plt.close()

    # recovery bars
    plt.figure(figsize=(6, 4))
    labels = []
    tvals = []
    for lbl in runs_df["label"].unique():
        sub = runs_df[runs_df["label"] == lbl]
        if sub.empty:
            continue
        # use max of per-seed rec times
        recs = [
            max(
                (a if a is not None else HORIZON + 1),
                (b if b is not None else HORIZON + 1),
            )
            for a, b in zip(sub["t_recover_CCI"], sub["t_recover_hazard"])
        ]
        labels.append(lbl)
        tvals.append(max(recs))
    plt.bar(labels, tvals)
    plt.ylabel("worst-case recovery epoch")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "recovery_bars.png")
    plt.close()

    # AUH compare
    plt.figure(figsize=(6, 4))
    labels = []
    auhvals = []
    for lbl in runs_df["label"].unique():
        sub = runs_df[runs_df["label"] == lbl]
        if sub.empty:
            continue
        labels.append(lbl)
        auhvals.append(float(np.mean(sub["AUH_6950_7050"])))
    plt.bar(labels, auhvals)
    plt.ylabel("AUH 6950-7050")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "auh_compare.png")
    plt.close()

    # report
    md = f"Shock HR results ({timestamp})\n\nWinner: {winner} (production_safe={winner_safe})\n\n"
    md += "Per-condition:\n"
    for k, v in summary["results"].items():
        md += f"- {k}: AUH_mean={v['AUH_6950_7050_mean']:.6f}, budgets_ok_count={v['budgets_ok_count']}\n"
    with open(OUT / "report" / "open_min_shock_hr.md", "w") as f:
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
    bundle_name = f"open_min_shock_hr_{timestamp}.zip"
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
    run_hr()


if __name__ == "__main__":
    main()

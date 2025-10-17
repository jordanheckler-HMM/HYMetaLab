#!/usr/bin/env python3
"""
Open-min budget-aware sweep

Implements:
- chemostat (periodic)
- chemostat controller (budget-aware)
- agent I/O (per-epoch need-based)
- guardrails, hygiene, expansion, fabric guards, early-stop

Exports CSV/JSON/PNGs/MD + ZIP under outputs/open_min_budget/<timestamp>/
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
OUT = Path(f"outputs/open_min_budget/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# CONFIG
SEEDS = [101, 202]
EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 5000
AGENTS_P1 = 64
AGENTS_P2 = 48
LOG_P1 = 1
LOG_P2 = 10
SNAPSHOT = 200

PLAN = [
    {"label": "closed", "mech": "none", "epsilon": 0.0, "period": None},
    {"label": "agentio_eps0.0025", "mech": "agent_io", "epsilon": 0.0025, "period": 20},
    {
        "label": "chemostat_eps0.0015_period40",
        "mech": "chemostat",
        "epsilon": 0.0015,
        "period": 40,
    },
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
agent_events = []
agents_snapshots = []
agent_messages = []
agents_observations = []
agent_predictions = []


def generate_agent_observations(state, agents, epoch):
    """Create per-agent noisy partial observations (obs_CCI, obs_energy) and log them."""
    for a in agents:
        obs_cci = float(a["CCI"] * np.random.normal(1.0, 0.02))
        obs_energy = float(a["energy"] + np.random.normal(0.0, 1e-8))
        # predictor: simple EWMA per-agent prediction stored on agent
        prev_pred = a.get("pred", None)
        if prev_pred is None:
            pred_before = obs_cci
        else:
            pred_before = float(prev_pred)
        # record observation
        agents_observations.append(
            {
                "run_id": state["run_id"],
                "seed": state["seed"],
                "epoch": epoch,
                "agent_id": a["id"],
                "obs_CCI": obs_cci,
                "obs_energy": obs_energy,
            }
        )
        # compute prediction error for previous prediction (if any)
        if prev_pred is not None:
            err = abs(obs_cci - float(prev_pred))
        else:
            err = None
        agent_predictions.append(
            {
                "run_id": state["run_id"],
                "seed": state["seed"],
                "epoch": epoch,
                "agent_id": a["id"],
                "pred_before": pred_before,
                "obs_CCI": obs_cci,
                "abs_error": err,
            }
        )
        # update agent's EWMA predictor for next epoch
        alpha = 0.5
        a["pred"] = alpha * obs_cci + (1.0 - alpha) * (
            prev_pred if prev_pred is not None else obs_cci
        )


def maybe_send_agent_messages(state, agents, epoch):
    """Simple rule: agents with CCI < 0.5 send a 'need' message to system (recipient_id=-1).
    Payload expresses normalized need magnitude. Messages logged to agent_messages."""
    for a in agents:
        # need messages for low CCI
        if a["CCI"] < 0.5:
            need = float(max(0.0, 0.5 - a["CCI"]))
            payload = {"need": need}
            agent_messages.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "sender_id": a["id"],
                    "recipient_id": -1,
                    "msg_type": "need",
                    "payload": json.dumps(payload),
                }
            )
        # offer messages for agents with surplus energy and high CCI
        elif (
            a.get("energy", 0.0) > (state["energy"] / max(1, len(agents))) * 1.2
            and a["CCI"] > 0.7
        ):
            surplus = float(
                max(0.0, a["energy"] - (state["energy"] / max(1, len(agents))))
            )
            payload = {"surplus": surplus}
            agent_messages.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "sender_id": a["id"],
                    "recipient_id": -1,
                    "msg_type": "offer",
                    "payload": json.dumps(payload),
                }
            )
        # occasional help solicitations
        elif 0.5 <= a["CCI"] < 0.7 and np.random.rand() < 0.02:
            payload = {"note": "help_request"}
            agent_messages.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "sender_id": a["id"],
                    "recipient_id": -1,
                    "msg_type": "help",
                    "payload": json.dumps(payload),
                }
            )


def hazard_from_survival(prev_s, s):
    if prev_s <= 0 or s <= 0:
        return 0.0
    return max(0.0, math.log(prev_s) - math.log(s))


def apply_chemostat(state, cond, epoch, reservoir):
    if cond["period"] is None or epoch % int(cond["period"]) != 0:
        return 0.0, 0.0
    eps = cond["epsilon"]
    imp = min(eps * state["energy"], reservoir["energy_left"])
    exp = min(eps * state["mass_high_noise"], reservoir["waste_left"])
    reservoir["energy_left"] -= imp
    reservoir["waste_left"] -= exp
    # apply
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


def apply_chemostat_ctrl(state, cond, epoch, reservoir):
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


def apply_agent_io(state, cond, epoch, reservoir):
    # Deprecated aggregated agent_io kept for compatibility; no-op
    return 0.0, 0.0


def apply_agent_io_agents(state, cond, epoch, reservoir, agents):
    """Distribute per-epoch intake to lowest-CCI agents proportionally to need.
    Log per-agent events to agent_events and return total intake & export.
    """
    eps = cond["epsilon"]
    budget = min((eps * state["energy"]) / 20.0, reservoir["energy_left"])
    if budget <= 0 or len(agents) == 0:
        return 0.0, 0.0
    # compute needs: lower CCI => higher need
    cci_vals = np.array([a["CCI"] for a in agents])
    needs = np.clip(1.0 - cci_vals, 0.0, 1.0)
    total_need = np.sum(needs)
    # identify prioritized agents (set by process_agent_messages_and_prioritize)
    prioritized_ids = [a["id"] for a in agents if a.get("priority")]
    if total_need <= 0:
        # small fallback: equal split
        alloc = budget / len(agents)
        for a in agents:
            a["energy"] += alloc
            agent_events.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "agent_id": a["id"],
                    "exergy_in": alloc,
                    "waste_out": 0.0,
                    "reservoir_energy_left": reservoir["energy_left"],
                    "reservoir_waste_fill": reservoir["waste_left"],
                }
            )
        reservoir["energy_left"] -= budget
        return budget, 0.0

    # allocate intake preferentially to prioritized agents if any
    allocated = []
    remain = budget
    if prioritized_ids:
        # compute needs only for prioritized group
        pri_agents = [a for a in agents if a["id"] in prioritized_ids]
        pri_needs = np.array([max(1e-8, 1.0 - a["CCI"]) for a in pri_agents])
        pri_total = np.sum(pri_needs)
        if pri_total > 0:
            pri_budget = min(remain, budget * 0.7)  # cap: give up to 70% to prioritized
            for a, need in zip(pri_agents, pri_needs):
                share = (need / pri_total) * pri_budget
                share = min(share, reservoir["energy_left"], max(0.0, remain))
                a["energy"] += share
                allocated.append(share)
                agent_events.append(
                    {
                        "run_id": state["run_id"],
                        "seed": state["seed"],
                        "epoch": epoch,
                        "agent_id": a["id"],
                        "exergy_in": share,
                        "waste_out": 0.0,
                        "reservoir_energy_left": reservoir["energy_left"],
                        "reservoir_waste_fill": reservoir["waste_left"],
                    }
                )
                remain -= share
    # allocate remaining to whole population proportional to needs
    if total_need > 0 and remain > 0:
        for a, need in zip(agents, needs):
            share = (need / total_need) * remain
            share = min(share, reservoir["energy_left"], max(0.0, remain))
            a["energy"] += share
            allocated.append(share)
            agent_events.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "agent_id": a["id"],
                    "exergy_in": share,
                    "waste_out": 0.0,
                    "reservoir_energy_left": reservoir["energy_left"],
                    "reservoir_waste_fill": reservoir["waste_left"],
                }
            )
            remain -= share
    # export from top-noise 30%: pick agents with highest noise
    noise_vals = np.array([a.get("mass_high_noise", 0.0) for a in agents])
    pct = 0.30
    k = max(1, int(np.ceil(len(agents) * pct)))
    top_idx = np.argsort(-noise_vals)[:k]
    exp_cap = min((eps * state["mass"]) / 20.0, reservoir["waste_left"])
    # split export proportionally by noise in top group
    top_noise = noise_vals[top_idx]
    top_total = np.sum(top_noise)
    exp_used = 0.0
    if top_total > 0:
        for idx in top_idx:
            a = agents[idx]
            portion = (noise_vals[idx] / top_total) * exp_cap
            portion = min(portion, a.get("mass_high_noise", 0.0))
            a["mass_high_noise"] = max(0.0, a.get("mass_high_noise", 0.0) - portion)
            exp_used += portion
            agent_events.append(
                {
                    "run_id": state["run_id"],
                    "seed": state["seed"],
                    "epoch": epoch,
                    "agent_id": a["id"],
                    "exergy_in": 0.0,
                    "waste_out": portion,
                    "reservoir_energy_left": reservoir["energy_left"],
                    "reservoir_waste_fill": reservoir["waste_left"],
                }
            )
    reservoir["energy_left"] -= budget - max(0.0, remain)
    reservoir["waste_left"] -= exp_used
    # update aggregated state for compatibility
    state["energy"] = sum(a["energy"] for a in agents)
    state["mass_high_noise"] = sum(a.get("mass_high_noise", 0.0) for a in agents)
    return (budget - max(0.0, remain)), exp_used


def process_agent_messages_and_prioritize(state, epoch, reservoir, agents):
    """Consume messages for this run/epoch and return a set of agent ids that should be prioritized for intake.
    Simple policy: prioritize 'need' messages by descending need magnitude, cap at top N (or until reservoir small).
    """
    # collect messages for this run and epoch
    msgs = [
        m
        for m in agent_messages
        if m["run_id"] == state["run_id"] and m["epoch"] == epoch
    ]
    needs = []
    for m in msgs:
        try:
            p = json.loads(m["payload"])
        except Exception:
            p = {}
        if m.get("msg_type") == "need" and "need" in p:
            needs.append((m["sender_id"], float(p["need"])))
    if not needs:
        return set()
    # sort by need desc
    needs.sort(key=lambda x: -x[1])
    prioritized = set([nid for nid, _ in needs[: max(1, int(len(needs) * 0.5))]])
    # mark agents for priority flag
    for a in agents:
        a["priority"] = a["id"] in prioritized
    return prioritized


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
        # make per-run deterministic
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

        # initialize per-agent population
        agents = []
        for i in range(DEFAULTS["agents_p1"]):
            agents.append(
                {
                    "id": i,
                    "CCI": state["avg_CCI"],
                    "energy": state["energy"] / DEFAULTS["agents_p1"],
                    "mass_high_noise": state["mass_high_noise"] / DEFAULTS["agents_p1"],
                }
            )

        # phase1
        for epoch in range(EPOCHS_PHASE1):
            log_every = LOG_P1
            # process messages and hooks
            prioritized = process_agent_messages_and_prioritize(
                state, epoch, reservoir, agents
            )
            if cond["mech"] == "chemostat":
                apply_chemostat(state, cond, epoch, reservoir)
            elif cond["mech"] == "chemostat_ctrl":
                apply_chemostat_ctrl(state, cond, epoch, reservoir)
            elif cond["mech"] == "agent_io":
                apply_agent_io_agents(state, cond, epoch, reservoir, agents)

            # expansion early
            expansion_boost = 1.0
            if (
                epoch >= DEFAULTS["expansion"]["early"]["every"]
                and epoch % DEFAULTS["expansion"]["early"]["every"] == 0
            ):
                expansion_boost *= 1.0 + DEFAULTS["expansion"]["early"]["pct"]
            # triggered expansion
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

            # compute metrics
            noise = np.random.normal(1.0, 0.01)
            state["avg_CCI"] = max(
                0.01, min(1.0, state["avg_CCI"] * 0.9999 * expansion_boost * noise)
            )
            survival = max(0.01, min(1.0, prev_survival * 0.9999 * expansion_boost))
            hazard = hazard_from_survival(prev_survival, survival)
            hazard_window.append(hazard)
            # per-agent observations & messages
            generate_agent_observations(state, agents, epoch)
            maybe_send_agent_messages(state, agents, epoch)

            # hygiene
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

            # fabric guards
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
                # snapshot agents
                if epoch % DEFAULTS["snapshot"] == 0:
                    for a in agents:
                        agents_snapshots.append(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "epoch": epoch,
                                "agent_id": a["id"],
                                "CCI": a["CCI"],
                                "energy": a["energy"],
                                "mass_high_noise": a["mass_high_noise"],
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

        # adjust agent count for phase2
        # trim or expand agents list to agents_p2
        if len(agents) > DEFAULTS["agents_p2"]:
            agents = agents[: DEFAULTS["agents_p2"]]
        else:
            # if fewer, add new agents with baseline
            start_idx = len(agents)
            for i in range(start_idx, DEFAULTS["agents_p2"]):
                agents.append(
                    {
                        "id": i,
                        "CCI": state["avg_CCI"],
                        "energy": state["energy"] / DEFAULTS["agents_p2"],
                        "mass_high_noise": state["mass_high_noise"]
                        / DEFAULTS["agents_p2"],
                    }
                )

        # phase2
        if not early_stopped:
            for epoch in range(EPOCHS_PHASE1, EPOCHS_PHASE2):
                log_every = LOG_P2
                prioritized = process_agent_messages_and_prioritize(
                    state, epoch, reservoir, agents
                )
                if cond["mech"] == "chemostat":
                    apply_chemostat(state, cond, epoch, reservoir)
                elif cond["mech"] == "chemostat_ctrl":
                    apply_chemostat_ctrl(state, cond, epoch, reservoir)
                elif cond["mech"] == "agent_io":
                    apply_agent_io_agents(state, cond, epoch, reservoir, agents)

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
                # per-agent observations & messages
                generate_agent_observations(state, agents, epoch)
                maybe_send_agent_messages(state, agents, epoch)

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
                    # snapshot agents in phase2 high frequency
                    if epoch % DEFAULTS["snapshot"] == 0:
                        for a in agents:
                            agents_snapshots.append(
                                {
                                    "run_id": run_id,
                                    "seed": seed,
                                    "epoch": epoch,
                                    "agent_id": a["id"],
                                    "CCI": a["CCI"],
                                    "energy": a["energy"],
                                    "mass_high_noise": a["mass_high_noise"],
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
            "lifespan_epochs": traj_local[-1]["epoch"] if traj_local else 0,
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

    # bootstrap <=100
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
        # agent messages and observations will be written in export section

    # winner selection
    winner = None
    safe_candidates = []
    for k, v in summary["results"].items():
        sc = v["stability_CCI_mean"]
        sub = pd.DataFrame(runs)[pd.DataFrame(runs)["label"] == k]
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
        best = None
        best_val = -9
        for k, v in summary["results"].items():
            if v["stability_CCI_mean"] > best_val:
                best = k
                best_val = v["stability_CCI_mean"]
        winner = best

    summary["winner"] = winner
    summary["results_table"] = pd.DataFrame(runs).to_dict(orient="records")

    # write exports
    runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
    traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)
    events_df.to_csv(OUT / "data" / "openness_events.csv", index=False)
    # per-agent exports
    if agent_events:
        pd.DataFrame(agent_events).to_csv(
            OUT / "data" / "openness_events_per_agent.csv", index=False
        )
    if agents_snapshots:
        pd.DataFrame(agents_snapshots).to_csv(
            OUT / "data" / "agents_snapshots.csv", index=False
        )
    # write agent messages and observations if any
    # always write agent messages/observations files (may be empty)
    pd.DataFrame(agent_messages).to_csv(
        OUT / "data" / "agent_messages.csv", index=False
    )
    pd.DataFrame(agents_observations).to_csv(
        OUT / "data" / "agents_observations.csv", index=False
    )
    pd.DataFrame(agent_predictions).to_csv(
        OUT / "data" / "agent_predictions.csv", index=False
    )
    conds_df.to_csv(OUT / "data" / "conditions_lookup.csv", index=False)
    with open(OUT / "data" / "open_min_budget_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # figures
    if not traj_df.empty:
        plt.figure(figsize=(10, 6))
        for lbl in runs_df["label"].unique():
            sub = traj_df[traj_df["run_id"].str.contains(lbl)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["CCI"].mean().reset_index()
            plt.plot(grp["epoch"], grp["CCI"], label=lbl)
        plt.legend(fontsize=8)
        plt.title("CCI open_min_budget")
        plt.savefig(OUT / "figures" / "cci_long.png")
        plt.close()

    # markdown report
    md = f"Open_min budget sweep results ({timestamp})\n\nPLAN: {[p['label'] for p in PLAN]}\n\nWinner: {winner}\n\nResults summary:\n"
    for k, v in summary["results"].items():
        md += f"- {k}: stability_mean={v['stability_CCI_mean']:.3f}, ci={v['stability_CCI_ci']}\n"
    md += "\nPASS/FAIL and reasons:\n"
    md += "|condition|PASS/FAIL|reason|\n|---|---:|---|\n"
    for k, v in summary["results"].items():
        sub = pd.DataFrame(runs)[pd.DataFrame(runs)["label"] == k]
        reservoirs_ok = all(sub["reservoirs_ok"]) if not sub.empty else False
        reason = "OK" if reservoirs_ok else "reservoirs exhausted"
        passfail = (
            "PASS" if (v["stability_CCI_mean"] >= 0.50 and reservoirs_ok) else "FAIL"
        )
        md += f"|{k}|{passfail}|{reason}|\n"
    with open(OUT / "report" / "open_min_budget_results.md", "w") as f:
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
    bundle_name = f"open_min_budget_{timestamp}.zip"
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

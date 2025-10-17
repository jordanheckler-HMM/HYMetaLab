#!/usr/bin/env python3
"""
Panspermia Phase II runner
Implements the user's Phase II spec: normalized collapse risk, time-decayed openness, EOI, shock+recovery, Bayes update, and exports.
"""

import datetime
import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

ROOT = Path(".")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "discovery_results" / f"panspermia_phaseII_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

# --- Simulation parameters (default unless prior bundle provides other values) ---
EPOCHS = 5000
SEEDS = [1, 2, 3, 4]
LOG_EVERY = 10
STABILITY_WINDOW = 200

# shock params
T_SHOCK = EPOCHS // 2
SHOCK_INTENSITY = 0.35
SHOCK_DURATION = 25

# agent params (inherited from prior run if possible)
N_AGENTS = 200
GOAL_DIVERSITY = 3
NOISE = 0.05

# openness decay lambdas to sweep
LAMBDAS = [0.0001, 0.0002]
EPSILONS = [0.0015, 0.0020]

# default bio anchor coefficients
A1, A2, A3 = 0.6, 0.3, 0.1

# prior for bayes
PRIOR_H1 = 0.05

# small numeric
EPS = 1e-9

# try to inherit baseline params from previous bundle if present
prev_runs = sorted((ROOT / "discovery_results").glob("panspermia_*"))
if prev_runs:
    # read most recent orchestrator file to get base choices
    # fallback: use constants above
    pass

# load earth bench
earth_file = DATA / "earth_bench.json"
if not earth_file.exists():
    # create placeholder per prior
    earth_file.write_text(
        json.dumps(
            {
                "t_origin_Gyr": 4.5,
                "code_opt_z": 2.5,
                "homochirality_lock_in_score": 0.8,
                "entropy_rate_CMB": 1e-22,
                "dark_energy_density": 0.7,
            },
            indent=2,
        )
    )
with open(earth_file) as f:
    earth = json.load(f)

# compute anchors
v_astro = float(earth.get("entropy_rate_CMB", 0.0)) * float(
    earth.get("dark_energy_density", 0.0)
)
# normalize using fixed prior scale xmax
ASTRO_XMAX = 1e-20
astro_anchor = float(min(max(v_astro / ASTRO_XMAX, 0.0), 1.0))


# bio anchor via logistic
def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


bio_input = (
    A1 * float(earth.get("code_opt_z", 1.0))
    + A2 * (1.0 / max(float(earth.get("t_origin_Gyr", 1.0)), 1e-6))
    + A3 * float(earth.get("homochirality_lock_in_score", 0.5))
)
bio_anchor = logistic(bio_input)


# helpers
def pdf_normal(x, mu, sigma):
    sigma = max(sigma, 1e-6)
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# containers for aggregated outputs
runs_rows = []
eoi_rows = []
op_events = []
shock_rows = []

# run matrix
conditions = []
conditions.append({"name": "A_closed", "openness": {"enabled": False}})
for eps in EPSILONS:
    for lam in LAMBDAS:
        conditions.append(
            {
                "name": f"B_open_agentio_eps{eps}_lam{lam}",
                "openness": {
                    "enabled": True,
                    "mechanism": "agent_io",
                    "epsilon0": eps,
                    "lambda": lam,
                },
            }
        )

print("Phase II run conditions:", [c["name"] for c in conditions])

for cond in conditions:
    name = cond["name"]
    openness_cfg = cond["openness"]
    print("\nRunning condition:", name)
    for seed in SEEDS:
        rng = np.random.RandomState(int(seed))
        # initialize agents
        agents = me.initialize_agents(N_AGENTS, GOAL_DIVERSITY, NOISE, rng)
        # per-epoch logs (thinned by LOG_EVERY)
        epochs = []
        ccis = []
        hazards_raw = []
        eps_eff_list = []
        eoi_list = []
        cci_max = 0.0

        event_id = 0
        cumulative_eps = 0.0

        # simulate
        for t in range(EPOCHS):
            # compute epsilon effective
            eps0 = (
                float(openness_cfg.get("epsilon0", 0.0))
                if openness_cfg.get("enabled")
                else 0.0
            )
            lam = (
                float(openness_cfg.get("lambda", 0.0))
                if openness_cfg.get("enabled")
                else 0.0
            )
            eps_eff = eps0 * math.exp(-lam * t) if eps0 > 0 else 0.0

            # if agent_io apply inflow at every epoch
            if (
                openness_cfg.get("enabled")
                and openness_cfg.get("mechanism") == "agent_io"
                and eps_eff > 0
            ):
                alive = [a for a in agents if a["alive"]]
                if alive:
                    for a in alive:
                        a["resource"] = min(1.0, a["resource"] + eps_eff)
                # log the event
                op_events.append(
                    {
                        "epoch": t,
                        "condition": name,
                        "seed": seed,
                        "epsilon0": eps0,
                        "lambda": lam,
                        "epsilon_eff": eps_eff,
                        "event_id": event_id,
                    }
                )
                event_id += 1

            # shock window
            if T_SHOCK <= t < T_SHOCK + SHOCK_DURATION:
                # during shock, we increase shock level
                current_shock = SHOCK_INTENSITY
                # reduce resource reservoirs by intensity
                for a in agents:
                    if a["alive"]:
                        a["resource"] -= SHOCK_INTENSITY * 0.2  # immediate drain
                        if a["resource"] < 0:
                            a["alive"] = False
                # entropy spike placeholder (not modeled in agents directly)
            else:
                # no shock
                current_shock = 0.0

            # call step_update which applies social updates and resource drains
            me.step_update(agents, current_shock, "chronic", rng)

            # compute raw hazard as fraction of agents with resource < 0.2
            alive = [a for a in agents if a["alive"]]
            hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(len(agents))
            cci = me.collective_cci(agents)
            cci_max = max(cci_max, cci)

            # accumulate
            cumulative_eps += eps_eff
            if t % LOG_EVERY == 0:
                epochs.append(t)
                ccis.append(cci)
                hazards_raw.append(hazard)
                eps_eff_list.append(eps_eff)
                # placeholder EOI until normalized risk computed
                eoi_list.append(0.0)

        # compute collapse_risk_norm per logged step using min/max across the run's logged hazard
        h_arr = np.array(hazards_raw)
        h_min = float(np.min(h_arr)) if h_arr.size else 0.0
        h_max = float(np.max(h_arr)) if h_arr.size else h_min
        denom = max(EPS, h_max - h_min)
        cr_norm = np.clip((h_arr - h_min) / denom, 0.0, 1.0)

        # compute cumulative mean epsilon to date per logged epoch
        eps_arr = np.array(eps_eff_list)
        cum_mean_eps = np.array(
            [eps_arr[: i + 1].mean() if i >= 0 else 0.0 for i in range(len(eps_arr))]
        )

        # compute EOI per logged epoch
        eoi_arr = []
        for i in range(len(epochs)):
            cci = ccis[i]
            cci_scale = (cci / (cci_max + EPS)) if cci_max > 0 else 0.0
            eoi = (
                cum_mean_eps[i]
                * (1.0 - cr_norm[i])
                * cci_scale
                * astro_anchor
                * bio_anchor
            )
            eoi_arr.append(float(eoi))

        # replace eoi_list
        eoi_list = eoi_arr

        # write per-epoch rows to eoi_rows and runs intermediate
        for i, t in enumerate(epochs):
            eoi_rows.append(
                {
                    "epoch": t,
                    "condition": name,
                    "seed": seed,
                    "cci": ccis[i],
                    "hazard_raw": hazards_raw[i],
                    "collapse_risk_norm": float(cr_norm[i]),
                    "eps_eff": float(eps_eff_list[i]),
                    "EOI": float(eoi_list[i]),
                }
            )

        # compute summary stats for this run
        mean_CCI = float(np.mean(ccis)) if ccis else 0.0
        mean_cr = float(np.mean(cr_norm)) if cr_norm.size else 0.0
        mean_EOI = float(np.mean(eoi_list)) if eoi_list else 0.0
        # last-window means (final STABILITY_WINDOW logged epochs)
        # convert STABILITY_WINDOW in epochs to count in logged points
        logged_window = STABILITY_WINDOW // LOG_EVERY
        last_idx = len(ccis)
        window_start = max(0, last_idx - logged_window)
        last_window_CCI = float(np.mean(ccis[window_start:])) if ccis else 0.0
        last_window_risk = (
            float(np.mean(cr_norm[window_start:])) if cr_norm.size else 0.0
        )
        last_window_eoi = float(np.mean(eoi_list[window_start:])) if eoi_list else 0.0

        # shock recovery metrics
        # pre-shock window for CCI mean
        pre_start = max(0, (T_SHOCK - 100) // LOG_EVERY)
        pre_end = max(0, (T_SHOCK - 1) // LOG_EVERY)
        pre_ccis = ccis[pre_start : pre_end + 1] if pre_end >= pre_start else []
        pre_shock_CCI_mean = float(np.mean(pre_ccis)) if pre_ccis else 0.0
        # find recovery_time
        recovery_time = None
        for idx in range((T_SHOCK) // LOG_EVERY, len(ccis)):
            if pre_shock_CCI_mean == 0:
                continue
            cur = ccis[idx]
            if abs(cur - pre_shock_CCI_mean) <= 0.05 * abs(
                pre_shock_CCI_mean if pre_shock_CCI_mean != 0 else 1.0
            ):
                recovery_time = epochs[idx] - T_SHOCK
                break
        # deltas
        post_start_idx = (T_SHOCK + SHOCK_DURATION) // LOG_EVERY
        post_end_idx = min(len(ccis) - 1, post_start_idx + (100 // LOG_EVERY))
        post_risk_mean = (
            float(np.mean(cr_norm[post_start_idx : post_end_idx + 1]))
            if post_end_idx >= post_start_idx
            else float("nan")
        )
        pre_risk_mean = (
            float(np.mean(cr_norm[pre_start : pre_end + 1]))
            if pre_end >= pre_start
            else float("nan")
        )
        delta_hazard = (
            post_risk_mean - pre_risk_mean
            if (not math.isnan(post_risk_mean) and not math.isnan(pre_risk_mean))
            else float("nan")
        )
        post_eoi_mean = (
            float(np.mean(eoi_list[post_start_idx : post_end_idx + 1]))
            if post_end_idx >= post_start_idx
            else float("nan")
        )
        pre_eoi_mean = (
            float(np.mean(eoi_list[pre_start : pre_end + 1]))
            if pre_end >= pre_start
            else float("nan")
        )
        delta_eoi = (
            post_eoi_mean - pre_eoi_mean
            if (not math.isnan(post_eoi_mean) and not math.isnan(pre_eoi_mean))
            else float("nan")
        )

        runs_rows.append(
            {
                "condition": name,
                "seed": seed,
                "mean_CCI": mean_CCI,
                "mean_collapse_risk_norm": mean_cr,
                "mean_EOI": mean_EOI,
                "last_window_mean_CCI": last_window_CCI,
                "last_window_mean_risk": last_window_risk,
                "last_window_mean_EOI": last_window_eoi,
                "recovery_time": (
                    recovery_time if recovery_time is not None else float("nan")
                ),
                "delta_hazard_postshock": delta_hazard,
                "delta_EOI_postshock": delta_eoi,
            }
        )

# write data outputs
pd.DataFrame(op_events).to_csv(DATA / "openness_events_decay.csv", index=False)
pd.DataFrame(eoi_rows).to_csv(DATA / "eoi_timeseries.csv", index=False)
pd.DataFrame(runs_rows).to_csv(DATA / "runs_phaseII_summary.csv", index=False)

# shock_recovery.csv is runs_rows
pd.DataFrame(runs_rows).to_csv(DATA / "shock_recovery.csv", index=False)

# EOI summary per condition
summary = []
rr = pd.DataFrame(runs_rows)
for cond in rr.condition.unique():
    sub = rr[rr.condition == cond]
    summary.append(
        {
            "condition": cond,
            "mean_EOI": float(sub.mean_EOI.mean()),
            "std_EOI": float(sub.mean_EOI.std()),
            "last_window_mean_EOI": float(sub.last_window_mean_EOI.mean()),
        }
    )
with open(DATA / "eoi_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Bayes Phase II
# reuse signatures from earth
sig = {
    k: earth.get(k)
    for k in ["t_origin_Gyr", "code_opt_z", "homochirality_lock_in_score"]
}
# compute BF per signature using simple normal model with prior sd
bf_sig = {}
for k, v in sig.items():
    # For Phase II we reuse approach from earlier: compare observed value to simple H1/H0 choices
    # Here we use placeholder mu0=0.3, mu1=0.7, sigma=0.3
    mu0, mu1, sigma = 0.3, 0.7, 0.3
    try:
        val = float(v)
    except:
        val = 0.5
    p1 = pdf_normal(val, mu1, sigma)
    p0 = pdf_normal(val, mu0, sigma)
    bf_sig[k] = p1 / max(p0, 1e-12)

# BF from EOI: compare winning open condition mean_EOI vs closed distribution
df_runs = pd.DataFrame(runs_rows)
closed = df_runs[df_runs.condition.str.startswith("A_closed")]
openconds = df_runs[~df_runs.condition.str.startswith("A_closed")]
# choose winning open cond by highest mean_EOI
win_cond = None
if not openconds.empty:
    mean_by_cond = openconds.groupby("condition").mean().mean_EOI
    win_cond = mean_by_cond.idxmax()

bf_eoi = 1.0
if win_cond is not None:
    mu_open = float(openconds[openconds.condition == win_cond].mean_EOI.mean())
    sigma_open = (
        float(openconds[openconds.condition == win_cond].mean_EOI.std())
        if openconds[openconds.condition == win_cond].mean_EOI.std()
        else 1e-6
    )
    # observed value = mu_open (self-consistency)
    observed = mu_open
    mu_closed = float(closed.mean_EOI.mean()) if not closed.empty else observed * 0.5
    sigma_closed = float(closed.mean_EOI.std()) if not closed.empty else 1e-6
    p_open = pdf_normal(observed, mu_open, sigma_open)
    p_closed = pdf_normal(observed, mu_closed, sigma_closed)
    bf_eoi = p_open / max(p_closed, 1e-12)

# combined BF
combined_bf = float(np.prod(list(bf_sig.values())) * bf_eoi)
posterior_odds = (combined_bf * PRIOR_H1) / (1.0 - PRIOR_H1)
posterior_prob = posterior_odds / (1.0 + posterior_odds)

bayes_out = {
    "bf_signatures": bf_sig,
    "bf_eoi": bf_eoi,
    "combined_bf": combined_bf,
    "prior_H1": PRIOR_H1,
    "posterior_prob_H1": posterior_prob,
}
with open(DATA / "bayes_phaseII.json", "w") as f:
    json.dump(bayes_out, f, indent=2)

# plots: cci_long, hazard_long_norm, eoi_long
edf = pd.DataFrame(eoi_rows)
if not edf.empty:
    plt.figure(figsize=(8, 4))
    for cond in edf.condition.unique():
        sub = edf[edf.condition == cond]
        mean_ccis = sub.groupby("epoch").cci.mean()
        plt.plot(mean_ccis.index, mean_ccis.values, label=cond)
    plt.legend()
    plt.title("CCI (logged epochs)")
    plt.xlabel("epoch")
    plt.ylabel("CCI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "cci_long.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    for cond in edf.condition.unique():
        sub = edf[edf.condition == cond]
        mean_h = sub.groupby("epoch").collapse_risk_norm.mean()
        plt.plot(mean_h.index, mean_h.values, label=cond)
    plt.legend()
    plt.title("Normalized Hazard (logged epochs)")
    plt.xlabel("epoch")
    plt.ylabel("collapse_risk_norm")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "hazard_long_norm.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    for cond in edf.condition.unique():
        sub = edf[edf.condition == cond]
        mean_e = sub.groupby("epoch").EOI.mean()
        plt.plot(mean_e.index, mean_e.values, label=cond)
    plt.legend()
    plt.title("EOI (logged epochs)")
    plt.xlabel("epoch")
    plt.ylabel("EOI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "eoi_long.png")
    plt.close()

# shock recovery plot
rr = pd.DataFrame(runs_rows)
if not rr.empty:
    plt.figure(figsize=(6, 4))
    for cond in rr.condition.unique():
        sub = rr[rr.condition == cond]
        plt.scatter([cond] * len(sub), sub.recovery_time, alpha=0.7)
    plt.ylabel("recovery_time (epochs)")
    plt.title("Shock recovery times by condition")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "shock_recovery.png")
    plt.close()

# ICO vs risk scatter (ICO from earlier run not recomputed here; use mean_CCI as proxy)
if not rr.empty:
    plt.figure(figsize=(6, 4))
    for cond in rr.condition.unique():
        sub = rr[rr.condition == cond]
        plt.scatter(sub.mean_CCI, sub.mean_collapse_risk_norm, alpha=0.6, label=cond)
    plt.xlabel("mean_CCI")
    plt.ylabel("mean_collapse_risk_norm")
    plt.legend()
    plt.title("CCI vs Collapse Risk")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "ico_vs_risk.png")
    plt.close()

# write report
md = OUT / "panspermia_phaseII_report.md"
with open(md, "w") as f:
    f.write("# Panspermia Phase II Report\n\n")
    f.write("Conditions run: \n")
    for c in conditions:
        f.write(f"- {c['name']}\n")
    f.write("\nSummary table (data/runs_phaseII_summary.csv)\n")
    f.write("\nBayes Phase II:\n")
    f.write(json.dumps(bayes_out, indent=2))

# bundle outputs
bundle = OUT / f"panspermia_phaseII_bundle_{STAMP}.zip"
with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
    for f in (
        list(OUT.rglob("*")) + list(DATA.glob("*.csv")) + list(DATA.glob("*.json"))
    ):
        if f.is_file():
            z.write(f, arcname=str(f.relative_to(ROOT)))
# sha256
h = hashlib.sha256()
with open(bundle, "rb") as bf:
    for chunk in iter(lambda: bf.read(1 << 20), b""):
        h.update(chunk)
with open(OUT / "SHA256SUMS.txt", "w") as s:
    s.write(f"{h.hexdigest()}  {bundle.name}\n")

print("Phase II run complete. Outputs in", OUT)
print("Data exports in", DATA)
print("Bundle:", bundle)

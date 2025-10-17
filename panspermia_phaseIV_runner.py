#!/usr/bin/env python3
"""
Panspermia Phase IV runner
Implements re-contact scenarios: baseline_closed, transient_recontact, entropy_feedback
Produces EOI, entropy profiles, recontact events, recovery metrics, Bayes update, plots, and a bundle.
"""
import datetime
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

ROOT = Path(".")
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "discovery_results" / f"panspermia_phaseIV_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

# --- Config defaults ---
EPOCHS = 6000
SEEDS = [1, 2, 3, 4]
LOG_EVERY = 10
STABILITY_WINDOW = 200
BASE_EPSILON = 0.0015
BASE_LAMBDA = 1e-5
SHOCK_ENABLED = True
T_SHOCK = 3000
SHOCK_INTENSITY = 0.35
SHOCK_DURATION = 25

RECONTACT = {
    "t_reopen": 4000,
    "duration": 50,
    "epsilon_boost_factor": 10.0,
    "decay_mode": "entropy",
}

MODES = ["baseline_closed", "transient_recontact", "entropy_feedback"]
ENTROPY_COUPLING = 0.2

# thresholds
PERSISTENCE_THRESHOLD_FRAC = 0.01  # 1% baseline closed value
PERSISTENCE_MIN_DELTA = 0.01

# Load anchors
earth_file = DATA / "earth_bench.json"
if earth_file.exists():
    earth = json.loads(earth_file.read_text())
else:
    earth = {
        "t_origin_Gyr": 4.5,
        "code_opt_z": 2.5,
        "homochirality_lock_in_score": 0.8,
        "entropy_rate_CMB": 1.0,
        "dark_energy_density": 0.69,
    }

v_astro = float(earth.get("entropy_rate_CMB", 1.0)) * float(
    earth.get("dark_energy_density", 1.0)
)
ASTRO_XMAX = max(v_astro, 1.0)
astro_anchor = v_astro / ASTRO_XMAX


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


bio_anchor = logistic(
    0.6 * float(earth.get("code_opt_z", 2.5))
    + 0.3 * (1.0 / float(earth.get("t_origin_Gyr", 4.5)))
    + 0.1 * float(earth.get("homochirality_lock_in_score", 0.8))
)

print("Phase IV anchors:", astro_anchor, bio_anchor)

# helpers
EPS = 1e-12


def clamp01(arr):
    return np.minimum(np.maximum(arr, 0.0), 1.0)


# containers for outputs
runs_summary = []
eoi_rows = []
entropy_rows = []
recontact_events = []
recovery_rows = []

# We'll first run baseline_closed across seeds and record baseline EOI per epoch (mean across seeds)
baseline_eoi_by_epoch = defaultdict(list)
baseline_ccis_by_epoch = defaultdict(list)

print("Running baseline_closed seeds...")
for seed in SEEDS:
    rng = np.random.RandomState(int(seed))
    agents = me.initialize_agents(200, 3, 0.05, rng)
    epochs_logged = []
    cci_logged = []
    hazard_logged = []
    eps_eff_logged = []
    entropy_logged = []
    cci_max = 0.0
    for t in range(EPOCHS):
        # baseline closed: time-decayed epsilon
        eps0 = BASE_EPSILON
        lam = BASE_LAMBDA
        eps_eff = eps0 * math.exp(-lam * t)
        # no explicit event logging for baseline beyond recording eps_eff
        # shock
        if SHOCK_ENABLED and T_SHOCK <= t < T_SHOCK + SHOCK_DURATION:
            current_shock = SHOCK_INTENSITY
            for a in agents:
                if a["alive"]:
                    a["resource"] -= SHOCK_INTENSITY * 0.2
                    if a["resource"] < 0:
                        a["alive"] = False
        else:
            current_shock = 0.0
        # openness inflow (closed but with eps_eff small)
        if eps_eff > 0:
            alive = [a for a in agents if a["alive"]]
            if alive:
                for a in alive:
                    a["resource"] = min(1.0, a["resource"] + eps_eff)
        # update
        me.step_update(agents, current_shock, "chronic", rng)
        # metrics
        alive = [a for a in agents if a["alive"]]
        hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(len(agents))
        cci = me.collective_cci(agents)
        cci_max = max(cci_max, cci)
        # entropy S(t): entropy of mean belief distribution
        B = np.array([a["belief"] for a in agents]) if agents else np.array([])
        mean_b = B.mean(axis=0) if B.size else np.array([])
        if mean_b.size:
            S = -np.sum(mean_b * np.log(mean_b + 1e-12))
            S_max = math.log(len(mean_b) + 1e-12)
        else:
            S = 0.0
            S_max = 1.0
        if t % LOG_EVERY == 0:
            epochs_logged.append(t)
            cci_logged.append(cci)
            hazard_logged.append(hazard)
            eps_eff_logged.append(eps_eff)
            entropy_logged.append(S)
    # compute normalized collapse risk per run (min-max over logged points)
    h_arr = np.array(hazard_logged)
    h_min = float(np.min(h_arr)) if h_arr.size else 0.0
    h_max = float(np.max(h_arr)) if h_arr.size else h_min
    denom = max(1e-9, h_max - h_min)
    cr_norm = clamp01((h_arr - h_min) / denom)
    eps_arr = np.array(eps_eff_logged)
    cum_mean_eps = (
        np.array([eps_arr[: i + 1].mean() for i in range(len(eps_arr))])
        if len(eps_arr) > 0
        else np.array([])
    )
    # compute EOI per logged epoch
    eoi_arr = []
    for i in range(len(epochs_logged)):
        cci = cci_logged[i]
        cci_scale = (cci / (cci_max + EPS)) if cci_max > 0 else 0.0
        eoi = (
            cum_mean_eps[i] * (1.0 - cr_norm[i]) * cci_scale * astro_anchor * bio_anchor
        )
        eoi_arr.append(float(eoi))
    # clamp EOI
    eoi_arr = np.array(eoi_arr)
    eoi_arr = clamp01(eoi_arr)
    # save baseline time series
    for i, t in enumerate(epochs_logged):
        baseline_eoi_by_epoch[t].append(float(eoi_arr[i]))
        baseline_ccis_by_epoch[t].append(float(cci_logged[i]))
        eoi_rows.append(
            {
                "mode": "baseline_closed",
                "epoch": t,
                "seed": seed,
                "cci": cci_logged[i],
                "hazard_raw": hazard_logged[i],
                "collapse_risk_norm": float(cr_norm[i]),
                "eps_eff": float(eps_eff_logged[i]),
                "EOI": float(eoi_arr[i]),
            }
        )
    # store entropy profile
    for i, t in enumerate(epochs_logged):
        entropy_rows.append(
            {
                "mode": "baseline_closed",
                "epoch": t,
                "seed": seed,
                "entropy": entropy_logged[i],
            }
        )
    # no per-run summary for baseline here; will compute later

# compute baseline mean EOI by epoch
baseline_mean_eoi = {
    t: (sum(vals) / len(vals) if vals else 0.0)
    for t, vals in baseline_eoi_by_epoch.items()
}


# function to run a mode for a seed
def run_mode_for_seed(mode, seed):
    rng = np.random.RandomState(int(seed))
    agents = me.initialize_agents(200, 3, 0.05, rng)
    epochs_logged = []
    cci_logged = []
    hazard_logged = []
    eps_eff_logged = []
    entropy_logged = []
    cci_max = 0.0
    events_local = []
    for t in range(EPOCHS):
        eps0 = BASE_EPSILON
        lam = BASE_LAMBDA
        # base eps eff
        base_eff = eps0 * math.exp(-lam * t)
        eps_eff = base_eff
        # apply mode adjustments
        if mode == "transient_recontact":
            tr = RECONTACT
            if tr["t_reopen"] <= t < tr["t_reopen"] + tr["duration"]:
                # boost proportional to eps0 * boost * exp(-lam * t_reopen)
                eps_eff = (
                    eps0 * tr["epsilon_boost_factor"] * math.exp(-lam * tr["t_reopen"])
                )
                events_local.append(
                    {
                        "epoch": t,
                        "mode": mode,
                        "seed": seed,
                        "epsilon0": eps0,
                        "lambda": lam,
                        "epsilon_eff": eps_eff,
                        "event": "recontact_burst",
                    }
                )
        elif mode == "entropy_feedback":
            # compute entropy S(t) of mean belief to compute f_entropy
            B = np.array([a["belief"] for a in agents]) if agents else np.array([])
            mean_b = B.mean(axis=0) if B.size else np.array([])
            if mean_b.size:
                S = -np.sum(mean_b * np.log(mean_b + 1e-12))
                S_max = math.log(len(mean_b) + 1e-12)
                f_entropy = ENTROPY_COUPLING * (1.0 - (S / (S_max + 1e-12)))
                eps_eff = base_eff * (1.0 + f_entropy)
            else:
                eps_eff = base_eff
        # shock
        if SHOCK_ENABLED and T_SHOCK <= t < T_SHOCK + SHOCK_DURATION:
            current_shock = SHOCK_INTENSITY
            for a in agents:
                if a["alive"]:
                    a["resource"] -= SHOCK_INTENSITY * 0.2
                    if a["resource"] < 0:
                        a["alive"] = False
        else:
            current_shock = 0.0
        # apply eps_eff inflow
        if eps_eff > 0:
            alive = [a for a in agents if a["alive"]]
            if alive:
                for a in alive:
                    a["resource"] = min(1.0, a["resource"] + eps_eff)
        # step update
        me.step_update(agents, current_shock, "chronic", rng)
        # metrics
        alive = [a for a in agents if a["alive"]]
        hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(len(agents))
        cci = me.collective_cci(agents)
        cci_max = max(cci_max, cci)
        # entropy
        B = np.array([a["belief"] for a in agents]) if agents else np.array([])
        mean_b = B.mean(axis=0) if B.size else np.array([])
        if mean_b.size:
            S = -np.sum(mean_b * np.log(mean_b + 1e-12))
        else:
            S = 0.0
        if t % LOG_EVERY == 0:
            epochs_logged.append(t)
            cci_logged.append(cci)
            hazard_logged.append(hazard)
            eps_eff_logged.append(eps_eff)
            entropy_logged.append(S)
    # normalize collapse risk per run
    h_arr = np.array(hazard_logged)
    h_min = float(np.min(h_arr)) if h_arr.size else 0.0
    h_max = float(np.max(h_arr)) if h_arr.size else h_min
    denom = max(1e-9, h_max - h_min)
    cr_norm = clamp01((h_arr - h_min) / denom)
    eps_arr = np.array(eps_eff_logged)
    cum_mean_eps = (
        np.array([eps_arr[: i + 1].mean() for i in range(len(eps_arr))])
        if len(eps_arr) > 0
        else np.array([])
    )
    eoi_arr = []
    for i in range(len(epochs_logged)):
        cci = cci_logged[i]
        cci_scale = (cci / (cci_max + EPS)) if cci_max > 0 else 0.0
        eoi = (
            cum_mean_eps[i] * (1.0 - cr_norm[i]) * cci_scale * astro_anchor * bio_anchor
        )
        eoi_arr.append(float(eoi))
    eoi_arr = clamp01(np.array(eoi_arr))
    # record rows
    for i, t in enumerate(epochs_logged):
        eoi_rows.append(
            {
                "mode": mode,
                "epoch": t,
                "seed": seed,
                "cci": cci_logged[i],
                "hazard_raw": hazard_logged[i],
                "collapse_risk_norm": float(cr_norm[i]),
                "eps_eff": float(eps_eff_logged[i]),
                "entropy": float(entropy_logged[i]),
                "EOI": float(eoi_arr[i]),
            }
        )
        entropy_rows.append(
            {
                "mode": mode,
                "epoch": t,
                "seed": seed,
                "entropy": float(entropy_logged[i]),
            }
        )
    # compute pre_recontact metrics
    tr = RECONTACT
    pre_start = max(0, (tr["t_reopen"] - 100) // LOG_EVERY)
    pre_end = max(0, (tr["t_reopen"] - 1) // LOG_EVERY)
    pre_ccis = cci_logged[pre_start : pre_end + 1] if pre_end >= pre_start else []
    pre_recontact_CCI_mean = float(np.mean(pre_ccis)) if pre_ccis else 0.0
    # post recontact window
    post_start_idx = (tr["t_reopen"] + tr["duration"]) // LOG_EVERY
    post_end_idx = min(len(cci_logged) - 1, post_start_idx + (100 // LOG_EVERY))
    delta_CCI_post = (
        float(np.mean(cci_logged[post_start_idx : post_end_idx + 1]))
        - pre_recontact_CCI_mean
        if post_end_idx >= post_start_idx and pre_ccis
        else float("nan")
    )
    delta_risk_post = (
        float(np.mean(cr_norm[post_start_idx : post_end_idx + 1]))
        - float(np.mean(cr_norm[pre_start : pre_end + 1]))
        if post_end_idx >= post_start_idx and pre_end >= pre_start
        else float("nan")
    )
    delta_eoi_post = (
        float(np.mean(eoi_arr[post_start_idx : post_end_idx + 1]))
        - float(np.mean(eoi_arr[pre_start : pre_end + 1]))
        if post_end_idx >= post_start_idx and pre_end >= pre_start
        else float("nan")
    )
    # recovery_time_post_recontact: smallest dt >=0 s.t. CCI at t_reopen+dt within 5% of pre_recontact_CCI_mean
    recovery_time = float("nan")
    for idx in range((tr["t_reopen"]) // LOG_EVERY, len(cci_logged)):
        if pre_recontact_CCI_mean == 0:
            break
        cur = cci_logged[idx]
        if abs(cur - pre_recontact_CCI_mean) <= 0.05 * abs(
            pre_recontact_CCI_mean if pre_recontact_CCI_mean != 0 else 1.0
        ):
            recovery_time = float(epochs_logged[idx] - tr["t_reopen"])
            break
    # persistence_time: epochs until EOI returns within 1% of baseline closed value at same epoch
    persistence_time = float("nan")
    baseline_map = baseline_mean_eoi
    # for each logged epoch after recontact end
    for idx in range(post_start_idx, len(epochs_logged)):
        t = epochs_logged[idx]
        baseline_val = baseline_map.get(t, None)
        if baseline_val is None:
            continue
        # if within 1% of baseline
        if abs(eoi_arr[idx] - baseline_val) <= max(
            PERSISTENCE_THRESHOLD_FRAC
            * abs(baseline_val if baseline_val != 0 else 1.0),
            0.01,
        ):
            persistence_time = float(t - tr["t_reopen"])
            break
    # record per-run summary
    runs_summary.append(
        {
            "mode": mode,
            "seed": seed,
            "mean_CCI": float(np.mean(cci_logged)) if cci_logged else 0.0,
            "mean_collapse_risk_norm": float(np.mean(cr_norm)) if cr_norm.size else 0.0,
            "mean_EOI": float(np.mean(eoi_arr)) if eoi_arr.size else 0.0,
            "last_window_mean_CCI": (
                float(np.mean(cci_logged[-(STABILITY_WINDOW // LOG_EVERY) :]))
                if cci_logged
                else 0.0
            ),
            "last_window_mean_risk": (
                float(np.mean(cr_norm[-(STABILITY_WINDOW // LOG_EVERY) :]))
                if cr_norm.size
                else 0.0
            ),
            "last_window_mean_EOI": (
                float(np.mean(eoi_arr[-(STABILITY_WINDOW // LOG_EVERY) :]))
                if eoi_arr.size
                else 0.0
            ),
            "pre_recontact_CCI_mean": pre_recontact_CCI_mean,
            "delta_CCI_post_recontact": delta_CCI_post,
            "delta_risk_post_recontact": delta_risk_post,
            "delta_EOI_post_recontact": delta_eoi_post,
            "recovery_time_post_recontact": recovery_time,
            "persistence_time": persistence_time,
        }
    )
    # append events
    for ev in events_local:
        recontact_events.append(ev)


# Run other modes for each seed
print("Running other modes:", MODES)
for mode in MODES:
    if mode == "baseline_closed":
        # baseline already run above and recorded in eoi_rows
        continue
    for seed in SEEDS:
        run_mode_for_seed(mode, seed)

# write outputs
pd.DataFrame(runs_summary).to_csv(DATA / "runs_phaseIV_summary.csv", index=False)
pd.DataFrame(recontact_events).to_csv(DATA / "recontact_events.csv", index=False)
pd.DataFrame(eoi_rows).to_csv(DATA / "eoi_recontact.csv", index=False)
pd.DataFrame(entropy_rows).to_csv(DATA / "entropy_profiles.csv", index=False)
pd.DataFrame(runs_summary).to_csv(DATA / "recontact_recovery.csv", index=False)

# Bayes Phase IV: L_recontact based on persistence > threshold
# define threshold persistence>0 (non-NaN positive) or > some epochs (e.g., 10)
PERSISTENCE_THRESHOLD_EPOCHS = 10
rs = pd.DataFrame(runs_summary)
# compute proportions per mode
modes = rs["mode"].unique()
persistence_stats = {}
for m in modes:
    sub = rs[rs["mode"] == m]
    total = len(sub)
    count = (
        sub["persistence_time"]
        .dropna()
        .apply(lambda x: x >= PERSISTENCE_THRESHOLD_EPOCHS)
        .sum()
    )
    persistence_stats[m] = {
        "total": int(total),
        "count_persistent": int(count),
        "prop": float(count) / max(1, total),
    }
# L_recontact = prop(mode=entropy_feedback)/prop(baseline_closed)
p_h1 = persistence_stats.get("entropy_feedback", {}).get("prop", 0.0)
p_h0 = persistence_stats.get("baseline_closed", {}).get("prop", 0.0)
L_recontact = (p_h1 + 1e-6) / (p_h0 + 1e-6)
# combine with signature BFs (reuse simple approach)
sig = {
    "t_origin_Gyr": earth.get("t_origin_Gyr", 4.5),
    "code_opt_z": earth.get("code_opt_z", 2.5),
    "homochirality_lock_in_score": earth.get("homochirality_lock_in_score", 0.8),
}
bf_sig = {}
for k, v in sig.items():
    try:
        val = float(v)
    except:
        val = 0.5
    p1 = pdf = math.exp(-0.5 * ((val - 0.7) / 0.3) ** 2) / (
        0.3 * math.sqrt(2 * math.pi)
    )
    p0 = math.exp(-0.5 * ((val - 0.3) / 0.3) ** 2) / (0.3 * math.sqrt(2 * math.pi))
    bf_sig[k] = p1 / max(p0, 1e-12)
combined_bf = float(np.prod(list(bf_sig.values())) * L_recontact)
posterior = (combined_bf * 0.05) / (1.0 - 0.05)

bayes_out = {
    "bf_signatures": bf_sig,
    "L_recontact": L_recontact,
    "combined_bf": combined_bf,
    "posterior_odds": posterior,
}
with open(DATA / "bayes_phaseIV.json", "w") as f:
    json.dump(bayes_out, f, indent=2)

# plots
edf = (
    pd.read_csv(DATA / "eoi_recontact.csv")
    if (DATA / "eoi_recontact.csv").exists()
    else pd.DataFrame()
)
if not edf.empty:
    plt.figure(figsize=(8, 4))
    for m in MODES:
        sub = edf[edf["mode"] == m]
        if sub.empty:
            continue
        grp = sub.groupby("epoch").EOI.mean()
        plt.plot(grp.index, grp.values, label=m)
    plt.legend()
    plt.title("EOI time series by mode")
    plt.xlabel("epoch")
    plt.ylabel("EOI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "eoi_recontact_timeseries.png")
    plt.close()

    # entropy vs eoi scatter
    plt.figure(figsize=(6, 4))
    for m in MODES:
        sub = edf[edf["mode"] == m]
        if sub.empty:
            continue
        plt.scatter(sub.entropy, sub.EOI, alpha=0.3, label=m)
    plt.legend()
    plt.xlabel("entropy")
    plt.ylabel("EOI")
    plt.title("Entropy vs EOI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "entropy_vs_eoi.png")
    plt.close()

# persistence decay plot
rs = pd.DataFrame(runs_summary)
if not rs.empty:
    plt.figure(figsize=(6, 4))
    for m in MODES:
        sub = rs[rs["mode"] == m]
        if sub.empty:
            continue
        plt.scatter([m] * len(sub), sub.persistence_time, alpha=0.6)
    plt.ylabel("persistence_time")
    plt.title("Persistence time by mode")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "persistence_decay.png")
    plt.close()

# recontact recovery plot
if not rs.empty:
    plt.figure(figsize=(6, 4))
    for m in MODES:
        sub = rs[rs["mode"] == m]
        if sub.empty:
            continue
        plt.scatter([m] * len(sub), sub.recovery_time_post_recontact, alpha=0.6)
    plt.ylabel("recovery_time_post_recontact")
    plt.title("Recovery time by mode")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "recontact_recovery.png")
    plt.close()

# save tables
pd.DataFrame(runs_summary).to_csv(OUT / "runs_phaseIV_summary.csv", index=False)
pd.DataFrame(recontact_events).to_csv(OUT / "recontact_events.csv", index=False)
pd.DataFrame(eoi_rows).to_csv(OUT / "eoi_recontact.csv", index=False)
pd.DataFrame(entropy_rows).to_csv(OUT / "entropy_profiles.csv", index=False)
pd.DataFrame(runs_summary).to_csv(OUT / "recontact_recovery.csv", index=False)

# report
md = OUT / "panspermia_phaseIV_report.md"
with open(md, "w") as f:
    f.write("# Panspermia Phase IV — Re-contact analysis\n\n")
    f.write("Anchors:\n")
    f.write(
        json.dumps({"astro_anchor": astro_anchor, "bio_anchor": bio_anchor}, indent=2)
    )
    f.write("\n\nBayes Phase IV summary:\n")
    f.write(json.dumps(bayes_out, indent=2))

# bundle
bundle = OUT / f"panspermia_phaseIV_bundle_{STAMP}.zip"
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

print("Phase IV complete. Outputs in", OUT)
print("Data exports in", DATA)
print("Bundle:", bundle)

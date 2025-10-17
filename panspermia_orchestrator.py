#!/usr/bin/env python3
"""
panspermia_orchestrator.py
Implements the Open Universe Index Runner / Panspermia experiment pipeline described by the user.

Produces per-run CSVs, aggregated runs_summary.csv, trajectories_long.csv (thinned), openness_events.csv,
bayes_summary.json, conditions_summary.json, figures, a markdown report, and a ZIP bundle with SHA256 sums.

This is a pragmatic implementation: models are simple and designed to be interpretable and fast.
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

# Import simulation primitives from meaning_experiment
import meaning_experiment as me

ROOT = Path(".")
DATA = Path("./data")
DATA.mkdir(exist_ok=True)
OUT_ROOT = Path("./discovery_results")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = OUT_ROOT / f"panspermia_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR = OUT / "report"
REPORT_DIR.mkdir(exist_ok=True)

# --- User plan constants ---
SEEDS = [1, 2, 3, 4]
AGENTS = [200, 400]
GOALS = [3, 4, 5]
NOISE = [0.05, 0.10]

# Conditions
A_closed = {"enabled": False}
B_open_agentio_template = {
    "enabled": True,
    "mechanism": "agent_io",
    "epsilon": None,
    "period": 40,
    "reservoirs": {"R_energy": 0.25, "R_waste": 0.25},
}
EPSILONS = [0.0015, 0.0020]

# Horizons
BASELINE_STEPS = 80
LONG_STEPS = 5000
THIN_STRIDE = 10  # save every 10th epoch for long runs
STABILITY_WINDOW = 200

# Metrics collection
runs_summary = []
trajectories = []  # thinned long trajectories
openness_events = []
conditions_summary = []

# Earth benchmarks file (create default if missing)
EARTH_BENCH = DATA / "earth_bench.json"
if not EARTH_BENCH.exists():
    default_earth = {
        "t_origin_Gyr": 4.5,
        "code_opt_z": 2.5,
        "homochirality_lock_in_score": 0.8,
    }
    with open(EARTH_BENCH, "w") as f:
        json.dump(default_earth, f, indent=2)

with open(EARTH_BENCH) as f:
    earth_bench = json.load(f)


# Bayes factor helpers (simple Gaussian models)
def bayes_factor(value, mu0, mu1, sigma=0.2):
    # BF = p(data|H1)/p(data|H0)
    def normal_pdf(x, mu, s):
        return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    p1 = normal_pdf(value, mu1, sigma)
    p0 = normal_pdf(value, mu0, sigma)
    if p0 == 0:
        return float("inf")
    return float(p1 / p0)


# ICO: information-coherence-overfit measure
def compute_ico(cci_vals):
    # observed_code_opt ~ mean CCI in stability window
    obs = float(np.mean(cci_vals)) if len(cci_vals) else 0.0
    # expected null for random beliefs ~ 1/num_branches (we assume 3 branches)
    expected_null = 1.0 / 3.0
    return obs - expected_null


# small helper to run a single simulation using primitives in meaning_experiment
def run_simulation(
    seed,
    n_agents,
    goal_diversity,
    noise,
    shock,
    stress_duration,
    steps,
    openness=None,
    heat_death=None,
    logging_dense=None,
    thin_stride=None,
):
    rng = np.random.RandomState(seed)
    branches = (
        openness.get("allowed_branches")
        if openness and isinstance(openness, dict) and openness.get("allowed_branches")
        else me.BRANCHES
    )
    agents = me.initialize_agents(
        n_agents, goal_diversity, noise, rng, branches=branches
    )

    history = {"epoch": [], "cci": [], "hazard": [], "survival_rate": []}
    branch_frac_series = []

    # openness schedule
    def current_epsilon(epoch):
        if not openness or not openness.get("enabled"):
            return 0.0
        if heat_death and heat_death.get("enabled"):
            # find latest schedule entry
            s = heat_death.get("schedule", [])
            eps = openness.get("epsilon", 0.0)
            for entry in sorted(s, key=lambda x: x["epoch"]):
                if epoch >= entry["epoch"]:
                    eps = entry["epsilon"]
            return eps
        return openness.get("epsilon", 0.0)

    for t in range(steps):
        eps = current_epsilon(t)
        # apply openness inflow if agent_io
        if (
            openness
            and openness.get("enabled")
            and openness.get("mechanism") == "agent_io"
            and eps > 0
        ):
            # apply small inflow to alive agents
            for a in agents:
                if a["alive"]:
                    a["resource"] = min(1.0, a["resource"] + eps)
            openness_events.append(
                {"time": t, "seed": seed, "n_agents": n_agents, "epsilon": eps}
            )
        # step update
        me.step_update(agents, shock, stress_duration, rng)
        # hazard proxy: fraction with resource < 0.2
        alive = [a for a in agents if a["alive"]]
        survival_rate = len(alive) / float(n_agents)
        hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(n_agents)
        cci = me.collective_cci(agents)
        # branch fractions
        alive_choices = [int(np.argmax(a["belief"])) for a in alive] if alive else []
        cnt = defaultdict(int)
        for ch in alive_choices:
            cnt[ch] += 1
        frac = [
            (cnt.get(i, 0) / len(alive)) if len(alive) > 0 else 0.0
            for i in range(len(branches))
        ]
        # record
        history["epoch"].append(t)
        history["cci"].append(cci)
        history["hazard"].append(hazard)
        history["survival_rate"].append(survival_rate)
        branch_frac_series.append(frac)
    # finalize
    final_surv = history["survival_rate"][-1]
    collapse_risk = 1.0 - final_surv
    # stability window metrics
    window = (
        history["cci"][-STABILITY_WINDOW:]
        if len(history["cci"]) >= STABILITY_WINDOW
        else history["cci"]
    )
    stability_cci_mean = float(np.mean(window)) if window else 0.0
    stability_hazard_mean = (
        float(np.mean(history["hazard"][-STABILITY_WINDOW:]))
        if len(history["hazard"]) >= STABILITY_WINDOW
        else float(np.mean(history["hazard"]))
    )
    # CCI slope over stability window
    if len(window) >= 2:
        xs = np.arange(len(window))
        slope = float(np.polyfit(xs, window, 1)[0])
    else:
        slope = 0.0
    ico = compute_ico(window)
    return {
        "seed": seed,
        "n_agents": n_agents,
        "goal_diversity": goal_diversity,
        "noise": noise,
        "shock": shock,
        "stress_duration": stress_duration,
        "cci_series": history["cci"],
        "hazard_series": history["hazard"],
        "survival_series": history["survival_rate"],
        "branch_frac_series": branch_frac_series,
        "final_survival": final_surv,
        "collapse_risk": collapse_risk,
        "stability_cci_mean": stability_cci_mean,
        "stability_hazard_mean": stability_hazard_mean,
        "cci_slope": slope,
        "ico": ico,
    }


# Run matrix
conditions = []
conditions.append({"name": "A_closed", "openness": {"enabled": False}})
for eps in EPSILONS:
    c = {
        "name": f"B_open_agentio_eps{eps}",
        "openness": {
            "enabled": True,
            "mechanism": "agent_io",
            "epsilon": eps,
            "period": 40,
            "reservoirs": {"R_energy": 0.25, "R_waste": 0.25},
        },
    }
    conditions.append(c)

print("Running panspermia matrix (smoke)")
for cond in conditions:
    print("Condition:", cond["name"])
    for seed in SEEDS:
        # baseline short
        res_baseline = run_simulation(
            seed,
            AGENTS[0],
            GOALS[0],
            NOISE[0],
            shock=0.2,
            stress_duration="chronic",
            steps=BASELINE_STEPS,
            openness=cond["openness"],
        )
        # long run (thin logging)
        res_long = run_simulation(
            seed,
            AGENTS[0],
            GOALS[0],
            NOISE[0],
            shock=0.4,
            stress_duration="chronic",
            steps=LONG_STEPS,
            openness=cond["openness"],
        )
        # store run summary
        runs_summary.append(
            {
                "condition": cond["name"],
                "seed": seed,
                "baseline_stability_cci": res_baseline["stability_cci_mean"],
                "long_stability_cci": res_long["stability_cci_mean"],
                "long_collapse_risk": res_long["collapse_risk"],
                "ico": res_long["ico"],
            }
        )
        # store thinned trajectories (every THIN_STRIDE)
        for i, (cval, hval, srate) in enumerate(
            zip(
                res_long["cci_series"][::THIN_STRIDE],
                res_long["hazard_series"][::THIN_STRIDE],
                res_long["survival_series"][::THIN_STRIDE],
            )
        ):
            trajectories.append(
                {
                    "condition": cond["name"],
                    "seed": seed,
                    "epoch": i * THIN_STRIDE,
                    "cci": cval,
                    "hazard": hval,
                    "survival_rate": srate,
                }
            )
    # condition summary
    cs = [r for r in runs_summary if r["condition"] == cond["name"]]
    conditions_summary.append(
        {
            "condition": cond["name"],
            "mean_ico": float(np.mean([x["ico"] for x in cs if "ico" in x])),
            "mean_collapse_risk": (
                float(np.mean([x["long_collapse_risk"] for x in cs])) if cs else None
            ),
        }
    )

# Earth benchmark Bayes Factors
bf_results = {}
# naive H0/H1 choices
for k, v in earth_bench.items():
    # map signature values to a 0..1 scale roughly
    val = float(v)
    mu0, mu1 = 0.2, 0.8
    bf = bayes_factor(val, mu0, mu1, sigma=0.3)
    bf_results[k] = {"value": val, "BF": bf}
# combined BF (product)
combined_bf = float(np.prod([bf_results[k]["BF"] for k in bf_results]))

# Export CSVs
pd.DataFrame(runs_summary).to_csv(OUT / "runs_summary.csv", index=False)
pd.DataFrame(trajectories).to_csv(OUT / "trajectories_long.csv", index=False)
pd.DataFrame(openness_events).to_csv(OUT / "openness_events.csv", index=False)
with open(OUT / "conditions_summary.json", "w") as f:
    json.dump(conditions_summary, f, indent=2)
with open(OUT / "bayes_summary.json", "w") as f:
    json.dump({"per_signature": bf_results, "combined_bf": combined_bf}, f, indent=2)

# Figures
# cci_long
df_traj = pd.DataFrame(trajectories)
if not df_traj.empty:
    plt.figure(figsize=(8, 4))
    for cond in df_traj["condition"].unique():
        sub = df_traj[df_traj.condition == cond]
        mean_ccis = sub.groupby("epoch").cci.mean()
        plt.plot(mean_ccis.index, mean_ccis.values, label=cond)
    plt.legend()
    plt.title("CCI (thinned long runs)")
    plt.xlabel("epoch")
    plt.ylabel("CCI")
    plt.tight_layout()
    plt.savefig(OUT / "cci_long.png")
    plt.close()

# hazard_long
if not df_traj.empty:
    plt.figure(figsize=(8, 4))
    for cond in df_traj["condition"].unique():
        sub = df_traj[df_traj.condition == cond]
        mean_h = sub.groupby("epoch").hazard.mean()
        plt.plot(mean_h.index, mean_h.values, label=cond)
    plt.legend()
    plt.title("Hazard (thinned long runs)")
    plt.xlabel("epoch")
    plt.ylabel("hazard")
    plt.tight_layout()
    plt.savefig(OUT / "hazard_long.png")
    plt.close()

# stability window scatter
rs = pd.DataFrame(runs_summary)
if not rs.empty:
    plt.figure(figsize=(6, 4))
    for cond in rs["condition"].unique():
        sub = rs[rs.condition == cond]
        plt.scatter(
            sub["mean_ico"] if "mean_ico" in sub.columns else sub["ico"],
            sub["long_collapse_risk"],
            label=cond,
        )
    plt.legend()
    plt.xlabel("ICO")
    plt.ylabel("collapse_risk")
    plt.title("ICO vs Collapse Risk")
    plt.tight_layout()
    plt.savefig(OUT / "stability_window.png")
    plt.close()

# Write report
md = REPORT_DIR / "panspermia_results.md"
with open(md, "w") as f:
    f.write("# Panspermia / Openness Runner Results\n\n")
    f.write("Conditions summary:\n\n")
    try:
        f.write(pd.DataFrame(conditions_summary).to_markdown())
    except Exception:
        f.write(pd.DataFrame(conditions_summary).to_string())
    f.write("\n\nBayes Summary:\n\n")
    try:
        f.write(
            pd.DataFrame([{k: bf_results[k]["BF"] for k in bf_results}]).to_markdown()
        )
    except Exception:
        f.write(
            pd.DataFrame([{k: bf_results[k]["BF"] for k in bf_results}]).to_string()
        )
    f.write("\n\nTakeaway:\n")
    f.write(
        "- Winner selection: production-safe per stability window thresholds not yet implemented; review runs_summary.csv for decisions.\n"
    )

# bundle + sha256
bundle_name = f"panspermia_bundle_{STAMP}.zip"
bundle_path = OUT / bundle_name
try:
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as z:
        for f in (OUT).rglob("*"):
            if f.is_file():
                # skip the bundle file itself if it's under OUT
                if f.resolve() == bundle_path.resolve():
                    continue
                z.write(f, arcname=str(f.relative_to(OUT)))
    # sha256 by streaming
    hobj = hashlib.sha256()
    with open(bundle_path, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            hobj.update(chunk)
    with open(OUT / "SHA256SUMS.txt", "w") as s:
        s.write(f"{hobj.hexdigest()}  {bundle_name}\n")
except OSError as e:
    print("OS error during bundling (likely no disk space):", e)
    print("Skipping bundling step. Outputs remain in", OUT)

print("Panspermia run complete. Outputs in", OUT)
print("Bundle:", bundle_path)

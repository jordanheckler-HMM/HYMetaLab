#!/usr/bin/env python3
"""
Temporal Ontology / Time-arrow sweep
"""
import copy
import datetime
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/time_arrow_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Experiment grid (per user request)
agent_counts = [50, 100, 200]
noise_levels = [0.0, 0.1]
shock_rules = ["reversible", "irreversible"]
shock_severities = [0.2, 0.5]
goal_diversities = [3, 4]
field_coherences = [0.6, 0.8]
# replicates: small number to compute means/CIs (assumption: 3 replicates)
replicates = [0, 1, 2]

epochs_cap = 1600
shock_epoch = 1000
pre_window = 50
post_window = 50


# helper: seed from params
def seed_for(x):
    s = json.dumps(x, sort_keys=True)
    return abs(hash(s)) % (2**32)


# wrapper runner
def run_once(
    n_agents, noise, shock_severity, goal_diversity, field_coherence, shock_rule, seed
):
    rng = np.random.RandomState(seed)
    agents = me.initialize_agents(n_agents, goal_diversity, noise, rng)
    # record avg resource initial
    avg_res_start = float(np.mean([a.get("resource", 0.0) for a in agents]))

    history = {"cci": [], "alive_frac": [], "hazard": [], "avg_resource": []}

    for t in range(0, epochs_cap + 1):
        # determine shock for this epoch
        if t == shock_epoch:
            shock_level = shock_severity
        elif t > shock_epoch:
            shock_level = shock_severity * 0.2
        else:
            shock_level = 0.0

        # For reversible rule, if at shock_epoch we should apply then rollback (i.e., robust undo)
        if shock_rule == "reversible" and t == shock_epoch:
            # snapshot
            snap_agents = copy.deepcopy(agents)
            # apply one epoch with shock to see the effect
            me.step_update(agents, shock_level, "acute", rng)
            # record 'would-be' metrics (not used downstream but kept for diagnostics)
            cci_would = me.collective_cci(agents)
            # rollback with slight decay: apply a tiny irreversible loss when restoring
            agents = rollback_state(snap_agents, decay=0.001)
            # after rollback, treat as no-shock epoch for actual metrics
            me.step_update(agents, 0.0, "acute", rng)
        else:
            # normal update
            me.step_update(
                agents,
                shock_level if shock_rule == "irreversible" else 0.0,
                "acute",
                rng,
            )

        # energy efficiency effect (field_coherence): higher coherence reduces net drain
        # implement as small per-agent resource compensation proportional to field_coherence
        for a in agents:
            if a.get("alive", True):
                a["resource"] = min(
                    1.0, a.get("resource", 0.0) + (field_coherence - 0.5) * 0.02
                )

        alive_frac = sum(1 for a in agents if a.get("alive", True)) / float(len(agents))
        hazard = 1.0 - alive_frac
        cci = me.collective_cci(agents)
        history["cci"].append(float(cci))
        history["alive_frac"].append(float(alive_frac))
        history["hazard"].append(float(hazard))
        history["avg_resource"].append(
            float(np.mean([a.get("resource", 0.0) for a in agents]))
        )

    # compute metrics
    # pre and post windows
    def mean_window(arr, start, length):
        s = max(0, start)
        e = min(len(arr), start + length)
        if e <= s:
            return float("nan")
        return float(np.mean(arr[s:e]))

    hazard_pre = mean_window(history["hazard"], shock_epoch - pre_window, pre_window)
    hazard_post = mean_window(history["hazard"], shock_epoch + 1, post_window)
    # asymmetry index
    if abs(hazard_pre) < 1e-12:
        if abs(hazard_post) < 1e-12:
            asymmetry = 0.0
        else:
            asymmetry = float("inf")
    else:
        asymmetry = (hazard_post - hazard_pre) / abs(hazard_pre)

    # recovery time: first epoch > shock_epoch where cci >= 0.95 * cci_pre
    cci_pre = mean_window(history["cci"], shock_epoch - pre_window, pre_window)
    recovery_time = float("inf")
    if not math.isnan(cci_pre) and cci_pre > 0:
        target = 0.95 * cci_pre
        for t in range(shock_epoch + 1, len(history["cci"])):
            if history["cci"][t] >= target:
                recovery_time = t - shock_epoch
                break

    avg_res_end = history["avg_resource"][-1]
    energy_drift = avg_res_end - avg_res_start

    survival_rate = history["alive_frac"][-1]

    return {
        "params": {
            "n_agents": n_agents,
            "noise": noise,
            "shock_severity": shock_severity,
            "goal_diversity": goal_diversity,
            "field_coherence": field_coherence,
            "shock_rule": shock_rule,
            "seed": seed,
        },
        "history": history,
        "survival_rate": survival_rate,
        "recovery_time": recovery_time,
        "energy_drift": energy_drift,
        "asymmetry_index": asymmetry,
        "hazard_pre": hazard_pre,
        "hazard_post": hazard_post,
    }


def rollback_state(agent_snapshot, decay=0.001):
    """Apply a small decay to a snapshot of agents to model imperfect rollback.

    Decay semantics:
    - reduce each agent's resource by a small fraction `decay`
    - slightly attenuate belief vectors toward uniform by mixing with uniform with weight `decay`
    - renormalize beliefs
    """
    import copy

    new_agents = copy.deepcopy(agent_snapshot)
    for a in new_agents:
        # resource loses a fraction of current value
        a["resource"] = max(0.0, a.get("resource", 0.0) * (1.0 - decay))
        # belief attenuation toward uniform
        b = np.array(a.get("belief", []), dtype=float)
        if b.size > 0:
            K = float(b.size)
            uniform = np.ones_like(b) / K
            b = b * (1.0 - decay) + uniform * decay
            b = np.clip(b, 1e-12, None)
            b = b / b.sum()
            a["belief"] = b
    return new_agents


# run grid
all_rows = []
per_config_trajs = []
start_total = time.time()
configs = []
for n in agent_counts:
    for noise in noise_levels:
        for shock_severity in shock_severities:
            for gd in goal_diversities:
                for fc in field_coherences:
                    configs.append((n, noise, shock_severity, gd, fc))

print("Total configurations (without replicates):", len(configs))

for n, noise, shock_severity, gd, fc in configs:
    for rep in replicates:
        seed = seed_for(
            {
                "n": n,
                "noise": noise,
                "shock_severity": shock_severity,
                "gd": gd,
                "fc": fc,
                "rep": rep,
            }
        )
        for shock_rule in ["reversible", "irreversible"]:
            t0 = time.time()
            res = run_once(n, noise, shock_severity, gd, fc, shock_rule, seed)
            t1 = time.time()
            run_id = f"N{n}_noise{noise}_sev{shock_severity}_gd{gd}_fc{fc}_rule{shock_rule}_rep{rep}"
            # save trajectory CSV
            hist = res["history"]
            traj_df = pd.DataFrame(
                {
                    "epoch": list(range(len(hist["cci"]))),
                    "cci": hist["cci"],
                    "alive_frac": hist["alive_frac"],
                    "hazard": hist["hazard"],
                    "avg_resource": hist["avg_resource"],
                }
            )
            traj_path = DATA_DIR / f"{run_id}_traj.csv"
            traj_df.to_csv(traj_path, index=False)
            per_config_trajs.append(str(traj_path))

            # summary row
            row = {
                "run_id": run_id,
                "n_agents": n,
                "noise": noise,
                "shock_severity": shock_severity,
                "goal_diversity": gd,
                "field_coherence": fc,
                "shock_rule": shock_rule,
                "seed": seed,
                "survival_rate": res["survival_rate"],
                "recovery_time": res["recovery_time"],
                "energy_drift": res["energy_drift"],
                "asymmetry_index": res["asymmetry_index"],
                "hazard_pre": res["hazard_pre"],
                "hazard_post": res["hazard_post"],
                "wall_time_sec": t1 - t0,
            }
            all_rows.append(row)
            print("Completed", run_id)

# write master CSV
master_csv = DATA_DIR / "time_arrow_runs.csv"
pd.DataFrame(all_rows).to_csv(master_csv, index=False)

# generate plots: for each unique (n,noise,shock_severity,gd,fc) make overlay of reversible vs irreversible using rep=0 (or mean across reps)
grouped = pd.DataFrame(all_rows).groupby(
    ["n_agents", "noise", "shock_severity", "goal_diversity", "field_coherence"]
)
heat_records = []
for name, group in grouped:
    n, noise, sev, gd, fc = name
    # for plotting overlay, pick rep=0 (or average across reps)
    rev_rows = [
        r
        for r in all_rows
        if r["n_agents"] == n
        and r["noise"] == noise
        and r["shock_severity"] == sev
        and r["goal_diversity"] == gd
        and r["field_coherence"] == fc
        and r["shock_rule"] == "reversible"
    ]
    irr_rows = [
        r
        for r in all_rows
        if r["n_agents"] == n
        and r["noise"] == noise
        and r["shock_severity"] == sev
        and r["goal_diversity"] == gd
        and r["field_coherence"] == fc
        and r["shock_rule"] == "irreversible"
    ]
    # pick first rep for both
    if not rev_rows or not irr_rows:
        continue
    rev_id = rev_rows[0]["run_id"]
    irr_id = irr_rows[0]["run_id"]
    rev_df = pd.read_csv(DATA_DIR / f"{rev_id}_traj.csv")
    irr_df = pd.read_csv(DATA_DIR / f"{irr_id}_traj.csv")
    # CCI plot
    plt.figure(figsize=(8, 4))
    plt.plot(rev_df.epoch, rev_df.cci, label="reversible")
    plt.plot(irr_df.epoch, irr_df.cci, label="irreversible")
    plt.xlabel("epoch")
    plt.ylabel("CCI")
    plt.title(f"N={n} noise={noise} sev={sev} gd={gd} fc={fc}")
    plt.legend()
    plt.tight_layout()
    p1 = FIG_DIR / f"{rev_id}_cci_overlay.png"
    plt.savefig(p1)
    plt.close()
    # hazard plot
    plt.figure(figsize=(8, 4))
    plt.plot(rev_df.epoch, rev_df.hazard, label="reversible")
    plt.plot(irr_df.epoch, irr_df.hazard, label="irreversible")
    plt.xlabel("epoch")
    plt.ylabel("hazard")
    plt.title(f"N={n} noise={noise} sev={sev} gd={gd} fc={fc}")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / f"{rev_id}_hazard_overlay.png"
    plt.savefig(p2)
    plt.close()

    # record asymmetry for heatmap aggregator later
    # take mean asymmetry across reps and shock_rule='irreversible'
    irr_asym = [
        r["asymmetry_index"]
        for r in all_rows
        if r["n_agents"] == n
        and r["noise"] == noise
        and r["shock_severity"] == sev
        and r["goal_diversity"] == gd
        and r["field_coherence"] == fc
        and r["shock_rule"] == "irreversible"
    ]
    mean_asym = (
        float(
            np.nanmean(
                [
                    x
                    for x in irr_asym
                    if not (
                        x is None
                        or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
                    )
                ]
            )
        )
        if irr_asym
        else float("nan")
    )
    heat_records.append(
        {
            "n_agents": n,
            "noise": noise,
            "shock_severity": sev,
            "goal_diversity": gd,
            "field_coherence": fc,
            "mean_asymmetry_irr": mean_asym,
        }
    )

# create asymmetry heatmap aggregated over n_agents,noise,goal_diversity: index=field_coherence x columns=shock_severity
heat_df = pd.DataFrame(heat_records)
if not heat_df.empty:
    hm = heat_df.groupby(["field_coherence", "shock_severity"]).mean().reset_index()
    pivot = hm.pivot(
        index="field_coherence", columns="shock_severity", values="mean_asymmetry_irr"
    )
    plt.figure(figsize=(6, 4))
    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[
            min(pivot.columns),
            max(pivot.columns),
            min(pivot.index),
            max(pivot.index),
        ],
    )
    plt.colorbar(label="mean asymmetry (irreversible)")
    plt.xlabel("shock_severity")
    plt.ylabel("field_coherence")
    plt.title("Asymmetry heatmap (avg across other dims)")
    plt.tight_layout()
    heat_path = FIG_DIR / "asymmetry_heatmap.png"
    plt.savefig(heat_path)
    plt.close()

# JSON summary: for each (n,noise,sev,gd,fc) compute mean and 95% CI of asymmetry_index across reps for irreversible
summary = []
for name, group in grouped:
    n, noise, sev, gd, fc = name
    values = [
        r["asymmetry_index"]
        for r in all_rows
        if r["n_agents"] == n
        and r["noise"] == noise
        and r["shock_severity"] == sev
        and r["goal_diversity"] == gd
        and r["field_coherence"] == fc
        and r["shock_rule"] == "irreversible"
    ]
    # filter inf
    vals = np.array(
        [
            v
            for v in values
            if (v is not None) and (not np.isinf(v)) and (not np.isnan(v))
        ]
    )
    if vals.size == 0:
        mean = None
        ci = [None, None]
    else:
        mean = float(np.mean(vals))
        boots = []
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            idx = rng.choice(len(vals), size=len(vals), replace=True)
            boots.append(np.mean(vals[idx]))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # ensure native Python types
    summary.append(
        {
            "n_agents": int(n),
            "noise": float(noise),
            "shock_severity": float(sev),
            "goal_diversity": int(gd),
            "field_coherence": float(fc),
            "mean_asymmetry": (None if mean is None else float(mean)),
            "asymmetry_CI": [
                None if ci[0] is None else float(ci[0]),
                None if ci[1] is None else float(ci[1]),
            ],
        }
    )

(JSON_DIR / "time_arrow_summary.json").write_text(json.dumps(summary, indent=2))

# write markdown report
md_lines = []
md_lines.append("# Time Arrow Sweep Results\n")
md_lines.append("Fast takeaways:")
# quick heuristic: count fraction of positive asymmetry in irreversible vs reversible
irr_vals = [r["asymmetry_index"] for r in all_rows if r["shock_rule"] == "irreversible"]
rev_vals = [r["asymmetry_index"] for r in all_rows if r["shock_rule"] == "reversible"]
irr_positive = sum(1 for v in irr_vals if v is not None and not np.isnan(v) and (v > 0))
irr_total = len([v for v in irr_vals if v is not None and not np.isnan(v)])
rev_positive = sum(1 for v in rev_vals if v is not None and not np.isnan(v) and (v > 0))
rev_total = len([v for v in rev_vals if v is not None and not np.isnan(v)])
md_lines.append(
    f"- Irreversible runs with positive asymmetry: {irr_positive}/{irr_total}"
)
md_lines.append(
    f"- Reversible runs with positive asymmetry: {rev_positive}/{rev_total}"
)
md_lines.append(
    "\nPASS/FAIL: Stable arrow iff majority irreversible positive and reversible near zero"
)
arrow_present = (irr_positive / irr_total if irr_total > 0 else 0) > 0.5 and (
    rev_positive / rev_total if rev_total > 0 else 0
) < 0.2
md_lines.append(f"- Time arrow detected (heuristic): {bool(arrow_present)}")
md_lines.append("\n## Embedded charts\n")
# attach some sample chart images (first few overlays)
sample_figs = sorted(FIG_DIR.glob("*_cci_overlay.png"))[:6]
for p in sample_figs:
    md_lines.append(f"![]({p.name})")
md_lines.append("\nAsymmetry heatmap:")
if "heat_path" in locals():
    md_lines.append(f"![]({heat_path.name})")
else:
    md_lines.append("No heatmap generated")

(REPORT_DIR / "time_arrow_results.md").write_text("\n".join(md_lines))

print("Done. Outputs in:", OUT_DIR)
print("Master CSV:", master_csv)
print("JSON summary:", JSON_DIR / "time_arrow_summary.json")
print("Report:", REPORT_DIR / "time_arrow_results.md")

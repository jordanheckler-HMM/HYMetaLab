#!/usr/bin/env python3
"""Metaphysics quick scan: Parts A, B, C

Low-cost exploratory runs (epochs=150, replicates=2). Exports CSV/JSON/PNGs/MD.
"""
import copy
import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/metaphysics_quickscan_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Common params
epochs = 150
replicates = [0, 1]
noise = 0.05
shock_epoch = 50


def seed_for(x):
    s = json.dumps(x, sort_keys=True)
    return abs(hash(s)) % (2**32)


def rollback_state(agent_snapshot, decay=0.001):
    new_agents = copy.deepcopy(agent_snapshot)
    for a in new_agents:
        a["resource"] = max(0.0, a.get("resource", 0.0) * (1.0 - decay))
        b = np.array(a.get("belief", []), dtype=float)
        if b.size > 0:
            K = float(b.size)
            uniform = np.ones_like(b) / K
            b = b * (1.0 - decay) + uniform * decay
            b = np.clip(b, 1e-12, None)
            b = b / b.sum()
            a["belief"] = b
    return new_agents


def run_run(agents, shock_level, rule, decay, rng):
    history = {"avg_resource": [], "cci": []}
    agents_local = copy.deepcopy(agents)
    for t in range(epochs):
        if t == shock_epoch:
            level = shock_level
        elif t > shock_epoch:
            level = shock_level * 0.2
        else:
            level = 0.0
        if rule == "reversible" and t == shock_epoch:
            snap = copy.deepcopy(agents_local)
            me.step_update(agents_local, level, "acute", rng)
            agents_local = rollback_state(snap, decay=decay)
            me.step_update(agents_local, 0.0, "acute", rng)
        else:
            me.step_update(
                agents_local, level if rule == "irreversible" else 0.0, "acute", rng
            )
        history["avg_resource"].append(
            float(np.mean([a.get("resource", 0.0) for a in agents_local]))
        )
        history["cci"].append(float(me.collective_cci(agents_local)))
    return history


def entropy_rate(series):
    x = np.arange(len(series))
    if len(x) < 2:
        return 0.0
    m, _ = np.polyfit(x, np.array(series), 1)
    return float(m)


####################
# Part A: Arrow Emergence
####################
partA_rows = []
field_coherences = [0.6, 0.7, 0.8]
shock_severities = [0.2, 0.3]
decays = [0.0, 0.001]
for fc in field_coherences:
    for shock in shock_severities:
        for decay in decays:
            for rep in replicates:
                seed = seed_for(
                    {"part": "A", "fc": fc, "shock": shock, "decay": decay, "rep": rep}
                )
                rng = np.random.RandomState(seed)
                agents = me.initialize_agents(100, 3, noise, rng)
                for a in agents:
                    a["resource"] = min(1.0, a.get("resource", 0.0) + (fc - 0.5) * 0.02)
                irr = run_run(agents, shock, "irreversible", decay=0.0, rng=rng)
                rng2 = np.random.RandomState(seed + 1)
                rev = run_run(agents, shock, "reversible", decay=decay, rng=rng2)
                entropy_irr = entropy_rate(irr["avg_resource"])
                entropy_rev = entropy_rate(rev["avg_resource"])
                entropy_diff = entropy_irr - entropy_rev
                energy_drift = float(
                    np.mean(irr["avg_resource"][-10:]) - irr["avg_resource"][0]
                )
                cci_stab = float(
                    np.mean(irr["cci"][-10:]) / (np.std(irr["cci"][-10:]) + 1e-12)
                )
                direction_index = 1 if entropy_diff > 0 else 0
                partA_rows.append(
                    {
                        "fc": fc,
                        "shock": shock,
                        "decay": decay,
                        "replicate": rep,
                        "entropy_rate_irrev": entropy_irr,
                        "entropy_rate_rev": entropy_rev,
                        "entropy_rate_diff": entropy_diff,
                        "energy_drift": energy_drift,
                        "CCI_stability": cci_stab,
                        "direction_index": direction_index,
                    }
                )

partA_df = pd.DataFrame(partA_rows)
partA_df.to_csv(DATA_DIR / "partA_runs.csv", index=False)
partA_summary = (
    partA_df.groupby(["fc", "shock", "decay"])
    .agg(
        {
            "entropy_rate_diff": ["mean", "std"],
            "energy_drift": ["mean", "std"],
            "CCI_stability": ["mean", "std"],
            "direction_index": "mean",
        }
    )
    .reset_index()
)
partA_summary.columns = [
    "fc",
    "shock",
    "decay",
    "entropy_rate_diff_mean",
    "entropy_rate_diff_std",
    "energy_drift_mean",
    "energy_drift_std",
    "CCI_stability_mean",
    "CCI_stability_std",
    "direction_index_mean",
]
partA_summary.to_json(JSON_DIR / "partA_summary.json", orient="records", indent=2)

plt.figure()
pivot = partA_summary.pivot_table(
    index="fc", columns="shock", values="direction_index_mean"
)
plt.imshow(
    pivot.values,
    origin="lower",
    aspect="auto",
    extent=[min(pivot.columns), max(pivot.columns), min(pivot.index), max(pivot.index)],
)
plt.colorbar(label="direction index mean")
plt.xlabel("shock")
plt.ylabel("fc")
plt.title("Arrow presence (Part A)")
plt.savefig(FIG_DIR / "arrow_heatmap.png")
plt.close()

####################
# Part B: High-CCI Agents Self-Direction
####################
partB_rows = []
for rep in replicates:
    seed = seed_for({"part": "B", "rep": rep})
    rng = np.random.RandomState(seed)
    agents = []
    K = 3
    for i in range(50):
        b = rng.rand(K)
        b = b / b.sum()
        agents.append(
            {
                "id": i,
                "belief": b,
                "optimism": 0.5,
                "resilience": 0.5,
                "alive": True,
                "resource": 1.0,
            }
        )
    for i in range(50, 100):
        b = np.zeros(K) + 1e-6
        b[0] = 1.0 - 1e-6
        agents.append(
            {
                "id": i,
                "belief": b,
                "optimism": 0.9,
                "resilience": 0.9,
                "alive": True,
                "resource": 1.0,
            }
        )
    for a in agents:
        a["resource"] = min(1.0, a.get("resource", 0.0) + (0.8 - 0.5) * 0.02)
    irr_low, irr_high = None, None

    def run_with_subgroups(agents_full, shock, rule, decay, rng):
        agents_local = copy.deepcopy(agents_full)
        hist_low = {"avg_resource": [], "cci": []}
        hist_high = {"avg_resource": [], "cci": []}
        for t in range(epochs):
            if t == shock_epoch:
                level = shock
            elif t > shock_epoch:
                level = shock * 0.2
            else:
                level = 0.0
            if rule == "reversible" and t == shock_epoch:
                snap = copy.deepcopy(agents_local)
                me.step_update(agents_local, level, "acute", rng)
                agents_local = rollback_state(snap, decay=decay)
                me.step_update(agents_local, 0.0, "acute", rng)
            else:
                me.step_update(
                    agents_local, level if rule == "irreversible" else 0.0, "acute", rng
                )
            low_agents = [a for a in agents_local if a["id"] < 50]
            high_agents = [a for a in agents_local if a["id"] >= 50]
            hist_low["avg_resource"].append(
                float(np.mean([a.get("resource", 0.0) for a in low_agents]))
            )
            hist_low["cci"].append(float(me.collective_cci(low_agents)))
            hist_high["avg_resource"].append(
                float(np.mean([a.get("resource", 0.0) for a in high_agents]))
            )
            hist_high["cci"].append(float(me.collective_cci(high_agents)))
        return hist_low, hist_high

    hlow_irr, hhigh_irr = run_with_subgroups(agents, 0.3, "irreversible", 0.0, rng)
    hlow_rev, hhigh_rev = run_with_subgroups(
        agents, 0.3, "reversible", 0.001, np.random.RandomState(seed + 1)
    )
    ent_low = entropy_rate(hlow_irr["avg_resource"]) - entropy_rate(
        hlow_rev["avg_resource"]
    )
    ent_high = entropy_rate(hhigh_irr["avg_resource"]) - entropy_rate(
        hhigh_rev["avg_resource"]
    )
    dir_low = 1 if ent_low > 0 else 0
    dir_high = 1 if ent_high > 0 else 0
    partB_rows.append(
        {
            "replicate": rep,
            "entropy_diff_low": ent_low,
            "entropy_diff_high": ent_high,
            "direction_low": dir_low,
            "direction_high": dir_high,
        }
    )

partB_df = pd.DataFrame(partB_rows)
partB_df.to_csv(DATA_DIR / "partB_runs.csv", index=False)

plt.figure()
plt.bar(
    [0, 1],
    [partB_df["entropy_diff_low"].mean(), partB_df["entropy_diff_high"].mean()],
    yerr=[partB_df["entropy_diff_low"].std(), partB_df["entropy_diff_high"].std()],
    tick_label=["lowCCI", "highCCI"],
)
plt.ylabel("entropy_rate_diff (irr - rev)")
plt.title("High-CCI vs Low-CCI (Part B)")
plt.savefig(FIG_DIR / "highCCI_vs_lowCCI.png")
plt.close()

####################
# Part C: Meaning Systems under Reversible Time
####################
partC_rows = []
from collections import Counter


def seed_agents_meaning(n_agents, rng, archetypes=["religion", "science", "art"]):
    agents = []
    K = len(archetypes)
    for i in range(n_agents):
        b = rng.rand(K)
        b = b / b.sum()
        agents.append(
            {
                "id": i,
                "belief": b,
                "optimism": 0.5,
                "resilience": 0.5,
                "alive": True,
                "resource": 1.0,
            }
        )
    return agents


for mode in ["normal_time", "reversible_time"]:
    for rep in replicates:
        seed = seed_for({"part": "C", "mode": mode, "rep": rep})
        rng = np.random.RandomState(seed)
        agents = seed_agents_meaning(150, rng)
        for a in agents:
            a["resource"] = min(1.0, a.get("resource", 0.0) + (0.8 - 0.5) * 0.02)
        hist_fraction = {k: [] for k in ["religion", "science", "art"]}
        for t in range(epochs):
            if t == shock_epoch:
                level = 0.3
            elif t > shock_epoch:
                level = 0.3 * 0.2
            else:
                level = 0.0
            if mode == "reversible_time" and t == shock_epoch:
                snap = copy.deepcopy(agents)
                me.step_update(agents, level, "acute", rng)
                agents = rollback_state(snap, decay=0.001)
                me.step_update(agents, 0.0, "acute", rng)
            else:
                me.step_update(
                    agents, level if mode == "normal_time" else 0.0, "acute", rng
                )
            choices = [int(np.argmax(a["belief"])) for a in agents if a["alive"]]
            cnt = Counter(choices)
            total = sum(cnt.values()) if cnt else 1
            for idx, name in enumerate(["religion", "science", "art"]):
                frac = float(cnt.get(idx, 0)) / total
                hist_fraction[name].append(frac)
        for t in range(epochs):
            partC_rows.append(
                {
                    "mode": mode,
                    "replicate": rep,
                    "epoch": t,
                    "religion_frac": hist_fraction["religion"][t],
                    "science_frac": hist_fraction["science"][t],
                    "art_frac": hist_fraction["art"][t],
                    "survival_rate": sum(1 for a in agents if a["alive"]) / len(agents),
                }
            )

partC_df = pd.DataFrame(partC_rows)
partC_df.to_csv(DATA_DIR / "partC_runs.csv", index=False)

partC_summary = (
    partC_df.groupby(["mode", "epoch"])
    .agg({"religion_frac": "mean", "science_frac": "mean", "art_frac": "mean"})
    .reset_index()
)
for mode in ["normal_time", "reversible_time"]:
    sub = partC_summary[partC_summary["mode"] == mode]
    plt.plot(sub.epoch, sub.religion_frac, label="religion")
    plt.plot(sub.epoch, sub.science_frac, label="science")
    plt.plot(sub.epoch, sub.art_frac, label="art")
    plt.title(f"Meaning fractions over time ({mode})")
    plt.xlabel("epoch")
    plt.ylabel("fraction")
    plt.legend()
    plt.savefig(FIG_DIR / f"meaning_fraction_{mode}.png")
    plt.clf()

plt.figure()
for mode in ["normal_time", "reversible_time"]:
    sub = partC_summary[partC_summary["mode"] == mode]
    plt.plot(sub.epoch, sub.religion_frac, label=f"religion ({mode})")
plt.plot(
    partC_summary[partC_summary["mode"] == "normal_time"].epoch,
    partC_summary[partC_summary["mode"] == "normal_time"].science_frac,
    label="science (normal)",
)
plt.title("Religion fraction (reversible vs normal)")
plt.xlabel("epoch")
plt.ylabel("fraction")
plt.legend()
plt.savefig(FIG_DIR / "meaning_fraction_plot.png")
plt.close()

####################
# Build report
####################
md = []
md.append("# Metaphysics Quickscan Results\n")
md.append(f"Date: {STAMP}\n")
md.append("## Part A: Arrow Emergence (quick)")
md.append("Arrow heatmap: figures/arrow_heatmap.png")
try:
    md.append(partA_summary.to_markdown())
except Exception:
    md.append(partA_summary.to_string(index=False))
md.append("\n## Part B: High-CCI vs Low-CCI")
try:
    md.append(partB_df.to_markdown())
except Exception:
    md.append(partB_df.to_string(index=False))
md.append("\nFigure: figures/highCCI_vs_lowCCI.png")
md.append("\n## Part C: Meaning systems under reversible vs normal time")
md.append(
    "Figures: figures/meaning_fraction_normal_time.png, figures/meaning_fraction_reversible_time.png, figures/meaning_fraction_plot.png"
)

passA = (partA_summary["direction_index_mean"] > 0).any()
passB = partB_df["direction_high"].mean() > partB_df["direction_low"].mean()
late = partC_summary[partC_summary["epoch"] == epochs - 1]
rev_rel = (
    float(late[late["mode"] == "reversible_time"].religion_frac.values[0])
    if not late[late["mode"] == "reversible_time"].empty
    else 0.0
)
norm_rel = (
    float(late[late["mode"] == "normal_time"].religion_frac.values[0])
    if not late[late["mode"] == "normal_time"].empty
    else 0.0
)
passC = rev_rel > norm_rel + 0.02

md.append("\n## Quick heuristics")
md.append(f"- Part A (arrow signal anywhere): {bool(passA)}")
md.append(f"- Part B (high-CCI shows stronger direction than low-CCI): {bool(passB)}")
md.append(f"- Part C (religion spikes under reversible > normal): {bool(passC)}")

(REPORT_DIR / "metaphysics_quickscan_results.md").write_text("\n".join(md))

print("Done. Outputs in:", OUT_DIR)
print("PartA CSV:", DATA_DIR / "partA_runs.csv")
print("PartB CSV:", DATA_DIR / "partB_runs.csv")
print("PartC CSV:", DATA_DIR / "partC_runs.csv")

#!/usr/bin/env python3
"""Entropy Ratchet test

Runs paired irreversible vs reversible_decay experiments and computes
entropy_rate = d(mean_energy_drift)/d(epoch) for each config.

Outputs:
 - discovery_results/entropy_ratchet_<STAMP>/data/*.csv
 - figures: entropy_vs_time.png, drift_surface.png
 - report: entropy_ratchet_results.md
"""
import copy
import datetime
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/entropy_ratchet_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Experiment grid (from user)
field_coherences = [0.5, 0.6, 0.7, 0.8]
shock_severities = [0.3, 0.5]
decays = [0.0005, 0.001, 0.002, 0.005]

# other params
n_agents = 100
noise = 0.05
replicates = [0, 1, 2]
run_length = 3000
shock_epoch = 1000


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


def run_single_config(
    n_agents, noise, shock_severity, field_coherence, shock_rule, decay, seed
):
    rng = np.random.RandomState(seed)
    agents = me.initialize_agents(n_agents, 3, noise, rng)
    avg_res_start = float(np.mean([a.get("resource", 0.0) for a in agents]))
    history = {"cci": [], "alive_frac": [], "hazard": [], "avg_resource": []}
    for t in range(0, run_length + 1):
        if t == shock_epoch:
            shock_level = shock_severity
        elif t > shock_epoch:
            shock_level = shock_severity * 0.2
        else:
            shock_level = 0.0

        if shock_rule == "reversible_decay" and t == shock_epoch:
            snap_agents = copy.deepcopy(agents)
            me.step_update(agents, shock_level, "acute", rng)
            # rollback with decay
            agents = rollback_state(snap_agents, decay=decay)
            me.step_update(agents, 0.0, "acute", rng)
        else:
            me.step_update(
                agents,
                shock_level if shock_rule == "irreversible" else 0.0,
                "acute",
                rng,
            )

        # field coherence boosts resource slightly
        for a in agents:
            if a.get("alive", True):
                a["resource"] = min(
                    1.0, a.get("resource", 0.0) + (field_coherence - 0.5) * 0.01
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

    # compute energy drift series relative to start
    avg_res_arr = np.array(history["avg_resource"], dtype=float)
    energy_drift_series = (avg_res_arr - avg_res_arr[0]) / (
        abs(avg_res_arr[0]) if abs(avg_res_arr[0]) > 1e-12 else 1.0
    )

    return {
        "history": history,
        "avg_res_start": float(avg_res_arr[0]),
        "avg_res_arr": avg_res_arr,
        "energy_drift_series": energy_drift_series,
    }


# Run grid
rows = []
traj_records = []
entropy_rates = []
paired = {}
start = time.time()
total_runs = (
    len(shock_severities) * len(field_coherences) * len(decays) * len(replicates)
)
count = 0
for shock in shock_severities:
    for fc in field_coherences:
        for decay in decays:
            for rep in replicates:
                count += 1
                seed = seed_for(
                    {
                        "n": n_agents,
                        "noise": noise,
                        "shock": shock,
                        "fc": fc,
                        "decay": decay,
                        "rep": rep,
                    }
                )
                # irreversible
                irr = run_single_config(
                    n_agents, noise, shock, fc, "irreversible", decay=0.0, seed=seed
                )
                # reversible with decay
                rev = run_single_config(
                    n_agents,
                    noise,
                    shock,
                    fc,
                    "reversible_decay",
                    decay=decay,
                    seed=seed,
                )

                # store per-epoch trajectory rows
                run_id_base = (
                    f"N{n_agents}_noise{noise}_sev{shock}_fc{fc}_decay{decay}_rep{rep}"
                )
                for epoch in range(len(irr["avg_res_arr"])):
                    traj_records.append(
                        {
                            "run_id": run_id_base,
                            "rule": "irreversible",
                            "shock_severity": shock,
                            "field_coherence": fc,
                            "decay": decay,
                            "replicate": rep,
                            "epoch": epoch,
                            "avg_resource": float(irr["avg_res_arr"][epoch]),
                            "energy_drift": float(irr["energy_drift_series"][epoch]),
                        }
                    )
                    traj_records.append(
                        {
                            "run_id": run_id_base,
                            "rule": "reversible_decay",
                            "shock_severity": shock,
                            "field_coherence": fc,
                            "decay": decay,
                            "replicate": rep,
                            "epoch": epoch,
                            "avg_resource": float(rev["avg_res_arr"][epoch]),
                            "energy_drift": float(rev["energy_drift_series"][epoch]),
                        }
                    )

                # compute mean energy drift across the run for this replicate (time series already)
                # For each rule compute slope of mean energy drift vs epoch (entropy_rate)
                # We'll compute later aggregated across replicates
                paired.setdefault((shock, fc, decay), []).append(
                    (irr["energy_drift_series"], rev["energy_drift_series"])
                )
                print(f"Completed [{count}/{total_runs}] {run_id_base}")

# write trajectory master CSV
traj_df = pd.DataFrame(traj_records)
traj_path = DATA_DIR / "entropy_ratchet_trajectories.csv"
traj_df.to_csv(traj_path, index=False)

# Aggregate across replicates: compute mean energy_drift per epoch for each (shock,fc,decay,rule)
summary_rows = []
for key, pairs in paired.items():
    shock, fc, decay = key
    # pairs is list of (irr_series, rev_series) per replicate
    irr_arrs = [p[0] for p in pairs]
    rev_arrs = [p[1] for p in pairs]
    # stack and mean across replicates (axis=0 => replicate, axis=1 => epoch)
    irr_stack = np.vstack(irr_arrs)
    rev_stack = np.vstack(rev_arrs)
    mean_irr = np.mean(irr_stack, axis=0)
    mean_rev = np.mean(rev_stack, axis=0)

    # compute entropy_rate as slope of linear fit of mean energy drift vs epoch
    epochs = np.arange(len(mean_irr))
    # fit slope for irreversible
    if len(epochs) >= 2:
        slope_irr, intercept = np.polyfit(epochs, mean_irr, 1)
        slope_rev, intercept2 = np.polyfit(epochs, mean_rev, 1)
    else:
        slope_irr = 0.0
        slope_rev = 0.0

    # store summary
    summary_rows.append(
        {
            "shock_severity": float(shock),
            "field_coherence": float(fc),
            "decay": float(decay),
            "entropy_rate_irreversible": float(slope_irr),
            "entropy_rate_reversible": float(slope_rev),
            "entropy_rate_diff": float(slope_irr - slope_rev),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_path = DATA_DIR / "entropy_ratchet_summary.csv"
summary_df.to_csv(summary_path, index=False)

# Plot: entropy_vs_time.png -> grid of subplots (rows=field_coherence, cols=shock_severity)
fig, axes = plt.subplots(
    len(field_coherences),
    len(shock_severities),
    figsize=(6 * len(shock_severities), 3 * len(field_coherences)),
    squeeze=False,
)
for i, fc in enumerate(field_coherences):
    for j, shock in enumerate(shock_severities):
        ax = axes[i][j]
        # find matching keys and plot
        for decay in decays:
            key = (shock, fc, decay)
            if key not in paired:
                continue
            irr_arrs = [p[0] for p in paired[key]]
            rev_arrs = [p[1] for p in paired[key]]
            mean_irr = np.mean(np.vstack(irr_arrs), axis=0)
            mean_rev = np.mean(np.vstack(rev_arrs), axis=0)
            ax.plot(mean_irr, label="irr", color="C0", alpha=0.9)
            ax.plot(mean_rev, label=f"rev d={decay}", linestyle="--", alpha=0.6)
        ax.axvline(shock_epoch, color="k", linestyle=":", linewidth=0.8)
        ax.set_title(f"fc={fc} shock={shock}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean energy drift")
        if i == 0 and j == 0:
            ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "entropy_vs_time.png")
plt.close()

# drift_surface.png -> heatmap of entropy_rate_diff (irreversible - reversible) with x=decay, y=field_coherence, one panel per shock
fig2, axes2 = plt.subplots(
    1, len(shock_severities), figsize=(5 * len(shock_severities), 4)
)
for k, shock in enumerate(shock_severities):
    ax = axes2[k] if len(shock_severities) > 1 else axes2
    sub = summary_df[summary_df.shock_severity == shock]
    if sub.empty:
        continue
    pivot = sub.pivot(
        index="field_coherence", columns="decay", values="entropy_rate_diff"
    )
    # sort
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            min(pivot.columns),
            max(pivot.columns),
            min(pivot.index),
            max(pivot.index),
        ],
    )
    ax.set_title(f"shock={shock} (entropy_rate diff)")
    ax.set_xlabel("decay")
    ax.set_ylabel("field_coherence")
    fig2.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(FIG_DIR / "drift_surface.png")
plt.close()

# Markdown report
md_lines = []
md_lines.append("# Entropy Ratchet Results\n")
md_lines.append(f"Date: {STAMP}\n")
md_lines.append(
    "Objective: detect whether cumulative energy drift under micro-decay creates an emergent time arrow.\n"
)
md_lines.append(
    'Success criterion: entropy_rate > 0 in irreversible but ≈ 0 in reversible_decay -> flag "arrow emergence via dissipation".\n'
)

flags = []
for idx, row in summary_df.iterrows():
    irr = row["entropy_rate_irreversible"]
    rev = row["entropy_rate_reversible"]
    diff = row["entropy_rate_diff"]
    flag = False
    # criterion: irreversible positive and reversible near zero (relative tolerance)
    if irr > 0 and abs(rev) < max(1e-8, 1e-3 * abs(irr)):
        flag = True
        flags.append((row["shock_severity"], row["field_coherence"], row["decay"]))
    md_lines.append(
        f"- shock={row['shock_severity']} fc={row['field_coherence']} decay={row['decay']}: entropy_rate_irrev={irr:.3e}, entropy_rate_rev={rev:.3e}, diff={diff:.3e}, flag={flag}"
    )

md_lines.append("\n## Figures")
md_lines.append(f"![]({FIG_DIR.name}/entropy_vs_time.png)")
md_lines.append(f"![]({FIG_DIR.name}/drift_surface.png)")

if flags:
    md_lines.append("\n## Conclusion: Arrow detected")
    md_lines.append(f"Configs passing criterion (shock, fc, decay): {flags}")
    md_lines.append('\nFlag: "arrow emergence via dissipation"')
else:
    md_lines.append(
        "\n## Conclusion: No clear arrow detected under the tested grid and thresholds"
    )

(REPORT_DIR / "entropy_ratchet_results.md").write_text("\n".join(md_lines))

print("Done. Outputs:")
print(" - Trajectories CSV:", traj_path)
print(" - Summary CSV:", summary_path)
print(" - Figures:", FIG_DIR)
print(" - Report:", REPORT_DIR / "entropy_ratchet_results.md")

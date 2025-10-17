#!/usr/bin/env python3
"""Time Arrow Reversal Threshold experiment

Runs grid over shocks and field_coherence to find fc_crit where direction_index flips.

Assumptions made:
- replicates=3 (not specified). Increase if you want tighter estimates.
- CCI_stability defined as mean(CCI over last window) / (std(CCI over last window)+1e-12).
  Higher means more stable (low variance relative to mean).
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
OUT_DIR = Path(f"./discovery_results/time_arrow_threshold_{STAMP}")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
JSON_DIR = OUT_DIR / "json"
for d in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Parameters per user
shock_severities = [0.2, 0.3, 0.4, 0.5]
field_coherences = list(np.linspace(0.5, 0.9, 9))
decays = [0.001]
modes = ["reversible_decay", "irreversible"]
run_length = 5000

# Other choices
n_agents = 100
noise = 0.05
replicates = [0, 1, 2]
shock_epoch = 1000
cci_post_window = 1000  # epochs at end to evaluate CCI stability


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
    history = {"cci": [], "avg_resource": []}
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
            agents = rollback_state(snap_agents, decay=decay)
            me.step_update(agents, 0.0, "acute", rng)
        else:
            me.step_update(
                agents,
                shock_level if shock_rule == "irreversible" else 0.0,
                "acute",
                rng,
            )

        # field coherence effect on resources
        for a in agents:
            if a.get("alive", True):
                a["resource"] = min(
                    1.0, a.get("resource", 0.0) + (field_coherence - 0.5) * 0.01
                )

        history["avg_resource"].append(
            float(np.mean([a.get("resource", 0.0) for a in agents]))
        )
        history["cci"].append(float(me.collective_cci(agents)))

    avg_res_arr = np.array(history["avg_resource"], dtype=float)
    energy_drift_series = (avg_res_arr - avg_res_arr[0]) / (
        abs(avg_res_arr[0]) if abs(avg_res_arr[0]) > 1e-12 else 1.0
    )

    return {
        "history": history,
        "avg_res_arr": avg_res_arr,
        "energy_drift_series": energy_drift_series,
    }


# Run grid
traj_rows = []
summary = []
paired = {}
count = 0
total = len(shock_severities) * len(field_coherences) * len(decays) * len(replicates)
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
                irr = run_single_config(
                    n_agents, noise, shock, fc, "irreversible", decay=0.0, seed=seed
                )
                rev = run_single_config(
                    n_agents,
                    noise,
                    shock,
                    fc,
                    "reversible_decay",
                    decay=decay,
                    seed=seed,
                )

                run_id = (
                    f"N{n_agents}_noise{noise}_sev{shock}_fc{fc}_decay{decay}_rep{rep}"
                )
                # save per-epoch trajectories
                for epoch in range(len(irr["energy_drift_series"])):
                    traj_rows.append(
                        {
                            "run_id": run_id,
                            "shock_severity": shock,
                            "field_coherence": fc,
                            "decay": decay,
                            "replicate": rep,
                            "epoch": epoch,
                            "rule": "irreversible",
                            "energy_drift": float(irr["energy_drift_series"][epoch]),
                            "avg_resource": float(irr["avg_res_arr"][epoch]),
                            "cci": float(irr["history"]["cci"][epoch]),
                        }
                    )
                    traj_rows.append(
                        {
                            "run_id": run_id,
                            "shock_severity": shock,
                            "field_coherence": fc,
                            "decay": decay,
                            "replicate": rep,
                            "epoch": epoch,
                            "rule": "reversible_decay",
                            "energy_drift": float(rev["energy_drift_series"][epoch]),
                            "avg_resource": float(rev["avg_res_arr"][epoch]),
                            "cci": float(rev["history"]["cci"][epoch]),
                        }
                    )

                paired.setdefault((shock, fc, decay), []).append(
                    (
                        irr["energy_drift_series"],
                        rev["energy_drift_series"],
                        irr["history"]["cci"],
                        rev["history"]["cci"],
                    )
                )
                print(f"Completed [{count}/{total}] {run_id}")

# write trajectories
traj_df = pd.DataFrame(traj_rows)
traj_path = DATA_DIR / "time_arrow_threshold_trajectories.csv"
traj_df.to_csv(traj_path, index=False)

# Aggregate per-config across replicates
summary_rows = []
for key, items in paired.items():
    shock, fc, decay = key
    # items: list of tuples (irr_series, rev_series, irr_cci, rev_cci)
    irr_arrs = np.vstack([it[0] for it in items])
    rev_arrs = np.vstack([it[1] for it in items])
    irr_ccis = np.vstack([it[2] for it in items])
    rev_ccis = np.vstack([it[3] for it in items])

    mean_irr = np.mean(irr_arrs, axis=0)
    mean_rev = np.mean(rev_arrs, axis=0)

    epochs = np.arange(len(mean_irr))
    if len(epochs) >= 2:
        slope_irr, _ = np.polyfit(epochs, mean_irr, 1)
        slope_rev, _ = np.polyfit(epochs, mean_rev, 1)
    else:
        slope_irr = 0.0
        slope_rev = 0.0

    # energy_drift metric: mean of last 200 epochs
    end_slice = max(1, int(len(mean_irr) * 0.04))
    energy_drift_irr = float(np.mean(mean_irr[-end_slice:]))
    energy_drift_rev = float(np.mean(mean_rev[-end_slice:]))

    # CCI stability: mean_post / (std_post + eps)
    last_window = cci_post_window if cci_post_window < mean_irr.size else mean_irr.size
    mean_cci_irr = float(np.mean(irr_ccis[:, -last_window:], axis=1).mean())
    std_cci_irr = float(np.std(irr_ccis[:, -last_window:], axis=1).mean())
    cci_stability_irr = float(mean_cci_irr / (std_cci_irr + 1e-12))

    mean_cci_rev = float(np.mean(rev_ccis[:, -last_window:], axis=1).mean())
    std_cci_rev = float(np.std(rev_ccis[:, -last_window:], axis=1).mean())
    cci_stability_rev = float(mean_cci_rev / (std_cci_rev + 1e-12))

    direction_index = 1 if (slope_irr - slope_rev) > 0 else 0

    summary_rows.append(
        {
            "shock_severity": float(shock),
            "field_coherence": float(fc),
            "decay": float(decay),
            "entropy_rate_irreversible": float(slope_irr),
            "entropy_rate_reversible": float(slope_rev),
            "entropy_rate_diff": float(slope_irr - slope_rev),
            "energy_drift_irreversible": energy_drift_irr,
            "energy_drift_reversible": energy_drift_rev,
            "CCI_stability_irreversible": cci_stability_irr,
            "CCI_stability_reversible": cci_stability_rev,
            "direction_index": int(direction_index),
        }
    )

# write outputs
summary_df = pd.DataFrame(summary_rows)
summary_csv = DATA_DIR / "time_arrow_threshold_summary.csv"
summary_df.to_csv(summary_csv, index=False)
summary_json = JSON_DIR / "time_arrow_threshold_summary.json"
summary_json.write_text(json.dumps(summary_rows, indent=2))

# Plot: entropy_rate_surface for each shock
fig, axes = plt.subplots(
    1, len(shock_severities), figsize=(5 * len(shock_severities), 4)
)
for i, shock in enumerate(shock_severities):
    ax = axes[i] if len(shock_severities) > 1 else axes
    sub = summary_df[summary_df.shock_severity == shock]
    pivot = sub.pivot(
        index="field_coherence", columns="decay", values="entropy_rate_diff"
    )
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
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(FIG_DIR / "entropy_rate_surface.png")
plt.close()

# direction_map: grid plot of direction_index as function of fc (y) and decay (x) per shock
fig2, axes2 = plt.subplots(
    1, len(shock_severities), figsize=(5 * len(shock_severities), 4)
)
for i, shock in enumerate(shock_severities):
    ax = axes2[i] if len(shock_severities) > 1 else axes2
    sub = summary_df[summary_df.shock_severity == shock]
    pivot = sub.pivot(
        index="field_coherence", columns="decay", values="direction_index"
    )
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
        cmap="RdYlBu",
    )
    ax.set_title(f"shock={shock} (direction index)")
    ax.set_xlabel("decay")
    ax.set_ylabel("field_coherence")
    fig2.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(FIG_DIR / "direction_map.png")
plt.close()

# Markdown summary
md = []
md.append("# Time Arrow Reversal Threshold Results\n")
md.append(f"Date: {STAMP}\n")
md.append(
    "Parameters: shocks=%s, fc grid=%s, decay=%s, run_length=%d, replicates=%d\n"
    % (shock_severities, field_coherences, decays, run_length, len(replicates))
)
for r in summary_rows:
    md.append(
        f"- shock={r['shock_severity']} fc={r['field_coherence']} decay={r['decay']}: entropy_rate_diff={r['entropy_rate_diff']:.3e}, direction_index={r['direction_index']}, energy_drift_irrev={r['energy_drift_irreversible']:.3e}, CCI_stability_irrev={r['CCI_stability_irreversible']:.3e}"
    )

# find fc_crit per shock: lowest fc where direction_index ==1 and previous fc==0
fc_crits = {}
for shock in shock_severities:
    sub = summary_df[summary_df.shock_severity == shock].sort_values("field_coherence")
    prev = 0
    fc_crit = None
    for idx, row in sub.iterrows():
        cur = int(row["direction_index"])
        if prev == 0 and cur == 1:
            fc_crit = float(row["field_coherence"])
            break
        prev = cur
    fc_crits[shock] = fc_crit

md.append("\n## fc_crit per shock")
for shock, fc_crit in fc_crits.items():
    md.append(f"- shock={shock} fc_crit={fc_crit}")

ok = all(
    (fc_crits[s] is not None and 0.77 <= fc_crits[s] <= 0.83) for s in shock_severities
)
if ok:
    md.append(
        "\nConclusion: Coherence-Induced Arrow law observed (fc_crit ≈ 0.77–0.83 across shocks)"
    )
else:
    md.append(
        "\nConclusion: No universal fc_crit in 0.77–0.83 across shocks; see fc_crit per shock above"
    )

md.append("\n## Figures")
md.append(f"![]({FIG_DIR.name}/entropy_rate_surface.png)")
md.append(f"![]({FIG_DIR.name}/direction_map.png)")
(REPORT_DIR / "time_arrow_threshold_results.md").write_text("\n".join(md))

print("Done. Outputs:")
print(" - Trajectories CSV:", traj_path)
print(" - Summary CSV:", summary_csv)
print(" - JSON summary:", summary_json)
print(" - Figures:", FIG_DIR)
print(" - Report:", REPORT_DIR / "time_arrow_threshold_results.md")

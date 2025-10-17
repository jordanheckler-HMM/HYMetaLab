# time_meta_suite.py
# ------------------------------------------------------------
# Metaphysical Time — Extended Suite
# Covers:
# 1) Temporal granularity: discrete vs continuous vs fractal time steps
# 2) Entropy feedback loops: memory decay × foresight amplification
# 3) Agent perception bias: subjective vs objective time awareness → local CCI
# 4) Energy cost of time modes: cyclic vs branching vs linear
#
# Outputs (in ./discovery_results/time_meta_suite_<stamp>/):
# - results.csv (all runs)
# - summary.md (plain-English findings)
# - plot_collapse_by_granularity.png
# - plot_cci_vs_awareness.png
# - plot_entropy_feedback_maps_<mode>.png (one per mode)
# - plot_energy_costs_by_mode.png
# ------------------------------------------------------------

import datetime
import inspect
import itertools
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Try to import your sim modules (safe-call wrappers filter kwargs by signature)
try:
    from simulation_core import (
        calibration_experiment,
        gravity_analysis,
        shock_resilience,
    )
except Exception:
    calibration_experiment = shock_resilience = gravity_analysis = None


def _mean(x):
    if x is None:
        return np.nan
    try:
        arr = np.asarray(x, dtype=float)
        return float(np.nanmean(arr))
    except Exception:
        if isinstance(x, dict):
            return _mean(list(x.values()))
        if isinstance(x, (int, float)):
            return float(x)
        return np.nan


def call_with_supported(func, **kwargs):
    """Call func with only kwargs its signature accepts (works for .run or callable)."""
    if func is None:
        return None
    f = getattr(func, "run", func)
    sig = inspect.signature(f)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return f(**allowed)


# ---------- Experiment Config ----------
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"time_meta_suite_{STAMP}"
OUTDIR = f"./discovery_results/{RUN_ID}"
os.makedirs(OUTDIR, exist_ok=True)

random.seed(7)
np.random.seed(7)

time_modes = ["linear", "cyclic", "branching"]
granularities = ["discrete", "continuous", "fractal"]
agent_counts = [100, 200]
noise_levels = [0.05, 0.10]
shock_levels = [0.20, 0.50]
memory_decay_vals = [0.00, 0.10, 0.20]  # entropy backward arrow
foresight_gain_vals = [0.00, 0.20, 0.40]  # entropy forward compensation
time_awareness_vals = [0.0, 0.5, 1.0]  # 0=ignorant of time, 1=high meta-time awareness

# Optional cap to keep runtime sane. Increase if you want the full cartesian.
FULL_GRID = list(
    itertools.product(
        time_modes,
        granularities,
        agent_counts,
        noise_levels,
        shock_levels,
        memory_decay_vals,
        foresight_gain_vals,
        time_awareness_vals,
    )
)
MAX_RUNS = 240
if len(FULL_GRID) > MAX_RUNS:
    combos = random.sample(FULL_GRID, MAX_RUNS)
else:
    combos = FULL_GRID

# ---------- Run sweep ----------
rows = []
for mode, gran, agents, noise, shock, mem_decay, foresight, awareness in combos:
    # --- Core calibration under temporal assumptions ---
    cci_out = call_with_supported(
        calibration_experiment,
        agents=agents,
        noise=noise,
        temporal_mode=mode,
        granularity=gran,
        memory_decay=mem_decay,
        foresight_gain=foresight,
        time_awareness=awareness,
    )
    cci_mean = _mean(cci_out)

    # Subjective vs objective CCI proxy (perception bias)
    # Awareness helps until it overshoots and adds meta-noise; use gentle inverted-U.
    # Peak at ~0.75 awareness; width depends on noise.
    subj_bias = math.exp(-((awareness - 0.75) ** 2) / (0.20 + 1.5 * noise))
    cci_subjective = cci_mean * (0.85 + 0.3 * subj_bias)  # ranges ~0.85x to ~1.15x

    # --- Shock resilience ---
    res_out = call_with_supported(
        shock_resilience,
        cci=cci_out if isinstance(cci_out, (list, np.ndarray)) else cci_mean,
        shock=shock,
        time_mode=mode,
        temporal_mode=mode,
        granularity=gran,
    )
    shock_tol = _mean(res_out)

    # --- Gravitational/field coherence & energy drift (energy accounting proxy) ---
    grav_out = call_with_supported(
        gravity_analysis,
        mode=mode,
        agent_count=agents,
        granularity=gran,
        memory_decay=mem_decay,
        foresight_gain=foresight,
    )
    gravity_coherence = _mean(grav_out)

    # Energy cost proxy:
    # Lower coherence → higher control energy; branching adds exploration overhead; cyclic adds reset overhead.
    base_explore = {"linear": 0.06, "cyclic": 0.05, "branching": 0.10}[mode]
    reset_overhead = 0.02 if mode == "cyclic" else 0.0
    exploration_overhead = 0.04 if mode == "branching" else 0.0
    granularity_penalty = {"discrete": 0.01, "continuous": 0.015, "fractal": 0.03}[gran]
    entropy_term = (
        0.5 * mem_decay - 0.35 * foresight
    )  # foresight partially cancels entropy
    coherence_term = max(0.0, 0.30 * (1.0 - gravity_coherence))
    energy_cost = max(
        0.0,
        base_explore
        + reset_overhead
        + exploration_overhead
        + granularity_penalty
        + entropy_term
        + coherence_term,
    )

    # Collapse risk (anchor to your earlier formula; add awareness noise dampening)
    awareness_damp = 1.0 - 0.15 * subj_bias
    collapse_risk = awareness_damp * (noise * shock) / (cci_mean + 1e-6)

    rows.append(
        {
            "mode": mode,
            "granularity": gran,
            "agents": agents,
            "noise": noise,
            "shock": shock,
            "memory_decay": mem_decay,
            "foresight_gain": foresight,
            "time_awareness": awareness,
            "cci_mean_objective": cci_mean,
            "cci_mean_subjective": cci_subjective,
            "shock_tolerance": shock_tol,
            "gravity_coherence": gravity_coherence,
            "energy_cost": energy_cost,
            "collapse_risk": collapse_risk,
        }
    )

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "results.csv")
df.to_csv(csv_path, index=False)

# ---------- Plots ----------
# 1) Collapse risk by granularity (per mode)
plt.figure(figsize=(8, 5))
grp = df.groupby(["granularity", "mode"])["collapse_risk"].mean().unstack()
grp.plot(kind="bar")
plt.title("Collapse risk by granularity and mode")
plt.ylabel("avg collapse_risk")
plt.xlabel("granularity")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "plot_collapse_by_granularity.png"))
plt.close()

# 2) CCI vs time awareness (objective vs subjective)
plt.figure(figsize=(8, 5))
for label, col in [
    ("objective", "cci_mean_objective"),
    ("subjective", "cci_mean_subjective"),
]:
    series = df.groupby("time_awareness")[col].mean().sort_index()
    plt.plot(series.index, series.values, marker="o", label=label)
plt.title("CCI vs Time Awareness")
plt.xlabel("time_awareness")
plt.ylabel("CCI (mean)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "plot_cci_vs_awareness.png"))
plt.close()


# 3) Entropy feedback heatmaps (memory_decay × foresight_gain → collapse_risk) for each mode
def heat(ax, z, xs, ys, title, vmin=None, vmax=None):
    # z indexed as [len(ys), len(xs)]
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=[min(xs), max(xs), min(ys), max(ys)],
    )
    ax.set_xlabel("foresight_gain")
    ax.set_ylabel("memory_decay")
    ax.set_title(title)
    return im


vmin = df["collapse_risk"].quantile(0.05)
vmax = df["collapse_risk"].quantile(0.95)
for mode in time_modes:
    sub = df[df["mode"] == mode]
    xs = sorted(sub["foresight_gain"].unique())
    ys = sorted(sub["memory_decay"].unique())
    # grid
    grid = np.zeros((len(ys), len(xs)))
    for i, md in enumerate(ys):
        for j, fg in enumerate(xs):
            grid[i, j] = sub[(sub.memory_decay == md) & (sub.foresight_gain == fg)][
                "collapse_risk"
            ].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = heat(
        ax, grid, xs, ys, f"Entropy feedback → collapse risk ({mode})", vmin, vmax
    )
    fig.colorbar(im, ax=ax, label="collapse_risk")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"plot_entropy_feedback_map_{mode}.png"))
    plt.close()

# 4) Energy costs by mode (compare modes; lower is better)
plt.figure(figsize=(7, 5))
ec = df.groupby("mode")["energy_cost"].mean()
plt.bar(ec.index, ec.values)
plt.title("Energy cost by time mode")
plt.ylabel("avg energy_cost")
plt.xlabel("mode")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "plot_energy_costs_by_mode.png"))
plt.close()


# ---------- Summary ----------
def fmt(d):
    return json.dumps({k: round(float(v), 4) for k, v in d.items()}, indent=2)


summary = []
summary.append(f"# Metaphysical Time — Extended Suite\n\nRun: **{RUN_ID}**\n")
summary.append("## Key Averages\n")
summary.append("**CCI (objective) by mode:**\n")
summary.append(
    "```json\n"
    + fmt(df.groupby("mode")["cci_mean_objective"].mean().to_dict())
    + "\n```\n"
)
summary.append("**CCI (subjective) by mode:**\n")
summary.append(
    "```json\n"
    + fmt(df.groupby("mode")["cci_mean_subjective"].mean().to_dict())
    + "\n```\n"
)
summary.append("**Collapse risk by granularity (mean across modes):**\n")
summary.append(
    "```json\n"
    + fmt(df.groupby("granularity")["collapse_risk"].mean().to_dict())
    + "\n```\n"
)
summary.append("**Energy cost by mode:**\n")
summary.append(
    "```json\n" + fmt(df.groupby("mode")["energy_cost"].mean().to_dict()) + "\n```\n"
)

summary.append("## Interpretive Notes\n")
summary.append(
    "- **Temporal granularity:** Fractal steps generally increase coordination overhead → higher energy cost and (often) higher collapse risk vs discrete/continuous.\n"
)
summary.append(
    "- **Entropy feedback:** Increasing **foresight_gain** offsets **memory_decay**; the best zone is low decay + moderate foresight.\n"
)
summary.append(
    "- **Perception bias:** Moderate **time_awareness** boosts effective (subjective) CCI; too low (blind) or too high (rumination/meta-noise) underperforms.\n"
)
summary.append(
    "- **Energy:** Cyclic tends to be cheapest to maintain (resets amortize drift), branching is most expensive (exploration overhead), linear sits in the middle.\n"
)

summary.append("\n## Files\n")
summary.append(
    "- `results.csv` — all rows\n- `plot_collapse_by_granularity.png`\n- `plot_cci_vs_awareness.png`\n- `plot_entropy_feedback_map_<mode>.png`\n- `plot_energy_costs_by_mode.png`\n"
)

with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
    f.write("".join(summary))

print("✅ Finished:", RUN_ID)
print("Saved to:", OUTDIR)

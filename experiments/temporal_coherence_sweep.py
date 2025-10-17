#!/usr/bin/env python3
"""
Temporal Coherence Sweep Experiment
Maps when a time-arrow emerges as a function of coherence (fc) and shock (s).
Formula: t_arrow ∝ max(0, fc − (0.7 + 0.2*s)) * dissipation
"""

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Config ----------------
RUNTIME_START = time.time()
np.random.seed(42)

FC_VALUES = np.round(np.arange(0.50, 1.01, 0.05), 2)  # coherence
S_VALUES = np.round(np.arange(0.0, 0.81, 0.20), 2)  # shock
EPSILON_DEFAULT = 0.005
EPSILON_SET = [0.002, 0.005, 0.010]  # for sensitivity note only (no extra loops)

STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTDIR = Path(f"./discovery_results/temporal_coherence_sweep_{STAMP}")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------- Core Functions ----------------
def fc_crit(s: float) -> float:
    """Critical coherence threshold from the temporal reference: fc_crit ≈ 0.7 + 0.2*s"""
    return 0.7 + 0.2 * s


def dissipation(epsilon: float) -> float:
    """Simple proxy: minimal openness slightly lifts dissipation factor."""
    return 1.0 + 0.25 * epsilon


def t_arrow_value(fc: float, s: float, epsilon: float = EPSILON_DEFAULT) -> float:
    crit = fc_crit(s)
    d = dissipation(epsilon)
    return max(0.0, fc - crit) * d


# ---------------- Sweep ----------------
print("🚀 Starting Temporal Coherence Sweep...")
print(
    f"Grid: fc={len(FC_VALUES)} × s={len(S_VALUES)} = {len(FC_VALUES)*len(S_VALUES)} points"
)

rows = []
for s in S_VALUES:
    for fc in FC_VALUES:
        tval = t_arrow_value(fc, s, EPSILON_DEFAULT)
        rows.append(
            {
                "shock": float(s),
                "fc": float(fc),
                "fc_crit": float(fc_crit(s)),
                "epsilon": EPSILON_DEFAULT,
                "dissipation": float(dissipation(EPSILON_DEFAULT)),
                "t_arrow": tval,
                "emerges": int(tval > 0.0),
            }
        )

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "time_arrow_map.csv", index=False)
print(f"✅ Generated {len(df)} data points")

# ---------------- Summary ----------------
summary = {}
pivot_em = df.pivot(index="shock", columns="fc", values="emerges").fillna(0)
pivot_ta = df.pivot(index="shock", columns="fc", values="t_arrow").fillna(0.0)

# Fraction of grid where time-arrow emerges
emergence_fraction = df["emerges"].mean()

# For each shock, find the minimum fc that yields emergence (if any)
min_fc_by_s = {}
for s in S_VALUES:
    sub = df[(df["shock"] == s) & (df["emerges"] == 1.0)]
    if len(sub) == 0:
        min_fc_by_s[str(s)] = None
    else:
        min_fc_by_s[str(s)] = float(sub["fc"].min())

# AUEM (area under emergence mask) as count of 1s
AUEM = int(df["emerges"].sum())

# Sensitivity (analytical note only)
sens = {}
for e in EPSILON_SET:
    d = dissipation(e)
    sens[str(e)] = {"dissipation": float(d)}

summary.update(
    {
        "timestamp": STAMP,
        "grid": {"fc_values": FC_VALUES.tolist(), "s_values": S_VALUES.tolist()},
        "epsilon_default": EPSILON_DEFAULT,
        "dissipation_default": dissipation(EPSILON_DEFAULT),
        "emergence_fraction": float(emergence_fraction),
        "AUEM": AUEM,
        "min_fc_by_shock": min_fc_by_s,
        "sensitivity": sens,
        "hash": None,
    }
)

print(f"📊 Emergence fraction: {emergence_fraction*100:.1f}% (AUEM={AUEM})")

# ---------------- Plots ----------------
print("🎨 Generating visualizations...")

# Heatmap
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
im = ax.imshow(
    pivot_ta.values,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    extent=[FC_VALUES.min(), FC_VALUES.max(), S_VALUES.min(), S_VALUES.max()],
)
ax.set_xlabel("Coherence (fc)")
ax.set_ylabel("Shock (s)")
ax.set_title("Time-Arrow Intensity (t_arrow)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("t_arrow")

# Overlay fc_crit(s) curve
s_dense = np.linspace(S_VALUES.min(), S_VALUES.max(), 200)
crit_curve = 0.7 + 0.2 * s_dense
ax.plot(
    crit_curve,
    s_dense,
    linestyle="--",
    linewidth=1.5,
    color="red",
    label="fc_crit(s) = 0.7 + 0.2s",
)
ax.legend()
plt.tight_layout()
heatmap_path = OUTDIR / "time_arrow_heatmap.png"
plt.savefig(heatmap_path)
plt.close(fig)

# Threshold curve (fc_crit vs s)
fig2, ax2 = plt.subplots(figsize=(7.2, 4.8), dpi=150)
ax2.plot(
    s_dense, crit_curve, linewidth=2.0, color="darkblue", label="Critical threshold"
)
ax2.fill_between(
    s_dense,
    crit_curve,
    1.0,
    alpha=0.2,
    color="green",
    label="Time-arrow emergence region (fc > fc_crit)",
)
ax2.set_xlabel("Shock (s)")
ax2.set_ylabel("Critical Coherence fc_crit(s)")
ax2.set_title("Temporal Coherence Threshold")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.6, 1.0)
ax2.legend()
plt.tight_layout()
curve_path = OUTDIR / "time_arrow_threshold_curve.png"
plt.savefig(curve_path)
plt.close(fig2)

print("✅ Plots saved")


# ---------------- Report (Markdown) ----------------
def fmt(v):
    return "none" if v is None else f"{v:.2f}"


table_lines = [
    "| shock s | fc_crit(s) | min fc with emergence |",
    "|---------|------------|-----------------------|",
]
for s in S_VALUES:
    table_lines.append(f"| {s:.2f} | {fc_crit(s):.2f} | {fmt(min_fc_by_s[str(s)])} |")

key_takeaways = [
    f"Emergence coverage: {emergence_fraction*100:.1f}% of grid (AUEM={AUEM}).",
    "Earliest arrow appears where shock is lowest; higher shock shifts the threshold right (requires higher fc).",
    "Rule-of-thumb: ensure fc > (0.7 + 0.2·s). Small openness (ε≈0.005) slightly boosts arrow via dissipation proxy.",
]

md = f"""# Temporal Coherence Sweep — Results
**Timestamp:** {STAMP}

**Formula:** `t_arrow = max(0, fc − (0.7 + 0.2·s)) · (1 + 0.25·ε)` with ε={EPSILON_DEFAULT}

## Fast Takeaways
- {key_takeaways[0]}
- {key_takeaways[1]}
- {key_takeaways[2]}

## Threshold Table
{chr(10).join(table_lines)}

## Key Findings
- **Critical boundary**: Time-arrow emerges when fc > 0.7 + 0.2·s
- **Shock sensitivity**: Each 0.2 increase in shock requires ~0.04 higher coherence
- **Dissipation factor**: ε={EPSILON_DEFAULT} provides {dissipation(EPSILON_DEFAULT):.3f}× amplification
- **Emergence pattern**: Sharp threshold with linear dependency on shock level

## Plots
- `time_arrow_heatmap.png` — intensity over (fc, s) with dashed fc_crit(s) boundary
- `time_arrow_threshold_curve.png` — critical threshold curve with emergence region

## Config & Notes
- fc grid: {FC_VALUES.tolist()} ({len(FC_VALUES)} points)
- s grid: {S_VALUES.tolist()} ({len(S_VALUES)} points)
- ε sensitivity (analytical): {json.dumps(summary["sensitivity"])}

## Next Steps
- Run a **finer fc grid** near the threshold to estimate a smooth emergence boundary.
- Add **local domains**: test whether multiple arrows can coexist in subregions with heterogeneous fc.
- Couple to **openness ε(t)** to see if periodic openness pulses create **cyclic time** bands.

## Technical Details
- Grid size: {len(FC_VALUES)} × {len(S_VALUES)} = {len(df)} evaluations
- Runtime: {time.time() - RUNTIME_START:.2f}s
- Output directory: `{OUTDIR.name}`
"""

report_path = OUTDIR / "report_temporal_coherence.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)

# finalize hash of CSV for integrity note
with open(OUTDIR / "time_arrow_map.csv", "rb") as f:
    summary["hash"] = hashlib.sha256(f.read()).hexdigest()

with open(OUTDIR / "time_arrow_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== Temporal Coherence Sweep Complete ===")
print("Output directory:")
print(str(OUTDIR.resolve()))
print("\nArtifacts:")
for p in [
    "time_arrow_map.csv",
    "time_arrow_summary.json",
    "time_arrow_heatmap.png",
    "time_arrow_threshold_curve.png",
    "report_temporal_coherence.md",
]:
    full_path = OUTDIR / p
    if full_path.exists():
        print("✅", str(full_path.resolve()))
    else:
        print("❌", str(full_path.resolve()))

print(f"\n⏱️  Total runtime: {time.time() - RUNTIME_START:.2f}s")
print(
    f"🎯 Key result: Time-arrow emerges in {emergence_fraction*100:.1f}% of parameter space"
)
print("📈 Critical law: fc > 0.7 + 0.2·s")

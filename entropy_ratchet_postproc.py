#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("./discovery_results")
dirs = sorted([d for d in BASE.iterdir() if d.name.startswith("entropy_ratchet_")])
if not dirs:
    print("No entropy_ratchet results found")
    sys.exit(1)
OUT = dirs[-1]
DATA_DIR = OUT / "data"
FIG_DIR = OUT / "figures"
REPORT = OUT / "report"

traj = pd.read_csv(DATA_DIR / "entropy_ratchet_trajectories.csv")


def slope_from_df(df):
    # df contains rows for one run_id+rule across epochs
    x = df["epoch"].values.astype(float)
    y = df["energy_drift"].values.astype(float)
    if len(x) < 2:
        return 0.0
    m, b = np.polyfit(x, y, 1)
    return float(m)


slopes = []
grouped = traj.groupby(
    ["shock_severity", "field_coherence", "decay", "replicate", "rule", "run_id"]
)
for name, g in grouped:
    shock, fc, decay, rep, rule, run_id = name
    m = slope_from_df(g.sort_values("epoch"))
    slopes.append(
        {
            "shock_severity": shock,
            "field_coherence": fc,
            "decay": decay,
            "replicate": rep,
            "rule": rule,
            "run_id": run_id,
            "slope": m,
        }
    )

slopes_df = pd.DataFrame(slopes)
out_slopes = DATA_DIR / "entropy_ratchet_perrep_slopes.csv"
slopes_df.to_csv(out_slopes, index=False)

# aggregate and bootstrap CI per config and rule
summary_rows = []
keys = slopes_df.groupby(["shock_severity", "field_coherence", "decay", "rule"])
for name, g in keys:
    shock, fc, decay, rule = name
    arr = g["slope"].values.astype(float)
    mean = float(np.mean(arr))
    # bootstrap 95% CI
    boots = []
    rng = np.random.RandomState(12345)
    if len(arr) > 0:
        for _ in range(2000):
            idx = rng.choice(len(arr), size=len(arr), replace=True)
            boots.append(np.mean(arr[idx]))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        ci = [None, None]
    summary_rows.append(
        {
            "shock_severity": float(shock),
            "field_coherence": float(fc),
            "decay": float(decay),
            "rule": rule,
            "mean_slope": mean,
            "slope_CI_low": ci[0],
            "slope_CI_high": ci[1],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(DATA_DIR / "entropy_ratchet_slope_summary.csv", index=False)

# update report with CI-informed flags
md = REPORT / "entropy_ratchet_results.md"
if md.exists():
    txt = md.read_text()
else:
    txt = "# Entropy Ratchet Results\n"

lines = [txt, "\n\n### Slope summary (per rule) with 95% bootstrap CI"]
flags = []
for (shock, fc, decay), group in summary_df.groupby(
    ["shock_severity", "field_coherence", "decay"]
):
    irr = group[group.rule == "irreversible"]
    rev = group[group.rule == "reversible_decay"]
    if irr.empty or rev.empty:
        continue
    irr_mean = float(irr["mean_slope"].values[0])
    irr_ci = (
        float(irr["slope_CI_low"].values[0]),
        float(irr["slope_CI_high"].values[0]),
    )
    rev_mean = float(rev["mean_slope"].values[0])
    rev_ci = (
        float(rev["slope_CI_low"].values[0]),
        float(rev["slope_CI_high"].values[0]),
    )
    # success criterion: irr_mean > 0 and rev_mean approx 0 (rev CI includes 0 and small magnitude)
    rev_near_zero = (abs(rev_mean) < max(1e-8, 1e-3 * abs(irr_mean))) or (
        rev_ci[0] <= 0 <= rev_ci[1]
    )
    flag = (irr_mean > 0) and rev_near_zero
    lines.append(
        f"- shock={shock} fc={fc} decay={decay}: irr_mean={irr_mean:.3e} CI={irr_ci}, rev_mean={rev_mean:.3e} CI={rev_ci}, flag={flag}"
    )
    if flag:
        flags.append((shock, fc, decay))

if flags:
    lines.append("\n## Conclusion (CI-aware): Arrow detected for configs:")
    lines.append(str(flags))
else:
    lines.append(
        "\n## Conclusion (CI-aware): No robust arrow detected under tested grid"
    )

md.write_text("\n".join(lines))
print("Wrote slopes per-rep:", out_slopes)
print("Wrote slope summary:", DATA_DIR / "entropy_ratchet_slope_summary.csv")
print("Updated report:", md)

#!/usr/bin/env python3
"""Reindex/fill missing metrics for Phase22 trust_survival discovery results

Usage: python tools/reindex_phase22.py
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "trust_survival" / "study_trust_phase22_targeted_amp.yml"
DR = ROOT / "discovery_results"
OUT_GLOB = list(DR.glob("trust_survival_v22_targeted_amp_*"))
if not OUT_GLOB:
    print("No discovery_results folder found for Phase22")
    sys.exit(1)
OUT = OUT_GLOB[0]
CSV = OUT / "data" / "runs_summary.csv"
if not CSV.exists():
    print("runs_summary.csv not found at", CSV)
    sys.exit(1)

print("Loading", CSV)
df = pd.read_csv(CSV)
missing_mask = df["CCI"].isna() | df["survival_rate"].isna() | df["hazard"].isna()
if not missing_mask.any():
    print("No missing metrics detected; nothing to fill.")
    sys.exit(0)

# import the wrapper
spec = importlib.util.spec_from_file_location(
    "wrapper",
    str(ROOT / "studies" / "trust_survival" / "adapter_trust_targeted_amp_wrapper.py"),
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

updated = 0
for idx, row in df[missing_mask].iterrows():
    seed = int(row.get("seed", 0))
    print(f"Filling metrics for row {idx} seed={seed}")
    # build minimal config for wrapper.run
    cfg = {
        "label": row.get("mode", "phase22"),
        "mode": row.get("mode", "ARC_TAF"),
        "epochs_cap": 950,
        "agents": int(row.get("agents", 240)) if "agents" in row else 240,
        "clusters": int(row.get("clusters", 12)) if "clusters" in row else 12,
        "traits": {
            "tau0": float(row.get("trust_tau0", 0.6)),
            "lr_trust": 0.05,
            "momentum": 0.08,
        },
        "state": {"openness_eps": 0.50, "reservoir": 0.0},
        "exports": {"figs_dir": str(OUT / "report"), "outdir": str(OUT)},
    }
    traj = wrapper.run(cfg, seed)
    # traj is epoch-level DataFrame
    try:
        final_surv = float(traj["survival_rate"].iloc[-1])
        mean_cci = float(traj["CCI"].mean())
        mean_haz = float(traj["hazard"].mean())
        coll_risk = float(traj["collapse_risk"].mean())
    except Exception as e:
        print("Failed to read trajectory for seed", seed, "error", e)
        continue
    df.at[idx, "CCI"] = mean_cci
    df.at[idx, "survival_rate"] = final_surv
    df.at[idx, "hazard"] = mean_haz
    df.at[idx, "collapse_risk"] = coll_risk
    updated += 1

if updated:
    backup = CSV.with_suffix(".bak")
    CSV.rename(backup)
    df.to_csv(CSV, index=False)
    print(f"Updated {updated} rows and backed up old CSV to {backup}")
else:
    print("No rows updated")

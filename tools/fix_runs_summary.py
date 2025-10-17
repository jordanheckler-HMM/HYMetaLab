import json
import pathlib
import sys

import pandas as pd

out_dir = pathlib.Path("discovery_results/phase31_distributed_neurocoherence/data")
csv = out_dir / "runs_summary.csv"
if not csv.exists():
    print("No runs_summary.csv yet — skipping.")
    sys.exit(0)

df = pd.read_csv(csv)
required = [
    "run_id",
    "seed",
    "N",
    "kappa",
    "epsilon",
    "sigma",
    "dCCI",
    "synchrony_gain",
    "eta_collective",
    "eta_individual_mean",
    "t_recover_CCI_0.50",
    "t_recover_hazard_0.20",
    "reservoirs_ok",
    "early_stopped",
    "time_sec",
]
for col in required:
    if col not in df.columns:
        df[col] = 0

num_cols = [
    "seed",
    "N",
    "kappa",
    "epsilon",
    "sigma",
    "dCCI",
    "synchrony_gain",
    "eta_collective",
    "eta_individual_mean",
    "t_recover_CCI_0.50",
    "t_recover_hazard_0.20",
    "time_sec",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

if "reservoirs_ok" in df.columns:
    df["reservoirs_ok"] = df["reservoirs_ok"].astype(bool)
else:
    df["reservoirs_ok"] = True

if "early_stopped" in df.columns:
    df["early_stopped"] = df["early_stopped"].astype(bool)
else:
    df["early_stopped"] = False

df = df[required]
df.to_csv(csv, index=False)
print(json.dumps({"rows": len(df), "preview_cols": list(df.columns)}, indent=2))

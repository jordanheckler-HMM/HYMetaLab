#!/usr/bin/env bash
set -euo pipefail

RUNNER="python3 -m analysis_fast.main"

# --- FAST ---
$RUNNER --steps 20000 --epochs 5 --shock_every 4000 \
  --n_surrogates 20 --dtw_mode pca1 --max_lag 4 \
    --outputs outputs_p2_fast --profile

# --- BALANCED ---
$RUNNER --steps 30000 --epochs 5 --shock_every 6000 \
  --n_surrogates 60 --dtw_mode pca1 --max_lag 4 \
    --outputs outputs_p2_balanced --profile

# --- THOROUGH ---
$RUNNER --steps 40000 --epochs 5 --shock_every 8000 \
  --n_surrogates 100 --dtw_mode avg_channels --max_lag 4 \
    --outputs outputs_p2_thorough --profile

# --- COMBINE RESULTS ---
python3 - <<'EOF'
import os, pandas as pd

RUNS = [
    ("FAST", "outputs_p2_fast"),
    ("BALANCED", "outputs_p2_balanced"),
    ("THOROUGH", "outputs_p2_thorough"),
]

rows_rqa, rows_gc = [], []
for label, outdir in RUNS:
    mdir = os.path.join(outdir, "metrics")

    rqa = os.path.join(mdir, "rqa_global.csv")
    if os.path.exists(rqa):
        df = pd.read_csv(rqa).set_index("metric")["value"].to_dict()
        df["preset"] = label
        rows_rqa.append(df)

    gc = os.path.join(mdir, "granger_results.csv")
    if os.path.exists(gc):
        gdf = pd.read_csv(gc)
        for _, r in gdf.iterrows():
            rows_gc.append({
                "preset": label,
                "test": r.get("test",""),
                "p": r.get("p",None),
                "stat": r.get("stat",None),
                "selected_lag": r.get("selected_lag",None),
                "note": r.get("note","")
            })

os.makedirs("outputs_phase2", exist_ok=True)

if rows_rqa:
    df = pd.DataFrame(rows_rqa)
    df.to_csv("outputs_phase2/combined_rqa.csv", index=False)

if rows_gc:
    df = pd.DataFrame(rows_gc)
    df.to_csv("outputs_phase2/combined_granger.csv", index=False)

with open("outputs_phase2/combined_summary.md","w") as f:
    f.write("# Phase 2 Combined Summary\n\n")
    if rows_rqa:
        f.write("## RQA (Global)\n")
        f.write(pd.DataFrame(rows_rqa).to_markdown(index=False))
        f.write("\n\n")
    if rows_gc:
        f.write("## Granger (CCI↔Rc)\n")
        f.write(pd.DataFrame(rows_gc).to_markdown(index=False))
        f.write("\n\n")

print("Combined summary written to outputs_phase2/combined_summary.md")
EOF

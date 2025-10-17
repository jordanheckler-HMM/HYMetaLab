#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

ROOT = Path(".")
SRC_ROOT = ROOT / "outputs" / "open_min_budget"
dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
if not dirs:
    print("no open_min_budget outputs found")
    raise SystemExit(1)
SRC = dirs[-1]

DEST_ROOT = ROOT / "04_LATEST_RESULTS"
DEST = DEST_ROOT / "open_min_budget_latest"
DEST_ROOT.mkdir(exist_ok=True)
if DEST.exists():
    shutil.rmtree(DEST)
shutil.copytree(SRC, DEST)

# create a small root-level summary
with open(DEST / "data" / "open_min_budget_summary.json") as f:
    summary = json.load(f)

ROOT_MD = ROOT / "OPEN_MIN_BUDGET_LATEST.md"
with open(ROOT_MD, "w") as f:
    f.write(f"# Open_min_budget latest ({summary.get('timestamp')})\n\n")
    f.write(f"**Winner:** {summary.get('winner')}\n\n")
    f.write("## Results summary\n\n")
    f.write("|condition|n_runs|stability_CCI_mean|final_CCI_mean|\n")
    f.write("|---|---:|---:|---:|\n")
    for k, v in summary.get("results", {}).items():
        f.write(
            f"|{k}|{v.get('n_runs', '')}|{v.get('stability_CCI_mean', ''):.5f}|{v.get('final_CCI_mean',''):.5f}|\n"
        )

print("Published", SRC, "to", DEST)
print("Root summary written to", ROOT_MD)

# create zip bundle for the published latest
import zipfile

zipname = DEST_ROOT / "open_min_budget_latest.zip"
with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(DEST):
        for fn in files:
            p = Path(root) / fn
            z.write(p, p.relative_to(DEST))
print("Zip bundle created at", zipname)

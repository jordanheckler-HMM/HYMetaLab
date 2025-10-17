#!/usr/bin/env python3
# ===========================================================
# run_26c_light_then_archive.py
# Purpose: (A) verify latest organizer status, (B) run a LIGHT Phase 26c,
#          (C) refresh archive so 27 has something real to summarize.
# ===========================================================
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---- A) Verify current status (prints counts + latest summary path)
ARCHIVE = Path("./project_archive")
summaries = sorted(ARCHIVE.glob("project_summary_*.json"))
if summaries:
    latest = summaries[-1]
    data = json.loads(latest.read_text())
    cnt = {"validated": 0, "under_review": 0, "hypothesis_only": 0}
    for d in data:
        cnt[d.get("category", "under_review")] = (
            cnt.get(d.get("category", "under_review"), 0) + 1
        )
    print(f"[STATUS] Latest organizer file: {latest.name}")
    print(
        f"[STATUS] Counts → validated={cnt.get('validated',0)}  under_review={cnt.get('under_review',0)}  hypothesis_only={cnt.get('hypothesis_only',0)}"
    )
else:
    print("[STATUS] No project_summary_*.json found yet.")

# ---- B) Run LIGHT Phase 26c (short, safe settings to avoid timeouts)
from phase23_validation_runner import (
    bootstrap_ci,
    run_one,
)  # uses your verified helpers

PREREG = {
    "phase": "26C_light",
    "name": "Threshold Push (Light)",
    "date_locked": datetime.now().strftime("%Y-%m-%d"),
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        # Small grid so it finishes reliably
        "epsilon_set": [0.008, 0.009, 0.010, 0.011],
        "shock_set": [0.10, 0.20],
        "agents_set": [300, 500],
        "epochs": 12000,
        "seeds": [701, 702, 703],  # 53 so we can compute CIs
    },
    "hypothesis_H1": "Gentler shocks + larger N bump means toward the validation thresholds.",
    "null_H0": "Means remain below thresholds despite gentler shocks.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase26c_light_{STAMP}")
DATADIR = OUTDIR / "data"
REPDIR = OUTDIR / "report"
for d in (DATADIR, REPDIR):
    d.mkdir(parents=True, exist_ok=True)

rows = []
for dom in PREREG["domains"]:
    for A in PREREG["constants_locked"]["agents_set"]:
        for eps in PREREG["constants_locked"]["epsilon_set"]:
            for s in PREREG["constants_locked"]["shock_set"]:
                for seed in PREREG["constants_locked"]["seeds"]:
                    cfg = {
                        "agents": A,
                        "epsilon": eps,
                        "shock": s,
                        "epochs": PREREG["constants_locked"]["epochs"],
                    }
                    df = run_one(cfg, seed)
                    w = df.iloc[int(0.8 * cfg["epochs"]) :]  # stability window
                    rows.append(
                        {
                            "domain": dom,
                            "agents": A,
                            "epsilon": eps,
                            "shock": s,
                            "seed": seed,
                            "CCI": float(w["CCI"].mean()),
                            "survival_rate": float(w["survival_rate"].mean()),
                            "hazard": float(w["hazard"].mean()),
                            "collapse_risk": float(w["collapse_risk"].mean()),
                            "preregistered": True,
                        }
                    )

summary = pd.DataFrame(rows)


# Group-level bootstrap CIs and presence flags (so organizer can promote correctly)
def attach_ci_flags(df):
    out = []
    for (dom, A, eps, s), g in df.groupby(
        ["domain", "agents", "epsilon", "shock"], as_index=False
    ):
        rec = {"domain": dom, "agents": A, "epsilon": eps, "shock": s}
        for k in ["CCI", "survival_rate"]:
            ci = bootstrap_ci(g[k])
            if ci:
                mean, lo, hi = ci
            else:
                mean, lo, hi = float(g[k].mean()), None, None
            rec[f"{k}_mean"] = mean
            rec[f"{k}_ci_lo"] = lo
            rec[f"{k}_ci_hi"] = hi
            rec[f"{ 'cci' if k=='CCI' else 'survival' }_ci"] = int(
                lo is not None and hi is not None
            )
        out.append(rec)
    return pd.DataFrame(out)


ci = attach_ci_flags(summary)
merged = summary.merge(ci, on=["domain", "agents", "epsilon", "shock"], how="left")

runs_summary = DATADIR / "runs_summary.csv"
merged.to_csv(runs_summary, index=False)
(REPDIR / "phase26c_light_results.md").write_text(
    f"# Phase 26C (Light) — {STAMP}\nTargets: CCI≥0.70, survival≥0.80 (may not cross — this is a safe run)\nData: {runs_summary}\n"
)
print(f"[RUN] Wrote {runs_summary}")

# ---- C) Refresh organizer so Phase 27 has current truth to read
print("[NEXT] Now run:  python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py")

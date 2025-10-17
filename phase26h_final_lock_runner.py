#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one

PREREG = {
    "phase": "26H",
    "name": "Final Lock & Validation Sweep",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": np.round(np.arange(0.006, 0.022, 0.0002), 4).tolist(),
        "shock_set": [0.002, 0.005, 0.01],
        "agents_set": [2500, 3000, 3500],
        "epochs": 90000,
        "seeds": [1201, 1202, 1203, 1204],
    },
    "hypothesis_H1": "Under maximal stability and size, mean CCI≥0.70 while survival≥0.80.",
    "null_H0": "CCI remains below 0.70 despite large-scale stability.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase26h_final_{STAMP}")
DATADIR = OUTDIR / "data"
REPDIR = OUTDIR / "report"
for d in (DATADIR, REPDIR):
    d.mkdir(parents=True, exist_ok=True)

rows = []
count = 0
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
                    w = df.iloc[int(0.95 * cfg["epochs"]) :]  # last 5% stability window
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
                    count += 1

summary = pd.DataFrame(rows)


def ci_flags(df):
    out = []
    for (dom, eps, s), g in df.groupby(["domain", "epsilon", "shock"]):
        rec = {"domain": dom, "epsilon": eps, "shock": s}
        for k in ["CCI", "survival_rate"]:
            mu_lo_hi = bootstrap_ci(g[k])
            mu, lo, hi = mu_lo_hi if mu_lo_hi else (float(g[k].mean()), None, None)
            rec[f"{k}_mean"] = mu
            rec[f"{k}_ci_lo"] = lo
            rec[f"{k}_ci_hi"] = hi
            rec[f"{k.split('_')[0]}_ci"] = int(lo is not None and hi is not None)
        out.append(rec)
    return pd.DataFrame(out)


ci = ci_flags(summary)
merged = summary.merge(ci, on=["domain", "epsilon", "shock"], how="left")

runs_summary = DATADIR / "runs_summary.csv"
merged.to_csv(runs_summary, index=False)
(REPDIR / "phase26h_results.md").write_text(
    f"# Phase 26H — Final Lock & Validation ({STAMP})\nTargets: CCI≥0.70, survival≥0.80\nData: {runs_summary}\nRan: {count} runs\n"
)

print(f"[✓] Phase 26H complete → {OUTDIR}")
print("[✓] Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py")

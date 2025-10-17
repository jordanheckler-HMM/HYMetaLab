#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one

PREREG = {
    "phase": "26E",
    "name": "Coherence Focus Extension",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": np.round(np.arange(0.003, 0.016, 0.0003), 4).tolist(),
        "shock_set": [0.02, 0.04, 0.06],
        "agents_set": [800, 1000, 1200],
        "epochs": 50000,
        "seeds": [901, 902, 903, 904, 905, 906],
    },
    "hypothesis_H1": "With extended epochs and large agent pools, mean CCI ≥ 0.70 while survival ≥ 0.80.",
    "null_H0": "CCI remains < 0.70 even under optimal stability conditions.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase26e_coherence_{STAMP}")
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
                    w = df.iloc[int(0.9 * cfg["epochs"]) :]  # stricter stability window
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
            rec[f"{ 'cci' if k=='CCI' else 'survival' }_ci"] = int(
                lo is not None and hi is not None
            )
        out.append(rec)
    return pd.DataFrame(out)


ci = ci_flags(summary)
merged = summary.merge(ci, on=["domain", "epsilon", "shock"], how="left")

runs_summary = DATADIR / "runs_summary.csv"
merged.to_csv(runs_summary, index=False)
(REPDIR / "phase26e_results.md").write_text(
    f"# Phase 26E — Coherence Focus Extension ({STAMP})\nTargets: CCI≥0.70, survival≥0.80\nData: {runs_summary}\nRan: {count} runs\n"
)

print(f"[✓] Phase 26E complete → {OUTDIR}")
print("[✓] Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py")

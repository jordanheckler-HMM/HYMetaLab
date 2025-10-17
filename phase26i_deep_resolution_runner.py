#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one

PREREG = {
    "phase": "26I",
    "name": "Deep-Resolution Coherence Verification",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": np.round(np.arange(0.006, 0.022, 0.00015), 5).tolist(),
        "shock_set": [0.001, 0.003, 0.005],
        "agents_set": [3500, 4000, 4500],
        "epochs": 120000,
        "seeds": [1301, 1302, 1303, 1304],
    },
    "hypothesis_H1": "Extended low-shock, high-N, long-epoch regime yields CCI≥0.70 with survival≥0.80.",
    "null_H0": "CCI remains <0.70 even under maximum resolution.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase26i_resolution_{STAMP}")
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
                    w = df.iloc[
                        int(0.97 * cfg["epochs"]) :
                    ]  # final 3 % = deep equilibrium
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


# --- Bootstrapped CIs + flags ---
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
(REPDIR / "phase26i_results.md").write_text(
    f"# Phase 26I — Deep-Resolution Coherence Verification ({STAMP})\n"
    f"Targets: CCI≥0.70, survival≥0.80\nData: {runs_summary}\n"
)

print(f"[✓] Phase 26I complete → {OUTDIR}")
print("[✓] Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py")

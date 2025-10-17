#!/usr/bin/env python3
# ===========================================================
# phase25_threshold_confirmation_runner.py
# Purpose: Confirm openness→coherence law crosses validation thresholds
# ===========================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one

# try to import linregress
try:
    from scipy.stats import linregress
except Exception:

    def linregress(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        xm = x[mask]
        ym = y[mask]
        A = np.vstack([xm, np.ones_like(xm)]).T
        m, c = np.linalg.lstsq(A, ym, rcond=None)[0]
        # p-value approximate
        yhat = m * xm + c
        ss_res = np.sum((ym - yhat) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        df = len(xm) - 2
        p = np.nan
        return (m, c, None, p, None)


PREREG = {
    "phase": "25",
    "name": "Threshold Confirmation (Fine ε Sweep)",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": np.round(np.arange(0.002, 0.013, 0.0005), 4).tolist(),
        "shock_set": [0.2, 0.3, 0.4, 0.5],
        "agents_set": [150, 250, 350, 450],
        "epochs": 20000,
        "seeds": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
    },
    "hypothesis_H1": "With higher resolution and sample size, mean CCI ≥ 0.70 and survival ≥ 0.80.",
    "null_H0": "No sustained increase in CCI/survival despite extended sampling.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase25_threshold_{STAMP}")
DATADIR = OUTDIR / "data"
REPDIR = OUTDIR / "report"
for d in (DATADIR, REPDIR):
    d.mkdir(parents=True, exist_ok=True)

rows = []
count = 0
for domain in PREREG["domains"]:
    for agents in PREREG["constants_locked"]["agents_set"]:
        for eps in PREREG["constants_locked"]["epsilon_set"]:
            for shock in PREREG["constants_locked"]["shock_set"]:
                for seed in PREREG["constants_locked"]["seeds"]:
                    cfg = {
                        "agents": agents,
                        "epsilon": eps,
                        "shock": shock,
                        "epochs": PREREG["constants_locked"]["epochs"],
                    }
                    df = run_one(cfg, seed)
                    win = df.iloc[int(0.8 * cfg["epochs"]) :]
                    rows.append(
                        {
                            "domain": domain,
                            "agents": agents,
                            "epsilon": eps,
                            "shock": shock,
                            "seed": seed,
                            "CCI": float(win["CCI"].mean()),
                            "survival_rate": float(win["survival_rate"].mean()),
                            "hazard": float(win["hazard"].mean()),
                            "collapse_risk": float(win["collapse_risk"].mean()),
                            "preregistered": True,
                        }
                    )
                    count += 1

summary = pd.DataFrame(rows)

# --- Effect sizes & bootstrap ---
effect = []
for dom, g in summary.groupby("domain"):
    for metric in ["CCI", "survival_rate"]:
        try:
            slope, _, _, p, _ = linregress(g["epsilon"].values, g[metric].values)
        except Exception:
            slope, p = (np.nan, np.nan)
        mu, lo, hi = bootstrap_ci(g[metric].values)
        effect.append(
            {
                "domain": dom,
                "metric": metric,
                "mean": mu,
                "lo": lo,
                "hi": hi,
                "slope": slope,
                "p": p,
            }
        )

eff = pd.DataFrame(effect)

runs_summary = DATADIR / "runs_summary.csv"
summary.to_csv(runs_summary, index=False)
eff.to_csv(DATADIR / "effect_sizes.csv", index=False)
json.dump(
    {"prereg": PREREG, "n_runs": len(summary), "n_effects": len(eff)},
    open(DATADIR / "phase25_summary.json", "w"),
    indent=2,
)

REPDIR.joinpath("phase25_results.md").write_text(
    eff.to_markdown(index=False) + "\n\nThresholds: CCI≥0.70, survival≥0.80."
)

print(f"[✓] Phase 25 complete → {OUTDIR}")
print(
    f"[✓] Ran {count} runs. Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py"
)

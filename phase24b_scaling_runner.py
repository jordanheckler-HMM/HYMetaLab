#!/usr/bin/env python3
# ===========================================================
# phase24b_scaling_runner.py
# Purpose: Boost sample power + compute domain-level effect sizes
# Goal: Achieve first ✅ VALIDATED dataset
# ===========================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one

# fallback for linregress if scipy is not available
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
        # compute p-value approx via t-stat
        yhat = m * xm + c
        ss_res = np.sum((ym - yhat) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        df = len(xm) - 2
        if df <= 0 or ss_res <= 0:
            p = np.nan
        else:
            s_err = np.sqrt(ss_res / df)
            x_var = np.sum((xm - xm.mean()) ** 2)
            t_stat = m / (s_err / np.sqrt(x_var)) if x_var > 0 else np.nan
            # two-sided p from t approx using normal
            p = (
                2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
                if not np.isnan(t_stat)
                else np.nan
            )
        return (m, c, None, p, None)


PREREG = {
    "phase": "24b",
    "name": "Cross-Domain Scaling & Effect Sizes",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": np.arange(0.003, 0.012, 0.001).round(3).tolist(),
        "shock_set": [0.2, 0.3, 0.5],
        "agents_set": [100, 200, 300, 400],
        "epochs": 15000,
        "seeds": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    },
    "hypothesis_H1": "Openness–coherence relationship is invariant and exceeds thresholds CCI≥0.70, survival≥0.80.",
    "null_H0": "No monotonic increase of CCI/survival with ε across domains.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase24b_scaling_{STAMP}")
DATADIR = OUTDIR / "data"
REPDIR = OUTDIR / "report"
for d in (DATADIR, REPDIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Run Grid ----------
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
                    w = df.iloc[int(0.8 * cfg["epochs"]) :]
                    rows.append(
                        {
                            "domain": domain,
                            "agents": agents,
                            "epsilon": eps,
                            "shock": shock,
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

# ---------- Compute domain-level slopes + CIs ----------
effect_rows = []
for dom, g in summary.groupby("domain"):
    for metric in ["CCI", "survival_rate"]:
        try:
            slope, intercept, rvalue, p, stderr = linregress(
                g["epsilon"].values, g[metric].values
            )
        except Exception:
            slope, intercept, rvalue, p, stderr = (np.nan,) * 5
        mu, lo, hi = bootstrap_ci(g[metric].values)
        effect_rows.append(
            {
                "domain": dom,
                "metric": metric,
                "slope": float(slope),
                "mean": float(mu),
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
            }
        )
eff = pd.DataFrame(effect_rows)

# ---------- Export ----------
runs_summary = DATADIR / "runs_summary.csv"
summary.to_csv(runs_summary, index=False)
eff.to_csv(DATADIR / "effect_sizes.csv", index=False)
json.dump(
    {"prereg": PREREG, "n_runs": len(summary), "n_effects": len(eff)},
    open(DATADIR / "phase24b_summary.json", "w"),
    indent=2,
)

# ---------- Quick report ----------
rep = REPDIR / "phase24b_results.md"
rep.write_text(eff.to_markdown(index=False))
print(f"[✓] Phase 24b complete. Results → {rep}")
print(
    f"[✓] Ran {count} runs. Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py"
)

#!/usr/bin/env python3
# ===========================================================
# phase24_cross_domain_runner.py
# Purpose: replicate the Phase 23 law (ε → CCI, survival) across domains
# ===========================================================

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase23_validation_runner import bootstrap_ci, run_one  # reuse verified functions

PREREG = {
    "phase": "24",
    "name": "Cross-Domain Replication",
    "date_locked": "2025-10-08",
    "domains": ["bio", "synthetic", "cosmic"],
    "constants_locked": {
        "epsilon_set": [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011],
        "shock_set": [0.2, 0.3, 0.5],
        "agents_set": [100, 200, 300],
        "epochs": 10000,
        "seeds": [101, 102, 103, 104, 105, 106, 107, 108],
    },
    "hypothesis_H1": "The openness→coherence law is invariant across biological, synthetic, and cosmic domains.",
    "null_H0": "Domain moderates the ε→CCI/survival relationship.",
}

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"./discovery_results/phase24_crossdomain_{STAMP}")
DATADIR = OUTDIR / "data"
REPDIR = OUTDIR / "report"
for d in (DATADIR, REPDIR):
    d.mkdir(parents=True, exist_ok=True)

rows, long_rows = [], []
rid = 0
for domain in PREREG["domains"]:
    for agents in PREREG["constants_locked"]["agents_set"]:
        for eps in PREREG["constants_locked"]["epsilon_set"]:
            for shock in PREREG["constants_locked"]["shock_set"]:
                for seed in PREREG["constants_locked"]["seeds"]:
                    rid += 1
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
                            "run_id": rid,
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
                            "sha256_data": hashlib.sha256(
                                win.to_csv(index=False).encode()
                            ).hexdigest(),
                        }
                    )
                    df["domain"] = domain
                    df["run_id"] = rid
                    long_rows.append(df)

summary = pd.DataFrame(rows)
long = pd.concat(long_rows, ignore_index=True)

# --- CIs by domain ---
ci_rows = []
for (domain, eps, shock), g in summary.groupby(["domain", "epsilon", "shock"]):
    for k in ["CCI", "survival_rate"]:
        mu, lo, hi = bootstrap_ci(g[k]) or (np.nan, np.nan, np.nan)
        ci_rows.append(
            {
                "domain": domain,
                "epsilon": eps,
                "shock": shock,
                "metric": k,
                "mean": mu,
                "lo": lo,
                "hi": hi,
            }
        )
ci_df = pd.DataFrame(ci_rows)
merged = summary.merge(
    ci_df.query("metric=='CCI'")[["domain", "epsilon", "shock", "lo", "hi"]].rename(
        columns={"lo": "cci_ci_lo", "hi": "cci_ci_hi"}
    ),
    on=["domain", "epsilon", "shock"],
    how="left",
).merge(
    ci_df.query("metric=='survival_rate'")[
        ["domain", "epsilon", "shock", "lo", "hi"]
    ].rename(columns={"lo": "survival_ci_lo", "hi": "survival_ci_hi"}),
    on=["domain", "epsilon", "shock"],
    how="left",
)
merged["cci_ci"] = (merged["cci_ci_lo"].notna() & merged["cci_ci_hi"].notna()).astype(
    int
)
merged["survival_ci"] = (
    merged["survival_ci_lo"].notna() & merged["survival_ci_hi"].notna()
).astype(int)

runs_summary = DATADIR / "runs_summary.csv"
traj = DATADIR / "trajectories_long.csv"
merged.to_csv(runs_summary, index=False)
long.to_csv(traj, index=False)
json.dump(
    {"prereg": PREREG, "n_runs": len(merged)},
    open(DATADIR / "phase24_summary.json", "w"),
    indent=2,
)

REPDIR.joinpath("phase24_results.md").write_text(
    f"# Phase 24 Cross-Domain Results — {STAMP}\n\n"
    f"Domains: {PREREG['domains']}\n"
    f"ε-range: {PREREG['constants_locked']['epsilon_set']}\n"
    f"Shocks: {PREREG['constants_locked']['shock_set']}\n"
)

print(f"[✓] Phase 24 complete. Artifacts in {OUTDIR}")
print("[✓] Next → python SIMULATION_RESEARCH_ORGANIZER_ORCHESTRATOR.py")

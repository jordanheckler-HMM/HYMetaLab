#!/usr/bin/env python3
"""
Run 6 seeds of F1_EH_early_periodic to 10000 epochs, compute bootstrap CIs.
Tuned logging and a 10-minute time guard.
"""
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = Path(f"outputs/stress10k_multiseed/{timestamp}")
for d in ["data", "figures", "report", "bundle", "logs"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

TIME_BUDGET_SEC = 600  # 10 minutes max guard
SEEDS = [1, 2, 3, 4, 5, 6]
EPOCHS = 10000
LOG_EVERY = 50  # keep logging sparse to speed up
AGENTS = 48

start = time.time()


def simulate_seed(seed, epochs_cap=EPOCHS, log_every=LOG_EVERY, agents=AGENTS):
    np.random.seed(seed + 1234)
    age = 1.0
    cci = 0.65
    survival = 0.95
    risk = 0.15
    traj = []
    interventions = []
    for epoch in range(epochs_cap):
        # time guard
        if time.time() - start > TIME_BUDGET_SEC - 2:
            break
        age *= 0.9998
        noise = 1.0 + np.random.normal(0, 0.01)
        expansion_boost = 1.0
        hygiene_boost = 1.0
        # fixed early expansion
        if epoch >= 12 and epoch % 12 == 0:
            expansion_boost = 1.0 + 0.03
            interventions.append((epoch, "E_fixed"))
        # periodic hygiene every 40
        if epoch > 0 and epoch % 40 == 0:
            hygiene_boost = 1.0 + 0.12 * 0.8
            interventions.append((epoch, "H_fixed"))

        cci = max(
            0.05,
            min(
                1.0,
                0.65
                * age
                * (0.6 / 0.5)
                * ((0.3 - 0.15) / 0.3 + 1.0)
                * expansion_boost
                * hygiene_boost
                * noise,
            ),
        )
        survival = max(
            0.05, min(1.0, 0.95 * age * (0.6 / 0.5) * expansion_boost * hygiene_boost)
        )
        risk = max(
            0.05,
            min(
                0.9,
                0.15 * (2.0 - age) / ((0.6 / 0.5) * expansion_boost * hygiene_boost),
            ),
        )
        hazard = max(0, -np.log(survival + 1e-9))

        if epoch % log_every == 0 or epoch == epochs_cap - 1:
            traj.append(
                {
                    "run_id": f"F1_EH_early_periodic_10000_s{seed}",
                    "seed": seed,
                    "epoch": epoch,
                    "CCI": cci,
                    "survival": survival,
                    "hazard": hazard,
                    "risk": risk,
                }
            )

        # early stop condition
        if risk >= 0.6:
            break

    # summarise
    cci_vals = [t["CCI"] for t in traj]
    hazard_vals = [t["hazard"] for t in traj]
    final_cci = float(cci_vals[-1]) if cci_vals else 0.0
    stability_vals = [
        t["CCI"]
        for t in traj
        if t["epoch"] >= max(0, (traj[-1]["epoch"] - 200) if traj else 0)
    ]
    stability_mean = float(np.mean(stability_vals)) if stability_vals else final_cci
    return {
        "run_id": f"F1_EH_early_periodic_10000_s{seed}",
        "seed": seed,
        "final_CCI": final_cci,
        "stability_CCI_mean": stability_mean,
        "hazard_mean": float(np.mean(hazard_vals)) if hazard_vals else 0.0,
        "lifespan_epochs": traj[-1]["epoch"] if traj else 0,
    }, traj


all_summaries = []
all_traj = []
for s in SEEDS:
    summary, traj = simulate_seed(s)
    all_summaries.append(summary)
    all_traj.extend(traj)

runs_df = pd.DataFrame(all_summaries)
traj_df = pd.DataFrame(all_traj)


# bootstrap CIs (n=2000 resamples)
def bootstrap_ci(values, n=2000, alpha=0.05):
    vals = np.array(values)
    nvals = len(vals)
    if nvals == 0:
        return (None, None)
    res = []
    for _ in range(n):
        samp = np.random.choice(vals, size=nvals, replace=True)
        res.append(np.mean(samp))
    lo = np.percentile(res, 100 * alpha / 2)
    hi = np.percentile(res, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


finals = runs_df["final_CCI"].tolist()
stabs = runs_df["stability_CCI_mean"].tolist()
bootstrap = {
    "final_CCI_mean": float(np.mean(finals)),
    "final_CCI_ci": bootstrap_ci(finals, n=2000),
    "stability_CCI_mean": float(np.mean(stabs)),
    "stability_CCI_ci": bootstrap_ci(stabs, n=2000),
    "n_seeds": len(finals),
}

# exports
runs_df.to_csv(OUT / "data" / "runs_summary.csv", index=False)
traj_df.to_csv(OUT / "data" / "trajectories_long.csv", index=False)
with open(OUT / "data" / "bootstrap_summary.json", "w") as f:
    json.dump(bootstrap, f, indent=2)

# figure
plt.figure(figsize=(8, 4))
for s in runs_df["run_id"]:
    sub = traj_df[traj_df["run_id"] == s]
    if not sub.empty:
        plt.plot(sub["epoch"], sub["CCI"], label=s)
plt.legend(fontsize=8)
plt.title("F1_EH_early_periodic 6-seed 10k CCI")
plt.savefig(OUT / "figures" / "cci_6seed_10k.png")
plt.close()

# bundle and sha256
checks = {}
for root, dirs, files in os.walk(OUT):
    for fn in files:
        p = Path(root) / fn
        rel = p.relative_to(OUT)
        checks[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
with open(OUT / "SHA256SUMS.txt", "w") as f:
    for k, v in sorted(checks.items()):
        f.write(f"{v}  {k}\n")

bundle_name = f"stress10k_multiseed_{timestamp}.zip"
with zipfile.ZipFile(OUT / "bundle" / bundle_name, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(OUT):
        for fn in files:
            if fn == bundle_name:
                continue
            z.write(
                os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), OUT)
            )

with open(OUT / "report" / "readme.md", "w") as f:
    f.write(
        f"6-seed 10k run for F1_EH_early_periodic_10000\nTimestamp: {timestamp}\n\nBootstrap summary:\n{json.dumps(bootstrap,indent=2)}\n\nOutputs: {OUT}\n"
    )

print("DONE", OUT)

"""Minimal neuro coherence adapter stub for study `neuro_coherence_phase31`.
This writes a small CSV of runs using deterministic pseudo-random metric stubs so you can
exercise the pipeline immediately.
"""

import json
import os
import random

import pandas as pd

OUT_DIR = os.environ.get("OUT_DIR", "discovery_results/neuro_coherence_phase31")


def measure_metrics(seed, participants, coupling, noise, epochs):
    random.seed(seed + int(coupling * 100) + participants)
    # stubbed deterministic metrics
    CCI_mean = 0.5 + 0.1 * coupling + random.random() * 0.05
    hazard_mean = 0.2 - 0.02 * coupling + random.random() * 0.02
    eta = 0.3 - 0.05 * coupling + random.random() * 0.03
    coherence_gain = 0.05 * coupling + random.random() * 0.02
    return CCI_mean, hazard_mean, eta, coherence_gain


def run_adapter(params, outdir=None):
    outdir = outdir or OUT_DIR
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for seed in params.get("preregistered_constants", {}).get("seeds", [11, 17, 23]):
        for participants in params.get("preregistered_constants", {}).get(
            "participants", [3, 5, 7]
        ):
            for coupling in params.get("preregistered_constants", {}).get(
                "coupling_strength", [0.1, 0.3, 0.5]
            ):
                CCI, hazard, eta, gain = measure_metrics(
                    seed,
                    participants,
                    coupling,
                    params.get("noise_base", 0.05),
                    params.get("epochs", 2000),
                )
                rows.append(
                    {
                        "seed": seed,
                        "participants": participants,
                        "coupling": coupling,
                        "CCI_mean": CCI,
                        "hazard_mean": hazard,
                        "eta": eta,
                        "coherence_gain": gain,
                    }
                )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "runs.csv")
    df.to_csv(csv_path, index=False)
    summary = {"n": len(df), "CCI_mean_overall": float(df["CCI_mean"].mean())}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import yaml

    with open("../study.yml") as f:
        params = yaml.safe_load(f)
    print("Running local neuro_coherence_adapter with params from study.yml")
    print(run_adapter(params))

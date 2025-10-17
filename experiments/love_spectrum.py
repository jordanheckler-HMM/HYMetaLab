#!/usr/bin/env python3
"""Love Spectrum Dynamics — Phase script

Runs a simple simulation that integrates different "types of love" into entropy/coherence shifts
and writes a CSV + a small figure into discovery_results/love_spectrum_<timestamp>/.

This script is robust: if `simulation_core` is not available it uses lightweight fallbacks.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from simulation_core import coherence_index, export_results, map_entropy
except Exception:
    # minimal fallbacks
    def map_entropy(source="local"):
        return 1.0

    def coherence_index(source="local"):
        return 0.5

    def export_results(obj, filename="results.json"):
        # write JSON or CSV depending on obj
        p = Path(filename)
        if isinstance(obj, dict):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(str(obj))
        return p


def love_index(love):
    c = love["calibration"]
    coh = love["coherence"]
    e = love["emergence"]
    n = love["noise"]
    return (c * coh * e) / n


def run(epochs=1000, dt=0.01, out_root=Path("discovery_results")):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = out_root / f"love_spectrum_{stamp}"
    figs = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    entropy = map_entropy(source="cosmic_local")
    coherence = coherence_index(source="human_network")

    love_types = {
        "Agape": {
            "calibration": 0.9,
            "coherence": 0.95,
            "emergence": 0.8,
            "noise": 0.2,
        },
        "Eros": {
            "calibration": 0.6,
            "coherence": 0.7,
            "emergence": 0.95,
            "noise": 0.45,
        },
        "Philia": {
            "calibration": 0.8,
            "coherence": 0.85,
            "emergence": 0.7,
            "noise": 0.25,
        },
        "Pragma": {
            "calibration": 0.95,
            "coherence": 0.9,
            "emergence": 0.65,
            "noise": 0.3,
        },
        "Storge": {
            "calibration": 0.85,
            "coherence": 0.8,
            "emergence": 0.6,
            "noise": 0.25,
        },
        "Ludus": {
            "calibration": 0.7,
            "coherence": 0.75,
            "emergence": 0.9,
            "noise": 0.4,
        },
    }

    love_scores = {k: love_index(v) for k, v in love_types.items()}
    max_score = max(love_scores.values())
    love_scores = {k: v / max_score for k, v in love_scores.items()}

    results = []
    for epoch in range(epochs):
        for lt, val in love_scores.items():
            entropy_shift = float(np.exp(-val * 0.5))
            coherence_shift = float(val * 0.7)
            openness_effect = float(coherence_shift - entropy_shift)
            results.append(
                {
                    "epoch": int(epoch),
                    "love_type": lt,
                    "entropy_delta": entropy_shift,
                    "coherence_delta": coherence_shift,
                    "openness_effect": openness_effect,
                }
            )

    # Aggregate
    agg = {}
    for lt in love_scores.keys():
        subset = [r for r in results if r["love_type"] == lt]
        mean_entropy = float(np.mean([r["entropy_delta"] for r in subset]))
        mean_coherence = float(np.mean([r["coherence_delta"] for r in subset]))
        mean_openness = float(np.mean([r["openness_effect"] for r in subset]))
        agg[lt] = {
            "mean_entropy_reduction": mean_entropy,
            "mean_coherence_boost": mean_coherence,
            "mean_openness_gain": mean_openness,
        }

    # write CSV of aggregated metrics
    csv_path = outdir / "love_spectrum_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "love_type",
                "mean_entropy_reduction",
                "mean_coherence_boost",
                "mean_openness_gain",
            ]
        )
        for lt, m in agg.items():
            w.writerow(
                [
                    lt,
                    m["mean_entropy_reduction"],
                    m["mean_coherence_boost"],
                    m["mean_openness_gain"],
                ]
            )

    # Save trajectories (thin sample) as CSV
    traj_path = outdir / "love_spectrum_trajectories.csv"
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "love_type",
                "entropy_delta",
                "coherence_delta",
                "openness_effect",
            ]
        )
        for r in results[:: max(1, int(len(results) / 1000))]:
            w.writerow(
                [
                    r["epoch"],
                    r["love_type"],
                    r["entropy_delta"],
                    r["coherence_delta"],
                    r["openness_effect"],
                ]
            )

    # Small figure: mean openness gain per love type
    labels = list(agg.keys())
    vals = [agg[k]["mean_openness_gain"] for k in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.xticks(rotation=45)
    plt.title("Mean Openness Gain by Love Type")
    plt.ylabel("Mean openness gain")
    plt.tight_layout()
    figp = figs / "love_openness_gain.png"
    plt.savefig(figp, dpi=200)
    plt.close()

    # Export aggregated JSON
    json_path = outdir / "love_spectrum_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # Use export_results if available
    try:
        export_results(agg, filename=str(outdir / "love_spectrum_summary_export.json"))
    except Exception:
        pass

    print("Love Spectrum Dynamics complete.")
    print("Outputs written to", outdir)
    return {
        "outdir": str(outdir),
        "summary_csv": str(csv_path),
        "trajectories_csv": str(traj_path),
        "summary_json": str(json_path),
        "figure": str(figp),
    }


if __name__ == "__main__":
    out = run(epochs=1000, dt=0.01)
    print(out)

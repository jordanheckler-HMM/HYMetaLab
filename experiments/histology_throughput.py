"""Throughput & scheduling optimizer demo for histology labs.

This synthetic demo simulates a simple pipeline: section -> stain -> scan.
It compares three policies: baseline, smaller batch size, and adding one technician.
Writes CSV/JSON and a plot to an outputs folder and returns the folder path.
"""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

from sim.io_utils import write_run_manifest

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def simulate_pipeline(
    n_days=30, n_techs=3, stainers=1, scanner=1, batch_size=16, rework_rate=0.05, seed=1
):
    random.seed(seed)
    # per-step durations in minutes
    section_time = 10
    stain_time = 45
    scan_time = 2

    results = []
    for day in range(n_days):
        # simple model: each tech can process 8 hours = 480 minutes per day
        capacity_minutes = n_techs * 480
        # approximate total minutes required per slide = section + stain/batch amortized + scan
        # stain amortized per slide = stain_time * (1 + rework_rate)  (approx)
        required_per_slide = section_time + stain_time * 1.0 + scan_time
        # assume additional overhead per batch: 5 minutes
        overhead_per_batch = 5
        # estimate slides processed
        slides = 0
        minutes_used = 0
        batches = 0
        while True:
            # check if next slide fits
            next_minutes = section_time + scan_time
            # if starting a new batch, include per-batch overhead and full stain time
            if slides % batch_size == 0:
                next_minutes += stain_time + overhead_per_batch
            else:
                next_minutes += 0  # already in batch

            if minutes_used + next_minutes > capacity_minutes:
                break
            minutes_used += next_minutes
            slides += 1
            if slides % batch_size == 0:
                batches += 1

        # account for rework slides
        rework_slides = int(round(slides * rework_rate))
        total_slides = slides + rework_slides

        results.append(
            {
                "day": day,
                "slides": slides,
                "rework_slides": rework_slides,
                "total_slides": total_slides,
                "minutes_used": minutes_used,
            }
        )

    return results


def run_demo(output_dir: Path = None, seed: int = 1):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/histology_throughput") / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"demo": "throughput", "seed": seed}
    write_run_manifest(str(output_dir), config, seed)

    policies = {
        "baseline": {"n_techs": 3, "batch_size": 16},
        "smaller_batch": {"n_techs": 3, "batch_size": 8},
        "plus_one_tech": {"n_techs": 4, "batch_size": 16},
    }

    summary = {}
    for name, params in policies.items():
        res = simulate_pipeline(
            n_days=30,
            n_techs=params["n_techs"],
            batch_size=params["batch_size"],
            seed=seed,
        )
        slides = [r["slides"] for r in res]
        total = sum(slides)
        median = sorted(slides)[len(slides) // 2]
        p95 = sorted(slides)[int(len(slides) * 0.95) - 1]
        summary[name] = {
            "total_slides_month": total,
            "median_slides_day": median,
            "p95_slides_day": p95,
        }
        # write per-policy CSV
        csvp = output_dir / f"{name}_daily.csv"
        with open(csvp, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=res[0].keys())
            writer.writeheader()
            for row in res:
                writer.writerow(row)

    # write summary JSON
    sj = output_dir / "summary.json"
    with open(sj, "w") as f:
        json.dump(summary, f, indent=2)

    # simple plot
    if plt is not None:
        plt.figure()
        for name, params in policies.items():
            res = simulate_pipeline(
                n_days=30,
                n_techs=params["n_techs"],
                batch_size=params["batch_size"],
                seed=seed,
            )
            slides = [r["slides"] for r in res]
            plt.plot(range(len(slides)), slides, label=name)
        plt.xlabel("day")
        plt.ylabel("slides processed")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "throughput_comparison.png")

    return str(output_dir)


if __name__ == "__main__":
    print("Run demo, output dir:", run_demo())

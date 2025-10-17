"""Rework vs QC tradeoff demo for histology labs."""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

from sim.io_utils import write_run_manifest


def simulate_rework_policy(
    n_slides=1000,
    qc_time_per_slide=0.0833,
    rework_rate_no_qc=0.08,
    rework_rate_with_qc=0.02,
    seed=1,
):
    # qc_time in hours (0.0833 = 5 minutes)
    random.seed(seed)
    # without QC
    rework_no_qc = int(round(n_slides * rework_rate_no_qc))
    time_no_qc = n_slides * (qc_time_per_slide) + rework_no_qc * (
        qc_time_per_slide + 0.5
    )  # assume rework costs extra 30 min

    # with QC
    rework_with_qc = int(round(n_slides * rework_rate_with_qc))
    time_with_qc = n_slides * (qc_time_per_slide) + rework_with_qc * (
        qc_time_per_slide + 0.5
    )

    return {
        "n_slides": n_slides,
        "rework_no_qc": rework_no_qc,
        "time_no_qc_hours": time_no_qc,
        "rework_with_qc": rework_with_qc,
        "time_with_qc_hours": time_with_qc,
    }


def run_demo(output_dir: Path = None, seed: int = 1):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/histology_rework_qc") / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"demo": "rework_qc", "seed": seed}
    write_run_manifest(str(output_dir), config, seed)

    res = simulate_rework_policy(seed=seed)
    with open(output_dir / "rework_qc_summary.json", "w") as f:
        json.dump(res, f, indent=2)

    # small CSV
    with open(output_dir / "rework_qc.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=res.keys())
        writer.writeheader()
        writer.writerow(res)

    return str(output_dir)


if __name__ == "__main__":
    print("Run demo, output dir:", run_demo())

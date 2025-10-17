"""Simulate lab pipeline throughput, QC and rework policies.

Writes per-policy CSVs and a summary JSON comparing throughput and defect rates.
"""

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.io_utils import write_run_manifest


def simulate_pipeline(
    n_items=200,
    staff=5,
    qc_threshold=0.95,
    rework_policy="retry",
    seed=0,
    output_dir=None,
):
    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/lab_pipeline") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Each item requires 1..3 processing steps; each step has base quality probability
    items = [
        {"id": i, "steps": random.randint(1, 3), "quality": 0.0, "done": False}
        for i in range(n_items)
    ]

    staff_capacity = staff
    time = 0
    records = []
    completed = 0

    while completed < n_items and time < 1000:
        # each staff processes one step from the next unfinished item
        for s in range(staff_capacity):
            # pick next item that is not done
            pending = [it for it in items if not it["done"]]
            if not pending:
                break
            item = pending[0]
            # process one step: increase quality probabilistically
            step_quality_gain = random.random() * 0.6
            item["quality"] += step_quality_gain
            # decrement steps
            item["steps"] -= 1
            if item["steps"] <= 0:
                # QC check
                passed = item["quality"] >= qc_threshold
                if passed:
                    item["done"] = True
                    completed += 1
                else:
                    if rework_policy == "retry":
                        # add extra step to rework
                        item["steps"] = 1
                    else:
                        # discard
                        item["done"] = True
                        completed += 1

        records.append(
            {"tick": time, "completed": completed, "pending": n_items - completed}
        )
        time += 1

    # write CSV
    csvp = out_root / "pipeline_timeseries.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["tick", "completed", "pending"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    summary = {
        "n_items": n_items,
        "staff": staff,
        "qc_threshold": qc_threshold,
        "rework_policy": rework_policy,
        "time_to_complete": time,
        "final_completed": completed,
        "seed": seed,
    }
    with open(out_root / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        str(out_root),
        {
            "experiment": "lab_pipeline",
            "n_items": n_items,
            "staff": staff,
            "qc_threshold": qc_threshold,
            "rework_policy": rework_policy,
        },
        seed,
    )
    print("Lab pipeline demo complete. Outputs in", str(out_root))
    return out_root


if __name__ == "__main__":
    simulate_pipeline(seed=1, output_dir="outputs/lab_pipeline/demo_run")

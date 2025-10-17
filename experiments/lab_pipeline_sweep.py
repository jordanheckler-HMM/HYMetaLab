"""Sweep staff and QC threshold for lab pipeline demo across rework policies.
Writes per-run CSV and aggregated summary + plots.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lab_pipeline_demo import simulate_pipeline

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def main():
    staffs = [3, 5, 8]
    qc_thresholds = [0.90, 0.95, 0.98]
    rework_policies = ["retry", "discard"]
    seeds = list(range(5))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs/lab_pipeline_sweeps") / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for staff in staffs:
        for qc in qc_thresholds:
            for policy in rework_policies:
                for seed in seeds:
                    od = out_root / f"staff_{staff}_qc_{qc}_policy_{policy}_seed_{seed}"
                    od.mkdir(parents=True, exist_ok=True)
                    simulate_pipeline(
                        n_items=100,
                        staff=staff,
                        qc_threshold=qc,
                        rework_policy=policy,
                        seed=seed,
                        output_dir=od,
                    )
                    try:
                        with open(od / "pipeline_summary.json") as f:
                            summ = json.load(f)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "staff": staff,
                            "qc_threshold": qc,
                            "policy": policy,
                            "seed": seed,
                            "time_to_complete": summ.get("time_to_complete"),
                            "final_completed": summ.get("final_completed"),
                        }
                    )

    csvp = out_root / "pipeline_sweep_results.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # aggregate mean time_to_complete per cell
    summary = {}
    for staff in staffs:
        for qc in qc_thresholds:
            for policy in rework_policies:
                cell = [
                    r
                    for r in rows
                    if r["staff"] == staff
                    and r["qc_threshold"] == qc
                    and r["policy"] == policy
                ]
                if not cell:
                    continue
                mean_time = sum(
                    r["time_to_complete"]
                    for r in cell
                    if r["time_to_complete"] is not None
                ) / len(cell)
                summary[f"staff_{staff}_qc_{qc}_policy_{policy}"] = {
                    "mean_time_to_complete": mean_time,
                    "n": len(cell),
                }

    with open(out_root / "pipeline_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # optional plotting: compare mean_time for retries vs discard across staff and qc
    if plt is not None:
        qc = 0.95
        xs = staffs
        ys_retry = [
            summary[f"staff_{s}_qc_{qc}_policy_retry"]["mean_time_to_complete"]
            for s in xs
        ]
        ys_discard = [
            summary[f"staff_{s}_qc_{qc}_policy_discard"]["mean_time_to_complete"]
            for s in xs
        ]
        plt.figure()
        plt.plot(xs, ys_retry, marker="o", label="retry")
        plt.plot(xs, ys_discard, marker="o", label="discard")
        plt.xlabel("staff")
        plt.ylabel("mean time to complete")
        plt.title("Pipeline: staff vs mean completion time (qc=0.95)")
        plt.legend()
        plt.savefig(out_root / "pipeline_staff_compare_qc95.png")
        try:
            plt.close("all")
        except Exception:
            pass

    print("Lab pipeline sweep complete. Outputs in", str(out_root))


if __name__ == "__main__":
    main()

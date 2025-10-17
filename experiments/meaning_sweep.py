"""Sweep social_weight and innovation_rate for the meaning experiment.

Outputs CSV of dominant-goal fractions per run and aggregated summary JSON + optional plot.
"""

import csv
import json

# allow running this script directly from the repo root (fix import path)
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.meaning_experiment import run_meaning_experiment

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def main():
    social_weights = [0.2, 0.6, 0.9]
    innovation_rates = [0.001, 0.01, 0.05]
    seeds = list(range(50))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs/meaning_sweeps") / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sw in social_weights:
        for ir in innovation_rates:
            for seed in seeds:
                od = out_root / f"sw_{sw}_ir_{ir}_seed_{seed}"
                od.mkdir(parents=True, exist_ok=True)
                # run experiment with moderate size for speed
                run_meaning_experiment(
                    n_agents=300,
                    n_steps=200,
                    social_weight=sw,
                    innovation_rate=ir,
                    seed=seed,
                    output_dir=od,
                )
                # read summary
                summp = od / "meaning_summary.json"
                try:
                    with open(summp) as f:
                        summ = json.load(f)
                except Exception:
                    continue
                final = summ.get("final_fractions", {})
                # dominant goal and fraction
                if final:
                    dom_goal = max(final.keys(), key=lambda k: final[k])
                    dom_frac = float(final[dom_goal])
                else:
                    dom_goal = None
                    dom_frac = None
                rows.append(
                    {
                        "social_weight": sw,
                        "innovation_rate": ir,
                        "seed": seed,
                        "dominant_goal": dom_goal,
                        "dominant_fraction": dom_frac,
                    }
                )

    # write CSV
    csvp = out_root / "meaning_sweep_results.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # aggregate
    summary = {}
    for sw in social_weights:
        for ir in innovation_rates:
            cell = [
                r
                for r in rows
                if r["social_weight"] == sw and r["innovation_rate"] == ir
            ]
            if not cell:
                continue
            mean_dom_frac = sum(
                r["dominant_fraction"]
                for r in cell
                if r["dominant_fraction"] is not None
            ) / len(cell)
            # mode of dominant_goal
            from collections import Counter

            mode_goal = Counter(r["dominant_goal"] for r in cell).most_common(1)[0][0]
            summary[f"sw_{sw}_ir_{ir}"] = {
                "mean_dominant_fraction": mean_dom_frac,
                "mode_dominant_goal": mode_goal,
                "n": len(cell),
            }

    with open(out_root / "meaning_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # plot heatmap of mean_dominant_fraction
    if plt is not None:
        xs = innovation_rates
        ys = social_weights
        Z = [
            [summary[f"sw_{y}_ir_{x}"]["mean_dominant_fraction"] for x in xs]
            for y in ys
        ]
        plt.figure()
        plt.imshow(Z, origin="lower", cmap="viridis", aspect="auto")
        plt.colorbar(label="mean dominant fraction")
        plt.xticks(range(len(xs)), xs)
        plt.yticks(range(len(ys)), ys)
        plt.xlabel("innovation_rate")
        plt.ylabel("social_weight")
        plt.title("Meaning sweep: mean dominant fraction")
        plt.savefig(out_root / "meaning_sweep_heatmap.png")
        try:
            plt.close("all")
        except Exception:
            pass

    print("Meaning sweep complete. Outputs in", str(out_root))


if __name__ == "__main__":
    main()

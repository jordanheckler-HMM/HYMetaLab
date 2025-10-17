"""Sweep social_weight and population size for the goal externalities experiment.
Writes per-run CSV and aggregated summary + heatmap.
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

from experiments.goal_externalities import run_goal_externalities

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def main():
    social_weights = [0.2, 0.5, 0.8]
    pop_sizes = [100, 300]
    seeds = list(range(10))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs/goal_externalities_sweeps") / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sw in social_weights:
        for n in pop_sizes:
            for seed in seeds:
                od = out_root / f"sw_{sw}_n_{n}_seed_{seed}"
                od.mkdir(parents=True, exist_ok=True)
                run_goal_externalities(
                    n_agents=n, n_steps=200, social_weight=sw, seed=seed, output_dir=od
                )
                try:
                    with open(od / "goal_summary.json") as f:
                        summ = json.load(f)
                except Exception:
                    continue
                rows.append(
                    {
                        "social_weight": sw,
                        "n_agents": n,
                        "seed": seed,
                        "final_mean_wealth": summ.get("final_mean_wealth"),
                        "final_gini": summ.get("final_gini"),
                    }
                )

    csvp = out_root / "goal_externalities_sweep_results.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # aggregate mean gini per cell
    summary = {}
    for sw in social_weights:
        for n in pop_sizes:
            cell = [r for r in rows if r["social_weight"] == sw and r["n_agents"] == n]
            if not cell:
                continue
            mean_gini = sum(
                r["final_gini"] for r in cell if r["final_gini"] is not None
            ) / len(cell)
            summary[f"sw_{sw}_n_{n}"] = {"mean_gini": mean_gini, "n": len(cell)}

    with open(out_root / "goal_externalities_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if plt is not None:
        xs = pop_sizes
        ys = social_weights
        Z = [[summary[f"sw_{y}_n_{x}"]["mean_gini"] for x in xs] for y in ys]
        plt.figure()
        plt.imshow(Z, origin="lower", cmap="coolwarm", aspect="auto")
        plt.colorbar(label="mean final Gini")
        plt.xticks(range(len(xs)), xs)
        plt.yticks(range(len(ys)), ys)
        plt.xlabel("population size")
        plt.ylabel("social_weight")
        plt.title("Goal externalities sweep: mean final Gini")
        plt.savefig(out_root / "goal_externalities_gini_heatmap.png")
        try:
            plt.close("all")
        except Exception:
            pass

    print("Goal externalities sweep complete. Outputs in", str(out_root))


if __name__ == "__main__":
    main()

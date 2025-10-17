"""Sweep shock severity and regrowth rate for the shock resilience experiment.
Produces CSV of per-run final alive fraction and an aggregated summary JSON + heatmap.
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

from experiments.shock_resilience import run_shock_experiment

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def main():
    severities = [0.2, 0.5, 0.8]
    regrowths = [0.05, 0.1, 0.2]
    seeds = list(range(10))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs/shock_sweeps") / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in severities:
        for r in regrowths:
            for seed in seeds:
                od = out_root / f"s_{s}_g_{r}_seed_{seed}"
                od.mkdir(parents=True, exist_ok=True)
                # use smaller population for speed
                run_shock_experiment(
                    n_agents=100,
                    n_steps=150,
                    shock_time=30,
                    shock_severity=s,
                    seed=seed,
                    output_dir=od,
                )
                try:
                    with open(od / "shock_summary.json") as f:
                        summ = json.load(f)
                except Exception:
                    continue
                rows.append(
                    {
                        "shock_severity": s,
                        "regrowth": r,
                        "seed": seed,
                        "final_alive_fraction": summ.get("final_alive_fraction"),
                    }
                )

    # write CSV
    csvp = out_root / "shock_sweep_results.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # aggregate mean per cell
    summary = {}
    for s in severities:
        for g in regrowths:
            cell = [r for r in rows if r["shock_severity"] == s and r["regrowth"] == g]
            if not cell:
                continue
            mean_alive = sum(
                r["final_alive_fraction"]
                for r in cell
                if r["final_alive_fraction"] is not None
            ) / len(cell)
            summary[f"s_{s}_g_{g}"] = {
                "mean_final_alive_fraction": mean_alive,
                "n": len(cell),
            }

    with open(out_root / "shock_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # optional heatmap
    if plt is not None:
        xs = regrowths
        ys = severities
        Z = [
            [summary[f"s_{y}_g_{x}"]["mean_final_alive_fraction"] for x in xs]
            for y in ys
        ]
        plt.figure()
        plt.imshow(Z, origin="lower", cmap="plasma", aspect="auto")
        plt.colorbar(label="mean final alive fraction")
        plt.xticks(range(len(xs)), xs)
        plt.yticks(range(len(ys)), ys)
        plt.xlabel("regrowth")
        plt.ylabel("shock_severity")
        plt.title("Shock sweep: mean final alive fraction")
        plt.savefig(out_root / "shock_sweep_heatmap.png")
        try:
            plt.close("all")
        except Exception:
            pass

    print("Shock sweep complete. Outputs in", str(out_root))


if __name__ == "__main__":
    main()

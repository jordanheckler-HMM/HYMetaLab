"""Run parameter sweeps for belief persistence and throughput demos.

Creates outputs in outputs/sweeps/<timestamp>/ with CSVs, JSON summaries, and plots.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from experiments.belief_experiment import run_belief_simulation
from experiments.histology_throughput import simulate_pipeline

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def run_belief_sweep(output_dir: Path, agents_list, steps_list, seeds):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_agents in agents_list:
        for n_steps in steps_list:
            vals = []
            for seed in seeds:
                od = output_dir / f"belief_agents{n_agents}_steps{n_steps}_seed{seed}"
                od.mkdir(parents=True, exist_ok=True)
                # run experiment
                run_belief_simulation(
                    n_agents=n_agents, n_steps=n_steps, seed=seed, output_dir=od
                )
                # read summary JSON
                summary_path = od / f"belief_summary_seed_{seed}.json"
                if not summary_path.exists():
                    # fallback: search for any json in folder
                    js = list(od.glob("*.json"))
                    if js:
                        summary_path = js[0]
                with open(summary_path) as f:
                    data = json.load(f)
                history = data.get("history", [])
                final_mean = history[-1]["mean_belief"] if history else None
                vals.append(final_mean)
                rows.append(
                    {
                        "experiment": "belief",
                        "n_agents": n_agents,
                        "n_steps": n_steps,
                        "seed": seed,
                        "final_mean_belief": final_mean,
                    }
                )
                # close any figures created by the belief experiment to avoid memory growth
                try:
                    if plt is not None:
                        plt.close("all")
                except Exception:
                    pass

    # write CSV
    csv_path = output_dir / "belief_sweep.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary JSON
    summary = {}
    for n_agents in agents_list:
        for n_steps in steps_list:
            vals = [
                r["final_mean_belief"]
                for r in rows
                if r["n_agents"] == n_agents and r["n_steps"] == n_steps
            ]
            summary[f"agents_{n_agents}_steps_{n_steps}"] = {
                "mean_final_mean_belief": (
                    float(sum(vals) / len(vals)) if vals else None
                ),
                "n": len(vals),
            }

    with open(output_dir / "belief_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # simple plot
    if plt is not None:
        plt.figure()
        for n_agents in agents_list:
            ys = []
            xs = steps_list
            for n_steps in steps_list:
                key = f"agents_{n_agents}_steps_{n_steps}"
                ys.append(summary[key]["mean_final_mean_belief"])
            plt.plot(xs, ys, label=f"agents={n_agents}")
        plt.xlabel("n_steps")
        plt.ylabel("mean final belief")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "belief_sweep.png")
        try:
            plt.close("all")
        except Exception:
            pass

    return csv_path


def run_throughput_sweep(output_dir: Path, techs_list, batch_list, seeds):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_techs in techs_list:
        for batch in batch_list:
            vals = []
            for seed in seeds:
                res = simulate_pipeline(
                    n_days=30, n_techs=n_techs, batch_size=batch, seed=seed
                )
                slides = [r["slides"] for r in res]
                total = sum(slides)
                vals.append(total)
                rows.append(
                    {
                        "experiment": "throughput",
                        "n_techs": n_techs,
                        "batch_size": batch,
                        "seed": seed,
                        "total_slides_month": total,
                    }
                )

    csv_path = output_dir / "throughput_sweep.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    summary = {}
    for n_techs in techs_list:
        for batch in batch_list:
            vals = [
                r["total_slides_month"]
                for r in rows
                if r["n_techs"] == n_techs and r["batch_size"] == batch
            ]
            summary[f"techs_{n_techs}_batch_{batch}"] = {
                "mean_total_slides": float(sum(vals) / len(vals)) if vals else None,
                "n": len(vals),
            }

    with open(output_dir / "throughput_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if plt is not None:
        plt.figure()
        for n_techs in techs_list:
            xs = batch_list
            ys = [
                summary[f"techs_{n_techs}_batch_{b}"]["mean_total_slides"]
                for b in batch_list
            ]
            plt.plot(xs, ys, label=f"techs={n_techs}")
        plt.xlabel("batch_size")
        plt.ylabel("mean monthly slides")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "throughput_sweep.png")

    return csv_path


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs/sweeps") / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    # simple 3x3 grids
    belief_agents = [50, 100, 200]
    belief_steps = [10, 25, 50]
    throughput_techs = [2, 3, 4]
    throughput_batches = [8, 16, 32]

    # seeds: 100 seeds
    seeds = list(range(100))

    print("Running belief sweep...")
    run_belief_sweep(out / "belief", belief_agents, belief_steps, seeds)
    print("Running throughput sweep...")
    run_throughput_sweep(
        out / "throughput", throughput_techs, throughput_batches, seeds
    )
    print("Sweeps complete. Outputs in", str(out))


if __name__ == "__main__":
    main()

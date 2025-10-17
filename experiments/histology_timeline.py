"""Study timeline certainty demo for histology labs."""

import json
import random
from datetime import datetime
from pathlib import Path

from sim.io_utils import write_run_manifest


def simulate_study_completion(
    n_sim=1000, base_days=10, per_step_variation=2, delay_chance=0.1, seed=1
):
    random.seed(seed)
    completions = []
    for i in range(n_sim):
        days = base_days + random.gauss(0, per_step_variation)
        # occasional upstream delay
        if random.random() < delay_chance:
            days += random.uniform(1, 5)
        completions.append(max(1, days))
    return completions


def run_demo(output_dir: Path = None, seed: int = 1):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/histology_timeline") / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"demo": "timeline", "seed": seed}
    write_run_manifest(str(output_dir), config, seed)

    completions = simulate_study_completion(n_sim=1000, seed=seed)
    # compute quantiles
    completions_sorted = sorted(completions)
    q50 = completions_sorted[int(len(completions_sorted) * 0.5)]
    q90 = completions_sorted[int(len(completions_sorted) * 0.9)]
    q95 = completions_sorted[int(len(completions_sorted) * 0.95)]

    summary = {
        "n_sim": len(completions),
        "median_days": q50,
        "p90_days": q90,
        "p95_days": q95,
    }
    with open(output_dir / "timeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # save the raw completions
    with open(output_dir / "completions.json", "w") as f:
        json.dump(completions, f)

    return str(output_dir)


if __name__ == "__main__":
    print("Run demo, output dir:", run_demo())

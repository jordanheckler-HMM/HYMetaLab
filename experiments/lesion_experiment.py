import os
import random
from datetime import datetime

import numpy as np

from sim.io_utils import write_jsonl, write_run_manifest


def run_lesion(
    n_agents=50,
    ticks=50,
    seeds=[42],
    workspace_enabled_values=[True, False],
    export_base="outputs/lesion",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(export_base, f"lesion_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    results = {}
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for ws in workspace_enabled_values:
            records = []
            for t in range(ticks):
                for i in range(n_agents):
                    # synthetic performance: workspace improves true_p by a small delta
                    base_p = 0.4 + 0.3 * np.sin((i + t) * 0.05)
                    delta = 0.15 if ws else 0.0
                    true_p = float(np.clip(base_p + delta, 0.01, 0.99))
                    outcome = 1 if random.random() < true_p else 0
                    reported = float(
                        np.clip(true_p + np.random.normal(0, 0.1), 0.0, 1.0)
                    )
                    rec = {
                        "tick": t,
                        "agent_id": f"A-{i}",
                        "reported_conf": reported,
                        "outcome_reward": outcome,
                        "workspace_enabled": ws,
                    }
                    records.append(rec)

            subdir = os.path.join(run_dir, f"seed_{seed}_ws_{int(ws)}")
            os.makedirs(subdir, exist_ok=True)
            decisions_path = os.path.join(subdir, "decisions.jsonl")
            write_jsonl(decisions_path, records)
            write_run_manifest(subdir, {"seed": seed, "workspace_enabled": ws}, seed)

            # summary: average reward
            avg_reward = float(np.mean([r["outcome_reward"] for r in records]))
            results[f"seed_{seed}_ws_{ws}"] = {
                "decisions": decisions_path,
                "avg_reward": avg_reward,
            }

    return {"run_dir": run_dir, "results": results}


if __name__ == "__main__":
    out = run_lesion()
    print(out)

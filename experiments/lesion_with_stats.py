import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json

import numpy as np

from experiments.lesion_experiment import run_lesion
from experiments.plot_lesion import plot_lesion_comparison


def run_lesion_with_stats(
    n_agents=50,
    ticks=50,
    seeds=[42],
    workspace_enabled_values=[True, False],
    export_base="outputs/lesion",
):
    out = run_lesion(
        n_agents=n_agents,
        ticks=ticks,
        seeds=seeds,
        workspace_enabled_values=workspace_enabled_values,
        export_base=export_base,
    )
    run_dir = out["run_dir"]
    # aggregate avg_rewards per condition
    cond_values = {}
    for k, v in out["results"].items():
        decisions = v.get("decisions")
        with open(decisions) as f:
            lines = [json.loads(line) for line in f]
        avg = float(np.mean([l["outcome_reward"] for l in lines]))
        if "_ws_1" in decisions or "_ws_True" in decisions or "/ws_1" in decisions:
            key = "workspace_on"
        else:
            key = "workspace_off"
        cond_values.setdefault(key, []).append(avg)

    on = cond_values.get("workspace_on", [])
    off = cond_values.get("workspace_off", [])
    stats_summary = {"n_on": len(on), "n_off": len(off)}
    if on and off:
        from scipy import stats

        t_stat, p_val = stats.ttest_ind(on, off, equal_var=False)
        stats_summary.update(
            {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "mean_on": float(np.mean(on)),
                "mean_off": float(np.mean(off)),
                "diff": float(np.mean(on) - np.mean(off)),
            }
        )
        boots = []
        for _ in range(1000):
            s_on = np.random.choice(on, size=len(on), replace=True)
            s_off = np.random.choice(off, size=len(off), replace=True)
            boots.append(float(np.mean(s_on) - np.mean(s_off)))
        stats_summary["ci_lower"] = float(np.percentile(boots, 2.5))
        stats_summary["ci_upper"] = float(np.percentile(boots, 97.5))
    else:
        stats_summary["note"] = "Insufficient data for stats"

    stats_path = os.path.join(run_dir, "lesion_stats.json")
    with open(stats_path, "w") as sf:
        json.dump(stats_summary, sf, indent=2)

    try:
        plot_lesion_comparison(on, off, os.path.join(run_dir, "lesion_comparison.png"))
    except Exception:
        pass

    return {"run_dir": run_dir, "results": out["results"], "stats": stats_path}


if __name__ == "__main__":
    print(run_lesion_with_stats())

"""Simulate resilience to sudden shocks (resource loss / increased mortality).

Writes a time series CSV of mean resources and alive fraction, and a summary JSON.
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


def run_shock_experiment(
    params=None,
    outdir=None,
    n_agents=200,
    n_steps=200,
    shock_time=50,
    shock_severity=0.5,
    seed=0,
    output_dir=None,
):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        n_agents = params.get("n_agents", n_agents)
        n_steps = params.get("n_steps", n_steps)
        shock_time = params.get("shock_time", shock_time)
        shock_severity = params.get("shock_severity", shock_severity)
        seed = params.get("seed", seed)
        output_dir = params.get("output_dir", output_dir)

    # Use outdir if provided
    if outdir:
        output_dir = outdir

    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/shock_resilience") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # simple agents with resource stock and baseline consumption
    agents = [{"id": i, "resource": 1.0, "alive": True} for i in range(n_agents)]
    common_pool = 1.0 * n_agents  # shared resource pool

    time_series = []
    recovered_at = None

    for t in range(n_steps):
        # shock
        if t == shock_time:
            # reduce common pool and randomly increase mortality for some
            common_pool *= 1.0 - shock_severity
            # instantaneous extra deaths proportional to severity
            for a in agents:
                if a["alive"] and random.random() < shock_severity * 0.2:
                    a["alive"] = False

        # each alive agent draws from pool up to 1.0, but pool is limited
        per_agent_share = common_pool / max(1, sum(1 for a in agents if a["alive"]))
        for a in agents:
            if not a["alive"]:
                continue
            take = min(0.5, per_agent_share)
            a["resource"] += take
            # consumption
            a["resource"] -= 0.3
            if a["resource"] <= 0:
                a["alive"] = False

        # lightly replenish pool each step (regrowth)
        common_pool += 0.1 * n_agents

        mean_res = sum(a["resource"] for a in agents if a["alive"]) / max(
            1, sum(1 for a in agents if a["alive"])
        )
        alive_frac = sum(1 for a in agents if a["alive"]) / n_agents
        time_series.append(
            {"tick": t, "mean_resource_alive": mean_res, "alive_fraction": alive_frac}
        )

        # detect recovery: alive_frac back above 90% of pre-shock mean
        if t > shock_time:
            if recovered_at is None and alive_frac >= 0.9:
                recovered_at = t

    # write outputs
    csvp = out_root / "shock_time_series.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["tick", "mean_resource_alive", "alive_fraction"]
        )
        writer.writeheader()
        for r in time_series:
            writer.writerow(r)

    summary = {
        "n_agents": n_agents,
        "n_steps": n_steps,
        "shock_time": shock_time,
        "shock_severity": shock_severity,
        "seed": seed,
        "final_alive_fraction": time_series[-1]["alive_fraction"],
        "recovered_at": recovered_at,
    }
    with open(out_root / "shock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        str(out_root),
        {
            "experiment": "shock_resilience",
            "n_agents": n_agents,
            "n_steps": n_steps,
            "shock_time": shock_time,
            "shock_severity": shock_severity,
        },
        seed,
    )
    print("Shock resilience run complete. Outputs in", str(out_root))
    return out_root


if __name__ == "__main__":
    run_shock_experiment(seed=1, output_dir="outputs/shock_resilience/demo_run")

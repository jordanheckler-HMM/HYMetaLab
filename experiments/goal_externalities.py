"""Simulate agents with competing intrinsic goals and measure inequality and externalities.

Each agent has utility components for goals (wealth, status, knowledge, pleasure). Agents act to increase their utility; social influence nudges goal salience.
Outputs: per-tick summary CSV and final JSON with inequality and externality metrics.
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

GOALS = ["wealth", "status", "knowledge", "pleasure"]


def run_goal_externalities(
    params=None,
    outdir=None,
    n_agents=300,
    n_steps=300,
    social_weight=0.5,
    seed=0,
    output_dir=None,
):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        n_agents = params.get("n_agents", n_agents)
        n_steps = params.get("n_steps", n_steps)
        social_weight = params.get("social_weight", social_weight)
        seed = params.get("seed", seed)
        output_dir = params.get("output_dir", output_dir)

    # Use outdir if provided
    if outdir:
        output_dir = outdir

    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/goal_externalities") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # agent state: endowment (wealth), status, knowledge, happiness
    agents = []
    for i in range(n_agents):
        # intrinsic salience weights for goals (sum to 1)
        w = [random.random() for _ in GOALS]
        total = sum(w)
        salience = {g: wi / total for g, wi in zip(GOALS, w)}
        agents.append(
            {
                "id": i,
                "wealth": random.random() * 1.0,
                "status": random.random() * 1.0,
                "knowledge": random.random() * 1.0,
                "pleasure": random.random() * 1.0,
                "salience": salience,
            }
        )

    records = []

    for t in range(n_steps):
        # each tick each agent acts to increase their most salient goal by investing effort
        for a in agents:
            # social influence nudges salience toward population mean
            mean_salience = {
                g: sum(x["salience"][g] for x in agents) / n_agents for g in GOALS
            }
            for g in GOALS:
                a["salience"][g] = (1.0 - social_weight) * a["salience"][
                    g
                ] + social_weight * mean_salience[g]
            # normalize
            ssum = sum(a["salience"].values())
            for g in GOALS:
                a["salience"][g] /= max(1e-12, ssum)

            # pick goal to act on (highest salience)
            goal = max(a["salience"].keys(), key=lambda gg: a["salience"][gg])
            # effect: invest random effort proportional to salience
            effort = 0.05 * (0.5 + random.random()) * a["salience"][goal]
            a[goal] += effort
            # side effects: pursuing some goals creates negative externalities on others
            if goal == "wealth":
                # wealth pursuit increases inequality: small negative impact on others at random
                victim = random.choice(agents)
                if victim is not a:
                    victim["wealth"] -= effort * 0.2
            if goal == "status":
                # status games reduce others' status slightly
                victim = random.choice(agents)
                if victim is not a:
                    victim["status"] -= effort * 0.1
            if goal == "knowledge":
                # knowledge spillover: small positive impact to random others
                for _ in range(2):
                    peer = random.choice(agents)
                    peer["knowledge"] += effort * 0.05

        # clip negatives
        for a in agents:
            for key in ["wealth", "status", "knowledge", "pleasure"]:
                if a[key] < 0:
                    a[key] = 0.0

        # compute inequality (Gini on wealth) and mean utilities
        wealths = [a["wealth"] for a in agents]
        mean_wealth = sum(wealths) / n_agents
        # Gini coefficient
        sorted_w = sorted(wealths)
        cumw = sum((i + 1) * w for i, w in enumerate(sorted_w))
        gini = (2 * cumw) / (n_agents * sum(sorted_w) + 1e-12) - (
            n_agents + 1
        ) / n_agents

        mean_utils = {g: sum(a[g] for a in agents) / n_agents for g in GOALS}
        records.append(
            {
                "tick": t,
                "mean_wealth": mean_wealth,
                "gini_wealth": gini,
                **{f"mean_{g}": mean_utils[g] for g in GOALS},
            }
        )

    # write CSV
    csvp = out_root / "goal_timeseries.csv"
    with open(csvp, "w", newline="") as cf:
        fieldnames = ["tick", "mean_wealth", "gini_wealth"] + [
            f"mean_{g}" for g in GOALS
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    summary = {
        "n_agents": n_agents,
        "n_steps": n_steps,
        "social_weight": social_weight,
        "seed": seed,
        "final_mean_wealth": records[-1]["mean_wealth"],
        "final_gini": records[-1]["gini_wealth"],
    }
    with open(out_root / "goal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        str(out_root),
        {
            "experiment": "goal_externalities",
            "n_agents": n_agents,
            "n_steps": n_steps,
            "social_weight": social_weight,
        },
        seed,
    )
    print("Goal externalities run complete. Outputs in", str(out_root))
    return out_root


if __name__ == "__main__":
    run_goal_externalities(seed=2, output_dir="outputs/goal_externalities/demo_run")

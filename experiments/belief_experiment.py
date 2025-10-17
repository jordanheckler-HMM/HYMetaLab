"""Belief dynamics experiment: test simple Bayesian updates and social influence.

Produces per-tick population belief summaries and saves a JSON summary and a plot
if matplotlib is available.
"""

import json
import random
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.agent import Agent

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def run_belief_simulation(
    params=None,
    outdir=None,
    n_agents: int = 50,
    n_steps: int = 50,
    seed: int = 42,
    output_dir: Path = None,
):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        n_agents = params.get("n_agents", n_agents)
        n_steps = params.get("n_steps", n_steps)
        seed = params.get("seed", seed)
        output_dir = params.get("output_dir", output_dir)

    # Use outdir if provided
    if outdir:
        output_dir = outdir

    random.seed(seed)
    agents: list[Agent] = [
        Agent(agent_id=i, belief=random.random(), openness=0.1 + random.random() * 0.4)
        for i in range(n_agents)
    ]

    history = []

    for t in range(n_steps):
        # synthetic global evidence at each step: either supports proposition (value near 1)
        # or contradicts it (value near 0). We encode as likelihood_ratio for Bayesian updates.
        event = random.random()
        if event < 0.05:
            # strong supporting evidence
            likelihood_ratio = 5.0
            evidence_value = 1.0
        elif event < 0.1:
            # strong contradicting evidence
            likelihood_ratio = 0.2
            evidence_value = 0.0
        else:
            # ambiguous / weak evidence
            likelihood_ratio = 1.0
            evidence_value = 0.5

        # Everyone updates from evidence (Bayesian or delta-rule randomly assigned)
        for a in agents:
            if random.random() < 0.6:
                a.update_belief_bayesian(likelihood_ratio)
            else:
                a.update_belief_delta(evidence_value, alpha=0.1)

        # Social influence: each agent sees k random neighbors
        for a in agents:
            k = 3
            neighbors = random.sample(agents, min(k, len(agents)))
            neighbor_beliefs = [
                nb.belief for nb in neighbors if nb.agent_id != a.agent_id
            ]
            a.social_influence(neighbor_beliefs)

        mean_belief = sum(a.belief for a in agents) / len(agents)
        median_belief = sorted(a.belief for a in agents)[len(agents) // 2]
        history.append(
            {"t": t, "mean_belief": mean_belief, "median_belief": median_belief}
        )

    out = {"n_agents": n_agents, "n_steps": n_steps, "seed": seed, "history": history}

    if output_dir is None:
        output_dir = Path("outputs/belief_sim")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"belief_summary_seed_{seed}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    if plt is not None:
        plt.figure()
        xs = [h["t"] for h in history]
        ys = [h["mean_belief"] for h in history]
        plt.plot(xs, ys, label="mean belief")
        ys2 = [h["median_belief"] for h in history]
        plt.plot(xs, ys2, label="median belief")
        plt.ylim(0, 1)
        plt.xlabel("t")
        plt.ylabel("belief")
        plt.legend()
        plt_path = output_dir / f"belief_trace_seed_{seed}.png"
        plt.savefig(plt_path)

    return json_path


if __name__ == "__main__":
    run_belief_simulation()

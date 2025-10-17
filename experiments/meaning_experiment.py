"""Meaning experiment: simulate emergence and persistence of shared 'goals' (purposes).

This is a toy, model-based exploration: agents have intrinsic preferences over a small set
of goals and can adopt others via social influence. The script writes a summary JSON and
time-series CSV and returns the output dir.
"""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

from sim.io_utils import write_run_manifest


def run_meaning_experiment(
    n_agents=200,
    n_steps=100,
    goals=None,
    social_weight=0.6,
    innovation_rate=0.01,
    seed=42,
    output_dir=None,
):
    random.seed(seed)
    if goals is None:
        goals = ["wealth", "community", "knowledge", "pleasure"]

    # each agent has an intrinsic preference score for each goal (0..1)
    agents_pref = [{g: random.random() for g in goals} for _ in range(n_agents)]
    # initialize current goal to highest intrinsic preference
    current_goal = [max(pref.keys(), key=lambda k: pref[k]) for pref in agents_pref]

    # simple random network neighbors: each agent has k random neighbors
    k = max(3, min(10, n_agents // 20))
    neighbors = [
        random.sample([j for j in range(n_agents) if j != i], k)
        for i in range(n_agents)
    ]

    history = []

    for t in range(n_steps):
        # each agent considers switching based on neighbors and intrinsic preference
        new_goal = current_goal.copy()
        for i in range(n_agents):
            if random.random() < innovation_rate:
                # try a random goal innovation
                new_goal[i] = random.choice(goals)
                continue

            neigh = neighbors[i]
            # count neighbor frequency of goals
            counts = {g: 0 for g in goals}
            for nb in neigh:
                counts[current_goal[nb]] += 1
            # compute social score and intrinsic score for each goal
            best_score = -1
            best_goal = current_goal[i]
            for g in goals:
                social_score = counts[g] / len(neigh)  # 0..1
                intrinsic = agents_pref[i][g]
                score = social_weight * social_score + (1 - social_weight) * intrinsic
                if score > best_score:
                    best_score = score
                    best_goal = g
            new_goal[i] = best_goal

        current_goal = new_goal

        # record fractions
        freqs = {g: current_goal.count(g) / n_agents for g in goals}
        freqs["t"] = t
        history.append(freqs)

    # output
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/meaning") / f"run_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "n_agents": n_agents,
        "n_steps": n_steps,
        "goals": goals,
        "social_weight": social_weight,
        "innovation_rate": innovation_rate,
        "seed": seed,
    }
    write_run_manifest(str(output_dir), config, seed)

    # write history CSV
    csvp = output_dir / "meaning_history.csv"
    with open(csvp, "w", newline="") as cf:
        fieldnames = ["t"] + goals
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    # summary JSON
    final = history[-1] if history else {}
    summary = {
        "final_fractions": {g: float(final.get(g, 0)) for g in goals},
        "config": config,
    }
    with open(output_dir / "meaning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return str(output_dir)


if __name__ == "__main__":
    out = run_meaning_experiment()
    print("meaning experiment output at", out)

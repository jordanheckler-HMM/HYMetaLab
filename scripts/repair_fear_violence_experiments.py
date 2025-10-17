#!/usr/bin/env python3
"""
Repair fear-violence experiments to ensure non-zero trajectories.

This script re-runs shock and inequality experiments with properly
integrated fear-violence adapters and exports updated figures.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psych.adapters.fear_core import FearParams
from psych.adapters.fear_hooks import FearChannel


def run_shock_experiment_with_fear(
    n_agents: int = 200,
    n_steps: int = 200,
    shock_time: int = 50,
    shock_severity: float = 0.5,
    seed: int = 42,
    output_dir: Path = None,
) -> dict[str, Any]:
    """
    Run shock experiment with integrated fear-violence dynamics.

    Returns results with fear trajectories and aggression rates.
    """
    import random

    random.seed(seed)

    if output_dir is None:
        output_dir = Path(
            f"discovery_results/fear_violence/v2_shock_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fear channel
    fear_params = FearParams()
    fear_channel = FearChannel(enable=True, params=fear_params, track_history=True)

    # Initialize agents with fear-violence state
    agents = []
    for i in range(n_agents):
        agent = {
            "id": i,
            "resource": 1.0,
            "alive": True,
            "fear": 0.0,
            "aggression_event": False,
            "aggression_intensity": 0.0,
            "cci": 0.5 + random.random() * 0.3,
            "rand_aggr": random.random(),
        }
        agents.append(agent)

    common_pool = 1.0 * n_agents
    time_series = []

    for t in range(n_steps):
        # Prepare group state for fear channel
        group_state = {
            "shock_level": 0.0,
            "gini": 0.0,
            "support_level": 0.0,
            "social_ties": 0.0,
        }

        # Apply shock
        if t == shock_time:
            common_pool *= 1.0 - shock_severity
            group_state["shock_level"] = shock_severity

            # Instantaneous deaths
            for a in agents:
                if a["alive"] and random.random() < shock_severity * 0.2:
                    a["alive"] = False

        # Update fear dynamics for all agents
        for agent in agents:
            if agent["alive"]:
                agent = fear_channel.before_commit(
                    t_now=t, agent_state=agent, group_state=group_state
                )

        # Resource dynamics
        alive_agents = [a for a in agents if a["alive"]]
        if alive_agents:
            per_agent_share = common_pool / len(alive_agents)

            for agent in alive_agents:
                # Base resource gain
                agent["resource"] += per_agent_share * 0.1

                # Fear-aggression effects on resource consumption
                if agent.get("aggression_event", False):
                    agent["resource"] -= 0.05  # Aggression costs resources

                # Fear effects on resource efficiency
                fear_efficiency = 1.0 - agent["fear"] * 0.3
                agent["resource"] *= fear_efficiency

                # Death from resource depletion
                if agent["resource"] <= 0:
                    agent["alive"] = False

        # Record time series
        alive_count = sum(1 for a in agents if a["alive"])
        fear_mean = (
            np.mean([a["fear"] for a in agents if a["alive"]])
            if alive_count > 0
            else 0.0
        )
        aggression_rate = (
            np.mean([a.get("aggression_intensity", 0.0) for a in agents if a["alive"]])
            if alive_count > 0
            else 0.0
        )

        time_series.append(
            {
                "t": t,
                "alive_fraction": alive_count / n_agents,
                "fear_mean": fear_mean,
                "aggression_rate": aggression_rate,
                "common_pool": common_pool,
                "shock_applied": t == shock_time,
            }
        )

    # Get fear channel statistics
    fear_stats = fear_channel.get_logging_stats()

    # Create results
    results = {
        "experiment_type": "shock_with_fear",
        "n_agents": n_agents,
        "n_steps": n_steps,
        "shock_time": shock_time,
        "shock_severity": shock_severity,
        "seed": seed,
        "final_alive_fraction": time_series[-1]["alive_fraction"],
        "max_fear": max(ts["fear_mean"] for ts in time_series),
        "max_aggression_rate": max(ts["aggression_rate"] for ts in time_series),
        "fear_stats": fear_stats,
        "time_series": time_series,
    }

    # Save results
    with open(output_dir / "shock_fear_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create CSV
    df = pd.DataFrame(time_series)
    df.to_csv(output_dir / "shock_fear_trajectory.csv", index=False)

    return results


def run_inequality_experiment_with_fear(
    n_agents: int = 300,
    n_steps: int = 300,
    social_weight: float = 0.5,
    seed: int = 42,
    output_dir: Path = None,
) -> dict[str, Any]:
    """
    Run inequality experiment with integrated fear-violence dynamics.

    Returns results with fear trajectories and aggression rates.
    """
    import random

    random.seed(seed)

    if output_dir is None:
        output_dir = Path(
            f"discovery_results/fear_violence/v2_inequality_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fear channel
    fear_params = FearParams()
    fear_channel = FearChannel(enable=True, params=fear_params, track_history=True)

    # Initialize agents with wealth and fear
    agents = []
    for i in range(n_agents):
        agent = {
            "id": i,
            "wealth": random.random() * 1.0,
            "status": random.random() * 1.0,
            "knowledge": random.random() * 1.0,
            "pleasure": random.random() * 1.0,
            "fear": 0.0,
            "aggression_event": False,
            "aggression_intensity": 0.0,
            "cci": 0.5 + random.random() * 0.3,
            "rand_aggr": random.random(),
        }

        # Initialize salience weights
        w = [random.random() for _ in range(4)]
        total = sum(w)
        agent["salience"] = {
            goal: wi / total
            for goal, wi in zip(["wealth", "status", "knowledge", "pleasure"], w)
        }

        agents.append(agent)

    time_series = []

    for t in range(n_steps):
        # Calculate current inequality (Gini coefficient)
        wealth_values = np.array([a["wealth"] for a in agents])
        gini = compute_gini(wealth_values)

        # Prepare group state for fear channel
        group_state = {
            "shock_level": 0.0,
            "gini": gini,
            "support_level": 1.0 - gini,  # Inverse of inequality
            "social_ties": social_weight,
        }

        # Update fear dynamics for all agents
        for agent in agents:
            agent = fear_channel.before_commit(
                t_now=t, agent_state=agent, group_state=group_state
            )

        # Goal pursuit dynamics with fear-aggression effects
        for agent in agents:
            # Social influence on salience
            mean_salience = {
                g: sum(a["salience"][g] for a in agents) / n_agents
                for g in ["wealth", "status", "knowledge", "pleasure"]
            }
            for g in ["wealth", "status", "knowledge", "pleasure"]:
                agent["salience"][g] = (1.0 - social_weight) * agent["salience"][
                    g
                ] + social_weight * mean_salience[g]

            # Normalize salience
            ssum = sum(agent["salience"].values())
            for g in ["wealth", "status", "knowledge", "pleasure"]:
                agent["salience"][g] /= max(1e-12, ssum)

            # Pick goal to act on
            goal = max(agent["salience"].keys(), key=lambda gg: agent["salience"][gg])

            # Fear effects on effort
            fear_efficiency = 1.0 - agent["fear"] * 0.2
            effort = (
                0.05
                * (0.5 + random.random())
                * agent["salience"][goal]
                * fear_efficiency
            )

            # Aggression effects on goal pursuit
            if agent.get("aggression_event", False):
                effort *= (
                    1.2  # Aggression can increase effort but also create externalities
                )

            agent[goal] += effort

            # Externalities with fear-aggression effects
            if goal == "wealth":
                victim = random.choice([a for a in agents if a is not agent])
                victim["wealth"] -= effort * 0.2 * (1.0 + agent["aggression_intensity"])
            elif goal == "status":
                victim = random.choice([a for a in agents if a is not agent])
                victim["status"] -= effort * 0.1 * (1.0 + agent["aggression_intensity"])
            elif goal == "knowledge":
                for _ in range(2):
                    peer = random.choice([a for a in agents if a is not agent])
                    peer["knowledge"] += effort * 0.1  # Knowledge spillover

        # Record time series
        fear_mean = np.mean([a["fear"] for a in agents])
        aggression_rate = np.mean([a.get("aggression_intensity", 0.0) for a in agents])
        wealth_gini = compute_gini(np.array([a["wealth"] for a in agents]))

        time_series.append(
            {
                "t": t,
                "fear_mean": fear_mean,
                "aggression_rate": aggression_rate,
                "wealth_gini": wealth_gini,
                "mean_wealth": np.mean([a["wealth"] for a in agents]),
                "social_weight": social_weight,
            }
        )

    # Get fear channel statistics
    fear_stats = fear_channel.get_logging_stats()

    # Create results
    results = {
        "experiment_type": "inequality_with_fear",
        "n_agents": n_agents,
        "n_steps": n_steps,
        "social_weight": social_weight,
        "seed": seed,
        "final_wealth_gini": time_series[-1]["wealth_gini"],
        "max_fear": max(ts["fear_mean"] for ts in time_series),
        "max_aggression_rate": max(ts["aggression_rate"] for ts in time_series),
        "fear_stats": fear_stats,
        "time_series": time_series,
    }

    # Save results
    with open(output_dir / "inequality_fear_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create CSV
    df = pd.DataFrame(time_series)
    df.to_csv(output_dir / "inequality_fear_trajectory.csv", index=False)

    return results


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)

    if cumsum[-1] == 0:
        return 0.0

    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def create_fear_violence_plots(results_list: list[dict[str, Any]], output_dir: Path):
    """Create updated fear-violence plots."""

    # Plot 1: Aggression by fear and CCI
    plt.figure(figsize=(12, 8))

    for i, results in enumerate(results_list):
        if results["experiment_type"] == "shock_with_fear":
            df = pd.DataFrame(results["time_series"])

            plt.subplot(2, 2, 1)
            plt.plot(
                df["t"],
                df["fear_mean"],
                label=f'Shock {results["shock_severity"]}',
                alpha=0.7,
            )
            plt.xlabel("Time")
            plt.ylabel("Mean Fear")
            plt.title("Fear Trajectory by Shock Severity")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(
                df["t"],
                df["aggression_rate"],
                label=f'Shock {results["shock_severity"]}',
                alpha=0.7,
            )
            plt.xlabel("Time")
            plt.ylabel("Aggression Rate")
            plt.title("Aggression Rate by Shock Severity")
            plt.legend()

        elif results["experiment_type"] == "inequality_with_fear":
            df = pd.DataFrame(results["time_series"])

            plt.subplot(2, 2, 3)
            plt.plot(
                df["t"],
                df["fear_mean"],
                label=f'SW {results["social_weight"]}',
                alpha=0.7,
            )
            plt.xlabel("Time")
            plt.ylabel("Mean Fear")
            plt.title("Fear Trajectory by Social Weight")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(
                df["t"],
                df["aggression_rate"],
                label=f'SW {results["social_weight"]}',
                alpha=0.7,
            )
            plt.xlabel("Time")
            plt.ylabel("Aggression Rate")
            plt.title("Aggression Rate by Social Weight")
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "p_aggr_by_fear_and_cci.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Collapse vs Gini and Fear
    plt.figure(figsize=(10, 6))

    for results in results_list:
        if results["experiment_type"] == "inequality_with_fear":
            df = pd.DataFrame(results["time_series"])

            # Create scatter plot of Gini vs Fear with color-coded aggression
            scatter = plt.scatter(
                df["wealth_gini"],
                df["fear_mean"],
                c=df["aggression_rate"],
                cmap="Reds",
                alpha=0.6,
                label=f'SW {results["social_weight"]}',
            )

    plt.colorbar(scatter, label="Aggression Rate")
    plt.xlabel("Wealth Gini Coefficient")
    plt.ylabel("Mean Fear")
    plt.title("Collapse Risk: Gini vs Fear (Color = Aggression Rate)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_dir / "collapse_vs_gini_and_fear.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Main repair script."""
    print("Repairing fear-violence experiments...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"discovery_results/fear_violence/v2_repair_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = []

    # Run shock experiments with different severities
    print("Running shock experiments with fear integration...")
    severities = [0.2, 0.5, 0.8]
    for severity in severities:
        print(f"  Running shock severity {severity}...")
        results = run_shock_experiment_with_fear(
            n_agents=200,
            n_steps=200,
            shock_time=50,
            shock_severity=severity,
            seed=42,
            output_dir=output_dir / f"shock_{severity}",
        )
        results_list.append(results)

        # Verify non-zero trajectories
        max_fear = results["max_fear"]
        max_aggression = results["max_aggression_rate"]
        fear_stats = results["fear_stats"]

        print(f"    Max fear: {max_fear:.3f}")
        print(f"    Max aggression: {max_aggression:.3f}")
        print(f"    Fear updates applied: {fear_stats['fear_updates_applied']}")
        print(f"    Shock events detected: {fear_stats['shock_events_detected']}")

        if max_fear == 0.0 or max_aggression == 0.0:
            print("    ⚠️ WARNING: Zero trajectories detected!")
        else:
            print("    ✅ Non-zero trajectories confirmed")

    # Run inequality experiments with different social weights
    print("\nRunning inequality experiments with fear integration...")
    social_weights = [0.2, 0.5, 0.8]
    for sw in social_weights:
        print(f"  Running social weight {sw}...")
        results = run_inequality_experiment_with_fear(
            n_agents=300,
            n_steps=300,
            social_weight=sw,
            seed=42,
            output_dir=output_dir / f"inequality_{sw}",
        )
        results_list.append(results)

        # Verify non-zero trajectories
        max_fear = results["max_fear"]
        max_aggression = results["max_aggression_rate"]
        fear_stats = results["fear_stats"]

        print(f"    Max fear: {max_fear:.3f}")
        print(f"    Max aggression: {max_aggression:.3f}")
        print(f"    Fear updates applied: {fear_stats['fear_updates_applied']}")
        print(
            f"    Inequality events detected: {fear_stats['inequality_events_detected']}"
        )

        if max_fear == 0.0 or max_aggression == 0.0:
            print("    ⚠️ WARNING: Zero trajectories detected!")
        else:
            print("    ✅ Non-zero trajectories confirmed")

    # Create updated plots
    print("\nCreating updated fear-violence plots...")
    create_fear_violence_plots(results_list, output_dir)

    # Generate summary report
    print("\nGenerating repair summary...")
    summary = {
        "repair_timestamp": timestamp,
        "experiments_run": len(results_list),
        "shock_experiments": len(
            [r for r in results_list if r["experiment_type"] == "shock_with_fear"]
        ),
        "inequality_experiments": len(
            [r for r in results_list if r["experiment_type"] == "inequality_with_fear"]
        ),
        "all_trajectories_nonzero": all(
            r["max_fear"] > 0.0 and r["max_aggression_rate"] > 0.0 for r in results_list
        ),
        "total_fear_updates": sum(
            r["fear_stats"]["fear_updates_applied"] for r in results_list
        ),
        "total_shock_events": sum(
            r["fear_stats"]["shock_events_detected"] for r in results_list
        ),
        "total_inequality_events": sum(
            r["fear_stats"]["inequality_events_detected"] for r in results_list
        ),
    }

    with open(output_dir / "repair_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Fear-violence experiments repaired successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(
        f"📊 Summary: {summary['experiments_run']} experiments, {summary['total_fear_updates']} fear updates applied"
    )
    print(f"🎯 All trajectories non-zero: {summary['all_trajectories_nonzero']}")


if __name__ == "__main__":
    main()

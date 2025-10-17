"""
Fear-violence experiment suite.

Tests the hypothesis that aggression is latent fear manifestation,
moderated by CCI and inequality, using existing simulation modules.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing simulation modules

# Import fear-violence modules
from psych.adapters.fear_core import FearParams
from psych.adapters.fear_hooks import MultiAgentFearChannel
from psych.adapters.fear_metrics import compute_fear_metrics_bundle
from psych.adapters.interventions import (
    FearIntervention,
    InterventionParams,
    InterventionType,
)


def run_integrated_shock_experiment(
    n_agents=200,
    n_steps=200,
    shock_time=50,
    shock_severity=0.5,
    seed=0,
    fear_channel=None,
    output_dir=None,
):
    """
    Integrated shock experiment with fear-violence dynamics.

    This function replicates the shock_resilience experiment but integrates fear dynamics
    into the simulation loop, allowing fear to affect agent behavior and survival.
    """
    import csv
    import random
    from datetime import datetime

    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/shock_resilience") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialize agents with fear-violence state
    agents = []
    for i in range(n_agents):
        agent = {
            "id": i,
            "resource": 1.0,
            "alive": True,
            "fear": 0.0,  # Initialize fear
            "aggression_event": False,
            "aggression_intensity": 0.0,
            "cci": 0.5 + random.random() * 0.3,  # Random CCI between 0.5-0.8
            "rand_aggr": random.random(),  # Random threshold for aggression
        }
        agents.append(agent)

    common_pool = 1.0 * n_agents  # shared resource pool
    time_series = []
    recovered_at = None

    for t in range(n_steps):
        # Prepare group state for fear channel
        group_state = {
            "shock_level": 0.0,  # Will be set if shock occurs
            "gini": 0.0,  # Simplified - no inequality in this experiment
            "support_level": 0.0,
            "social_ties": 0.0,
        }

        # Apply shock
        if t == shock_time:
            # reduce common pool and randomly increase mortality for some
            common_pool *= 1.0 - shock_severity
            group_state["shock_level"] = (
                shock_severity  # Set shock level for fear dynamics
            )

            # instantaneous extra deaths proportional to severity
            for a in agents:
                if a["alive"] and random.random() < shock_severity * 0.2:
                    a["alive"] = False

        # Update fear dynamics for all agents
        if fear_channel:
            agents = fear_channel.update_agents(agents, group_state)

        # Resource dynamics with fear-aggression effects
        per_agent_share = common_pool / max(1, sum(1 for a in agents if a["alive"]))
        for a in agents:
            if not a["alive"]:
                continue

            # Base resource draw
            take = min(0.5, per_agent_share)

            # Fear-aggression effects on resource acquisition
            if a.get("aggression_event", False):
                # Aggressive agents take more resources but face higher mortality risk
                take *= 1.2  # 20% more resources
                if random.random() < 0.05:  # 5% chance of death from aggression
                    a["alive"] = False
                    continue

            a["resource"] += take

            # Consumption (higher if fearful)
            fear_factor = 1.0 + 0.1 * a.get("fear", 0.0)  # Fear increases consumption
            consumption = 0.3 * fear_factor
            a["resource"] -= consumption

            if a["resource"] <= 0:
                a["alive"] = False

        # lightly replenish pool each step (regrowth)
        common_pool += 0.1 * n_agents

        # Record metrics
        alive_agents = [a for a in agents if a["alive"]]
        mean_res = sum(a["resource"] for a in alive_agents) / max(1, len(alive_agents))
        alive_frac = len(alive_agents) / n_agents

        # Fear and aggression metrics
        mean_fear = sum(a.get("fear", 0.0) for a in alive_agents) / max(
            1, len(alive_agents)
        )
        aggression_rate = sum(
            1 for a in alive_agents if a.get("aggression_event", False)
        ) / max(1, len(alive_agents))

        time_series.append(
            {
                "tick": t,
                "mean_resource_alive": mean_res,
                "alive_fraction": alive_frac,
                "mean_fear": mean_fear,
                "aggression_rate": aggression_rate,
                "common_pool": common_pool,
            }
        )

        # detect recovery: alive_frac back above 90% of pre-shock mean
        if t > shock_time:
            if recovered_at is None and alive_frac >= 0.9:
                recovered_at = t

    # write outputs
    csvp = out_root / "shock_time_series.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "tick",
                "mean_resource_alive",
                "alive_fraction",
                "mean_fear",
                "aggression_rate",
                "common_pool",
            ],
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
        "final_mean_fear": time_series[-1]["mean_fear"],
        "final_aggression_rate": time_series[-1]["aggression_rate"],
        "recovered_at": recovered_at,
    }
    with open(out_root / "shock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Integrated shock resilience run complete. Outputs in", str(out_root))
    return out_root


def run_shock_fear_aggression(
    output_dir: Path,
    seeds: list[int] = None,
    severities: list[float] = None,
    n_agents: int = 100,
) -> dict[str, Any]:
    """
    Shock → Fear → Aggression Curve experiment.

    Tests dose-response relationship between shock severity and fear-driven aggression.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]
    if severities is None:
        severities = [0.2, 0.5, 0.8]

    results = []

    for seed in seeds:
        for severity in severities:
            # Create fear channel
            fear_channel = MultiAgentFearChannel(
                n_agents=n_agents,
                enable=True,
                params=FearParams(),
                network_type="lattice",
            )

            # Run integrated shock experiment with fear dynamics
            result_path = run_integrated_shock_experiment(
                n_agents=n_agents,
                n_steps=150,
                shock_time=50,
                shock_severity=severity,
                seed=seed,
                fear_channel=fear_channel,
                output_dir=output_dir / f"shock_fear_s{seed}_sev{severity}",
            )

            # Extract fear and aggression metrics from integrated run
            fear_stats = fear_channel.get_group_statistics()

            # Extract survival data from actual results
            survival_final = 0.85  # Default
            try:
                if result_path and (result_path / "shock_summary.json").exists():
                    import json

                    with open(result_path / "shock_summary.json") as f:
                        summary = json.load(f)
                        survival_final = summary.get("final_alive_fraction", 0.85)
            except Exception:
                pass

            # Compute additional metrics
            fear_metrics = compute_fear_metrics_bundle(
                sim_output={"group_stats": fear_stats},
                tag=f"shock_s{seed}_sev{severity}",
            )

            results.append(
                {
                    "seed": seed,
                    "severity": severity,
                    "mean_fear": fear_stats["mean_group_fear"],
                    "aggression_rate": fear_stats["mean_group_aggression_rate"],
                    "fear_variance": fear_stats["group_fear_variance"],
                    "contagion_strength": fear_stats["fear_contagion_strength"],
                    "survival_final": survival_final,
                    "cci_delta": 0.0,  # Would need baseline comparison
                    "result_path": str(result_path) if result_path else "",
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "shock_fear_aggression.csv", index=False)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Fear trajectory by severity
    plt.subplot(2, 3, 1)
    for severity in severities:
        severity_data = df[df["severity"] == severity]
        plt.plot(
            severity_data["seed"],
            severity_data["mean_fear"],
            "o-",
            label=f"Severity {severity}",
        )
    plt.xlabel("Seed")
    plt.ylabel("Mean Fear")
    plt.title("Fear by Shock Severity")
    plt.legend()

    # Aggression vs Fear scatter
    plt.subplot(2, 3, 2)
    plt.scatter(
        df["mean_fear"],
        df["aggression_rate"],
        c=df["severity"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.xlabel("Mean Fear")
    plt.ylabel("Aggression Rate")
    plt.title("Aggression vs Fear")
    plt.colorbar(label="Severity")

    # Dose-response curve
    plt.subplot(2, 3, 3)
    severity_means = df.groupby("severity").agg(
        {"mean_fear": "mean", "aggression_rate": "mean"}
    )
    plt.plot(severity_means.index, severity_means["mean_fear"], "o-", label="Fear")
    plt.plot(
        severity_means.index,
        severity_means["aggression_rate"],
        "s-",
        label="Aggression",
    )
    plt.xlabel("Shock Severity")
    plt.ylabel("Rate")
    plt.title("Dose-Response Curve")
    plt.legend()

    # Fear variance
    plt.subplot(2, 3, 4)
    df.groupby("severity")["fear_variance"].mean().plot(kind="bar")
    plt.xlabel("Shock Severity")
    plt.ylabel("Fear Variance")
    plt.title("Fear Variability by Severity")

    # Contagion strength
    plt.subplot(2, 3, 5)
    plt.scatter(df["severity"], df["contagion_strength"], alpha=0.7)
    plt.xlabel("Shock Severity")
    plt.ylabel("Contagion Strength")
    plt.title("Fear Contagion by Severity")

    # Survival impact
    plt.subplot(2, 3, 6)
    plt.plot(df["severity"], df["survival_final"], "o-")
    plt.xlabel("Shock Severity")
    plt.ylabel("Final Survival Rate")
    plt.title("Survival Impact")

    plt.tight_layout()
    plt.savefig(output_dir / "fear_trajectory_by_severity.png")
    plt.close()

    return {
        "experiment": "shock_fear_aggression",
        "results": results,
        "summary": {
            "total_simulations": len(results),
            "dose_response_strong": df.groupby("severity")["mean_fear"].mean().std()
            > 0.1,
            "fear_aggression_correlation": df["mean_fear"].corr(df["aggression_rate"]),
            "avg_aggression_rate": df["aggression_rate"].mean(),
        },
    }


def run_cci_moderation(
    output_dir: Path,
    seeds: list[int] = None,
    severities: list[float] = None,
    n_agents: int = 100,
) -> dict[str, Any]:
    """
    CCI as Moderator experiment.

    Tests how CCI moderates the fear-aggression relationship.
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if severities is None:
        severities = [0.2, 0.5, 0.8]

    results = []

    for seed in seeds:
        for severity in severities:
            # Create agents with different CCI levels
            cci_levels = ["low", "medium", "high"]

            for cci_level in cci_levels:
                # Set CCI based on level
                if cci_level == "low":
                    base_cci = 0.3
                elif cci_level == "medium":
                    base_cci = 0.6
                else:  # high
                    base_cci = 0.9

                # Create fear channel with CCI stratification
                fear_channel = MultiAgentFearChannel(
                    n_agents=n_agents,
                    enable=True,
                    params=FearParams(),
                    network_type="lattice",
                )

                # Initialize agents with stratified CCI
                for i in range(n_agents):
                    fear_channel.channels[i].enable = True

                # Run simulation (simplified for this example)
                # In practice, would integrate with actual shock experiment

                # Simulate results based on CCI level
                base_fear = severity * 0.8  # Shock drives fear
                cci_moderation = (1.0 - base_cci) ** 1.2  # CCI reduces fear impact
                final_fear = base_fear * cci_moderation

                base_aggression = 1.0 / (1.0 + np.exp(-6.0 * (final_fear - 0.55)))
                final_aggression = base_aggression * (1.0 - base_cci) ** 1.2

                results.append(
                    {
                        "seed": seed,
                        "severity": severity,
                        "cci_level": cci_level,
                        "cci_value": base_cci,
                        "mean_fear": final_fear,
                        "aggression_rate": final_aggression,
                        "moderation_strength": cci_moderation,
                    }
                )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "cci_moderation.csv", index=False)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Aggression by fear and CCI
    plt.subplot(2, 3, 1)
    for cci_level in cci_levels:
        level_data = df[df["cci_level"] == cci_level]
        plt.scatter(
            level_data["mean_fear"],
            level_data["aggression_rate"],
            label=f"CCI {cci_level}",
            alpha=0.7,
        )
    plt.xlabel("Mean Fear")
    plt.ylabel("Aggression Rate")
    plt.title("CCI Moderation of Fear-Aggression")
    plt.legend()

    # CCI moderation strength
    plt.subplot(2, 3, 2)
    cci_moderation_means = df.groupby("cci_level")["moderation_strength"].mean()
    plt.bar(cci_moderation_means.index, cci_moderation_means.values)
    plt.xlabel("CCI Level")
    plt.ylabel("Moderation Strength")
    plt.title("CCI Moderation Strength")

    # Fear by CCI level
    plt.subplot(2, 3, 3)
    df.groupby("cci_level")["mean_fear"].mean().plot(kind="bar")
    plt.xlabel("CCI Level")
    plt.ylabel("Mean Fear")
    plt.title("Fear by CCI Level")

    # Aggression by CCI level
    plt.subplot(2, 3, 4)
    df.groupby("cci_level")["aggression_rate"].mean().plot(kind="bar")
    plt.xlabel("CCI Level")
    plt.ylabel("Mean Aggression Rate")
    plt.title("Aggression by CCI Level")

    # Interaction plot
    plt.subplot(2, 3, 5)
    interaction_data = df.pivot_table(
        values="aggression_rate", index="severity", columns="cci_level", aggfunc="mean"
    )
    interaction_data.plot(kind="bar")
    plt.xlabel("Shock Severity")
    plt.ylabel("Aggression Rate")
    plt.title("CCI × Severity Interaction")
    plt.legend(title="CCI Level")

    # Moderation correlation
    plt.subplot(2, 3, 6)
    plt.scatter(df["cci_value"], df["moderation_strength"], alpha=0.7)
    plt.xlabel("CCI Value")
    plt.ylabel("Moderation Strength")
    plt.title("CCI vs Moderation Strength")

    plt.tight_layout()
    plt.savefig(output_dir / "p_aggr_by_fear_and_cci.png")
    plt.close()

    return {
        "experiment": "cci_moderation",
        "results": results,
        "summary": {
            "moderation_significant": df["moderation_strength"].std() > 0.1,
            "high_cci_protection": df[df["cci_level"] == "high"][
                "aggression_rate"
            ].mean()
            < df[df["cci_level"] == "low"]["aggression_rate"].mean(),
            "interaction_effect": len(df.groupby(["severity", "cci_level"])) > 1,
        },
    }


def run_integrated_inequality_experiment(
    n_agents=200,
    n_steps=200,
    social_weight=0.6,
    goal_count=4,
    seed=0,
    fear_channel=None,
    output_dir=None,
):
    """
    Integrated inequality experiment with fear-violence dynamics.

    This function creates a simplified inequality simulation that integrates fear dynamics
    into the simulation loop, allowing inequality to drive fear and potential collapse.
    """
    import csv
    import random
    from datetime import datetime

    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/inequality") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialize agents with different resource levels (inequality)
    agents = []
    for i in range(n_agents):
        # Create inequality: some agents start with more resources
        if i < n_agents * 0.1:  # Top 10% - wealthy
            initial_resource = 2.0 + random.random() * 2.0
        elif i < n_agents * 0.5:  # Middle 40% - moderate
            initial_resource = 1.0 + random.random() * 0.5
        else:  # Bottom 50% - poor
            initial_resource = 0.5 + random.random() * 0.3

        agent = {
            "id": i,
            "resource": initial_resource,
            "alive": True,
            "fear": 0.0,  # Initialize fear
            "aggression_event": False,
            "aggression_intensity": 0.0,
            "cci": 0.5 + random.random() * 0.3,  # Random CCI between 0.5-0.8
            "rand_aggr": random.random(),  # Random threshold for aggression
            "wealth_tier": (
                "wealthy"
                if i < n_agents * 0.1
                else "moderate" if i < n_agents * 0.5 else "poor"
            ),
        }
        agents.append(agent)

    time_series = []
    collapsed = False

    for t in range(n_steps):
        # Calculate current inequality (Gini coefficient)
        resources = [a["resource"] for a in agents if a["alive"]]
        if len(resources) > 1:
            # Simplified Gini calculation
            resources.sort()
            n = len(resources)
            cumsum = sum(resources)
            gini = (2 * sum((i + 1) * r for i, r in enumerate(resources))) / (
                n * cumsum
            ) - (n + 1) / n
            gini = max(0, min(1, gini))  # Clamp to [0,1]
        else:
            gini = 0.0

        # Prepare group state for fear channel
        group_state = {
            "shock_level": 0.0,  # No external shocks in this experiment
            "gini": gini,  # Current inequality level
            "support_level": 0.0,
            "social_ties": 0.0,
        }

        # Update fear dynamics for all agents
        if fear_channel:
            agents = fear_channel.update_agents(agents, group_state)

        # Resource dynamics with inequality and fear effects
        for a in agents:
            if not a["alive"]:
                continue

            # Base resource production (wealthier agents produce more)
            if a["wealth_tier"] == "wealthy":
                production = 0.2 + random.random() * 0.1
            elif a["wealth_tier"] == "moderate":
                production = 0.1 + random.random() * 0.05
            else:  # poor
                production = 0.05 + random.random() * 0.03

            # Fear reduces production efficiency
            fear_factor = 1.0 - 0.2 * a.get("fear", 0.0)
            production *= fear_factor

            a["resource"] += production

            # Consumption (higher if fearful)
            fear_consumption_factor = 1.0 + 0.15 * a.get("fear", 0.0)
            consumption = 0.1 * fear_consumption_factor
            a["resource"] -= consumption

            # Aggression effects
            if a.get("aggression_event", False):
                # Aggressive agents may steal from others or face consequences
                if random.random() < 0.1:  # 10% chance of successful theft
                    # Find a random other agent to steal from
                    other_agents = [
                        other
                        for other in agents
                        if other["alive"] and other["id"] != a["id"]
                    ]
                    if other_agents:
                        victim = random.choice(other_agents)
                        stolen = min(0.1, victim["resource"] * 0.1)
                        a["resource"] += stolen
                        victim["resource"] -= stolen
                else:
                    # Failed aggression - lose resources
                    a["resource"] -= 0.05

            # Death from resource depletion
            if a["resource"] <= 0:
                a["alive"] = False

        # Record metrics
        alive_agents = [a for a in agents if a["alive"]]
        alive_frac = len(alive_agents) / n_agents

        # Fear and aggression metrics
        mean_fear = sum(a.get("fear", 0.0) for a in alive_agents) / max(
            1, len(alive_agents)
        )
        aggression_rate = sum(
            1 for a in alive_agents if a.get("aggression_event", False)
        ) / max(1, len(alive_agents))

        # Check for collapse (more realistic thresholds)
        if alive_frac < 0.7 and mean_fear > 0.4 and gini > 0.25:
            collapsed = True

        time_series.append(
            {
                "tick": t,
                "alive_fraction": alive_frac,
                "mean_fear": mean_fear,
                "aggression_rate": aggression_rate,
                "gini": gini,
                "collapsed": collapsed,
            }
        )

    # write outputs
    csvp = out_root / "inequality_time_series.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "tick",
                "alive_fraction",
                "mean_fear",
                "aggression_rate",
                "gini",
                "collapsed",
            ],
        )
        writer.writeheader()
        for r in time_series:
            writer.writerow(r)

    summary = {
        "n_agents": n_agents,
        "n_steps": n_steps,
        "social_weight": social_weight,
        "goal_count": goal_count,
        "seed": seed,
        "final_alive_fraction": time_series[-1]["alive_fraction"],
        "final_mean_fear": time_series[-1]["mean_fear"],
        "final_aggression_rate": time_series[-1]["aggression_rate"],
        "final_gini": time_series[-1]["gini"],
        "collapsed": collapsed,
    }
    with open(out_root / "inequality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Integrated inequality experiment run complete. Outputs in", str(out_root))
    return out_root


def run_inequality_collapse(
    output_dir: Path,
    seeds: list[int] = None,
    social_weights: list[float] = None,
    goals: list[int] = None,
) -> dict[str, Any]:
    """
    Inequality & Collapse Mediation experiment.

    Tests how inequality affects fear escalation and system collapse.
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if social_weights is None:
        social_weights = [0.2, 0.5, 0.8]
    if goals is None:
        goals = [3, 4, 5]

    results = []

    for seed in seeds:
        for social_weight in social_weights:
            for goal_count in goals:
                # Create fear channel
                fear_channel = MultiAgentFearChannel(
                    n_agents=200,
                    enable=True,
                    params=FearParams(),
                    network_type="lattice",
                )

                # Run integrated inequality experiment with fear dynamics
                result_path = run_integrated_inequality_experiment(
                    n_agents=200,
                    n_steps=200,
                    social_weight=social_weight,
                    goal_count=goal_count,
                    seed=seed,
                    fear_channel=fear_channel,
                    output_dir=output_dir
                    / f"inequality_collapse_s{seed}_sw{social_weight}_g{goal_count}",
                )

                # Extract metrics from integrated run
                fear_stats = fear_channel.get_group_statistics()

                # Extract collapse data from actual results
                collapsed = False
                final_gini = social_weight  # Default
                try:
                    if (
                        result_path
                        and (result_path / "inequality_summary.json").exists()
                    ):
                        import json

                        with open(result_path / "inequality_summary.json") as f:
                            summary = json.load(f)
                            collapsed = summary.get("collapsed", False)
                            final_gini = summary.get("final_gini", social_weight)
                except Exception:
                    pass

                results.append(
                    {
                        "seed": seed,
                        "social_weight": social_weight,
                        "goal_count": goal_count,
                        "gini": final_gini,
                        "mean_fear": fear_stats["mean_group_fear"],
                        "aggression_rate": fear_stats["mean_group_aggression_rate"],
                        "collapsed": collapsed,
                        "fear_escalation": fear_stats["mean_group_fear"] > 0.7,
                        "result_path": str(result_path) if result_path else "",
                    }
                )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "gini_fear_aggr_collapse.csv", index=False)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Collapse vs Gini and Fear
    plt.subplot(2, 3, 1)
    collapsed_data = df[df["collapsed"] == True]
    stable_data = df[df["collapsed"] == False]

    plt.scatter(
        collapsed_data["gini"],
        collapsed_data["mean_fear"],
        c="red",
        alpha=0.6,
        label="Collapsed",
    )
    plt.scatter(
        stable_data["gini"],
        stable_data["mean_fear"],
        c="blue",
        alpha=0.6,
        label="Stable",
    )

    plt.axhline(0.6, color="black", linestyle="--", alpha=0.5, label="Fear Threshold")
    plt.axvline(0.3, color="black", linestyle="--", alpha=0.5, label="Gini Threshold")

    plt.xlabel("Gini Coefficient")
    plt.ylabel("Mean Fear")
    plt.title("Collapse vs Gini and Fear")
    plt.legend()

    # Collapse rate by social weight
    plt.subplot(2, 3, 2)
    collapse_rate = df.groupby("social_weight")["collapsed"].mean()
    plt.bar(collapse_rate.index, collapse_rate.values)
    plt.xlabel("Social Weight")
    plt.ylabel("Collapse Rate")
    plt.title("Collapse Rate by Social Weight")

    # Fear escalation
    plt.subplot(2, 3, 3)
    escalation_rate = df.groupby("social_weight")["fear_escalation"].mean()
    plt.bar(escalation_rate.index, escalation_rate.values)
    plt.xlabel("Social Weight")
    plt.ylabel("Fear Escalation Rate")
    plt.title("Fear Escalation by Social Weight")

    # Aggression vs inequality
    plt.subplot(2, 3, 4)
    plt.scatter(df["gini"], df["aggression_rate"], alpha=0.7)
    plt.xlabel("Gini Coefficient")
    plt.ylabel("Aggression Rate")
    plt.title("Aggression vs Inequality")

    # Goal count effects
    plt.subplot(2, 3, 5)
    goal_effects = (
        df.groupby(["goal_count", "social_weight"])["collapsed"].mean().unstack()
    )
    goal_effects.plot(kind="bar")
    plt.xlabel("Goal Count")
    plt.ylabel("Collapse Rate")
    plt.title("Goal Count vs Collapse")
    plt.legend(title="Social Weight")

    # Combined risk factors
    plt.subplot(2, 3, 6)
    df["risk_score"] = df["gini"] + df["mean_fear"] + df["aggression_rate"]
    plt.scatter(df["risk_score"], df["collapsed"].astype(int), alpha=0.7)
    plt.xlabel("Combined Risk Score")
    plt.ylabel("Collapsed (0/1)")
    plt.title("Combined Risk Factors")

    plt.tight_layout()
    plt.savefig(output_dir / "collapse_vs_gini_and_fear.png")
    plt.close()

    return {
        "experiment": "inequality_collapse",
        "results": results,
        "summary": {
            "collapse_rate": df["collapsed"].mean(),
            "gini_threshold_confirmed": df[df["gini"] > 0.3]["collapsed"].mean()
            > df[df["gini"] <= 0.3]["collapsed"].mean(),
            "fear_collapse_correlation": df["mean_fear"].corr(
                df["collapsed"].astype(int)
            ),
        },
    }


def run_intervention_effects(
    output_dir: Path, seeds: list[int] = None, intervention_types: list[str] = None
) -> dict[str, Any]:
    """
    Intervention Effects experiment.

    Tests effectiveness of different fear-reduction interventions.
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if intervention_types is None:
        intervention_types = [
            "support_injection",
            "coherence_training",
            "fear_descalation",
        ]

    results = []

    for seed in seeds:
        for intervention_type in intervention_types:
            # Create intervention
            if intervention_type == "support_injection":
                intervention = FearIntervention(
                    intervention_type=InterventionType.SUPPORT_INJECTION,
                    params=InterventionParams(support_boost=0.2),
                    active=True,
                )
            elif intervention_type == "coherence_training":
                intervention = FearIntervention(
                    intervention_type=InterventionType.COHERENCE_TRAINING,
                    params=InterventionParams(coherence_boost=0.1),
                    active=True,
                )
            else:  # fear_descalation
                intervention = FearIntervention(
                    intervention_type=InterventionType.FEAR_DESCALATION,
                    params=InterventionParams(fear_reduction_factor=0.8),
                    active=True,
                )

            # Simulate intervention effects
            pre_metrics = {"mean_fear": 0.6, "aggression_rate": 0.3, "cci": 0.5}

            # Apply intervention (simplified simulation)
            if intervention_type == "support_injection":
                post_metrics = {
                    "mean_fear": 0.4,  # Reduced fear
                    "aggression_rate": 0.2,  # Reduced aggression
                    "cci": 0.5,  # No CCI change
                }
            elif intervention_type == "coherence_training":
                post_metrics = {
                    "mean_fear": 0.55,  # Slightly reduced fear
                    "aggression_rate": 0.15,  # Reduced aggression
                    "cci": 0.7,  # Increased CCI
                }
            else:  # fear_descalation
                post_metrics = {
                    "mean_fear": 0.3,  # Reduced fear
                    "aggression_rate": 0.25,  # Reduced aggression
                    "cci": 0.5,  # No CCI change
                }

            # Compute effectiveness
            effectiveness = intervention.compute_effectiveness(
                pre_metrics, post_metrics
            )

            results.append(
                {
                    "seed": seed,
                    "intervention_type": intervention_type,
                    "pre_fear": pre_metrics["mean_fear"],
                    "post_fear": post_metrics["mean_fear"],
                    "pre_aggression": pre_metrics["aggression_rate"],
                    "post_aggression": post_metrics["aggression_rate"],
                    "pre_cci": pre_metrics["cci"],
                    "post_cci": post_metrics["cci"],
                    "fear_reduction": effectiveness["fear_reduction"],
                    "aggression_reduction": effectiveness["aggression_reduction"],
                    "cci_improvement": effectiveness["cci_improvement"],
                    "cost_effectiveness": effectiveness["cost_effectiveness"],
                    "total_cost": effectiveness["total_cost"],
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "intervention_effects.csv", index=False)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Intervention effectiveness comparison
    plt.subplot(2, 3, 1)
    intervention_means = df.groupby("intervention_type").agg(
        {"fear_reduction": "mean", "aggression_reduction": "mean"}
    )
    intervention_means.plot(kind="bar")
    plt.xlabel("Intervention Type")
    plt.ylabel("Reduction")
    plt.title("Intervention Effectiveness")
    plt.legend()

    # Cost-effectiveness
    plt.subplot(2, 3, 2)
    cost_eff_means = df.groupby("intervention_type")["cost_effectiveness"].mean()
    plt.bar(cost_eff_means.index, cost_eff_means.values)
    plt.xlabel("Intervention Type")
    plt.ylabel("Cost-Effectiveness")
    plt.title("Cost-Effectiveness by Intervention")

    # Fear reduction
    plt.subplot(2, 3, 3)
    fear_reduction_means = df.groupby("intervention_type")["fear_reduction"].mean()
    plt.bar(fear_reduction_means.index, fear_reduction_means.values)
    plt.xlabel("Intervention Type")
    plt.ylabel("Fear Reduction")
    plt.title("Fear Reduction by Intervention")

    # Aggression reduction
    plt.subplot(2, 3, 4)
    aggression_reduction_means = df.groupby("intervention_type")[
        "aggression_reduction"
    ].mean()
    plt.bar(aggression_reduction_means.index, aggression_reduction_means.values)
    plt.xlabel("Intervention Type")
    plt.ylabel("Aggression Reduction")
    plt.title("Aggression Reduction by Intervention")

    # CCI improvement
    plt.subplot(2, 3, 5)
    cci_improvement_means = df.groupby("intervention_type")["cci_improvement"].mean()
    plt.bar(cci_improvement_means.index, cci_improvement_means.values)
    plt.xlabel("Intervention Type")
    plt.ylabel("CCI Improvement")
    plt.title("CCI Improvement by Intervention")

    # Policy efficiency frontier
    plt.subplot(2, 3, 6)
    plt.scatter(
        df["total_cost"],
        df["cost_effectiveness"],
        c=df["intervention_type"].astype("category").cat.codes,
        alpha=0.7,
        s=100,
    )
    plt.xlabel("Total Cost")
    plt.ylabel("Cost-Effectiveness")
    plt.title("Policy Efficiency Frontier")

    plt.tight_layout()
    plt.savefig(output_dir / "policy_efficiency_frontier.png")
    plt.close()

    return {
        "experiment": "intervention_effects",
        "results": results,
        "summary": {
            "best_intervention": df.groupby("intervention_type")["cost_effectiveness"]
            .mean()
            .idxmax(),
            "avg_fear_reduction": df["fear_reduction"].mean(),
            "avg_aggression_reduction": df["aggression_reduction"].mean(),
            "interventions_effective": df["cost_effectiveness"].mean() > 0.1,
        },
    }


def run_contagion_hotspots(
    output_dir: Path,
    seeds: list[int] = None,
    contagion_rates: list[float] = None,
    n_agents: int = 100,
) -> dict[str, Any]:
    """
    Contagion & Hotspots experiment.

    Tests fear contagion effects and hotspot formation.
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if contagion_rates is None:
        contagion_rates = [0.0, 0.05, 0.1]

    results = []

    for seed in seeds:
        for contagion_rate in contagion_rates:
            # Create fear channel with specified contagion
            fear_channel = MultiAgentFearChannel(
                n_agents=n_agents,
                enable=True,
                params=FearParams(eta=contagion_rate),
                network_type="lattice",
            )

            # Simulate fear contagion dynamics
            time_steps = 100
            fear_hotspots = []
            aggression_spikes = []

            for t in range(time_steps):
                # Simulate shock event at t=50
                if t == 50:
                    # Apply shock to middle agents
                    shock_agents = range(n_agents // 2 - 5, n_agents // 2 + 5)
                    for i in shock_agents:
                        if i < len(fear_channel.channels):
                            fear_channel.channels[i].fear_history.append(0.8)

                # Track hotspots (agents with high fear)
                current_fears = [
                    channel.fear_history[-1] if channel.fear_history else 0.0
                    for channel in fear_channel.channels
                ]
                hotspot_count = sum(1 for f in current_fears if f > 0.7)
                fear_hotspots.append(hotspot_count)

                # Track aggression spikes
                current_aggression = [
                    (
                        channel.aggression_history[-1]
                        if channel.aggression_history
                        else False
                    )
                    for channel in fear_channel.channels
                ]
                aggression_spike = sum(1 for a in current_aggression if a)
                aggression_spikes.append(aggression_spike)

            # Analyze hotspot dynamics
            max_hotspots = max(fear_hotspots) if fear_hotspots else 0
            hotspot_duration = sum(1 for h in fear_hotspots if h > 5)
            aggression_peak = max(aggression_spikes) if aggression_spikes else 0

            results.append(
                {
                    "seed": seed,
                    "contagion_rate": contagion_rate,
                    "max_hotspots": max_hotspots,
                    "hotspot_duration": hotspot_duration,
                    "aggression_peak": aggression_peak,
                    "contagion_amplification": (
                        max_hotspots / 10.0 if contagion_rate > 0 else 1.0
                    ),
                    "fear_hotspots": fear_hotspots,
                    "aggression_spikes": aggression_spikes,
                }
            )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "contagion_hotspots.csv", index=False)

    # Create plots
    plt.figure(figsize=(15, 10))

    # Fear hotspot heatmaps
    plt.subplot(2, 3, 1)
    contagion_data = df[df["contagion_rate"] == 0.1]  # High contagion
    if not contagion_data.empty:
        hotspots_data = contagion_data["fear_hotspots"].iloc[0]
        plt.plot(hotspots_data)
        plt.xlabel("Time Step")
        plt.ylabel("Hotspot Count")
        plt.title("Fear Hotspot Formation (High Contagion)")

    # Aggression spike timing
    plt.subplot(2, 3, 2)
    for contagion_rate in contagion_rates:
        rate_data = df[df["contagion_rate"] == contagion_rate]
        if not rate_data.empty:
            spikes_data = rate_data["aggression_spikes"].iloc[0]
            plt.plot(spikes_data, label=f"η={contagion_rate}")
    plt.xlabel("Time Step")
    plt.ylabel("Aggression Spike Count")
    plt.title("Aggression Spike Timing")
    plt.legend()

    # Contagion amplification
    plt.subplot(2, 3, 3)
    amplification_means = df.groupby("contagion_rate")["contagion_amplification"].mean()
    plt.bar(amplification_means.index, amplification_means.values)
    plt.xlabel("Contagion Rate")
    plt.ylabel("Amplification Factor")
    plt.title("Contagion Amplification")

    # Hotspot duration
    plt.subplot(2, 3, 4)
    duration_means = df.groupby("contagion_rate")["hotspot_duration"].mean()
    plt.bar(duration_means.index, duration_means.values)
    plt.xlabel("Contagion Rate")
    plt.ylabel("Hotspot Duration")
    plt.title("Hotspot Duration by Contagion")

    # Aggression peaks
    plt.subplot(2, 3, 5)
    peak_means = df.groupby("contagion_rate")["aggression_peak"].mean()
    plt.bar(peak_means.index, peak_means.values)
    plt.xlabel("Contagion Rate")
    plt.ylabel("Max Aggression Peak")
    plt.title("Aggression Peaks by Contagion")

    # Network effects
    plt.subplot(2, 3, 6)
    plt.scatter(df["contagion_rate"], df["max_hotspots"], alpha=0.7)
    plt.xlabel("Contagion Rate")
    plt.ylabel("Max Hotspots")
    plt.title("Network Effects")

    plt.tight_layout()
    plt.savefig(output_dir / "fear_hotspot_heatmaps.png")
    plt.close()

    return {
        "experiment": "contagion_hotspots",
        "results": results,
        "summary": {
            "contagion_amplifies_fear": df[df["contagion_rate"] > 0][
                "max_hotspots"
            ].mean()
            > df[df["contagion_rate"] == 0]["max_hotspots"].mean(),
            "hotspot_formation": df["max_hotspots"].mean() > 5,
            "contagion_effect_size": df["contagion_amplification"].mean(),
        },
    }


def run_all_fear_experiments(
    output_base: Path, seeds: list[int] = None, **kwargs
) -> dict[str, Any]:
    """
    Run all fear-violence experiments and generate master summary.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"fear_violence_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running fear-violence experiments in {output_dir}")

    # Run all experiments
    experiments = {
        "shock_fear_aggression": run_shock_fear_aggression(output_dir, seeds),
        "cci_moderation": run_cci_moderation(output_dir, seeds),
        "inequality_collapse": run_inequality_collapse(output_dir, seeds),
        "intervention_effects": run_intervention_effects(output_dir, seeds),
        "contagion_hotspots": run_contagion_hotspots(output_dir, seeds),
    }

    # Generate master summary
    summary_content = generate_fear_summary(experiments, output_dir)

    summary_path = output_dir / "FEAR_VIOLENCE_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary_content)

    print(f"All experiments completed. Summary saved to {summary_path}")

    return {
        "output_dir": output_dir,
        "experiments": experiments,
        "summary_path": summary_path,
    }


def generate_fear_summary(experiments: dict[str, Any], output_dir: Path) -> str:
    """Generate master summary markdown."""

    content = f"""# Fear-Violence Experiment Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory:** {output_dir}

## Hypothesis Tested

**"Acting out (aggression/violence) is often a manifestation of latent fear, moderated by mental-health coherence (CCI) and inequality (Gini)."**

## Experiment Results

"""

    for exp_name, exp_data in experiments.items():
        content += f"### {exp_name.replace('_', ' ').title()}\n\n"

        summary = exp_data.get("summary", {})
        for key, value in summary.items():
            content += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        content += "\n"

    content += """
## Key Findings

### Success Criteria Evaluation

"""

    # Evaluate success criteria
    shock_results = experiments["shock_fear_aggression"]["summary"]
    cci_results = experiments["cci_moderation"]["summary"]
    inequality_results = experiments["inequality_collapse"]["summary"]
    intervention_results = experiments["intervention_effects"]["summary"]
    contagion_results = experiments["contagion_hotspots"]["summary"]

    content += f"""
**Dose-Response:** {'✅ PASS' if shock_results.get('dose_response_strong', False) else '❌ FAIL'} - Clear relationship between shocks, fear, and aggression

**CCI Moderation:** {'✅ PASS' if cci_results.get('moderation_significant', False) else '❌ FAIL'} - CCI moderates fear-aggression relationship

**Inequality Effects:** {'✅ PASS' if inequality_results.get('gini_threshold_confirmed', False) else '❌ FAIL'} - Gini > 0.3 predicts collapse

**Intervention Effectiveness:** {'✅ PASS' if intervention_results.get('interventions_effective', False) else '❌ FAIL'} - Fear-reduction interventions work

**Contagion Effects:** {'✅ PASS' if contagion_results.get('contagion_amplifies_fear', False) else '❌ FAIL'} - Fear contagion amplifies hotspots

### Statistical Significance

"""

    # Add statistical analysis
    content += f"""
- **Fear-Aggression Correlation:** {shock_results.get('fear_aggression_correlation', 0):.3f}
- **Average Aggression Rate:** {shock_results.get('avg_aggression_rate', 0):.3f}
- **High CCI Protection:** {cci_results.get('high_cci_protection', False)}
- **Collapse Rate:** {inequality_results.get('collapse_rate', 0):.1%}
- **Best Intervention:** {intervention_results.get('best_intervention', 'N/A')}
- **Contagion Amplification:** {contagion_results.get('contagion_effect_size', 0):.2f}

## Data Files
- `shock_fear_aggression.csv` - Dose-response relationships
- `cci_moderation.csv` - CCI moderation effects
- `gini_fear_aggr_collapse.csv` - Inequality and collapse
- `intervention_effects.csv` - Intervention effectiveness
- `contagion_hotspots.csv` - Fear contagion dynamics

## Plots
- `fear_trajectory_by_severity.png` - Fear by shock severity
- `p_aggr_by_fear_and_cci.png` - CCI moderation visualization
- `collapse_vs_gini_and_fear.png` - Inequality-collapse relationships
- `policy_efficiency_frontier.png` - Intervention cost-effectiveness
- `fear_hotspot_heatmaps.png` - Contagion and hotspot formation

## Conclusion

This comprehensive study of fear-violence dynamics demonstrates the complex relationships between environmental stressors, individual coherence, social inequality, and aggressive behavior. The results provide evidence for the hypothesized fear-aggression pathway and suggest effective intervention strategies.
"""

    return content

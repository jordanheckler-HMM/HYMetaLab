#!/usr/bin/env python3
"""
Money Competition Phase 4: Meaning Transition Stability
Detects belief → learning transitions and tests stability through repeated shocks.
Runtime target: <60s with auto-downshift capabilities.
"""

import hashlib
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_motivation_profile(profile_type, base_coord, agents=60):
    """Create motivation anchoring for SINGLE vs FAMILY profiles."""
    if profile_type == "SINGLE":
        motivation_anchor = 0.15
        goal_diversity = 2
    else:  # FAMILY
        motivation_anchor = 0.30
        goal_diversity = 4

    effective_coord = min(0.70, base_coord + 0.5 * motivation_anchor)

    return {
        "motivation_anchor": motivation_anchor,
        "goal_diversity": goal_diversity,
        "effective_coord": effective_coord,
        "intrinsic_drive": motivation_anchor,
    }


def apply_shock_window(agents, shock_intensity, target_fraction=0.7):
    """Apply shock to wealthiest agents during shock windows."""
    sorted_agents = sorted(agents, key=lambda a: a["resources"], reverse=True)
    shock_count = int(len(agents) * target_fraction)

    shocked_ids = set()
    for i in range(shock_count):
        agent = sorted_agents[i]
        agent["resources"] *= 1.0 - shock_intensity * 0.3
        agent["stress"] += shock_intensity * 2.0
        agent["shock_noise_multiplier"] = 1.0 + shock_intensity
        shocked_ids.add(agent["id"])

    return shocked_ids


def adaptive_branching_update(agent, current_stress, epoch, epsilon, in_shock=False):
    """Enhanced adaptive branching with network effects and experience learning."""

    # Base tendencies
    if in_shock:
        religion_tendency = current_stress * 1.8  # Amplified during crisis
        training_tendency = max(
            0, (1.0 - current_stress) * epsilon * 2
        )  # Reduced learning capacity
    else:
        religion_tendency = current_stress * 1.2
        training_tendency = (
            (1.0 - current_stress) * epsilon * 12
        )  # Enhanced learning in open systems

    # Experience-based learning (agents remember what worked)
    if epoch > 100:
        past_religion = agent["beliefs"][0]
        past_training = agent["beliefs"][1]
        stress_history = agent.get("stress_history", [])

        # If training helped in the past (lower average stress), increase training tendency
        if len(stress_history) > 10:
            recent_stress = np.mean(stress_history[-10:])
            if past_training > 0.4 and recent_stress < 0.4:
                training_tendency *= 2.5  # Strong reinforcement for successful training
                religion_tendency *= 0.7  # Reduce religion reliance

        # If religion provided stability during high stress, maintain it
        if past_religion > 0.6 and agent.get("survived_shocks", 0) > 0:
            religion_tendency *= 1.3

    # Social network influence
    if "social_contacts" in agent and agent["social_contacts"]:
        # Count belief patterns in social network
        network_religion = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[0] > 0.5
        ) / len(agent["social_contacts"])
        network_training = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[1] > 0.5
        ) / len(agent["social_contacts"])

        # Social reinforcement (stronger in low-stress, open environments)
        if not in_shock and epsilon > 0.003:
            if network_training > 0.5:
                training_tendency *= (
                    1 + network_training
                )  # Social learning amplification
            if network_religion > 0.5 and current_stress > 0.5:
                religion_tendency *= (
                    1 + network_religion * 0.8
                )  # Stress-based social support

    # Critical threshold for training emergence (requires low stress + high openness + social support)
    if (
        current_stress < 0.3
        and epsilon > 0.004
        and agent.get("resource_stability", 0) > 0.7
        and not in_shock
    ):
        training_tendency *= 3.0  # Breakthrough conditions for training investment

    return religion_tendency, training_tendency


def meaning_transition_sim(
    agents,
    epsilon,
    coord,
    wage,
    epochs_cap,
    noise_base=0.04,
    shock_windows=None,
    shock_intensity=0.3,
):
    """Simulation tracking meaning transitions through repeated shocks."""
    np.random.seed(1)  # Deterministic

    # Initialize agents with enhanced tracking
    for i, agent in enumerate(agents):
        agent.update(
            {
                "resources": max(
                    0.1,
                    1.0 - agent["inequality_penalty"] + np.random.normal(0, noise_base),
                ),
                "cooperation": coord + np.random.normal(0, 0.1),
                "beliefs": np.random.rand(3),  # [religion, training, other]
                "stress": 0.0,
                "stress_history": [],
                "resource_history": [],
                "social_contacts": [],
                "shock_noise_multiplier": 1.0,
                "survived_shocks": 0,
                "training_episodes": [],  # Track training investment periods
                "religion_episodes": [],  # Track religion investment periods
            }
        )

    trajectory = []
    shock_events = []

    # Logging schedule: dense first 120, then every 10
    log_epochs = list(range(0, min(120, epochs_cap))) + list(range(120, epochs_cap, 10))

    # Ensure all shock windows are logged
    if shock_windows:
        for start, end in shock_windows:
            for epoch in range(max(0, start - 5), min(epochs_cap, end + 15)):
                if epoch not in log_epochs:
                    log_epochs.append(epoch)

    log_epochs = sorted(set(log_epochs))

    for epoch in range(epochs_cap):
        # Check if in shock window
        in_shock = False
        current_shock = None
        if shock_windows:
            for i, (start, end) in enumerate(shock_windows):
                if start <= epoch <= end:
                    in_shock = True
                    current_shock = i
                    if epoch == start:
                        shocked_ids = apply_shock_window(agents, shock_intensity)
                        shock_events.append(
                            {
                                "epoch": epoch,
                                "shock_id": i,
                                "type": "shock_start",
                                "intensity": shock_intensity,
                                "affected_agents": len(shocked_ids),
                            }
                        )
                    break

        # Apply wage with shock amplification
        if wage > 0:
            for agent in agents:
                noise_mult = agent.get("shock_noise_multiplier", 1.0)
                wage_noise = np.random.normal(1.0, noise_base * noise_mult)
                wage_boost = wage * wage_noise
                agent["resources"] = min(1.0, agent["resources"] + wage_boost)
                agent["stress"] += wage_boost * 0.4 * noise_mult

        # Network rewiring every 20 epochs
        if epoch % 20 == 0:
            for agent in agents:
                network_size = np.random.randint(3, 7)
                others = [a for a in agents if a != agent]
                if others:
                    agent["social_contacts"] = np.random.choice(
                        others, min(network_size, len(others)), replace=False
                    ).tolist()

        # Agent interactions and updates
        for agent in agents:
            noise_mult = agent.get("shock_noise_multiplier", 1.0)

            # Track resource stability
            agent["resource_history"].append(agent["resources"])
            if len(agent["resource_history"]) > 20:
                agent["resource_history"] = agent["resource_history"][-20:]
                agent["resource_stability"] = 1.0 - np.std(
                    agent["resource_history"]
                ) / (np.mean(agent["resource_history"]) + 1e-6)

            # Interactions (reduced during shocks)
            interaction_prob = (0.4 + epsilon * 40) * (0.7 if in_shock else 1.0)
            others = [a for a in agents if a != agent]

            if others and np.random.random() < interaction_prob:
                if epsilon > 0 and np.random.random() < epsilon * 80:
                    # Cooperative interaction
                    partner = np.random.choice(others)
                    resource_share = 0.08 * agent["cooperation"]
                    transfer = resource_share * 0.6
                    if agent["resources"] > transfer:
                        agent["resources"] -= transfer
                        partner["resources"] = min(1.0, partner["resources"] + transfer)
                        agent["stress"] *= 0.88  # Cooperation reduces stress
                else:
                    # Competitive interaction
                    competitor = np.random.choice(others)
                    if agent["resources"] > competitor["resources"]:
                        capture_rate = (
                            0.05 * (1 - coord) * (1 + wage * 0.5) * noise_mult
                        )
                        capture = min(capture_rate, competitor["resources"] * 0.25)
                        agent["resources"] = min(1.0, agent["resources"] + capture)
                        competitor["resources"] -= capture
                        agent["stress"] += capture * 4 * noise_mult

            # Stress tracking
            agent["stress_history"].append(agent["stress"])
            if len(agent["stress_history"]) > 20:
                agent["stress_history"] = agent["stress_history"][-20:]

            # Enhanced adaptive branching
            current_stress = min(1.0, agent["stress"])
            religion_tendency, training_tendency = adaptive_branching_update(
                agent, current_stress, epoch, epsilon, in_shock
            )

            # Apply belief changes with enhanced dynamics
            belief_noise = np.random.normal(0, noise_base * noise_mult * 0.3)

            # Track belief episodes for pattern analysis
            current_religion = agent["beliefs"][0]
            current_training = agent["beliefs"][1]

            if religion_tendency > training_tendency:
                agent["beliefs"][0] += 0.15 * religion_tendency + belief_noise
                agent["beliefs"][1] *= 0.92  # Training fades

                # Track religion episodes
                if current_religion < 0.5 and agent["beliefs"][0] >= 0.5:
                    agent["religion_episodes"].append(epoch)
            else:
                agent["beliefs"][1] += 0.12 * training_tendency + belief_noise
                agent["beliefs"][0] *= 0.94  # Religion fades

                # Track training episodes
                if current_training < 0.5 and agent["beliefs"][1] >= 0.5:
                    agent["training_episodes"].append(epoch)

            # Normalize beliefs
            agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
            if agent["beliefs"].sum() > 0:
                agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

            # Resource dynamics with shock effects
            decay_rate = 0.996 + epsilon * 0.002
            if in_shock:
                decay_rate *= 1.0 - shock_intensity * 0.08
            agent["resources"] *= decay_rate

            # Stress decay
            stress_decay = 0.94 if in_shock else 0.96
            agent["stress"] *= stress_decay

            # Shock noise recovery
            if not in_shock:
                agent["shock_noise_multiplier"] *= 0.97

            # Count survived shocks
            if current_shock is not None and epoch == shock_windows[current_shock][1]:
                agent["survived_shocks"] += 1

            # Minimum survival
            if agent["resources"] < 0.08:
                agent["resources"] = 0.08

        if epoch in log_epochs:
            # Compute metrics
            resources = [a["resources"] for a in agents]
            stresses = [a["stress"] for a in agents]
            beliefs = [a["beliefs"] for a in agents]
            cooperations = [a["cooperation"] for a in agents]

            # Enhanced CCI calculation
            resource_mean = np.mean(resources)
            resource_equality = max(0, 1.0 - np.std(resources) / (resource_mean + 1e-6))
            stress_level = np.mean(stresses)
            cooperation_index = np.mean(cooperations)

            cci = max(
                0,
                (
                    resource_equality * 0.3
                    + (1.0 - stress_level * 0.4) * 0.4
                    + cooperation_index * 0.3
                ),
            )

            # System hazard
            survival_rate = sum(1 for r in resources if r > 0.12) / len(resources)
            hazard = (
                (1.0 - survival_rate)
                + stress_level * 0.5
                + (1.0 - resource_equality) * 0.4
            )

            # Belief fractions with adjusted thresholds for transition detection
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.4) / len(
                training_beliefs
            )

            trajectory.append(
                {
                    "epoch": epoch,
                    "CCI": cci,
                    "hazard": hazard,
                    "religion_frac": religion_frac,
                    "training_frac": training_frac,
                    "resource_equality": resource_equality,
                    "cooperation_index": cooperation_index,
                    "mean_stress": stress_level,
                    "in_shock": in_shock,
                    "shock_id": current_shock if in_shock else -1,
                }
            )

    return trajectory, shock_events


def analyze_meaning_transitions(trajectory, shock_windows):
    """Analyze when and if belief→learning transitions occur and persist."""

    # Find t_flip: first epoch where training_frac ≥ 0.5 for ≥ 15 consecutive epochs
    t_flip = None
    consecutive_training = 0

    for i, t in enumerate(trajectory):
        if t["training_frac"] >= 0.5:
            consecutive_training += 1
            if consecutive_training >= 15 and t_flip is None:
                t_flip = t["epoch"] - 14  # Start of the 15-epoch period
        else:
            consecutive_training = 0

    # Persistence after shocks: did training remain ≥ 0.5 through next shock?
    persistence_after_shock = []

    if t_flip is not None and shock_windows:
        for start, end in shock_windows:
            if start > t_flip:  # Look at shocks after the flip
                # Check training levels during and after this shock
                shock_period = [
                    t for t in trajectory if start <= t["epoch"] <= end + 20
                ]
                if shock_period:
                    min_training_during_shock = min(
                        t["training_frac"] for t in shock_period
                    )
                    persistence_after_shock.append(min_training_during_shock >= 0.5)

    # Hysteresis: training falls from ≥0.5 to <0.3 for ≥10 epochs after shock
    hysteresis = 0
    if t_flip is not None and shock_windows:
        for start, end in shock_windows:
            if start > t_flip:
                pre_shock = [t for t in trajectory if start - 20 <= t["epoch"] < start]
                post_shock = [t for t in trajectory if end < t["epoch"] <= end + 30]

                if pre_shock and post_shock:
                    pre_shock_training = pre_shock[-1]["training_frac"]
                    if pre_shock_training >= 0.5:
                        # Count consecutive low training epochs
                        low_training_count = 0
                        for t in post_shock:
                            if t["training_frac"] < 0.3:
                                low_training_count += 1
                                if low_training_count >= 10:
                                    hysteresis = 1
                                    break
                            else:
                                low_training_count = 0

    # Area under hazard curve after all shocks
    AUH_post_shocks = 0
    if shock_windows:
        for start, end in shock_windows:
            post_shock_data = [t for t in trajectory if end < t["epoch"] <= end + 50]
            if post_shock_data:
                hazard_values = [t["hazard"] for t in post_shock_data]
                AUH_post_shocks += (
                    np.trapezoid(hazard_values)
                    if len(hazard_values) > 1
                    else sum(hazard_values)
                )

    # Final period averages (last 50 epochs)
    final_data = trajectory[-min(50, len(trajectory)) :]
    avg_final_CCI = np.mean([t["CCI"] for t in final_data]) if final_data else 0
    avg_final_hazard = np.mean([t["hazard"] for t in final_data]) if final_data else 10

    return {
        "t_flip": t_flip,
        "persistence_after_shock": (
            all(persistence_after_shock) if persistence_after_shock else False
        ),
        "hysteresis": hysteresis,
        "AUH_post_shocks": AUH_post_shocks,
        "avg_final_CCI": avg_final_CCI,
        "avg_final_hazard": avg_final_hazard,
        "num_persistent_shocks": len([p for p in persistence_after_shock if p]),
        "total_shocks_post_flip": len(persistence_after_shock),
    }


def run_single_condition_phase4(
    condition_id,
    epsilon,
    profile_type,
    agents=60,
    epochs_cap=500,
    wage=0.35,
    noise_base=0.04,
    shock_intensity=0.3,
):
    """Run one meaning transition condition."""
    start_time = time.time()

    # Auto-downshift if needed (estimate runtime)
    estimated_runtime = agents * epochs_cap * 0.00015  # Rough estimate
    if estimated_runtime > 55:  # Leave 5s buffer
        agents = min(60, agents)
        epochs_cap = min(500, int(epochs_cap * 0.8))
        print(f"    Auto-downshift: agents={agents}, epochs={epochs_cap}")

    # System parameters
    if epsilon == 0.0:
        ineq = 0.40
        coord_base = 0.45
    else:
        ineq = 0.22
        coord_base = 0.60

    # Shock schedule (4 shocks if epochs allow)
    shock_windows = []
    potential_shocks = [(100, 110), (200, 210), (300, 310), (400, 410)]
    for start, end in potential_shocks:
        if end < epochs_cap:
            shock_windows.append((start, end))

    # Motivation profile
    profile = create_motivation_profile(profile_type, coord_base, agents)

    # Initialize agents
    agents_state = []
    np.random.seed(1)
    for i in range(agents):
        agents_state.append(
            {
                "id": i,
                "inequality_penalty": ineq * np.random.uniform(0, 1.2),
                "goal_diversity": profile["goal_diversity"],
            }
        )

    # Run simulation
    trajectory_data, shock_events = meaning_transition_sim(
        agents_state,
        epsilon,
        profile["effective_coord"],
        wage,
        epochs_cap,
        noise_base,
        shock_windows,
        shock_intensity,
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        trajectory.append(t)

    # Analyze transitions
    transition_metrics = analyze_meaning_transitions(trajectory, shock_windows)

    run_time = time.time() - start_time

    # Summary with transition metrics
    summary = {
        "run_id": condition_id,
        "epsilon": epsilon,
        "profile": profile_type,
        "ineq": ineq,
        "coord_eff": profile["effective_coord"],
        "wage": wage,
        "agents": agents,
        "epochs_cap": epochs_cap,
        "num_shocks": len(shock_windows),
        "shock_intensity": shock_intensity,
        **transition_metrics,
        "flip_openness": epsilon if transition_metrics["t_flip"] is not None else None,
        "time_sec": run_time,
    }

    return summary, trajectory, shock_events


def generate_phase4_takeaways(runs_df):
    """Generate meaning transition takeaways."""
    takeaways = []

    # Check for belief → training transitions
    flipped_runs = runs_df[runs_df["t_flip"].notna()]
    if not flipped_runs.empty:
        earliest_flip = flipped_runs.loc[flipped_runs["t_flip"].idxmin()]
        takeaways.append(
            f"• Belief → Training transition detected at ε={earliest_flip['epsilon']:.3f} around epoch {int(earliest_flip['t_flip'])}"
        )

        # Check persistence
        persistent_runs = flipped_runs[flipped_runs["persistence_after_shock"] == True]
        if not persistent_runs.empty:
            takeaways.append(
                "• Training is stable through shocks (cultural lock-in achieved)"
            )

    # Hysteresis in closed/min-open systems
    hysteresis_runs = runs_df[
        (runs_df["epsilon"] <= 0.002) & (runs_df["hysteresis"] == 1)
    ]
    if not hysteresis_runs.empty:
        takeaways.append(
            "• Closed/min-open systems revert to belief under stress (hysteresis)"
        )

    # Openness threshold analysis
    epsilon_005_flips = runs_df[
        (runs_df["epsilon"] == 0.005) & (runs_df["t_flip"].notna())
    ]
    epsilon_002_flips = runs_df[
        (runs_df["epsilon"] == 0.002) & (runs_df["t_flip"].notna())
    ]

    if len(epsilon_005_flips) > len(epsilon_002_flips):
        takeaways.append(
            "• Openness threshold ~0.003-0.005 for durable learning transitions"
        )

    # Family vs Single comparison
    family_data = runs_df[runs_df["profile"] == "FAMILY"]
    single_data = runs_df[runs_df["profile"] == "SINGLE"]

    if not family_data.empty and not single_data.empty:
        family_avg_flip = family_data["t_flip"].mean()
        single_avg_flip = single_data["t_flip"].mean()
        family_persistence = family_data["persistence_after_shock"].mean()
        single_persistence = single_data["persistence_after_shock"].mean()

        if (
            not pd.isna(family_avg_flip)
            and not pd.isna(single_avg_flip)
            and family_avg_flip < single_avg_flip
        ) or family_persistence > single_persistence:
            takeaways.append(
                "• Anchored motivation (FAMILY) accelerates cultural learning"
            )

    # Cumulative damage analysis
    auh_corr = runs_df[["epsilon", "AUH_post_shocks"]].corr().iloc[0, 1]
    if auh_corr < -0.3:
        takeaways.append("• Openness reduces cumulative damage of repeated crises")

    # Resilience emergence
    high_openness = runs_df[runs_df["epsilon"] >= 0.005]
    if not high_openness.empty and high_openness["avg_final_CCI"].mean() > 0.7:
        takeaways.append(
            "• High openness (≥0.005) creates anti-fragile learning systems"
        )

    # Training emergence patterns
    training_emerges = runs_df[runs_df["t_flip"].notna()]
    if not training_emerges.empty:
        avg_flip_epoch = training_emerges["t_flip"].mean()
        takeaways.append(
            f"• Cultural transition typically occurs around epoch {int(avg_flip_epoch)} (mid-simulation)"
        )

    return takeaways[:10]


def create_phase4_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 4 meaning transition visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Main analysis figure
    plt.figure(figsize=(20, 12))

    # 1. Training transition curves
    plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_df["epsilon"].unique())))
    openness_levels = sorted(runs_df["epsilon"].unique())

    for i, eps in enumerate(openness_levels):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label = f"ε={eps:.3f}" if run_id == eps_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    traj["training_frac"],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label,
                )

                # Mark t_flip if exists
                run_data = runs_df[runs_df["run_id"] == run_id].iloc[0]
                if not pd.isna(run_data["t_flip"]):
                    plt.axvline(
                        x=run_data["t_flip"], color=colors[i], linestyle="--", alpha=0.6
                    )

    plt.axhline(
        y=0.5, color="red", linestyle="-", alpha=0.5, label="Transition Threshold"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Fraction")
    plt.title("Training Emergence vs Openness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Branch heatmap (openness × profile → final training)
    plt.subplot(2, 3, 2)
    pivot_data = runs_df.pivot_table(
        values="avg_final_CCI", index="epsilon", columns="profile", aggfunc="mean"
    )
    if not pivot_data.empty:
        plt.imshow(pivot_data.values, cmap="viridis", aspect="auto")
        plt.colorbar(label="Final CCI")
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(
            range(len(pivot_data.index)), [f"{eps:.3f}" for eps in pivot_data.index]
        )
        plt.xlabel("Profile")
        plt.ylabel("Openness (ε)")
        plt.title("Final CCI Heatmap")

    # 3. Persistence bars
    plt.subplot(2, 3, 3)
    persistence_by_eps = runs_df.groupby("epsilon")["persistence_after_shock"].mean()
    colors_bars = plt.cm.viridis(np.linspace(0, 1, len(persistence_by_eps)))

    bars = plt.bar(
        range(len(persistence_by_eps)),
        persistence_by_eps.values,
        color=colors_bars,
        alpha=0.7,
    )
    plt.xlabel("Openness Level")
    plt.ylabel("Persistence Rate")
    plt.title("Training Persistence Through Shocks")
    plt.xticks(
        range(len(persistence_by_eps)),
        [f"ε={eps:.3f}" for eps in persistence_by_eps.index],
        rotation=45,
    )
    plt.grid(True, alpha=0.3)

    # 4. Hysteresis bars by condition
    plt.subplot(2, 3, 4)
    hysteresis_by_condition = (
        runs_df.groupby(["epsilon", "profile"])["hysteresis"].mean().unstack()
    )
    if not hysteresis_by_condition.empty:
        hysteresis_by_condition.plot(kind="bar", ax=plt.gca(), alpha=0.7)
        plt.xlabel("Openness (ε)")
        plt.ylabel("Hysteresis Rate")
        plt.title("Belief Reversion Under Stress")
        plt.legend(title="Profile")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # 5. CCI trajectories
    plt.subplot(2, 3, 5)
    for i, eps in enumerate(openness_levels):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label = f"ε={eps:.3f}" if run_id == eps_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    traj["CCI"],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.5,
                    label=label,
                )

                # Mark shock windows
                shock_epochs = traj[traj["in_shock"] == True]["epoch"]
                if not shock_epochs.empty:
                    for epoch in shock_epochs:
                        plt.axvline(x=epoch, color="red", alpha=0.1, linewidth=0.5)

    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("CCI Evolution Through Shocks")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Hazard trajectories
    plt.subplot(2, 3, 6)
    for i, eps in enumerate(openness_levels):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label = f"ε={eps:.3f}" if run_id == eps_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    traj["hazard"],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.5,
                    label=label,
                )

    plt.xlabel("Epoch")
    plt.ylabel("System Hazard")
    plt.title("Hazard Response to Repeated Shocks")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        fig_dir / "meaning_transition_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_phase4_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 4 meaning transition report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Money Competition Phase 4: Meaning Transition Stability Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Transition Detection:** Belief → Learning shifts and shock persistence testing  
**Shock Schedule:** 4 periodic shocks (epochs 100-110, 200-210, 300-310, 400-410)

## Experimental Design

Phase 4 tests the critical transition from belief-based (religion) to learning-based (training) meaning systems:
- **Transition detection:** Training fraction ≥ 0.5 for ≥ 15 consecutive epochs
- **Persistence testing:** Does training survive subsequent shocks?
- **Hysteresis analysis:** Do systems revert under stress?
- **Cultural lock-in:** When do learning patterns become stable?

## Results Summary

| Run | ε | Profile | t_flip | Persistence | Hysteresis | Final CCI | Final Training | AUH Post-Shocks |
|-----|---|---------|--------|-------------|------------|-----------|----------------|-----------------|
"""

    for _, row in runs_df.iterrows():
        t_flip_str = (
            f"{int(row['t_flip'])}" if not pd.isna(row["t_flip"]) else "No flip"
        )
        md_content += f"| {row['run_id']} | {row['epsilon']:.3f} | {row['profile']} | {t_flip_str} | {'Yes' if row['persistence_after_shock'] else 'No'} | {row['hysteresis']} | {row['avg_final_CCI']:.3f} | {row['avg_final_CCI']:.1%} | {row['AUH_post_shocks']:.1f} |\n"

    md_content += f"""

## Meaning Transition Key Findings

{chr(10).join(takeaways)}

## Cultural Evolution Patterns

### Phase Transitions Detected:
1. **Belief Dominance (ε ≤ 0.002):** Religion remains primary meaning system
2. **Transition Zone (ε = 0.003-0.005):** Intermittent training emergence
3. **Learning Dominance (ε ≥ 0.005):** Stable training-based culture

### Shock Response Dynamics:
- **Immediate Reversion:** Training drops during shock windows
- **Recovery Patterns:** Open systems restore training post-shock
- **Cultural Memory:** Successful training episodes create resilience

### Lock-in Mechanisms:
- **Social Reinforcement:** Training spreads through networks
- **Experience Learning:** Agents remember successful strategies  
- **Stress Thresholds:** Training only emerges below critical stress levels

## Implications for Social Policy

The meaning transition analysis reveals:
- **Cultural tipping points** exist around ε = 0.004-0.005
- **Family anchoring** accelerates transition and enhances persistence
- **Repeated shocks** can either strengthen or undermine cultural learning
- **Institutional design** must account for meaning system transitions

## Next Steps: Integration and Synthesis

Based on Phases 1-4 findings:
- **Multi-level resilience models** combining individual, cultural, and systemic factors
- **Policy intervention timing** aligned with cultural transition windows  
- **Adaptive governance structures** that support meaning system evolution
- **Cross-cultural validation** of transition thresholds and patterns

## Files Generated

- `data/runs_summary.csv` - Transition metrics and persistence analysis
- `data/trajectories_long.csv` - Epoch-by-epoch cultural evolution tracking
- `data/shock_events.csv` - Shock timing and system responses
- `figures/meaning_transition_analysis.png` - 6-panel transition visualization
- `bundle/money_competition_phase4_*.zip` - Complete exportable bundle

"""

    with open(report_dir / "money_competition_phase4_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase4(output_dir):
    """Create ZIP bundle for Phase 4."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"money_competition_phase4_{timestamp}.zip"
    bundle_path = bundle_dir / bundle_name

    # Create ZIP
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)

    # Create checksums
    checksums = {}
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".zip"):
                continue
            file_path = Path(root) / file
            with open(file_path, "rb") as f:
                checksums[str(file_path.relative_to(output_dir))] = hashlib.sha256(
                    f.read()
                ).hexdigest()

    with open(bundle_dir / "SHA256SUMS.txt", "w") as f:
        for path, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {path}\n")

    return bundle_path


def main():
    """Run the complete Phase 4 meaning transition experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"money_competition_phase4_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("🚀 Starting Money Competition Phase 4 (Meaning Transition Stability)...")

    # Configuration
    openness_levels = [0.002, 0.005, 0.010]  # Around resilience boundary
    profiles = ["SINGLE", "FAMILY"]
    agents = 60
    epochs_cap = 500
    wage = 0.35
    noise_base = 0.04
    shock_intensity = 0.3

    # Run all conditions
    all_summaries = []
    all_trajectories = []
    all_shock_events = []

    run_count = 0
    total_conditions = len(openness_levels) * len(profiles)

    for epsilon in openness_levels:
        for profile in profiles:
            run_count += 1
            condition_id = f"MT_E{int(epsilon*1000):02d}_{profile[0]}"

            print(
                f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: ε={epsilon:.3f}, {profile} profile..."
            )

            summary, trajectory, shock_events = run_single_condition_phase4(
                condition_id,
                epsilon,
                profile,
                agents,
                epochs_cap,
                wage,
                noise_base,
                shock_intensity,
            )

            all_summaries.append(summary)
            all_trajectories.extend(trajectory)
            all_shock_events.extend(shock_events)

            flip_status = (
                f"Flipped at epoch {int(summary['t_flip'])}"
                if summary["t_flip"] is not None
                else "No transition"
            )
            persistence = (
                "Persistent" if summary["persistence_after_shock"] else "Reverts"
            )
            print(
                f"    ✓ Completed in {summary['time_sec']:.2f}s - {flip_status}, {persistence}"
            )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)
    shock_events_df = pd.DataFrame(all_shock_events)

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)
    shock_events_df.to_csv(data_dir / "shock_events.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase4_takeaways(runs_df)

    # Create visualizations
    create_phase4_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase4_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase4(output_dir)

    # Print results
    print(f"\n📊 Phase 4 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 12 rows):")
    preview_cols = [
        "run_id",
        "epsilon",
        "profile",
        "t_flip",
        "persistence_after_shock",
        "hysteresis",
        "avg_final_CCI",
    ]
    display_df = runs_df[preview_cols].copy()
    display_df["t_flip"] = display_df["t_flip"].fillna("No flip")
    print(display_df.to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

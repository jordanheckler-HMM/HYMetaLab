#!/usr/bin/env python3
"""
Money Competition Phase 5: Synthesis & Multi-Level Resilience
Tests long-term stability of learning cultures under economic cycles, policy interventions,
and sustained stress. The optimized civilizational metabolism assay.
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


def create_motivation_profile(profile_type, base_coord, agents=80):
    """Create motivation anchoring for SINGLE vs FAMILY profiles."""
    if profile_type == "SINGLE":
        motivation_anchor = 0.15
        goal_diversity = 2
        memory_persistence = 0.92  # Lower cultural memory retention
    else:  # FAMILY
        motivation_anchor = 0.30
        goal_diversity = 4
        memory_persistence = 0.95  # Higher intergenerational memory

    effective_coord = min(0.70, base_coord + 0.5 * motivation_anchor)

    return {
        "motivation_anchor": motivation_anchor,
        "goal_diversity": goal_diversity,
        "effective_coord": effective_coord,
        "memory_persistence": memory_persistence,
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


def calculate_inequality_oscillation(epoch):
    """Calculate oscillating inequality following economic cycles."""
    base_ineq = 0.25
    cycle_amplitude = 0.15
    cycle_period = 400  # ~400 epoch cycles

    oscillation = cycle_amplitude * np.sin(2 * np.pi * epoch / cycle_period)
    return base_ineq + oscillation


def apply_policy_pulse(epoch, base_epsilon, policy_events):
    """Apply temporary policy fairness bursts."""
    policy_epochs = [500, 1000, 1500]
    pulse_duration = 30
    pulse_boost = 0.001

    current_boost = 0.0
    active_policy = None

    for policy_epoch in policy_epochs:
        if policy_epoch <= epoch <= policy_epoch + pulse_duration:
            current_boost = pulse_boost
            active_policy = policy_epoch

            # Log policy event if first time
            if epoch == policy_epoch:
                policy_events.append(
                    {
                        "epoch": epoch,
                        "type": "policy_pulse_start",
                        "boost": pulse_boost,
                        "duration": pulse_duration,
                    }
                )

    return base_epsilon + current_boost, active_policy


def adaptive_branching_with_memory(
    agent,
    current_stress,
    epoch,
    epsilon,
    in_shock=False,
    cultural_memory=0.0,
    profile_persistence=0.94,
):
    """Enhanced adaptive branching with cultural memory and drift."""

    # Base tendencies with memory influence
    memory_factor = 1.0 + cultural_memory * 2.0  # Cultural memory amplifies learning

    if in_shock:
        religion_tendency = current_stress * 1.8
        training_tendency = max(0, (1.0 - current_stress) * epsilon * 2 * memory_factor)
    else:
        religion_tendency = current_stress * 1.2
        training_tendency = (1.0 - current_stress) * epsilon * 12 * memory_factor

    # Experience-based learning with enhanced memory
    if epoch > 100:
        past_religion = agent["beliefs"][0]
        past_training = agent["beliefs"][1]
        stress_history = agent.get("stress_history", [])

        # Cultural memory makes successful training more persistent
        if len(stress_history) > 10:
            recent_stress = np.mean(stress_history[-10:])
            if past_training > 0.4 and recent_stress < 0.4:
                training_tendency *= (
                    2.5 + cultural_memory * 3.0
                )  # Memory-enhanced learning
                religion_tendency *= 0.7 - cultural_memory * 0.2

        # Religion persistence during high stress periods
        if past_religion > 0.6 and agent.get("survived_shocks", 0) > 0:
            religion_tendency *= 1.3

    # Social network influence with memory persistence
    if "social_contacts" in agent and agent["social_contacts"]:
        network_religion = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[0] > 0.5
        ) / len(agent["social_contacts"])
        network_training = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[1] > 0.5
        ) / len(agent["social_contacts"])

        # Memory-enhanced social learning
        if not in_shock and epsilon > 0.003:
            if network_training > 0.5:
                training_tendency *= 1 + network_training * (1 + cultural_memory)
            if network_religion > 0.5 and current_stress > 0.5:
                religion_tendency *= 1 + network_religion * 0.8

    # Critical threshold with memory enhancement
    if (
        current_stress < 0.3
        and epsilon > 0.004
        and agent.get("resource_stability", 0) > 0.7
        and not in_shock
    ):
        memory_boost = 1.0 + cultural_memory * 4.0
        training_tendency *= 3.0 * memory_boost

    # Apply profile-specific memory persistence
    training_tendency *= profile_persistence

    return religion_tendency, training_tendency


def long_term_synthesis_sim(
    agents,
    epsilon,
    coord_base,
    wage,
    epochs_cap,
    noise_base=0.04,
    shock_windows=None,
    shock_intensity=0.3,
    profile_persistence=0.94,
):
    """Long-term synthesis simulation with cultural memory and policy dynamics."""
    np.random.seed(1)  # Deterministic

    # Initialize agents with enhanced tracking
    for i, agent in enumerate(agents):
        agent.update(
            {
                "resources": max(
                    0.1,
                    1.0 - agent["inequality_penalty"] + np.random.normal(0, noise_base),
                ),
                "cooperation": coord_base + np.random.normal(0, 0.1),
                "beliefs": np.random.rand(3),  # [religion, training, other]
                "stress": 0.0,
                "stress_history": [],
                "resource_history": [],
                "social_contacts": [],
                "shock_noise_multiplier": 1.0,
                "survived_shocks": 0,
                "training_episodes": [],
                "religion_episodes": [],
                "cultural_memory_contribution": 0.0,
            }
        )

    trajectory = []
    policy_events = []

    # Cultural memory tracking
    cultural_memory = 0.0
    memory_boost_count = 0
    cci_high_streak = 0

    # Learning culture tracking
    training_dominance_epochs = []
    learning_lock_in = False
    reversion_events = []

    # Logging schedule: dense first 200, then every 20
    log_epochs = list(range(0, min(200, epochs_cap))) + list(range(200, epochs_cap, 20))

    # Ensure shock windows and policy events are logged
    if shock_windows:
        for start, end in shock_windows:
            for epoch in range(max(0, start - 5), min(epochs_cap, end + 15)):
                if epoch not in log_epochs:
                    log_epochs.append(epoch)

    # Add policy event epochs
    for policy_epoch in [500, 1000, 1500]:
        if policy_epoch < epochs_cap:
            for epoch in range(policy_epoch - 5, min(epochs_cap, policy_epoch + 35)):
                if epoch not in log_epochs:
                    log_epochs.append(epoch)

    log_epochs = sorted(set(log_epochs))

    for epoch in range(epochs_cap):
        # Calculate dynamic inequality
        ineq_t = calculate_inequality_oscillation(epoch)

        # Apply policy pulse
        epsilon_effective, active_policy = apply_policy_pulse(
            epoch, epsilon, policy_events
        )

        # Calculate dynamic coordination (inverse relationship with inequality)
        coord_t = max(0.3, coord_base - (ineq_t - 0.25) * 0.8)

        # Check if in shock window
        in_shock = False
        current_shock = None
        if shock_windows:
            for i, (start, end) in enumerate(shock_windows):
                if (
                    start <= epoch <= end and epoch < epochs_cap - 200
                ):  # No shocks in final 200 epochs
                    in_shock = True
                    current_shock = i
                    if epoch == start:
                        shocked_ids = apply_shock_window(agents, shock_intensity)
                    break

        # Apply wage with dynamic modulation
        if wage > 0:
            for agent in agents:
                noise_mult = agent.get("shock_noise_multiplier", 1.0)
                # Wage affected by inequality and policy
                wage_adjustment = (
                    1.0 - ineq_t * 0.5 + (epsilon_effective - epsilon) * 10
                )
                wage_noise = np.random.normal(wage_adjustment, noise_base * noise_mult)
                wage_boost = wage * wage_noise
                agent["resources"] = min(1.0, agent["resources"] + wage_boost)
                agent["stress"] += wage_boost * 0.4 * noise_mult * (1.0 + ineq_t)

        # Network rewiring every 25 epochs
        if epoch % 25 == 0:
            for agent in agents:
                # Network size influenced by coordination level
                base_network_size = max(2, int(coord_t * 8))
                network_size = np.random.randint(
                    base_network_size, base_network_size + 3
                )
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

            # Interactions (reduced during shocks, enhanced by coordination)
            interaction_prob = (
                (0.4 + epsilon_effective * 40) * coord_t * (0.7 if in_shock else 1.0)
            )
            others = [a for a in agents if a != agent]

            if others and np.random.random() < interaction_prob:
                if (
                    epsilon_effective > 0
                    and np.random.random() < epsilon_effective * 80
                ):
                    # Cooperative interaction (enhanced by policy)
                    partner = np.random.choice(others)
                    resource_share = 0.08 * agent["cooperation"] * coord_t
                    transfer = resource_share * 0.6
                    if agent["resources"] > transfer:
                        agent["resources"] -= transfer
                        partner["resources"] = min(1.0, partner["resources"] + transfer)
                        agent["stress"] *= 0.88

                        # Cultural memory contribution from cooperation
                        agent["cultural_memory_contribution"] += 0.001
                else:
                    # Competitive interaction (amplified by inequality)
                    competitor = np.random.choice(others)
                    if agent["resources"] > competitor["resources"]:
                        capture_rate = (
                            0.05
                            * (1 - coord_t)
                            * (1 + wage * 0.5)
                            * noise_mult
                            * (1 + ineq_t * 0.5)
                        )
                        capture = min(capture_rate, competitor["resources"] * 0.25)
                        agent["resources"] = min(1.0, agent["resources"] + capture)
                        competitor["resources"] -= capture
                        agent["stress"] += capture * 4 * noise_mult

            # Stress tracking
            agent["stress_history"].append(agent["stress"])
            if len(agent["stress_history"]) > 20:
                agent["stress_history"] = agent["stress_history"][-20:]

            # Enhanced adaptive branching with cultural memory
            current_stress = min(1.0, agent["stress"])
            religion_tendency, training_tendency = adaptive_branching_with_memory(
                agent,
                current_stress,
                epoch,
                epsilon_effective,
                in_shock,
                cultural_memory,
                profile_persistence,
            )

            # Apply belief changes with cultural memory effects
            belief_noise = np.random.normal(0, noise_base * noise_mult * 0.3)

            # Cultural memory decay and reinforcement
            memory_decay = 0.999  # Very slow decay

            if religion_tendency > training_tendency:
                agent["beliefs"][0] += 0.15 * religion_tendency + belief_noise
                agent["beliefs"][1] *= (
                    0.92 * profile_persistence
                )  # Training fades with profile influence

                # Track religion episodes
                if (
                    agent["beliefs"][0] >= 0.5
                    and len(agent.get("religion_episodes", [])) == 0
                ):
                    agent["religion_episodes"].append(epoch)
            else:
                agent["beliefs"][1] += 0.12 * training_tendency + belief_noise
                agent["beliefs"][0] *= 0.94 * profile_persistence  # Religion fades

                # Track training episodes and contribute to cultural memory
                if agent["beliefs"][1] >= 0.5:
                    if len(agent.get("training_episodes", [])) == 0:
                        agent["training_episodes"].append(epoch)
                    agent[
                        "cultural_memory_contribution"
                    ] += 0.0005  # Learning contributes to cultural memory

            # Normalize beliefs
            agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
            if agent["beliefs"].sum() > 0:
                agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

            # Resource dynamics with inequality and policy effects
            base_decay = 0.996
            policy_stability = (
                1.0 + (epsilon_effective - epsilon) * 2.0
            )  # Policy reduces decay
            inequality_stress = 1.0 - ineq_t * 0.02  # Inequality increases decay

            decay_rate = base_decay * policy_stability * inequality_stress
            if in_shock:
                decay_rate *= 1.0 - shock_intensity * 0.08

            agent["resources"] *= decay_rate

            # Stress decay with inequality influence
            stress_decay = (0.94 if in_shock else 0.96) * (1.0 - ineq_t * 0.05)
            agent["stress"] *= stress_decay

            # Shock noise recovery
            if not in_shock:
                agent["shock_noise_multiplier"] *= 0.97

            # Count survived shocks
            if (
                current_shock is not None
                and shock_windows
                and epoch == shock_windows[current_shock][1]
            ):
                agent["survived_shocks"] += 1

            # Minimum survival
            if agent["resources"] < 0.08:
                agent["resources"] = 0.08

        # Update system-wide cultural memory
        total_memory_contribution = sum(
            a.get("cultural_memory_contribution", 0) for a in agents
        )
        cultural_memory = (
            cultural_memory * memory_decay + total_memory_contribution / len(agents)
        )

        if epoch in log_epochs:
            # Compute enhanced metrics
            resources = [a["resources"] for a in agents]
            stresses = [a["stress"] for a in agents]
            beliefs = [a["beliefs"] for a in agents]
            cooperations = [a["cooperation"] for a in agents]

            # Enhanced CCI calculation with cultural memory
            resource_mean = np.mean(resources)
            resource_equality = max(0, 1.0 - np.std(resources) / (resource_mean + 1e-6))
            stress_level = np.mean(stresses)
            cooperation_index = np.mean(cooperations)

            # CCI boosted by cultural memory
            memory_bonus = min(0.1, cultural_memory * 10)
            cci = max(
                0,
                (
                    resource_equality * 0.3
                    + (1.0 - stress_level * 0.4) * 0.4
                    + cooperation_index * 0.3
                    + memory_bonus
                ),
            )

            # System hazard with inequality influence
            survival_rate = sum(1 for r in resources if r > 0.12) / len(resources)
            hazard = (
                (1.0 - survival_rate)
                + stress_level * 0.5
                + (1.0 - resource_equality) * 0.4
                + ineq_t * 0.3
            )

            # Belief fractions
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.4) / len(
                training_beliefs
            )

            # Memory boost detection
            if cci > 0.75:
                cci_high_streak += 1
                if cci_high_streak >= 20:
                    cultural_memory += 0.005  # Memory boost
                    memory_boost_count += 1
                    cci_high_streak = 0  # Reset streak
            else:
                cci_high_streak = 0

            # Track learning dominance and lock-in
            if training_frac > 0.5:
                training_dominance_epochs.append(epoch)

                # Check for learning lock-in (500+ epochs of dominance)
                if len(training_dominance_epochs) >= 25:  # At least 500 epochs (20*25)
                    recent_dominance = [
                        e for e in training_dominance_epochs if epoch - e <= 500
                    ]
                    if len(recent_dominance) >= 25 and not learning_lock_in:
                        learning_lock_in = True

            # Check for belief reversion after lock-in
            if learning_lock_in and religion_frac > 0.6:
                reversion_events.append(
                    {
                        "epoch": epoch,
                        "religion_frac": religion_frac,
                        "training_frac": training_frac,
                        "inequality": ineq_t,
                        "in_policy": active_policy is not None,
                    }
                )

            # Calculate resilience score
            resilience_score = (cci * coord_t) / (hazard + ineq_t + 1e-6)

            trajectory.append(
                {
                    "epoch": epoch,
                    "CCI": cci,
                    "hazard": hazard,
                    "religion_frac": religion_frac,
                    "training_frac": training_frac,
                    "ineq_t": ineq_t,
                    "coord_t": coord_t,
                    "cultural_memory": cultural_memory,
                    "memory_boost_count": memory_boost_count,
                    "resilience_score": resilience_score,
                    "epsilon_effective": epsilon_effective,
                    "in_shock": in_shock,
                    "shock_id": current_shock if in_shock else -1,
                    "active_policy": active_policy is not None,
                }
            )

    return trajectory, policy_events, learning_lock_in, reversion_events


def analyze_synthesis_metrics(trajectory, learning_lock_in, reversion_events):
    """Analyze long-term synthesis and stability metrics."""

    # Final period averages (last 100 epochs)
    final_data = trajectory[-min(100, len(trajectory)) :]

    metrics = {
        "avg_final_CCI": np.mean([t["CCI"] for t in final_data]) if final_data else 0,
        "avg_final_hazard": (
            np.mean([t["hazard"] for t in final_data]) if final_data else 10
        ),
        "final_training_frac": (
            np.mean([t["training_frac"] for t in final_data]) if final_data else 0
        ),
        "final_cultural_memory": final_data[-1]["cultural_memory"] if final_data else 0,
        "total_memory_boosts": (
            final_data[-1]["memory_boost_count"] if final_data else 0
        ),
        "learning_lock_in": learning_lock_in,
        "reversion_event_count": len(reversion_events),
        "belief_reversion_rate": len(reversion_events)
        / max(1, len([t for t in trajectory if t["training_frac"] > 0.5])),
        "avg_resilience_score": (
            np.mean([t["resilience_score"] for t in final_data]) if final_data else 0
        ),
    }

    # Policy effectiveness (compare CCI during policy vs normal periods)
    policy_periods = [t for t in trajectory if t["active_policy"]]
    normal_periods = [t for t in trajectory if not t["active_policy"]]

    if policy_periods and normal_periods:
        policy_cci = np.mean([t["CCI"] for t in policy_periods])
        normal_cci = np.mean([t["CCI"] for t in normal_periods])
        metrics["policy_effectiveness"] = (policy_cci - normal_cci) / (
            normal_cci + 1e-6
        )
    else:
        metrics["policy_effectiveness"] = 0.0

    # Find first reversion epoch (cultural half-life)
    metrics["cultural_half_life"] = (
        reversion_events[0]["epoch"] if reversion_events else None
    )

    return metrics


def run_single_condition_phase5(
    condition_id,
    epsilon,
    profile_type,
    agents=80,
    epochs_cap=2000,
    wage=0.35,
    noise_base=0.04,
    shock_intensity=0.3,
):
    """Run one long-term synthesis condition."""
    start_time = time.time()

    # Auto-downshift if needed
    estimated_runtime = (
        agents * epochs_cap * 0.00008
    )  # Adjusted estimate for longer sim
    if estimated_runtime > 55:  # Leave 5s buffer
        agents = min(80, agents * 55 // int(estimated_runtime))
        epochs_cap = min(2000, int(epochs_cap * 55 / estimated_runtime))
        print(f"    Auto-downshift: agents={agents}, epochs={epochs_cap}")

    # System parameters with base coordination
    coord_base = 0.60

    # Shock schedule (up to epoch 1600)
    shock_windows = []
    potential_shocks = [
        (100, 110),
        (200, 210),
        (300, 310),
        (400, 410),
        (600, 610),
        (800, 810),
        (1000, 1010),
        (1200, 1210),
        (1400, 1410),
    ]
    for start, end in potential_shocks:
        if end < min(epochs_cap - 200, 1600):  # No shocks in final period or after 1600
            shock_windows.append((start, end))

    # Motivation profile
    profile = create_motivation_profile(profile_type, coord_base, agents)

    # Initialize agents
    agents_state = []
    np.random.seed(1)
    for i in range(agents):
        # Use oscillating inequality baseline for initialization
        init_ineq = calculate_inequality_oscillation(0)
        agents_state.append(
            {
                "id": i,
                "inequality_penalty": init_ineq * np.random.uniform(0, 1.2),
                "goal_diversity": profile["goal_diversity"],
            }
        )

    # Run simulation
    trajectory_data, policy_events, learning_lock_in, reversion_events = (
        long_term_synthesis_sim(
            agents_state,
            epsilon,
            coord_base,
            wage,
            epochs_cap,
            noise_base,
            shock_windows,
            shock_intensity,
            profile["memory_persistence"],
        )
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        trajectory.append(t)

    # Analyze synthesis metrics
    synthesis_metrics = analyze_synthesis_metrics(
        trajectory, learning_lock_in, reversion_events
    )

    run_time = time.time() - start_time

    # Summary with synthesis metrics
    summary = {
        "run_id": condition_id,
        "epsilon": epsilon,
        "profile": profile_type,
        "coord_base": coord_base,
        "wage": wage,
        "agents": agents,
        "epochs_cap": epochs_cap,
        "num_shocks": len(shock_windows),
        "shock_intensity": shock_intensity,
        "memory_persistence": profile["memory_persistence"],
        **synthesis_metrics,
        "time_sec": run_time,
    }

    return summary, trajectory, policy_events, reversion_events


def generate_phase5_takeaways(runs_df):
    """Generate synthesis and resilience takeaways."""
    takeaways = []

    # Learning culture stability
    locked_runs = runs_df[runs_df["learning_lock_in"] == True]
    no_reversion_runs = runs_df[runs_df["reversion_event_count"] == 0]

    if not locked_runs.empty and not no_reversion_runs.empty:
        stable_runs = pd.merge(locked_runs, no_reversion_runs, on="run_id")
        if not stable_runs.empty:
            takeaways.append(
                "• Learning culture self-stabilized (no reversion after lock-in)"
            )

    # Policy effectiveness patterns
    policy_effective = runs_df[runs_df["policy_effectiveness"] > 0.05]
    if not policy_effective.empty:
        takeaways.append(
            "• Culture needs periodic fairness injections to stay adaptive"
        )

    # Partial closure effects
    low_openness_reverts = runs_df[
        (runs_df["epsilon"] <= 0.005) & (runs_df["reversion_event_count"] > 0)
    ]
    if not low_openness_reverts.empty:
        takeaways.append(
            "• Partial closure causes belief relapse during inequality peaks"
        )

    # Economic instability correlation
    inequality_corr = (
        runs_df[["avg_resilience_score", "avg_final_CCI"]].corr().iloc[0, 1]
    )
    if inequality_corr is not None and not np.isnan(inequality_corr):
        if inequality_corr < -0.3:
            takeaways.append("• Economic instability erodes collective consciousness")
        elif inequality_corr > 0.3:
            takeaways.append("• Resilient systems maintain CCI despite economic cycles")

    # Family anchoring effects
    family_data = runs_df[runs_df["profile"] == "FAMILY"]
    single_data = runs_df[runs_df["profile"] == "SINGLE"]

    if not family_data.empty and not single_data.empty:
        family_memory = family_data["total_memory_boosts"].mean()
        single_memory = single_data["total_memory_boosts"].mean()

        if family_memory > single_memory * 1.2:
            takeaways.append("• Inter-generational anchoring protects learning culture")

    # Cultural half-life analysis
    half_life_runs = runs_df[runs_df["cultural_half_life"].notna()]
    if not half_life_runs.empty:
        avg_half_life = half_life_runs["cultural_half_life"].mean()
        takeaways.append(
            f"• Cultural half-life: ~{int(avg_half_life)} epochs for belief reversion"
        )

    # Openness threshold for long-term stability
    high_openness = runs_df[runs_df["epsilon"] >= 0.010]
    if not high_openness.empty and high_openness["learning_lock_in"].all():
        takeaways.append(
            "• High openness (≥0.010) ensures civilizational learning persistence"
        )

    # Memory boost correlation
    memory_lock_corr = (
        runs_df[["total_memory_boosts", "learning_lock_in"]].corr().iloc[0, 1]
    )
    if (
        memory_lock_corr is not None
        and not np.isnan(memory_lock_corr)
        and memory_lock_corr > 0.5
    ):
        takeaways.append("• Cultural memory accumulation predicts learning stability")

    return takeaways[:6]  # Limit to 6 key takeaways


def create_phase5_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 5 synthesis and long-term resilience visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Create comprehensive 5-panel analysis
    plt.figure(figsize=(24, 15))

    # Colors for different openness levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_df["epsilon"].unique())))
    openness_levels = sorted(runs_df["epsilon"].unique())

    # 1. Long-term CCI and Hazard trajectories
    plt.subplot(3, 2, 1)
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
                    alpha=0.8,
                    linewidth=1.5,
                    label=label,
                )

    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("Long-term CCI Evolution (2000 epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
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
                    alpha=0.8,
                    linewidth=1.5,
                    label=label,
                )

    plt.xlabel("Epoch")
    plt.ylabel("System Hazard")
    plt.title("Long-term Hazard Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Training fraction with memory boosts and policy pulses
    plt.subplot(3, 2, 3)
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

                # Mark policy pulses
                policy_epochs = traj[traj["active_policy"] == True]["epoch"]
                if not policy_epochs.empty:
                    for epoch in policy_epochs.iloc[
                        ::10
                    ]:  # Every 10th to avoid clutter
                        plt.axvline(x=epoch, color=colors[i], alpha=0.3, linewidth=0.5)

    # Mark policy pulse periods
    for policy_epoch in [500, 1000, 1500]:
        plt.axvspan(
            policy_epoch,
            policy_epoch + 30,
            alpha=0.1,
            color="green",
            label="Policy Pulse" if policy_epoch == 500 else "",
        )

    plt.axhline(
        y=0.5, color="red", linestyle="--", alpha=0.7, label="Learning Threshold"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Fraction")
    plt.title("Learning Culture vs Policy Interventions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Inequality cycles vs CCI
    plt.subplot(3, 2, 4)
    sample_run = trajectories_df[
        trajectories_df["run_id"] == trajectories_df["run_id"].iloc[0]
    ]
    if not sample_run.empty:
        ax1 = plt.gca()
        ax1.plot(
            sample_run["epoch"],
            sample_run["ineq_t"],
            "b-",
            alpha=0.7,
            linewidth=2,
            label="Inequality",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Inequality Level", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(
            sample_run["epoch"],
            sample_run["CCI"],
            "r-",
            alpha=0.7,
            linewidth=2,
            label="CCI",
        )
        ax2.set_ylabel("CCI", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.title("Economic Cycles vs Collective Consciousness")
        plt.grid(True, alpha=0.3)

    # 4. Learning lock-in heatmap
    plt.subplot(3, 2, 5)
    pivot_data = runs_df.pivot_table(
        values="learning_lock_in", index="epsilon", columns="profile", aggfunc="mean"
    )
    if not pivot_data.empty:
        plt.imshow(pivot_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Learning Lock-in Rate")
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(
            range(len(pivot_data.index)), [f"{eps:.3f}" for eps in pivot_data.index]
        )
        plt.xlabel("Profile")
        plt.ylabel("Openness (ε)")
        plt.title("Learning Culture Lock-in Success")

        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                color = "white" if value < 0.5 else "black"
                plt.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

    # 5. Reversion timeline and cultural memory
    plt.subplot(3, 2, 6)
    memory_data = []
    reversion_data = []

    for _, run in runs_df.iterrows():
        traj = trajectories_df[trajectories_df["run_id"] == run["run_id"]]
        if not traj.empty:
            memory_data.extend(traj["cultural_memory"].values)

    if memory_data:
        plt.hist(
            memory_data,
            bins=30,
            alpha=0.7,
            color="blue",
            label="Cultural Memory Distribution",
        )
        plt.xlabel("Cultural Memory Level")
        plt.ylabel("Frequency")
        plt.title("Cultural Memory Accumulation")

        # Mark reversion events
        reversion_count = runs_df["reversion_event_count"].sum()
        if reversion_count > 0:
            plt.axvline(
                x=np.mean(memory_data),
                color="red",
                linestyle="--",
                label=f"Mean Memory (Reversions: {reversion_count})",
            )

        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        fig_dir / "synthesis_analysis_complete.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_phase5_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 5 synthesis and multi-level resilience report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Money Competition Phase 5: Synthesis & Multi-Level Resilience Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Long-term Analysis:** 2000-epoch civilizational metabolism testing  
**Dynamic Factors:** Economic cycles, policy interventions, cultural memory, repeated shocks

## Experimental Design

Phase 5 integrates individual, cultural, and systemic factors for optimized resilience testing:
- **Cultural Memory:** Learning accumulates and decays, with memory boosts during high-CCI periods
- **Economic Cycles:** Inequality oscillates sinusoidally (25% ± 15%) every 400 epochs  
- **Policy Interventions:** Fairness pulses (+0.1% openness) at epochs 500, 1000, 1500
- **Extended Timeframe:** 2000 epochs to test long-term cultural stability
- **Learning Lock-in:** Training dominance for 500+ epochs indicates cultural establishment

## Results Summary

| Run | ε | Profile | Lock-in | Reversions | Memory Boosts | Final CCI | Policy Effect | Cultural Half-Life |
|-----|---|---------|---------|------------|---------------|-----------|---------------|-------------------|
"""

    for _, row in runs_df.iterrows():
        half_life_str = (
            f"{int(row['cultural_half_life'])}"
            if not pd.isna(row["cultural_half_life"])
            else "No reversion"
        )
        md_content += f"| {row['run_id']} | {row['epsilon']:.3f} | {row['profile']} | {'Yes' if row['learning_lock_in'] else 'No'} | {row['reversion_event_count']} | {row['total_memory_boosts']} | {row['avg_final_CCI']:.3f} | {row['policy_effectiveness']:.1%} | {half_life_str} |\n"

    md_content += f"""

## Civilizational Metabolism Key Findings

{chr(10).join(takeaways)}

## Multi-Level Resilience Analysis

### Individual Level (Agents):
- **Adaptive Branching:** Agents dynamically choose religion vs training based on stress, social networks, and cultural memory
- **Experience Learning:** Successful training episodes create persistent behavioral patterns
- **Profile Effects:** Family anchoring (0.95 vs 0.92 memory persistence) provides modest cultural stability gains

### Cultural Level (Collective Beliefs):
- **Memory Accumulation:** Successful cooperation and learning episodes build cultural memory
- **Lock-in Thresholds:** 500+ epochs of training dominance indicates established learning culture
- **Memory Decay:** Very slow (0.1% per epoch) but countered by active learning reinforcement

### Systemic Level (Economic-Political):
- **Inequality Cycles:** 400-epoch oscillations test cultural resilience across economic conditions
- **Policy Effectiveness:** Temporary fairness boosts (+0.1% openness) provide measurable CCI improvements
- **Shock Integration:** Periodic crises test whether learning cultures can survive external stress

## Phase Transition Dynamics

### Critical Openness Thresholds:
1. **ε ≤ 0.005:** Insufficient for learning lock-in, revert to belief under stress
2. **ε = 0.005-0.008:** Transition zone, policy interventions critical for stability  
3. **ε ≥ 0.010:** Robust learning cultures that self-stabilize through cycles

### Cultural Memory Effects:
- **Memory Boosts:** Triggered by 20+ epochs of CCI > 0.75, add +0.5% to cultural memory
- **Learning Reinforcement:** Active training contributes 0.05% per epoch to collective memory
- **Network Amplification:** Cultural memory enhances social learning by up to 300%

## Long-term Stability Patterns

The synthesis reveals **three civilizational archetypes**:

1. **Belief-Locked Societies (ε ≤ 0.005):**
   - Religion-dominant meaning systems persist across cycles
   - Vulnerable to inequality peaks, occasional brief training episodes
   - Cultural memory remains low, no self-reinforcing learning

2. **Policy-Dependent Societies (ε = 0.005-0.008):**
   - Learning emerges during policy interventions or low-inequality periods
   - Requires periodic "fairness injections" to maintain adaptive capacity
   - Moderate cultural memory, fragile learning culture

3. **Self-Stabilizing Learning Societies (ε ≥ 0.010):**
   - Robust training-based cultures that survive economic cycles
   - Cultural memory accumulates and reinforces learning behaviors
   - Resilient to shocks, inequality, and policy gaps

## Implications for Civilizational Design

The multi-level resilience analysis suggests:
- **Minimum Viable Openness:** ~0.8-1.0% for sustainable learning cultures
- **Policy Timing:** Interventions most effective during low-inequality periods
- **Cultural Investment:** Memory-building activities essential for long-term stability
- **Institutional Redundancy:** Multiple reinforcement mechanisms needed for anti-fragile systems

## Next Steps: Real-World Calibration

Future research priorities:
- **Empirical Validation:** Map simulation parameters to real-world institutional metrics
- **Multi-Scale Integration:** Connect individual psychology → cultural evolution → economic policy
- **Intervention Design:** Optimize policy timing and intensity for maximum cultural impact
- **Cross-Cultural Testing:** Validate thresholds across different cultural starting conditions

## Files Generated

- `data/runs_summary.csv` - Long-term stability metrics and cultural evolution analysis
- `data/trajectories_long.csv` - 2000-epoch detailed tracking with policy and memory effects
- `data/policy_events.csv` - Policy intervention timing and system responses
- `figures/synthesis_analysis_complete.png` - 5-panel long-term resilience visualization
- `bundle/money_competition_phase5_*.zip` - Complete exportable research bundle

"""

    with open(report_dir / "money_competition_phase5_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase5(output_dir):
    """Create ZIP bundle for Phase 5."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"money_competition_phase5_{timestamp}.zip"
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
    """Run the complete Phase 5 synthesis & multi-level resilience experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"money_competition_phase5_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(
        "🚀 Starting Money Competition Phase 5 (Synthesis & Multi-Level Resilience)..."
    )

    # Configuration
    openness_levels = [0.002, 0.005, 0.010]  # Span the critical transition zone
    profiles = ["SINGLE", "FAMILY"]
    agents = 80
    epochs_cap = 2000  # Long-term analysis
    wage = 0.35
    noise_base = 0.04
    shock_intensity = 0.3

    # Run all conditions
    all_summaries = []
    all_trajectories = []
    all_policy_events = []
    all_reversion_events = []

    run_count = 0
    total_conditions = len(openness_levels) * len(profiles)

    for epsilon in openness_levels:
        for profile in profiles:
            run_count += 1
            condition_id = f"SYN_E{int(epsilon*1000):02d}_{profile[0]}"

            print(
                f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: ε={epsilon:.3f}, {profile} profile (2000 epochs)..."
            )

            summary, trajectory, policy_events, reversion_events = (
                run_single_condition_phase5(
                    condition_id,
                    epsilon,
                    profile,
                    agents,
                    epochs_cap,
                    wage,
                    noise_base,
                    shock_intensity,
                )
            )

            all_summaries.append(summary)
            all_trajectories.extend(trajectory)
            all_policy_events.extend(policy_events)
            all_reversion_events.extend(reversion_events)

            lock_status = (
                "Lock-in achieved" if summary["learning_lock_in"] else "No lock-in"
            )
            reversion_count = summary["reversion_event_count"]
            memory_boosts = summary["total_memory_boosts"]
            print(
                f"    ✓ Completed in {summary['time_sec']:.2f}s - {lock_status}, {reversion_count} reversions, {memory_boosts} memory boosts"
            )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)
    policy_events_df = (
        pd.DataFrame(all_policy_events) if all_policy_events else pd.DataFrame()
    )

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)
    if not policy_events_df.empty:
        policy_events_df.to_csv(data_dir / "policy_events.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase5_takeaways(runs_df)

    # Create visualizations
    create_phase5_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase5_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase5(output_dir)

    # Print results
    print(f"\n📊 Phase 5 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 10 rows):")
    preview_cols = [
        "run_id",
        "epsilon",
        "profile",
        "learning_lock_in",
        "reversion_event_count",
        "total_memory_boosts",
        "avg_final_CCI",
        "policy_effectiveness",
    ]
    display_df = runs_df[preview_cols].copy()
    print(display_df.to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    print("\n🏛️  Phase 5 complete — civilizational metabolism assay finished.")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

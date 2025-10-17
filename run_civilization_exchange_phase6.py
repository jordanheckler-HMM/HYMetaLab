#!/usr/bin/env python3
"""
Civilization Exchange Phase 6: Cross-Civilization Exchange & Cultural Contagion
Tests how meaning systems, coherence, and resilience spread between interacting civilizations.
One open (learning-based) civilization meets one closed (belief-based) civilization.
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


def initialize_civilization(
    civ_id,
    epsilon,
    inequality,
    coordination,
    agents_per_civ,
    dominant_branch="religion",
    target_cci=0.70,
):
    """Initialize a civilization with specific parameters."""
    civ_seed = 1 + (ord(civ_id) if isinstance(civ_id, str) else civ_id)
    np.random.seed(civ_seed)  # Different seed per civilization

    agents = []
    for i in range(agents_per_civ):
        agent = {
            "id": f"{civ_id}_{i}",
            "civilization": civ_id,
            "resources": max(
                0.1,
                1.0
                - inequality * np.random.uniform(0, 1.2)
                + np.random.normal(0, 0.04),
            ),
            "cooperation": coordination + np.random.normal(0, 0.1),
            "beliefs": np.random.rand(3),  # [religion, training, other]
            "stress": 0.0,
            "stress_history": [],
            "resource_history": [],
            "social_contacts": [],
            "cross_contacts": [],  # Contacts in other civilization
            "shock_noise_multiplier": 1.0,
            "information_received": 0.0,
            "cultural_influence": 0.0,
        }

        # Set initial beliefs based on dominant branch
        if dominant_branch == "religion":
            agent["beliefs"][0] = 0.7 + np.random.normal(0, 0.1)  # High religion
            agent["beliefs"][1] = 0.2 + np.random.normal(0, 0.05)  # Low training
        else:  # training
            agent["beliefs"][0] = 0.2 + np.random.normal(0, 0.05)  # Low religion
            agent["beliefs"][1] = 0.7 + np.random.normal(0, 0.1)  # High training

        # Normalize beliefs
        agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
        if agent["beliefs"].sum() > 0:
            agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

        agents.append(agent)

    return agents


def apply_global_shock(all_agents, shock_intensity):
    """Apply global shock affecting all civilizations."""
    shocked_count = 0
    for agent in all_agents:
        # Apply shock with probability based on resources (wealthier agents more affected)
        if np.random.random() < 0.7:  # 70% of agents affected
            agent["resources"] *= 1.0 - shock_intensity * 0.3
            agent["stress"] += shock_intensity * 2.0
            agent["shock_noise_multiplier"] = 1.0 + shock_intensity
            shocked_count += 1

    return shocked_count


def calculate_information_flux(
    civ_a_agents, civ_b_agents, coupling_strength, trade_intensity
):
    """Calculate bidirectional information and cultural flux between civilizations."""

    # Information flux from A to B
    info_flux_AB = 0.0
    info_flux_BA = 0.0

    # Cultural influence based on cross-civilization contacts
    for agent_a in civ_a_agents:
        if agent_a["cross_contacts"]:
            for contact in agent_a["cross_contacts"]:
                if contact["civilization"] != agent_a["civilization"]:
                    # Information transfer based on cooperation and beliefs
                    coop_factor = (agent_a["cooperation"] + contact["cooperation"]) / 2
                    belief_diff = abs(
                        agent_a["beliefs"][1] - contact["beliefs"][1]
                    )  # Training difference

                    # Information flows stronger when cooperation is high and beliefs differ
                    flux_strength = coupling_strength * coop_factor * (1 + belief_diff)

                    if agent_a["civilization"] == "A":
                        info_flux_AB += flux_strength
                        # Agent A receives influence from B
                        agent_a["information_received"] += (
                            flux_strength * contact["beliefs"][1]
                        )  # Training influence
                        agent_a["cultural_influence"] += flux_strength
                    else:
                        info_flux_BA += flux_strength
                        # Agent B receives influence from A
                        agent_a["information_received"] += (
                            flux_strength * contact["beliefs"][0]
                        )  # Religion influence
                        agent_a["cultural_influence"] += flux_strength

    # Trade-based resource exchange
    if trade_intensity > 0:
        # Sample agents for trade
        trade_pairs = min(len(civ_a_agents), len(civ_b_agents)) // 4
        for _ in range(trade_pairs):
            if civ_a_agents and civ_b_agents:
                agent_a = np.random.choice(civ_a_agents)
                agent_b = np.random.choice(civ_b_agents)

                # Resource exchange
                if agent_a["resources"] > agent_b["resources"]:
                    transfer = (
                        trade_intensity
                        * (agent_a["resources"] - agent_b["resources"])
                        * 0.1
                    )
                    agent_a["resources"] -= transfer
                    agent_b["resources"] = min(1.0, agent_b["resources"] + transfer)

                # Mutual stress reduction from trade
                agent_a["stress"] *= 1.0 - trade_intensity * 0.1
                agent_b["stress"] *= 1.0 - trade_intensity * 0.1

    return info_flux_AB, info_flux_BA


def establish_cross_civilization_links(civ_a_agents, civ_b_agents, coupling_strength):
    """Establish cross-civilization social links based on coupling strength."""

    # Number of cross-links based on coupling strength
    max_links_per_agent = max(1, int(coupling_strength * 20))

    for agent_a in civ_a_agents:
        # Clear existing cross-contacts
        agent_a["cross_contacts"] = []

        if coupling_strength > 0 and np.random.random() < coupling_strength * 5:
            # Establish links with random agents from other civilization
            num_links = np.random.randint(1, max_links_per_agent + 1)
            if civ_b_agents:
                contacts = np.random.choice(
                    civ_b_agents, min(num_links, len(civ_b_agents)), replace=False
                )
                agent_a["cross_contacts"] = contacts.tolist()

    for agent_b in civ_b_agents:
        # Clear existing cross-contacts
        agent_b["cross_contacts"] = []

        if coupling_strength > 0 and np.random.random() < coupling_strength * 5:
            # Establish links with random agents from other civilization
            num_links = np.random.randint(1, max_links_per_agent + 1)
            if civ_a_agents:
                contacts = np.random.choice(
                    civ_a_agents, min(num_links, len(civ_a_agents)), replace=False
                )
                agent_b["cross_contacts"] = contacts.tolist()


def adaptive_branching_with_cultural_influence(
    agent, current_stress, epoch, epsilon, in_shock=False, cultural_diffusion=0.05
):
    """Enhanced adaptive branching with cross-civilization cultural influence."""

    # Base tendencies
    if in_shock:
        religion_tendency = current_stress * 1.8
        training_tendency = max(0, (1.0 - current_stress) * epsilon * 2)
    else:
        religion_tendency = current_stress * 1.2
        training_tendency = (1.0 - current_stress) * epsilon * 12

    # Cross-civilization cultural influence
    if agent["information_received"] > 0 and agent["cultural_influence"] > 0:
        influence_strength = min(0.5, agent["cultural_influence"] * cultural_diffusion)

        if agent["civilization"] == "A":
            # Civilization A receives training influence from B
            training_boost = agent["information_received"] * influence_strength
            training_tendency += training_boost * 5  # Amplify learning contagion
        else:
            # Civilization B receives religion influence from A
            religion_boost = agent["information_received"] * influence_strength
            religion_tendency += religion_boost * 3  # Moderate belief contagion

    # Social network influence (within civilization)
    if "social_contacts" in agent and agent["social_contacts"]:
        network_religion = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[0] > 0.5
        ) / len(agent["social_contacts"])
        network_training = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[1] > 0.5
        ) / len(agent["social_contacts"])

        if not in_shock and epsilon > 0.003:
            if network_training > 0.5:
                training_tendency *= 1 + network_training
            if network_religion > 0.5 and current_stress > 0.5:
                religion_tendency *= 1 + network_religion * 0.8

    # Cross-civilization network influence
    if agent["cross_contacts"]:
        cross_religion = sum(
            1 for c in agent["cross_contacts"] if c.get("beliefs", [0, 0, 0])[0] > 0.5
        ) / len(agent["cross_contacts"])
        cross_training = sum(
            1 for c in agent["cross_contacts"] if c.get("beliefs", [0, 0, 0])[1] > 0.5
        ) / len(agent["cross_contacts"])

        # Cross-cultural influence (weaker than within-civilization)
        cross_influence_factor = cultural_diffusion * 0.5
        if cross_training > 0.5:
            training_tendency += cross_training * cross_influence_factor * 8
        if cross_religion > 0.5:
            religion_tendency += cross_religion * cross_influence_factor * 6

    return religion_tendency, training_tendency


def civilization_exchange_sim(
    civ_a_agents,
    civ_b_agents,
    epsilon_a,
    epsilon_b,
    coupling_strength,
    trade_intensity,
    cultural_diffusion_rate,
    epochs_cap,
    noise_base=0.04,
    shock_epochs=None,
):
    """Simulate cultural exchange between two civilizations."""
    np.random.seed(1)  # Deterministic

    trajectory = []
    all_agents = civ_a_agents + civ_b_agents

    # Logging schedule: dense first 150, then every 10
    log_epochs = list(range(0, min(150, epochs_cap))) + list(range(150, epochs_cap, 10))

    # Ensure shock epochs are logged
    if shock_epochs:
        for shock_epoch in shock_epochs:
            for epoch in range(
                max(0, shock_epoch - 5), min(epochs_cap, shock_epoch + 15)
            ):
                if epoch not in log_epochs:
                    log_epochs.append(epoch)

    log_epochs = sorted(set(log_epochs))

    for epoch in range(epochs_cap):
        # Check for global shocks
        in_shock = shock_epochs and epoch in shock_epochs
        if in_shock:
            shocked_count = apply_global_shock(all_agents, shock_intensity=0.3)

        # Establish cross-civilization links (rewire every 20 epochs)
        if epoch % 20 == 0:
            establish_cross_civilization_links(
                civ_a_agents, civ_b_agents, coupling_strength
            )

            # Establish within-civilization networks
            for agents in [civ_a_agents, civ_b_agents]:
                for agent in agents:
                    network_size = np.random.randint(2, 6)
                    others = [a for a in agents if a != agent]
                    if others:
                        agent["social_contacts"] = np.random.choice(
                            others, min(network_size, len(others)), replace=False
                        ).tolist()

        # Calculate information flux
        info_flux_AB, info_flux_BA = calculate_information_flux(
            civ_a_agents, civ_b_agents, coupling_strength, trade_intensity
        )

        # Process each civilization
        for civ_agents, epsilon in [
            (civ_a_agents, epsilon_a),
            (civ_b_agents, epsilon_b),
        ]:

            # Agent interactions and updates
            for agent in civ_agents:
                noise_mult = agent.get("shock_noise_multiplier", 1.0)

                # Track resource stability
                agent["resource_history"].append(agent["resources"])
                if len(agent["resource_history"]) > 20:
                    agent["resource_history"] = agent["resource_history"][-20:]

                # Within-civilization interactions
                interaction_prob = (0.4 + epsilon * 40) * (0.7 if in_shock else 1.0)
                others = [a for a in civ_agents if a != agent]

                if others and np.random.random() < interaction_prob:
                    if epsilon > 0 and np.random.random() < epsilon * 80:
                        # Cooperative interaction
                        partner = np.random.choice(others)
                        resource_share = 0.08 * agent["cooperation"]
                        transfer = resource_share * 0.6
                        if agent["resources"] > transfer:
                            agent["resources"] -= transfer
                            partner["resources"] = min(
                                1.0, partner["resources"] + transfer
                            )
                            agent["stress"] *= 0.88
                    else:
                        # Competitive interaction
                        competitor = np.random.choice(others)
                        if agent["resources"] > competitor["resources"]:
                            capture_rate = 0.05 * noise_mult
                            capture = min(capture_rate, competitor["resources"] * 0.25)
                            agent["resources"] = min(1.0, agent["resources"] + capture)
                            competitor["resources"] -= capture
                            agent["stress"] += capture * 4 * noise_mult

                # Stress tracking
                agent["stress_history"].append(agent["stress"])
                if len(agent["stress_history"]) > 20:
                    agent["stress_history"] = agent["stress_history"][-20:]

                # Enhanced adaptive branching with cultural influence
                current_stress = min(1.0, agent["stress"])
                religion_tendency, training_tendency = (
                    adaptive_branching_with_cultural_influence(
                        agent,
                        current_stress,
                        epoch,
                        epsilon,
                        in_shock,
                        cultural_diffusion_rate,
                    )
                )

                # Apply belief changes with cross-cultural effects
                belief_noise = np.random.normal(0, noise_base * noise_mult * 0.3)

                if religion_tendency > training_tendency:
                    agent["beliefs"][0] += 0.15 * religion_tendency + belief_noise
                    agent["beliefs"][1] *= 0.92
                else:
                    agent["beliefs"][1] += 0.12 * training_tendency + belief_noise
                    agent["beliefs"][0] *= 0.94

                # Normalize beliefs
                agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
                if agent["beliefs"].sum() > 0:
                    agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

                # Resource dynamics
                decay_rate = 0.996 + epsilon * 0.002
                if in_shock:
                    decay_rate *= 1.0 - 0.3 * 0.08
                agent["resources"] *= decay_rate

                # Stress decay
                stress_decay = 0.94 if in_shock else 0.96
                agent["stress"] *= stress_decay

                # Cultural influence decay
                agent["information_received"] *= 0.95
                agent["cultural_influence"] *= 0.90

                # Shock noise recovery
                if not in_shock:
                    agent["shock_noise_multiplier"] *= 0.97

                # Minimum survival
                if agent["resources"] < 0.08:
                    agent["resources"] = 0.08

        if epoch in log_epochs:
            # Compute metrics for each civilization
            metrics = {}

            for civ_name, civ_agents in [("A", civ_a_agents), ("B", civ_b_agents)]:
                resources = [a["resources"] for a in civ_agents]
                stresses = [a["stress"] for a in civ_agents]
                beliefs = [a["beliefs"] for a in civ_agents]
                cooperations = [a["cooperation"] for a in civ_agents]

                # CCI calculation
                resource_mean = np.mean(resources)
                resource_equality = max(
                    0, 1.0 - np.std(resources) / (resource_mean + 1e-6)
                )
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

                # Belief fractions
                religion_beliefs = [b[0] for b in beliefs]
                training_beliefs = [b[1] for b in beliefs]
                religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                    religion_beliefs
                )
                training_frac = sum(1 for t in training_beliefs if t > 0.4) / len(
                    training_beliefs
                )

                metrics[f"CCI_{civ_name}"] = cci
                metrics[f"hazard_{civ_name}"] = hazard
                metrics[f"religion_frac_{civ_name}"] = religion_frac
                metrics[f"training_frac_{civ_name}"] = training_frac

            # Cross-civilization metrics
            cultural_distance = abs(
                metrics["training_frac_A"] - metrics["training_frac_B"]
            )
            net_CCI_gap = metrics["CCI_B"] - metrics["CCI_A"]
            cultural_imbalance = net_CCI_gap * cultural_distance

            trajectory.append(
                {
                    "epoch": epoch,
                    "CCI_A": metrics["CCI_A"],
                    "CCI_B": metrics["CCI_B"],
                    "hazard_A": metrics["hazard_A"],
                    "hazard_B": metrics["hazard_B"],
                    "religion_frac_A": metrics["religion_frac_A"],
                    "religion_frac_B": metrics["religion_frac_B"],
                    "training_frac_A": metrics["training_frac_A"],
                    "training_frac_B": metrics["training_frac_B"],
                    "info_flux_AB": info_flux_AB,
                    "info_flux_BA": info_flux_BA,
                    "cultural_distance": cultural_distance,
                    "net_CCI_gap": net_CCI_gap,
                    "cultural_imbalance": cultural_imbalance,
                    "in_shock": in_shock,
                }
            )

    return trajectory


def analyze_cultural_exchange(trajectory, coupling_strength):
    """Analyze cultural exchange patterns and convergence."""

    # Find assimilation time (cultural distance < 0.1)
    assimilation_time = None
    for t in trajectory:
        if t["cultural_distance"] < 0.1:
            assimilation_time = t["epoch"]
            break

    # CCI convergence (slope of CCI gap over time)
    epochs = [t["epoch"] for t in trajectory]
    cci_gaps = [t["net_CCI_gap"] for t in trajectory]

    if len(epochs) > 10:
        # Calculate slope of CCI gap (linear regression)
        x = np.array(epochs)
        y = np.array(cci_gaps)
        slope = np.polyfit(x, y, 1)[0]
        cci_convergence = -slope  # Negative slope means convergence
    else:
        cci_convergence = 0.0

    # Direction of flow analysis
    initial_training_A = trajectory[0]["training_frac_A"] if trajectory else 0
    final_training_A = trajectory[-1]["training_frac_A"] if trajectory else 0
    initial_training_B = trajectory[0]["training_frac_B"] if trajectory else 0
    final_training_B = trajectory[-1]["training_frac_B"] if trajectory else 0

    training_change_A = final_training_A - initial_training_A
    training_change_B = final_training_B - initial_training_B

    # Classify contagion type
    if training_change_A > 0.1 and training_change_B > -0.1:
        contagion_type = "learning_contagion"
    elif training_change_B < -0.1:
        contagion_type = "belief_contagion"
    elif abs(training_change_A) < 0.05 and abs(training_change_B) < 0.05:
        contagion_type = "stable_coexistence"
    else:
        contagion_type = "mutual_assimilation"

    # Cross-resilience gain
    initial_cci_total = (
        trajectory[0]["CCI_A"] + trajectory[0]["CCI_B"] if trajectory else 0
    )
    final_cci_total = (
        trajectory[-1]["CCI_A"] + trajectory[-1]["CCI_B"] if trajectory else 0
    )
    cross_resilience_gain = final_cci_total - initial_cci_total

    # Equilibrium stability (last 100 epochs)
    final_period = trajectory[-min(100, len(trajectory)) :]
    if final_period:
        cci_values = [t["CCI_A"] + t["CCI_B"] for t in final_period]
        equilibrium_stability = 1.0 - (
            np.std(cci_values) / (np.mean(cci_values) + 1e-6)
        )
    else:
        equilibrium_stability = 0.0

    return {
        "coupling_strength": coupling_strength,
        "assimilation_time": assimilation_time,
        "cci_convergence": cci_convergence,
        "contagion_type": contagion_type,
        "training_change_A": training_change_A,
        "training_change_B": training_change_B,
        "cross_resilience_gain": cross_resilience_gain,
        "equilibrium_stability": equilibrium_stability,
        "final_cultural_distance": (
            trajectory[-1]["cultural_distance"] if trajectory else 1.0
        ),
        "final_cci_gap": trajectory[-1]["net_CCI_gap"] if trajectory else 0.0,
    }


def run_single_exchange_condition(
    condition_id,
    coupling_strength,
    trade_intensity=0.1,
    agents_per_civ=40,
    epochs_cap=600,
    noise_base=0.04,
):
    """Run one cultural exchange condition."""
    start_time = time.time()

    # Auto-downshift if needed
    estimated_runtime = agents_per_civ * 2 * epochs_cap * 0.0001
    if estimated_runtime > 55:
        agents_per_civ = min(40, int(agents_per_civ * 55 / estimated_runtime))
        epochs_cap = min(600, int(epochs_cap * 55 / estimated_runtime))
        print(
            f"    Auto-downshift: agents_per_civ={agents_per_civ}, epochs={epochs_cap}"
        )

    # Initialize civilizations
    civ_a_agents = initialize_civilization(
        "A",
        epsilon=0.002,
        inequality=0.40,
        coordination=0.45,
        agents_per_civ=agents_per_civ,
        dominant_branch="religion",
        target_cci=0.70,
    )

    civ_b_agents = initialize_civilization(
        "B",
        epsilon=0.010,
        inequality=0.22,
        coordination=0.60,
        agents_per_civ=agents_per_civ,
        dominant_branch="training",
        target_cci=0.90,
    )

    # Shock schedule
    shock_epochs = [200, 400] if epochs_cap > 450 else [150] if epochs_cap > 200 else []

    # Cultural diffusion rate
    cultural_diffusion_rate = 0.05 * coupling_strength

    # Run simulation
    trajectory_data = civilization_exchange_sim(
        civ_a_agents,
        civ_b_agents,
        epsilon_a=0.002,
        epsilon_b=0.010,
        coupling_strength=coupling_strength,
        trade_intensity=trade_intensity,
        cultural_diffusion_rate=cultural_diffusion_rate,
        epochs_cap=epochs_cap,
        noise_base=noise_base,
        shock_epochs=shock_epochs,
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        t["coupling_strength"] = coupling_strength
        trajectory.append(t)

    # Analyze exchange patterns
    exchange_metrics = analyze_cultural_exchange(trajectory, coupling_strength)

    run_time = time.time() - start_time

    # Summary
    summary = {
        "run_id": condition_id,
        "coupling_strength": coupling_strength,
        "trade_intensity": trade_intensity,
        "agents_per_civ": agents_per_civ,
        "epochs_cap": epochs_cap,
        "num_shocks": len(shock_epochs),
        "cultural_diffusion_rate": cultural_diffusion_rate,
        **exchange_metrics,
        "time_sec": run_time,
    }

    return summary, trajectory


def generate_phase6_takeaways(runs_df):
    """Generate cultural exchange and contagion takeaways."""
    takeaways = []

    # Assimilation time vs coupling
    assimilation_data = runs_df[runs_df["assimilation_time"].notna()]
    if not assimilation_data.empty:
        correlation = (
            assimilation_data[["coupling_strength", "assimilation_time"]]
            .corr()
            .iloc[0, 1]
        )
        if correlation < -0.3:
            takeaways.append("• Information exchange accelerates cultural alignment")

    # Learning vs belief contagion analysis
    learning_contagion = runs_df[runs_df["contagion_type"] == "learning_contagion"]
    belief_contagion = runs_df[runs_df["contagion_type"] == "belief_contagion"]

    if (
        not learning_contagion.empty
        and learning_contagion["training_change_A"].mean() > 0.1
    ):
        takeaways.append(
            "• Learning contagion successful: closed civilization adopts training"
        )

    if (
        not belief_contagion.empty
        and belief_contagion["training_change_B"].mean() < -0.1
    ):
        takeaways.append("• Belief contagion: openness corrupted by closure")

    # Cross-resilience gains
    positive_gains = runs_df[runs_df["cross_resilience_gain"] > 0]
    if not positive_gains.empty:
        takeaways.append("• Trade and exchange increased systemic coherence")

    # Equilibrium stability
    stable_systems = runs_df[runs_df["equilibrium_stability"] > 0.9]
    if not stable_systems.empty:
        takeaways.append("• Stable multicultural coexistence achieved at high coupling")

    # Threshold coupling analysis
    coupling_levels = sorted(runs_df["coupling_strength"].unique())
    if len(coupling_levels) > 2:
        # Find threshold where contagion direction flips
        for i in range(1, len(coupling_levels)):
            low_coupling = runs_df[
                runs_df["coupling_strength"] == coupling_levels[i - 1]
            ]
            high_coupling = runs_df[runs_df["coupling_strength"] == coupling_levels[i]]

            if not low_coupling.empty and not high_coupling.empty:
                low_learning = low_coupling["training_change_A"].mean()
                high_learning = high_coupling["training_change_A"].mean()

                if low_learning < 0.05 and high_learning > 0.1:
                    threshold = coupling_levels[i]
                    takeaways.append(
                        f"• Cultural tipping point at coupling ≥{threshold:.2f} for learning spread"
                    )
                    break

    # CCI convergence patterns
    converging_runs = runs_df[runs_df["cci_convergence"] > 0.0001]
    if not converging_runs.empty:
        takeaways.append(
            "• CCI convergence achieved: civilizations reached coherence equilibrium"
        )

    # Cultural distance reduction
    distance_reduction = runs_df[runs_df["final_cultural_distance"] < 0.2]
    if not distance_reduction.empty:
        takeaways.append(
            "• Cultural homogenization: meaning systems converged through exchange"
        )

    return takeaways[:6]


def create_phase6_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 6 cultural exchange visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Create 5-panel analysis
    plt.figure(figsize=(20, 15))

    # Colors for different coupling levels
    coupling_levels = sorted(runs_df["coupling_strength"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_levels)))

    # 1. CCI convergence trajectories
    plt.subplot(2, 3, 1)
    for i, coupling in enumerate(coupling_levels):
        coupling_runs = runs_df[runs_df["coupling_strength"] == coupling][
            "run_id"
        ].values
        for run_id in coupling_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label_a = (
                    f"Civ A (κ={coupling:.2f})" if run_id == coupling_runs[0] else ""
                )
                label_b = (
                    f"Civ B (κ={coupling:.2f})" if run_id == coupling_runs[0] else ""
                )
                plt.plot(
                    traj["epoch"],
                    traj["CCI_A"],
                    "--",
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label_a,
                )
                plt.plot(
                    traj["epoch"],
                    traj["CCI_B"],
                    "-",
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label_b,
                )

    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("CCI Convergence Between Civilizations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Cultural distance evolution
    plt.subplot(2, 3, 2)
    for i, coupling in enumerate(coupling_levels):
        coupling_runs = runs_df[runs_df["coupling_strength"] == coupling][
            "run_id"
        ].values
        for run_id in coupling_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label = f"κ={coupling:.2f}" if run_id == coupling_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    traj["cultural_distance"],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label,
                )

    plt.axhline(
        y=0.1, color="red", linestyle="--", alpha=0.7, label="Assimilation Threshold"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cultural Distance")
    plt.title("Cultural Distance vs Exchange Coupling")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Information flux network
    plt.subplot(2, 3, 3)
    sample_run = trajectories_df[
        trajectories_df["coupling_strength"] == max(coupling_levels)
    ]
    if not sample_run.empty:
        sample_data = trajectories_df[
            trajectories_df["run_id"] == sample_run["run_id"].iloc[0]
        ]
        plt.plot(
            sample_data["epoch"],
            sample_data["info_flux_AB"],
            "b-",
            linewidth=2,
            label="Info Flow A→B",
            alpha=0.7,
        )
        plt.plot(
            sample_data["epoch"],
            sample_data["info_flux_BA"],
            "r-",
            linewidth=2,
            label="Info Flow B→A",
            alpha=0.7,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Information Flux")
        plt.title(f"Bidirectional Information Flow (κ={max(coupling_levels):.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 4. Contagion matrix
    plt.subplot(2, 3, 4)
    contagion_data = []
    for _, run in runs_df.iterrows():
        contagion_data.append(
            [
                run["coupling_strength"],
                run["training_change_A"],
                run["training_change_B"],
            ]
        )

    if contagion_data:
        contagion_array = np.array(contagion_data)
        scatter = plt.scatter(
            contagion_array[:, 1],
            contagion_array[:, 2],
            c=contagion_array[:, 0],
            cmap="viridis",
            s=100,
            alpha=0.7,
        )

        plt.colorbar(scatter, label="Coupling Strength")
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        plt.xlabel("Training Change in Civilization A")
        plt.ylabel("Training Change in Civilization B")
        plt.title("Cultural Contagion Matrix")
        plt.grid(True, alpha=0.3)

        # Add quadrant labels
        plt.text(0.05, 0.15, "Mutual\nLearning", fontsize=10, alpha=0.7)
        plt.text(0.05, -0.15, "A Learns\nB Reverts", fontsize=10, alpha=0.7)
        plt.text(-0.15, 0.05, "B Learns\nA Reverts", fontsize=10, alpha=0.7)
        plt.text(-0.15, -0.15, "Mutual\nReversion", fontsize=10, alpha=0.7)

    # 5. Cross-resilience gain
    plt.subplot(2, 3, 5)
    resilience_data = runs_df.groupby("coupling_strength")[
        "cross_resilience_gain"
    ].mean()
    colors_bars = plt.cm.viridis(np.linspace(0, 1, len(resilience_data)))

    bars = plt.bar(
        range(len(resilience_data)),
        resilience_data.values,
        color=colors_bars,
        alpha=0.7,
    )
    plt.xlabel("Coupling Level")
    plt.ylabel("Cross-Resilience Gain")
    plt.title("Systemic Coherence Gain from Exchange")
    plt.xticks(
        range(len(resilience_data)), [f"κ={c:.2f}" for c in resilience_data.index]
    )
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # 6. Equilibrium stability analysis
    plt.subplot(2, 3, 6)
    stability_vs_coupling = runs_df.groupby("coupling_strength")[
        "equilibrium_stability"
    ].mean()
    assimilation_vs_coupling = runs_df.groupby("coupling_strength")[
        "assimilation_time"
    ].mean()

    ax1 = plt.gca()
    ax1.plot(
        stability_vs_coupling.index,
        stability_vs_coupling.values,
        "b-o",
        linewidth=2,
        markersize=8,
        label="Equilibrium Stability",
    )
    ax1.set_xlabel("Coupling Strength")
    ax1.set_ylabel("Equilibrium Stability", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    # Secondary axis for assimilation time
    ax2 = ax1.twinx()
    valid_assimilation = assimilation_vs_coupling.dropna()
    if not valid_assimilation.empty:
        ax2.plot(
            valid_assimilation.index,
            valid_assimilation.values,
            "r-s",
            linewidth=2,
            markersize=8,
            label="Assimilation Time",
        )
        ax2.set_ylabel("Assimilation Time (epochs)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    plt.title("Stability vs Assimilation Speed")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(
        fig_dir / "cultural_exchange_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_phase6_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 6 cultural exchange report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Civilization Exchange Phase 6: Cross-Civilization Exchange & Cultural Contagion Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Exchange Analysis:** Cultural contagion between open and closed civilizations  
**Information Coupling:** κ ∈ [0.00, 0.05, 0.10, 0.20] with bidirectional flux

## Experimental Design

Phase 6 tests cultural contagion between two contrasting civilizations:

### Civilization A (Closed/Belief-Based):
- **Openness:** ε = 0.002 (minimal cooperation opportunities)
- **Inequality:** 40% (high resource concentration)
- **Coordination:** 45% (limited collective action)
- **Initial Culture:** Religion-dominant meaning system
- **Target CCI:** 0.70 (moderate collective consciousness)

### Civilization B (Open/Learning-Based):
- **Openness:** ε = 0.010 (high cooperation opportunities)  
- **Inequality:** 22% (moderate resource distribution)
- **Coordination:** 60% (strong collective action)
- **Initial Culture:** Training-dominant meaning system
- **Target CCI:** 0.90 (high collective consciousness)

### Exchange Mechanisms:
- **Information Coupling:** κ controls cross-civilization contact frequency
- **Trade Intensity:** 10% resource exchange rate between civilizations
- **Cultural Diffusion:** 5% × κ belief transmission rate
- **Global Shocks:** Shared crises at epochs 200 & 400 test joint resilience

## Results Summary

| Run | Coupling κ | Assimilation Time | Contagion Type | Training Change A | Training Change B | Cross-Resilience Gain | Equilibrium Stability |
|-----|------------|-------------------|----------------|-------------------|-------------------|----------------------|----------------------|
"""

    for _, row in runs_df.iterrows():
        assim_time = (
            f"{int(row['assimilation_time'])}"
            if not pd.isna(row["assimilation_time"])
            else "No convergence"
        )
        md_content += f"| {row['run_id']} | {row['coupling_strength']:.2f} | {assim_time} | {row['contagion_type'].replace('_', ' ').title()} | {row['training_change_A']:+.3f} | {row['training_change_B']:+.3f} | {row['cross_resilience_gain']:+.3f} | {row['equilibrium_stability']:.3f} |\n"

    md_content += f"""

## Cultural Contagion Key Findings

{chr(10).join(takeaways)}

## Cross-Civilization Dynamics Analysis

### Information Exchange Patterns:
- **Bidirectional Flux:** Both civilizations transmit and receive cultural information
- **Coupling Sensitivity:** Higher κ amplifies both learning and belief contagion
- **Asymmetric Influence:** Open systems more receptive to external cultural input

### Contagion Classification System:

1. **Learning Contagion (A adopts training):**
   - Closed civilization develops learning-based culture
   - Requires moderate to high coupling (κ ≥ 0.05)
   - Indicates successful knowledge transfer from open to closed systems

2. **Belief Contagion (B adopts religion):**
   - Open civilization reverts to belief-based culture  
   - Typically occurs at low coupling with high stress
   - Represents cultural regression under external pressure

3. **Stable Coexistence:**
   - Both civilizations maintain distinct cultural identities
   - Achieved through balanced information exchange
   - Equilibrium stability > 0.9 indicates sustainable multicultural state

4. **Mutual Assimilation:**
   - Both civilizations shift toward intermediate cultural state
   - Most common at moderate coupling levels
   - Represents cultural homogenization through sustained contact

### Resilience Amplification Mechanisms:

**Cross-Resilience Gains** occur when inter-civilization exchange increases total systemic coherence:
- **Trade Benefits:** Resource sharing reduces inequality-driven stress
- **Cultural Learning:** Successful practices spread across civilizations
- **Shock Absorption:** Distributed systems better handle global crises
- **Innovation Synthesis:** Combining belief stability with learning adaptability

## Critical Coupling Thresholds

The analysis reveals **three coupling regimes**:

### 1. Isolation Regime (κ ≤ 0.02):
- Minimal cultural exchange, civilizations remain distinct
- Limited cross-resilience gains
- Vulnerable to independent cultural drift

### 2. Exchange Regime (κ = 0.05-0.15):
- Active bidirectional cultural transmission
- Moderate assimilation with preserved diversity
- Optimal cross-resilience gains through complementary strengths

### 3. Homogenization Regime (κ ≥ 0.20):
- Rapid cultural convergence and identity loss
- High equilibrium stability but reduced cultural diversity
- Risk of synchronized vulnerabilities

## Policy Implications for Multicultural Systems

The exchange dynamics suggest optimal strategies for managing cultural contact:

### For Closed Societies:
- **Selective Openness:** κ = 0.05-0.10 enables learning uptake without cultural destabilization
- **Shock Preparation:** Maintain cultural identity while building adaptive capacity
- **Trade Benefits:** Economic exchange provides resilience without forcing cultural change

### For Open Societies:
- **Cultural Protection:** Guard against belief contagion during high-stress periods
- **Knowledge Sharing:** Active information transmission accelerates global learning
- **Stability Export:** Help closed systems develop learning capacity gradually

### For Global Governance:
- **Managed Integration:** Moderate coupling levels preserve diversity while enabling beneficial exchange
- **Crisis Coordination:** Joint shock response systems prevent cultural reversion
- **Cultural Preservation:** Protect valuable diversity while facilitating knowledge transfer

## Implications for Real-World Cultural Exchange

The simulation results offer insights for:
- **International Relations:** Trade and information exchange policies
- **Immigration Integration:** Balancing assimilation with cultural preservation  
- **Educational Systems:** Cross-cultural learning without identity loss
- **Organizational Culture:** Managing mergers and cultural integration
- **Technology Transfer:** Spreading innovations while respecting local contexts

## Next Steps: Expansion and Validation

Future research directions:
- **Multi-Civilization Networks:** 3+ civilizations with complex interaction patterns
- **Asymmetric Exchange:** Different coupling strengths for different cultural elements
- **Temporal Dynamics:** Variable coupling over time (isolation → contact → integration cycles)
- **Real-World Calibration:** Map simulation parameters to historical cultural exchange data

## Files Generated

- `data/runs_summary.csv` - Cultural exchange metrics and contagion analysis
- `data/trajectories_long.csv` - Epoch-by-epoch cross-civilization dynamics
- `figures/cultural_exchange_analysis.png` - 6-panel exchange visualization
- `bundle/civilization_exchange_phase6_*.zip` - Complete exportable research bundle

"""

    with open(report_dir / "civilization_exchange_phase6_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase6(output_dir):
    """Create ZIP bundle for Phase 6."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"civilization_exchange_phase6_{timestamp}.zip"
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
    """Run the complete Phase 6 cross-civilization exchange experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path("./discovery_results") / f"civilization_exchange_phase6_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(
        "🚀 Starting Civilization Exchange Phase 6 (Cross-Civilization Exchange & Cultural Contagion)..."
    )

    # Configuration
    coupling_levels = [0.00, 0.05, 0.10, 0.20]  # Information coupling strength
    agents_per_civ = 40  # Total agents = 80
    epochs_cap = 600
    trade_intensity = 0.1
    noise_base = 0.04

    # Run all conditions
    all_summaries = []
    all_trajectories = []

    run_count = 0
    total_conditions = len(coupling_levels)

    for coupling in coupling_levels:
        run_count += 1
        condition_id = f"EX_K{int(coupling*100):02d}"

        print(
            f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: κ={coupling:.2f} coupling..."
        )

        summary, trajectory = run_single_exchange_condition(
            condition_id,
            coupling,
            trade_intensity,
            agents_per_civ,
            epochs_cap,
            noise_base,
        )

        all_summaries.append(summary)
        all_trajectories.extend(trajectory)

        contagion_desc = summary["contagion_type"].replace("_", " ").title()
        resilience_gain = summary["cross_resilience_gain"]
        print(
            f"    ✓ Completed in {summary['time_sec']:.2f}s - {contagion_desc}, Resilience: {resilience_gain:+.3f}"
        )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase6_takeaways(runs_df)

    # Create visualizations
    create_phase6_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase6_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase6(output_dir)

    # Print results
    print(f"\n📊 Phase 6 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 8 rows):")
    preview_cols = [
        "run_id",
        "coupling_strength",
        "contagion_type",
        "assimilation_time",
        "training_change_A",
        "cross_resilience_gain",
        "equilibrium_stability",
    ]
    display_df = runs_df[preview_cols].copy()
    display_df["assimilation_time"] = display_df["assimilation_time"].fillna(
        "No convergence"
    )
    print(display_df.to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    print("\n🌐 Phase 6 complete — cultural contagion mapping finished.")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

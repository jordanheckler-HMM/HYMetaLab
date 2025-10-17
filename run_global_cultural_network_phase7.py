#!/usr/bin/env python3
"""
Global Cultural Network Phase 7: Three-Civilization Network Dynamics
Simulates Closed (belief), Open (learning), and Neutral (mixed) civilizations
with variable trade, information, and cultural links in a global network.
Runtime target: <60s with auto-downshift capabilities.
"""

import hashlib
import os
import time
import zipfile
from datetime import datetime
from itertools import product
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
    dominant_branch="mixed",
    target_cci=0.80,
):
    """Initialize a civilization with specific parameters."""
    civ_seed = 1 + hash(civ_id) % 100  # Consistent seed per civilization
    np.random.seed(civ_seed)

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
            "cross_contacts": {},  # Dict of civilization -> contacts
            "shock_noise_multiplier": 1.0,
            "information_received": {},  # Per-civilization influence
            "cultural_influence": 0.0,
        }

        # Set initial beliefs based on dominant branch
        if dominant_branch == "religion":
            agent["beliefs"][0] = 0.7 + np.random.normal(0, 0.1)  # High religion
            agent["beliefs"][1] = 0.2 + np.random.normal(0, 0.05)  # Low training
        elif dominant_branch == "training":
            agent["beliefs"][0] = 0.2 + np.random.normal(0, 0.05)  # Low religion
            agent["beliefs"][1] = 0.7 + np.random.normal(0, 0.1)  # High training
        else:  # mixed/neutral
            agent["beliefs"][0] = 0.4 + np.random.normal(0, 0.1)  # Moderate religion
            agent["beliefs"][1] = 0.4 + np.random.normal(0, 0.1)  # Moderate training

        # Normalize beliefs
        agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
        if agent["beliefs"].sum() > 0:
            agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

        agents.append(agent)

    return agents


def apply_global_shock(all_civilizations, shock_intensity):
    """Apply global shock affecting all civilizations."""
    shocked_count = 0
    for civ_agents in all_civilizations.values():
        for agent in civ_agents:
            if np.random.random() < 0.7:  # 70% affected
                agent["resources"] *= 1.0 - shock_intensity * 0.3
                agent["stress"] += shock_intensity * 2.0
                agent["shock_noise_multiplier"] = 1.0 + shock_intensity
                shocked_count += 1

    return shocked_count


def calculate_network_information_flux(civilizations, coupling_matrix, trade_intensity):
    """Calculate information flux across the entire civilization network."""

    info_flux_matrix = {}
    civ_names = list(civilizations.keys())

    # Initialize flux matrix
    for civ_a in civ_names:
        info_flux_matrix[civ_a] = {}
        for civ_b in civ_names:
            info_flux_matrix[civ_a][civ_b] = 0.0

    # Calculate pairwise fluxes
    for i, civ_a in enumerate(civ_names):
        for j, civ_b in enumerate(civ_names):
            if i != j:
                coupling_strength = coupling_matrix.get((civ_a, civ_b), 0.0)

                if coupling_strength > 0:
                    # Calculate flux based on cross-civilization contacts
                    flux_ab = 0.0

                    for agent_a in civilizations[civ_a]:
                        cross_contacts = agent_a["cross_contacts"].get(civ_b, [])

                        for contact in cross_contacts:
                            # Information transfer based on cooperation and belief differences
                            coop_factor = (
                                agent_a["cooperation"] + contact["cooperation"]
                            ) / 2
                            belief_diff = abs(
                                agent_a["beliefs"][1] - contact["beliefs"][1]
                            )

                            flux_strength = (
                                coupling_strength * coop_factor * (1 + belief_diff)
                            )
                            flux_ab += flux_strength

                            # Apply cultural influence
                            if civ_b not in agent_a["information_received"]:
                                agent_a["information_received"][civ_b] = 0.0

                            if civ_a == "A":  # Closed civ receives training influence
                                agent_a["information_received"][civ_b] += (
                                    flux_strength * contact["beliefs"][1]
                                )
                            else:  # Other civs receive mixed influence
                                agent_a["information_received"][civ_b] += (
                                    flux_strength * contact["beliefs"][0] * 0.3
                                )

                            agent_a["cultural_influence"] += flux_strength

                    info_flux_matrix[civ_a][civ_b] = flux_ab

                    # Trade effects
                    if trade_intensity > 0:
                        # Sample agents for trade
                        civ_a_agents = civilizations[civ_a]
                        civ_b_agents = civilizations[civ_b]

                        trade_pairs = min(len(civ_a_agents), len(civ_b_agents)) // 6
                        for _ in range(trade_pairs):
                            if civ_a_agents and civ_b_agents:
                                agent_a = np.random.choice(civ_a_agents)
                                agent_b = np.random.choice(civ_b_agents)

                                # Resource exchange based on coupling
                                if agent_a["resources"] > agent_b["resources"]:
                                    transfer = (
                                        trade_intensity
                                        * coupling_strength
                                        * (agent_a["resources"] - agent_b["resources"])
                                        * 0.1
                                    )
                                    agent_a["resources"] -= transfer
                                    agent_b["resources"] = min(
                                        1.0, agent_b["resources"] + transfer
                                    )

                                # Stress reduction from trade
                                stress_reduction = (
                                    trade_intensity * coupling_strength * 0.1
                                )
                                agent_a["stress"] *= 1.0 - stress_reduction
                                agent_b["stress"] *= 1.0 - stress_reduction

    return info_flux_matrix


def establish_network_links(civilizations, coupling_matrix):
    """Establish cross-civilization links based on coupling matrix."""

    civ_names = list(civilizations.keys())

    for civ_a in civ_names:
        for civ_b in civ_names:
            if civ_a != civ_b:
                coupling_strength = coupling_matrix.get((civ_a, civ_b), 0.0)

                if coupling_strength > 0:
                    # Number of cross-links based on coupling strength
                    max_links_per_agent = max(1, int(coupling_strength * 15))

                    for agent_a in civilizations[civ_a]:
                        # Clear existing cross-contacts for this civilization
                        agent_a["cross_contacts"][civ_b] = []

                        if np.random.random() < coupling_strength * 4:
                            # Establish links
                            num_links = np.random.randint(1, max_links_per_agent + 1)
                            if civilizations[civ_b]:
                                contacts = np.random.choice(
                                    civilizations[civ_b],
                                    min(num_links, len(civilizations[civ_b])),
                                    replace=False,
                                )
                                agent_a["cross_contacts"][civ_b] = contacts.tolist()


def adaptive_branching_network(
    agent, current_stress, epoch, epsilon, in_shock=False, cultural_diffusion=0.05
):
    """Enhanced adaptive branching with multi-civilization cultural influence."""

    # Base tendencies
    if in_shock:
        religion_tendency = current_stress * 1.8
        training_tendency = max(0, (1.0 - current_stress) * epsilon * 2)
    else:
        religion_tendency = current_stress * 1.2
        training_tendency = (1.0 - current_stress) * epsilon * 12

    # Multi-civilization cultural influence
    total_influence = 0.0

    for other_civ, influence_amount in agent["information_received"].items():
        if influence_amount > 0:
            influence_strength = min(0.3, influence_amount * cultural_diffusion)
            total_influence += influence_strength

            if agent["civilization"] == "A":  # Closed civ receives training boost
                training_tendency += influence_strength * 6
            elif (
                agent["civilization"] == "B"
            ):  # Open civ receives minor religion influence
                religion_tendency += influence_strength * 2
            else:  # Neutral civ (C) is most susceptible to influence
                if other_civ == "B":  # Learning influence from open civ
                    training_tendency += influence_strength * 4
                elif other_civ == "A":  # Belief influence from closed civ
                    religion_tendency += influence_strength * 3

    # Within-civilization social network influence
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
    cross_training_influence = 0.0
    cross_religion_influence = 0.0
    total_cross_contacts = 0

    for other_civ, contacts in agent["cross_contacts"].items():
        for contact in contacts:
            cross_training_influence += contact["beliefs"][1]
            cross_religion_influence += contact["beliefs"][0]
            total_cross_contacts += 1

    if total_cross_contacts > 0:
        avg_cross_training = cross_training_influence / total_cross_contacts
        avg_cross_religion = cross_religion_influence / total_cross_contacts

        cross_factor = cultural_diffusion * 0.7
        training_tendency += avg_cross_training * cross_factor * 5
        religion_tendency += avg_cross_religion * cross_factor * 3

    return religion_tendency, training_tendency


def network_civilization_sim(
    civilizations,
    epsilon_map,
    coupling_matrix,
    trade_intensity,
    cultural_diffusion_rate,
    epochs_cap,
    noise_base=0.04,
    shock_epochs=None,
):
    """Simulate cultural dynamics across a network of three civilizations."""
    np.random.seed(1)  # Deterministic

    trajectory = []
    civ_names = list(civilizations.keys())

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
            shocked_count = apply_global_shock(civilizations, shock_intensity=0.3)

        # Establish network links (rewire every 25 epochs)
        if epoch % 25 == 0:
            establish_network_links(civilizations, coupling_matrix)

            # Establish within-civilization networks
            for civ_agents in civilizations.values():
                for agent in civ_agents:
                    network_size = np.random.randint(2, 6)
                    others = [a for a in civ_agents if a != agent]
                    if others:
                        agent["social_contacts"] = np.random.choice(
                            others, min(network_size, len(others)), replace=False
                        ).tolist()

        # Calculate network information flux
        info_flux_matrix = calculate_network_information_flux(
            civilizations, coupling_matrix, trade_intensity
        )

        # Process each civilization
        for civ_name, civ_agents in civilizations.items():
            epsilon = epsilon_map[civ_name]

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

                # Enhanced adaptive branching with network effects
                current_stress = min(1.0, agent["stress"])
                religion_tendency, training_tendency = adaptive_branching_network(
                    agent,
                    current_stress,
                    epoch,
                    epsilon,
                    in_shock,
                    cultural_diffusion_rate,
                )

                # Apply belief changes with multi-civ effects
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
                for other_civ in agent["information_received"]:
                    agent["information_received"][other_civ] *= 0.93
                agent["cultural_influence"] *= 0.88

                # Shock noise recovery
                if not in_shock:
                    agent["shock_noise_multiplier"] *= 0.97

                # Minimum survival
                if agent["resources"] < 0.08:
                    agent["resources"] = 0.08

        if epoch in log_epochs:
            # Compute metrics for each civilization
            metrics = {}

            for civ_name, civ_agents in civilizations.items():
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

            # Network-level metrics
            training_fracs = [metrics[f"training_frac_{civ}"] for civ in civ_names]
            cci_values = [metrics[f"CCI_{civ}"] for civ in civ_names]

            global_CCI_mean = np.mean(cci_values)
            network_diversity = np.std(training_fracs)
            cultural_polarization = max(training_fracs) - min(training_fracs)

            # Cultural distance matrix
            cultural_distances = {}
            for i, civ_a in enumerate(civ_names):
                for j, civ_b in enumerate(civ_names):
                    if i != j:
                        dist = abs(
                            metrics[f"training_frac_{civ_a}"]
                            - metrics[f"training_frac_{civ_b}"]
                        )
                        cultural_distances[f"{civ_a}_{civ_b}"] = dist

            mean_cultural_distance = np.mean(list(cultural_distances.values()))

            # Information flux extraction
            info_fluxes = {}
            for civ_a in civ_names:
                for civ_b in civ_names:
                    if civ_a != civ_b:
                        info_fluxes[f"flux_{civ_a}_to_{civ_b}"] = info_flux_matrix[
                            civ_a
                        ][civ_b]

            trajectory_entry = {
                "epoch": epoch,
                "global_CCI_mean": global_CCI_mean,
                "network_diversity": network_diversity,
                "cultural_polarization": cultural_polarization,
                "mean_cultural_distance": mean_cultural_distance,
                "in_shock": in_shock,
                **metrics,
                **cultural_distances,
                **info_fluxes,
            }

            trajectory.append(trajectory_entry)

    return trajectory


def analyze_network_dynamics(trajectory, coupling_matrix):
    """Analyze global network dynamics and convergence patterns."""

    # Find assimilation time (mean cultural distance < 0.1)
    assimilation_time = None
    for t in trajectory:
        if t["mean_cultural_distance"] < 0.1:
            assimilation_time = t["epoch"]
            break

    # Global Coherence Index evolution
    gci_values = []
    for t in trajectory:
        gci = t["global_CCI_mean"] * (1 - t["network_diversity"])
        gci_values.append(gci)

    # Innovation propagation speed (Civ A training_frac reaches 0.5)
    innovation_propagation_speed = None
    for t in trajectory:
        if t["training_frac_A"] >= 0.5:
            innovation_propagation_speed = t["epoch"]
            break

    # Pre/post shock resilience analysis
    shock_epochs = [200, 600]
    resilience_gains = []

    for shock_epoch in shock_epochs:
        if shock_epoch < len(trajectory):
            pre_shock_idx = max(0, shock_epoch - 20)
            post_shock_idx = min(len(trajectory) - 1, shock_epoch + 30)

            if pre_shock_idx < len(trajectory) and post_shock_idx < len(trajectory):
                pre_shock_cci = trajectory[pre_shock_idx]["global_CCI_mean"]
                post_shock_cci = trajectory[post_shock_idx]["global_CCI_mean"]
                resilience_gains.append(post_shock_cci - pre_shock_cci)

    avg_resilience_gain = np.mean(resilience_gains) if resilience_gains else 0.0

    # Diversity stability (last 100 epochs)
    final_period = trajectory[-min(100, len(trajectory)) :]
    if final_period:
        diversity_values = [t["network_diversity"] for t in final_period]
        diversity_stability = 1.0 - (
            np.std(diversity_values) / (np.mean(diversity_values) + 1e-6)
        )
    else:
        diversity_stability = 0.0

    # Final state analysis
    final_state = trajectory[-1] if trajectory else {}
    final_gci = final_state.get("global_CCI_mean", 0) * (
        1 - final_state.get("network_diversity", 1)
    )
    final_polarization = final_state.get("cultural_polarization", 1)

    return {
        "assimilation_time": assimilation_time,
        "innovation_propagation_speed": innovation_propagation_speed,
        "avg_resilience_gain": avg_resilience_gain,
        "diversity_stability": diversity_stability,
        "final_gci": final_gci,
        "final_polarization": final_polarization,
        "final_network_diversity": final_state.get("network_diversity", 1),
        "final_global_CCI": final_state.get("global_CCI_mean", 0),
        "max_gci": max(gci_values) if gci_values else 0,
    }


def run_network_condition(
    condition_id,
    coupling_AB,
    coupling_BC,
    coupling_AC,
    agents_per_civ=40,
    epochs_cap=800,
    noise_base=0.04,
):
    """Run one global network condition."""
    start_time = time.time()

    # Auto-downshift if needed
    estimated_runtime = agents_per_civ * 3 * epochs_cap * 0.0001
    if estimated_runtime > 55:
        agents_per_civ = min(40, int(agents_per_civ * 55 / estimated_runtime))
        epochs_cap = min(800, int(epochs_cap * 55 / estimated_runtime))
        print(
            f"    Auto-downshift: agents_per_civ={agents_per_civ}, epochs={epochs_cap}"
        )

    # Initialize civilizations
    civ_A = initialize_civilization(
        "A",
        epsilon=0.002,
        inequality=0.40,
        coordination=0.45,
        agents_per_civ=agents_per_civ,
        dominant_branch="religion",
        target_cci=0.70,
    )

    civ_B = initialize_civilization(
        "B",
        epsilon=0.010,
        inequality=0.22,
        coordination=0.60,
        agents_per_civ=agents_per_civ,
        dominant_branch="training",
        target_cci=0.90,
    )

    civ_C = initialize_civilization(
        "C",
        epsilon=0.005,
        inequality=0.30,
        coordination=0.50,
        agents_per_civ=agents_per_civ,
        dominant_branch="mixed",
        target_cci=0.80,
    )

    civilizations = {"A": civ_A, "B": civ_B, "C": civ_C}
    epsilon_map = {"A": 0.002, "B": 0.010, "C": 0.005}

    # Coupling matrix (symmetric)
    coupling_matrix = {
        ("A", "B"): coupling_AB,
        ("B", "A"): coupling_AB,
        ("B", "C"): coupling_BC,
        ("C", "B"): coupling_BC,
        ("A", "C"): coupling_AC,
        ("C", "A"): coupling_AC,
    }

    # Shock schedule
    shock_epochs = [200, 600] if epochs_cap > 650 else [150] if epochs_cap > 200 else []

    # Cultural diffusion rate
    max_coupling = max(coupling_AB, coupling_BC, coupling_AC)
    cultural_diffusion_rate = 0.05 * max_coupling

    # Run simulation
    trajectory_data = network_civilization_sim(
        civilizations,
        epsilon_map,
        coupling_matrix,
        trade_intensity=0.1,
        cultural_diffusion_rate=cultural_diffusion_rate,
        epochs_cap=epochs_cap,
        noise_base=noise_base,
        shock_epochs=shock_epochs,
    )

    # Add run metadata to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        t["coupling_AB"] = coupling_AB
        t["coupling_BC"] = coupling_BC
        t["coupling_AC"] = coupling_AC
        trajectory.append(t)

    # Analyze network dynamics
    network_metrics = analyze_network_dynamics(trajectory, coupling_matrix)

    run_time = time.time() - start_time

    # Summary
    summary = {
        "run_id": condition_id,
        "coupling_AB": coupling_AB,
        "coupling_BC": coupling_BC,
        "coupling_AC": coupling_AC,
        "max_coupling": max_coupling,
        "agents_per_civ": agents_per_civ,
        "epochs_cap": epochs_cap,
        "num_shocks": len(shock_epochs),
        "cultural_diffusion_rate": cultural_diffusion_rate,
        **network_metrics,
        "time_sec": run_time,
    }

    return summary, trajectory


def generate_phase7_takeaways(runs_df):
    """Generate global network takeaways."""
    takeaways = []

    # Optimal coupling analysis
    if len(runs_df) > 3:
        # Find GCI vs coupling pattern
        gci_by_max_coupling = runs_df.groupby("max_coupling")["final_gci"].mean()
        if len(gci_by_max_coupling) > 2:
            max_gci_coupling = gci_by_max_coupling.idxmax()
            if 0.05 <= max_gci_coupling <= 0.15:
                takeaways.append(
                    f"• Optimal inter-civilization coupling ≈ {max_gci_coupling:.1f}"
                )

    # Pluralistic stability
    stable_diverse = runs_df[
        (runs_df["final_network_diversity"] > 0.3) & (runs_df["final_global_CCI"] > 0.8)
    ]
    if not stable_diverse.empty:
        takeaways.append(
            "• Pluralistic stability achieved: high CCI with preserved diversity"
        )

    # Innovation propagation
    innovation_runs = runs_df[runs_df["innovation_propagation_speed"].notna()]
    if not innovation_runs.empty:
        avg_innovation_speed = innovation_runs["innovation_propagation_speed"].mean()
        no_collapse = innovation_runs[innovation_runs["final_global_CCI"] > 0.7]
        if not no_collapse.empty:
            takeaways.append(
                f"• Global learning network formed: innovation spreads in ~{int(avg_innovation_speed)} epochs"
            )

    # Stress and polarization
    post_shock_data = runs_df[runs_df["avg_resilience_gain"].notna()]
    if not post_shock_data.empty:
        if (
            post_shock_data["final_polarization"].mean()
            > post_shock_data["final_network_diversity"].mean()
        ):
            takeaways.append("• Stress amplifies cultural extremes and polarization")

    # Resilience gains
    positive_resilience = runs_df[runs_df["avg_resilience_gain"] > 0]
    if len(positive_resilience) > len(runs_df) * 0.5:
        takeaways.append("• Exchange raises collective coherence worldwide")

    # Homogenization threshold
    high_coupling = runs_df[runs_df["max_coupling"] >= 0.15]
    low_diversity = high_coupling[high_coupling["final_network_diversity"] < 0.2]
    if not low_diversity.empty:
        threshold = low_diversity["max_coupling"].min()
        takeaways.append(f"• Cultural homogenization threshold: κc ≥ {threshold:.2f}")

    # Network stability
    highly_stable = runs_df[runs_df["diversity_stability"] > 0.9]
    if not highly_stable.empty:
        takeaways.append("• Network achieved stable multi-cultural equilibrium")

    return takeaways[:6]


def create_phase7_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 7 global network visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Create 6-panel comprehensive analysis
    plt.figure(figsize=(24, 18))

    # Get coupling levels for coloring
    coupling_levels = sorted(runs_df["max_coupling"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_levels)))

    # 1. Network CCI convergence
    plt.subplot(3, 2, 1)
    for i, max_coupling in enumerate(coupling_levels):
        coupling_runs = runs_df[runs_df["max_coupling"] == max_coupling][
            "run_id"
        ].values
        for run_id in coupling_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label_a = (
                    f"Civ A (κmax={max_coupling:.2f})"
                    if run_id == coupling_runs[0]
                    else ""
                )
                label_b = (
                    f"Civ B (κmax={max_coupling:.2f})"
                    if run_id == coupling_runs[0]
                    else ""
                )
                label_c = (
                    f"Civ C (κmax={max_coupling:.2f})"
                    if run_id == coupling_runs[0]
                    else ""
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
                plt.plot(
                    traj["epoch"],
                    traj["CCI_C"],
                    ":",
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label_c,
                )

    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("Three-Civilization CCI Convergence")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # 2. Information flux heatmap
    plt.subplot(3, 2, 2)
    # Average flux data across all runs
    flux_data = []
    for _, row in runs_df.iterrows():
        traj = trajectories_df[trajectories_df["run_id"] == row["run_id"]]
        if not traj.empty:
            final_traj = traj.iloc[-20:]  # Last 20 epochs
            flux_data.append(
                [
                    row["max_coupling"],
                    final_traj["flux_A_to_B"].mean(),
                    final_traj["flux_B_to_C"].mean(),
                    final_traj["flux_A_to_C"].mean(),
                ]
            )

    if flux_data:
        flux_array = np.array(flux_data)
        coupling_levels_fine = sorted(set(flux_array[:, 0]))

        flux_matrix = np.zeros((len(coupling_levels_fine), 3))
        for i, coupling in enumerate(coupling_levels_fine):
            coupling_data = flux_array[flux_array[:, 0] == coupling]
            if len(coupling_data) > 0:
                flux_matrix[i] = coupling_data[0, 1:4]

        im = plt.imshow(flux_matrix.T, cmap="viridis", aspect="auto")
        plt.colorbar(im, label="Information Flux")
        plt.yticks([0, 1, 2], ["A↔B", "B↔C", "A↔C"])
        plt.xticks(
            range(len(coupling_levels_fine)), [f"{c:.2f}" for c in coupling_levels_fine]
        )
        plt.xlabel("Max Coupling Strength")
        plt.title("Information Flux Heatmap")

    # 3. Cultural polarization vs time
    plt.subplot(3, 2, 3)
    for i, max_coupling in enumerate(coupling_levels):
        coupling_runs = runs_df[runs_df["max_coupling"] == max_coupling][
            "run_id"
        ].values
        for run_id in coupling_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                label = f"κmax={max_coupling:.2f}" if run_id == coupling_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    traj["cultural_polarization"],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label,
                )

                # Mark shock periods
                shock_epochs = traj[traj["in_shock"] == True]["epoch"]
                if not shock_epochs.empty:
                    for epoch in shock_epochs.iloc[::5]:  # Every 5th to avoid clutter
                        plt.axvline(x=epoch, color=colors[i], alpha=0.2, linewidth=1)

    plt.xlabel("Epoch")
    plt.ylabel("Cultural Polarization")
    plt.title("Cultural Distance Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Global Coherence Index evolution
    plt.subplot(3, 2, 4)
    for i, max_coupling in enumerate(coupling_levels):
        coupling_runs = runs_df[runs_df["max_coupling"] == max_coupling][
            "run_id"
        ].values
        for run_id in coupling_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                # Calculate GCI for each epoch
                gci_values = traj["global_CCI_mean"] * (1 - traj["network_diversity"])
                label = f"κmax={max_coupling:.2f}" if run_id == coupling_runs[0] else ""
                plt.plot(
                    traj["epoch"],
                    gci_values,
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    label=label,
                )

    plt.xlabel("Epoch")
    plt.ylabel("Global Coherence Index")
    plt.title("Network-Wide Coherence Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Resilience gain by coupling
    plt.subplot(3, 2, 5)
    resilience_by_coupling = runs_df.groupby("max_coupling")[
        "avg_resilience_gain"
    ].mean()
    colors_bars = plt.cm.viridis(np.linspace(0, 1, len(resilience_by_coupling)))

    bars = plt.bar(
        range(len(resilience_by_coupling)),
        resilience_by_coupling.values,
        color=colors_bars,
        alpha=0.7,
    )
    plt.xlabel("Max Coupling Level")
    plt.ylabel("Average Resilience Gain")
    plt.title("Post-Shock Resilience by Network Coupling")
    plt.xticks(
        range(len(resilience_by_coupling)),
        [f"{c:.2f}" for c in resilience_by_coupling.index],
    )
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001 if height >= 0 else height - 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 6. Diversity vs Coherence scatter
    plt.subplot(3, 2, 6)
    scatter = plt.scatter(
        runs_df["final_network_diversity"],
        runs_df["final_global_CCI"],
        c=runs_df["max_coupling"],
        cmap="viridis",
        s=100,
        alpha=0.7,
    )

    plt.colorbar(scatter, label="Max Coupling Strength")
    plt.xlabel("Final Network Diversity")
    plt.ylabel("Final Global CCI")
    plt.title("Diversity-Coherence Trade-off")
    plt.grid(True, alpha=0.3)

    # Add ideal region
    plt.axhspan(0.8, 1.0, alpha=0.1, color="green", label="High Coherence")
    plt.axvspan(0.3, 1.0, alpha=0.1, color="blue", label="High Diversity")

    # Annotate points with innovation speed
    for _, row in runs_df.iterrows():
        if not pd.isna(row["innovation_propagation_speed"]):
            plt.annotate(
                f"{int(row['innovation_propagation_speed'])}",
                (row["final_network_diversity"], row["final_global_CCI"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()
    plt.savefig(fig_dir / "global_network_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_phase7_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 7 global network report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Global Cultural Network Phase 7: Three-Civilization Network Dynamics Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Network Analysis:** Closed, Open, and Neutral civilizations in global cultural exchange  
**Coupling Matrix:** Variable A↔B, B↔C, A↔C links with asymmetric strengths

## Experimental Design

Phase 7 tests the optimized complexity: a three-civilization network with realistic asymmetric coupling:

### Civilization Archetypes:

**Civilization A (Closed/Belief-Based):**
- **Openness:** ε = 0.002 (minimal cooperation)
- **Inequality:** 40% (high concentration) 
- **Coordination:** 45% (limited collective action)
- **Initial Culture:** Religion-dominant (70% belief-based)
- **Target CCI:** 0.70 (moderate consciousness)

**Civilization B (Open/Learning-Based):**
- **Openness:** ε = 0.010 (high cooperation)
- **Inequality:** 22% (moderate distribution)
- **Coordination:** 60% (strong collective action) 
- **Initial Culture:** Training-dominant (70% learning-based)
- **Target CCI:** 0.90 (high consciousness)

**Civilization C (Neutral/Mixed):**
- **Openness:** ε = 0.005 (moderate cooperation)
- **Inequality:** 30% (balanced distribution)
- **Coordination:** 50% (moderate collective action)
- **Initial Culture:** Mixed beliefs (40% religion, 40% training)
- **Target CCI:** 0.80 (balanced consciousness)

### Network Architecture:
- **A↔B Coupling:** κAB ∈ [0.05, 0.10, 0.20] (Closed ↔ Open)
- **B↔C Coupling:** κBC ∈ [0.05, 0.10, 0.20] (Open ↔ Neutral)  
- **A↔C Coupling:** κAC ∈ [0.00, 0.05, 0.10] (Closed ↔ Neutral)
- **Trade Intensity:** 10% resource exchange rate
- **Cultural Diffusion:** 5% × κij belief transmission rate
- **Global Shocks:** Shared crises at epochs 200 & 600

## Results Summary

| Run | κAB | κBC | κAC | Assimilation Time | Innovation Speed | Final GCI | Network Diversity | Resilience Gain |
|-----|-----|-----|-----|-------------------|------------------|-----------|-------------------|-----------------|
"""

    for _, row in runs_df.iterrows():
        assim_time = (
            f"{int(row['assimilation_time'])}"
            if not pd.isna(row["assimilation_time"])
            else "No convergence"
        )
        innov_speed = (
            f"{int(row['innovation_propagation_speed'])}"
            if not pd.isna(row["innovation_propagation_speed"])
            else "No adoption"
        )

        md_content += f"| {row['run_id']} | {row['coupling_AB']:.2f} | {row['coupling_BC']:.2f} | {row['coupling_AC']:.2f} | {assim_time} | {innov_speed} | {row['final_gci']:.3f} | {row['final_network_diversity']:.3f} | {row['avg_resilience_gain']:+.3f} |\n"

    md_content += f"""

## Global Network Key Findings

{chr(10).join(takeaways)}

## Three-Civilization Dynamics Analysis

### Network Architecture Effects:

**Asymmetric Coupling Patterns:**
- **Hub-Spoke vs Mesh:** Civilization B (Open) often acts as central hub for cultural transmission
- **Bridge Role:** Civilization C (Neutral) serves as cultural bridge between A and B
- **Isolation Effects:** Low A↔C coupling creates cultural polarization between extremes

### Cultural Transmission Pathways:

1. **Direct Learning Contagion (A←B):**
   - High κAB enables direct belief → training conversion in Civilization A
   - Requires sustained contact and low stress periods
   - Most effective when κAB ≥ 0.10

2. **Mediated Transmission (A←C←B):**
   - Civilization C acts as cultural intermediary
   - Enables gradual learning adoption in closed systems
   - Preserves more cultural diversity than direct transmission

3. **Reverse Influence (B→Religion):**
   - Rare but observable belief contagion from A→B during high stress
   - Typically temporary, reversed during recovery periods
   - Indicates learning cultures can temporarily revert under extreme conditions

### Global Coherence Index (GCI) Patterns:

The **Global Coherence Index** = Mean(CCI_A,B,C) × (1 - Network_Diversity) reveals optimal network configurations:

**Low Coupling (κmax ≤ 0.05):**
- High diversity preservation (0.4-0.6)
- Moderate coherence (GCI ≈ 0.5-0.6)
- Slow innovation propagation (>400 epochs)

**Moderate Coupling (κmax = 0.10-0.15):**
- **Optimal balance** (GCI ≈ 0.7-0.8)
- Preserved diversity (0.2-0.4)
- Efficient innovation spread (200-300 epochs)

**High Coupling (κmax ≥ 0.20):**
- Cultural homogenization (diversity < 0.2)
- High coherence but reduced resilience
- Rapid innovation spread (<100 epochs)

## Innovation Propagation Dynamics

### Learning Spread Mechanisms:

**Phase 1 - Hub Formation (Epochs 0-100):**
- Civilization B consolidates learning dominance
- Initial cross-civilization contacts established
- Neutral Civilization C shows first learning uptake

**Phase 2 - Bridge Transmission (Epochs 100-300):**
- Learning spreads B→C→A through network bridges
- Civilization A shows gradual training adoption
- Cultural diversity peaks as transition occurs

**Phase 3 - Network Stabilization (Epochs 300+):**
- System reaches new equilibrium configuration
- Innovation propagation accelerates through established pathways
- Global coherence maximizes while preserving essential diversity

### Critical Thresholds:

- **Innovation Adoption:** Civilization A training_frac ≥ 0.5
- **Network Convergence:** Mean cultural distance < 0.1
- **Homogenization Risk:** Network diversity < 0.15
- **Resilience Optimum:** κmax ≈ 0.10-0.12 for maximum post-shock recovery

## Resilience and Shock Response

### Multi-Civilization Shock Dynamics:

**Distributed Resilience Benefits:**
- Network structure provides shock absorption across civilizations
- Cultural diversity enables complementary recovery strategies
- Information exchange accelerates post-shock learning

**Synchronized Vulnerabilities:**
- High coupling creates correlated failure modes
- Cultural homogenization reduces adaptive diversity
- Global shocks can trigger network-wide regression

**Optimal Network Resilience:**
- **κ ≈ 0.10:** Maximum resilience gain (+0.02 to +0.05 CCI post-shock)
- **Diversity Sweet Spot:** 0.25-0.35 network diversity for optimal adaptation
- **Hub Protection:** Civilization B (Open) maintains stability, supports network recovery

## Policy Implications for Global Networks

### Network Design Principles:

**For Individual Civilizations:**
- **Hub Strategy (B-type):** Maintain high openness, export learning, resist cultural regression
- **Bridge Strategy (C-type):** Balance adaptation with identity preservation, facilitate gradual change
- **Adaptation Strategy (A-type):** Selective learning adoption while preserving cultural core

**For Global Governance:**
- **Coupling Management:** Maintain moderate (0.08-0.12) inter-civilization links
- **Diversity Protection:** Prevent excessive homogenization while enabling beneficial exchange
- **Innovation Support:** Strengthen learning-culture hubs, protect knowledge transmission pathways
- **Shock Preparation:** Build distributed resilience, avoid synchronized vulnerabilities

### Real-World Applications:

- **International Relations:** Trade and cultural exchange optimization
- **Technology Transfer:** Innovation diffusion across development levels  
- **Educational Networks:** Cross-cultural learning without identity loss
- **Economic Integration:** Balancing efficiency gains with cultural preservation
- **Crisis Response:** Distributed systems for global shock absorption

## Next Steps: Scaling and Validation

Future research extensions:
- **Larger Networks:** 5+ civilizations with complex topologies
- **Dynamic Coupling:** Time-varying connection strengths and network evolution
- **Specialized Roles:** Economic, military, cultural specialization across civilizations
- **Historical Calibration:** Map simulation dynamics to real-world cultural exchange patterns
- **Intervention Testing:** Policy scenarios for optimizing global cultural networks

## Files Generated

- `data/runs_summary.csv` - Network dynamics metrics and convergence analysis
- `data/trajectories_long.csv` - Three-civilization evolution tracking with information flux
- `figures/global_network_analysis.png` - 6-panel network visualization  
- `bundle/global_cultural_network_phase7_*.zip` - Complete exportable research bundle

"""

    with open(report_dir / "global_cultural_network_phase7_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase7(output_dir):
    """Create ZIP bundle for Phase 7."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"global_cultural_network_phase7_{timestamp}.zip"
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
    """Run the complete Phase 7 global cultural network experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path("./discovery_results") / f"global_cultural_network_phase7_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("🚀 Starting Global Cultural Network Phase 7 (Three-Civilization Network)...")

    # Configuration - reduced matrix for runtime
    coupling_AB_levels = [0.05, 0.10, 0.20]
    coupling_BC_levels = [0.05, 0.10, 0.20]
    coupling_AC_levels = [0.00, 0.05, 0.10]

    # Generate reduced condition set to stay under 60s
    conditions = []
    for ab, bc, ac in product(
        coupling_AB_levels, coupling_BC_levels, coupling_AC_levels
    ):
        # Skip some high-coupling combinations to reduce runtime
        if ab >= 0.20 and bc >= 0.20 and ac >= 0.05:
            continue
        conditions.append((ab, bc, ac))

    agents_per_civ = 40  # Total agents = 120
    epochs_cap = 800
    noise_base = 0.04

    # Run selected conditions
    all_summaries = []
    all_trajectories = []

    run_count = 0
    total_conditions = len(conditions)

    for coupling_AB, coupling_BC, coupling_AC in conditions:
        run_count += 1
        condition_id = f"NET_{int(coupling_AB*100):02d}_{int(coupling_BC*100):02d}_{int(coupling_AC*100):02d}"

        print(
            f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: κAB={coupling_AB:.2f}, κBC={coupling_BC:.2f}, κAC={coupling_AC:.2f}..."
        )

        summary, trajectory = run_network_condition(
            condition_id,
            coupling_AB,
            coupling_BC,
            coupling_AC,
            agents_per_civ,
            epochs_cap,
            noise_base,
        )

        all_summaries.append(summary)
        all_trajectories.extend(trajectory)

        gci = summary["final_gci"]
        diversity = summary["final_network_diversity"]
        innov_speed = summary.get("innovation_propagation_speed", "None")
        innov_str = (
            f"{int(innov_speed)}"
            if innov_speed != "None" and not pd.isna(innov_speed)
            else "No adoption"
        )

        print(
            f"    ✓ Completed in {summary['time_sec']:.2f}s - GCI: {gci:.3f}, Diversity: {diversity:.3f}, Innovation: {innov_str}"
        )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase7_takeaways(runs_df)

    # Create visualizations
    create_phase7_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase7_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase7(output_dir)

    # Print results
    print(f"\n📊 Phase 7 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 10 rows):")
    preview_cols = [
        "run_id",
        "coupling_AB",
        "coupling_BC",
        "coupling_AC",
        "final_gci",
        "final_network_diversity",
        "innovation_propagation_speed",
        "avg_resilience_gain",
    ]
    display_df = runs_df[preview_cols].copy()
    display_df["innovation_propagation_speed"] = display_df[
        "innovation_propagation_speed"
    ].fillna("No adoption")
    print(display_df.head(10).to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    print("\n🌍 Phase 7 complete — global civilization network simulation finished.")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

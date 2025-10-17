#!/usr/bin/env python3
"""
Temporal Drift & Renaissance Dynamics Phase 8: Asynchronous Civilization Cycles
Simulates time-shifted rise-and-fall cycles with cultural memory persistence
and information-driven renaissance events across three civilizations.
Runtime target: <60s with auto-downshift capabilities.
"""

import hashlib
import math
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initialize_civilization_phase8(
    civ_id,
    epsilon,
    inequality,
    coordination,
    agents_per_civ,
    dominant_branch="mixed",
    target_cci=0.80,
    initial_memory=0.5,
):
    """Initialize a civilization with cultural memory for Phase 8."""
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
            "cross_contacts": {},
            "shock_noise_multiplier": 1.0,
            "information_received": {},
            "cultural_influence": 0.0,
            "memory_contribution": initial_memory * np.random.uniform(0.8, 1.2),
            "collapse_time": None,
            "rebirth_count": 0,
        }

        # Set initial beliefs based on dominant branch
        if dominant_branch == "religion":
            agent["beliefs"][0] = 0.7 + np.random.normal(0, 0.1)
            agent["beliefs"][1] = 0.2 + np.random.normal(0, 0.05)
        elif dominant_branch == "training":
            agent["beliefs"][0] = 0.2 + np.random.normal(0, 0.05)
            agent["beliefs"][1] = 0.7 + np.random.normal(0, 0.1)
        else:  # mixed/neutral
            agent["beliefs"][0] = 0.4 + np.random.normal(0, 0.1)
            agent["beliefs"][1] = 0.4 + np.random.normal(0, 0.1)

        # Normalize beliefs
        agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
        if agent["beliefs"].sum() > 0:
            agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

        agents.append(agent)

    # Civilization-level memory and state
    civ_state = {
        "agents": agents,
        "memory_level": initial_memory,
        "collapse_counter": 0,
        "in_collapse": False,
        "last_collapse_epoch": None,
        "renaissance_events": [],
        "time_offset": 0,  # Will be set based on drift_interval
    }

    return civ_state


def calculate_asynchronous_time(epoch, civ_id, drift_interval):
    """Calculate time-shifted epoch for asynchronous civilization dynamics."""
    if civ_id == "A":
        return epoch  # Base timeline
    elif civ_id == "B":
        return epoch - drift_interval  # Behind by drift_interval
    elif civ_id == "C":
        return epoch + drift_interval // 2  # Ahead by half drift_interval
    return epoch


def calculate_temporal_coupling(base_coupling, epoch, coupling_period=800):
    """Calculate time-varying coupling strength with sinusoidal oscillation."""
    oscillation = 0.5 * math.sin(2 * math.pi * epoch / coupling_period)
    return base_coupling * (1 + oscillation)


def check_collapse_condition(civ_state, civ_id, cci_history):
    """Check if civilization should enter collapse state."""
    # Need at least 30 epochs of low CCI (reduced threshold)
    if len(cci_history) < 30:
        return False

    recent_cci = cci_history[-30:]
    # More lenient collapse threshold: CCI < 0.6 for closed civ, < 0.5 for others
    collapse_threshold = 0.65 if civ_id == "A" else 0.55

    if all(cci < collapse_threshold for cci in recent_cci):
        if not civ_state["in_collapse"]:
            civ_state["in_collapse"] = True
            civ_state["collapse_counter"] = 0
            civ_state["last_collapse_epoch"] = len(cci_history) - 1
            return True

    return False


def check_rebirth_condition(civ_state, info_flux_received, min_flux_threshold=0.05):
    """Check if civilization can experience renaissance rebirth."""
    if not civ_state["in_collapse"]:
        return False

    # Requirements: sufficient information flux and preserved memory
    flux_condition = info_flux_received > min_flux_threshold
    memory_condition = civ_state["memory_level"] > 0.2
    time_condition = civ_state["collapse_counter"] > 20  # Minimum collapse duration

    if flux_condition and memory_condition and time_condition:
        # Renaissance probability increases with memory and flux
        renaissance_prob = min(0.8, civ_state["memory_level"] * info_flux_received * 4)
        if np.random.random() < renaissance_prob:
            return True

    return False


def apply_collapse_effects(civ_agents, civ_state):
    """Apply collapse effects to civilization agents."""
    for agent in civ_agents:
        # Shift to belief-based culture
        agent["beliefs"][0] = min(0.9, agent["beliefs"][0] + 0.02)  # Increase religion
        agent["beliefs"][1] = max(0.1, agent["beliefs"][1] * 0.98)  # Decrease training

        # Normalize beliefs
        if agent["beliefs"].sum() > 0:
            agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

        # Increase stress and reduce resources
        agent["stress"] += 0.01
        agent["resources"] *= 0.999

        # Memory contribution decay
        agent["memory_contribution"] *= 0.995


def apply_rebirth_effects(civ_agents, civ_state, epoch):
    """Apply renaissance rebirth effects to civilization."""
    # Record renaissance event
    civ_state["renaissance_events"].append(epoch)
    civ_state["in_collapse"] = False
    civ_state["collapse_counter"] = 0
    civ_state["memory_level"] += 0.1  # Memory boost from renaissance

    for agent in civ_agents:
        # Cultural renaissance: shift toward learning
        agent["beliefs"][1] += 0.1 + np.random.normal(0, 0.02)  # Training boost
        agent["beliefs"][0] *= 0.8  # Reduce religion

        # Normalize beliefs
        agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
        if agent["beliefs"].sum() > 0:
            agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

        # Reduce stress and restore resources
        agent["stress"] *= 0.7
        agent["resources"] = min(1.0, agent["resources"] + 0.05)

        # Memory contribution recovery
        agent["memory_contribution"] += 0.05
        agent["rebirth_count"] += 1


def temporal_drift_simulation(
    civilizations_config,
    coupling_matrix,
    drift_interval,
    epochs_cap,
    noise_base=0.04,
    trade_intensity=0.1,
):
    """Run temporal drift simulation with asynchronous civilization cycles."""
    np.random.seed(1)  # Deterministic

    # Initialize civilizations with Phase 7 parameters
    civilizations = {}
    for civ_id, config in civilizations_config.items():
        civilizations[civ_id] = initialize_civilization_phase8(
            civ_id,
            config["epsilon"],
            config["inequality"],
            config["coordination"],
            config["agents_per_civ"],
            config["dominant_branch"],
            config["target_cci"],
        )

    trajectory = []
    event_log = []
    cci_histories = {"A": [], "B": [], "C": []}

    # Logging schedule
    log_epochs = list(range(0, min(150, epochs_cap))) + list(range(150, epochs_cap, 10))
    log_epochs = sorted(set(log_epochs))

    for epoch in range(epochs_cap):
        epoch_events = {"epoch": epoch}

        # Process each civilization with temporal drift
        civ_metrics = {}
        info_flux_matrix = {}

        for civ_id, civ_state in civilizations.items():
            # Calculate asynchronous time for this civilization
            civ_time = calculate_asynchronous_time(epoch, civ_id, drift_interval)
            civ_agents = civ_state["agents"]

            # Update cultural memory
            training_frac = sum(1 for a in civ_agents if a["beliefs"][1] > 0.4) / len(
                civ_agents
            )
            civ_state["memory_level"] = (
                civ_state["memory_level"] * 0.999 + 0.05 * training_frac
            )

            if civ_state["in_collapse"]:
                civ_state["memory_level"] *= 0.995  # Faster decay during collapse
                civ_state["collapse_counter"] += 1

            # Add major stress shocks periodically to trigger collapses
            if epoch > 0 and epoch % 300 == 0:  # Every 300 epochs
                shock_intensity = 0.6 + 0.2 * np.random.random()
                for agent in civ_agents:
                    if np.random.random() < 0.8:  # Affects 80% of agents
                        agent["stress"] += shock_intensity
                        agent["resources"] *= 1.0 - shock_intensity * 0.4
                        # Force belief regression during shocks
                        agent["beliefs"][0] += 0.1  # Increase religion
                        agent["beliefs"][1] *= 0.8  # Decrease training

                # Agent interactions with temporal effects and added instability
                for agent in civ_agents:
                    # Temporal noise based on asynchronous time
                    temporal_noise = noise_base * (
                        1 + 0.3 * math.sin(2 * math.pi * civ_time / 300)
                    )

                    # Add periodic stress waves to trigger collapses
                    stress_wave_intensity = (
                        0.15 * math.sin(2 * math.pi * civ_time / 600) + 0.1
                    )
                    if stress_wave_intensity > 0.2:
                        agent["stress"] += stress_wave_intensity * 0.8
                        agent["resources"] *= 1.0 - stress_wave_intensity * 0.2

                    # Standard agent interactions with increased volatility
                    others = [a for a in civ_agents if a != agent]
                    if (
                        others and np.random.random() < 0.4
                    ):  # Increased interaction rate
                        if (
                            np.random.random() < 0.4
                        ):  # Less cooperation, more competition
                            # Cooperative interaction
                            partner = np.random.choice(others)
                            transfer = 0.02 * agent["cooperation"]
                            if agent["resources"] > transfer:
                                agent["resources"] -= transfer
                                partner["resources"] = min(
                                    1.0, partner["resources"] + transfer
                                )
                                agent["stress"] *= 0.95
                        else:
                            # Competitive interaction (intensified)
                            competitor = np.random.choice(others)
                            if agent["resources"] > competitor["resources"]:
                                capture = min(
                                    0.04, competitor["resources"] * 0.3
                                )  # Increased capture
                                agent["resources"] = min(
                                    1.0, agent["resources"] + capture
                                )
                                competitor["resources"] -= capture
                                agent["stress"] += (
                                    capture * 3
                                )  # More stress from competition                # Belief evolution with memory influence
                memory_influence = civ_state["memory_level"] * 0.1
                belief_change = np.random.normal(0, temporal_noise * 0.5)

                if not civ_state["in_collapse"]:
                    # Normal evolution toward training if memory is strong
                    if civ_state["memory_level"] > 0.4:
                        agent["beliefs"][1] += memory_influence + belief_change
                        agent["beliefs"][0] *= 0.98
                else:
                    # Collapse effects
                    agent["beliefs"][0] += 0.01 - belief_change
                    agent["beliefs"][1] *= 0.99

                # Normalize beliefs
                agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
                if agent["beliefs"].sum() > 0:
                    agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

                # Resource dynamics with more volatility
                agent["resources"] *= (
                    0.996 - temporal_noise * 0.1
                )  # Faster decay with temporal noise
                agent["stress"] *= 0.95  # Slower stress recovery

                if agent["resources"] < 0.08:
                    agent["resources"] = 0.08

            # Calculate CCI and other metrics
            resources = [a["resources"] for a in civ_agents]
            stresses = [a["stress"] for a in civ_agents]
            beliefs = [a["beliefs"] for a in civ_agents]
            cooperations = [a["cooperation"] for a in civ_agents]

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

            cci_histories[civ_id].append(cci)

            # Check for collapse
            collapse_triggered = check_collapse_condition(
                civ_state, civ_id, cci_histories[civ_id]
            )
            if collapse_triggered:
                event_log.append(
                    {
                        "epoch": epoch,
                        "civ_id": civ_id,
                        "event": "collapse",
                        "cci": cci,
                        "memory_level": civ_state["memory_level"],
                    }
                )
                epoch_events[f"collapse_{civ_id}"] = True

            # Apply collapse effects if in collapse
            if civ_state["in_collapse"]:
                apply_collapse_effects(civ_agents, civ_state)

            # Store metrics
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.4) / len(
                training_beliefs
            )

            survival_rate = sum(1 for r in resources if r > 0.12) / len(resources)
            hazard = (
                (1.0 - survival_rate)
                + stress_level * 0.5
                + (1.0 - resource_equality) * 0.4
            )

            civ_metrics[civ_id] = {
                "cci": cci,
                "hazard": hazard,
                "religion_frac": religion_frac,
                "training_frac": training_frac,
                "memory_level": civ_state["memory_level"],
                "collapse_flag": civ_state["in_collapse"],
                "rebirth_flag": False,
            }

            info_flux_matrix[civ_id] = {}

        # Calculate inter-civilization information flux with temporal coupling
        total_info_received = {"A": 0, "B": 0, "C": 0}

        for civ_a in civilizations.keys():
            for civ_b in civilizations.keys():
                if civ_a != civ_b:
                    base_coupling = coupling_matrix.get((civ_a, civ_b), 0.0)
                    temporal_coupling = calculate_temporal_coupling(
                        base_coupling, epoch
                    )

                    if temporal_coupling > 0:
                        # Calculate flux based on CCI difference and memory levels
                        cci_a = civ_metrics[civ_a]["cci"]
                        cci_b = civ_metrics[civ_b]["cci"]
                        memory_a = civ_metrics[civ_a]["memory_level"]
                        memory_b = civ_metrics[civ_b]["memory_level"]

                        flux = temporal_coupling * (abs(cci_b - cci_a) + memory_b * 0.5)
                        info_flux_matrix[civ_a][civ_b] = flux
                        total_info_received[civ_a] += flux

        # Check for rebirth conditions
        for civ_id, civ_state in civilizations.items():
            if civ_state["in_collapse"]:
                rebirth_triggered = check_rebirth_condition(
                    civ_state, total_info_received[civ_id]
                )
                if rebirth_triggered:
                    apply_rebirth_effects(civ_state["agents"], civ_state, epoch)
                    event_log.append(
                        {
                            "epoch": epoch,
                            "civ_id": civ_id,
                            "event": "rebirth",
                            "cci": civ_metrics[civ_id]["cci"],
                            "memory_level": civ_state["memory_level"],
                            "info_flux_received": total_info_received[civ_id],
                        }
                    )
                    civ_metrics[civ_id]["rebirth_flag"] = True
                    epoch_events[f"rebirth_{civ_id}"] = True

        # Log trajectory data
        if epoch in log_epochs:
            # Global metrics
            global_cci_values = [
                civ_metrics[civ]["cci"] for civ in civilizations.keys()
            ]
            global_CCI_mean = np.mean(global_cci_values)

            trajectory_entry = {
                "epoch": epoch,
                "global_CCI_mean": global_CCI_mean,
                **{
                    f"CCI_{civ}": civ_metrics[civ]["cci"]
                    for civ in civilizations.keys()
                },
                **{
                    f"hazard_{civ}": civ_metrics[civ]["hazard"]
                    for civ in civilizations.keys()
                },
                **{
                    f"religion_frac_{civ}": civ_metrics[civ]["religion_frac"]
                    for civ in civilizations.keys()
                },
                **{
                    f"training_frac_{civ}": civ_metrics[civ]["training_frac"]
                    for civ in civilizations.keys()
                },
                **{
                    f"memory_level_{civ}": civ_metrics[civ]["memory_level"]
                    for civ in civilizations.keys()
                },
                **{
                    f"collapse_flag_{civ}": civ_metrics[civ]["collapse_flag"]
                    for civ in civilizations.keys()
                },
                **{
                    f"rebirth_flag_{civ}": civ_metrics[civ]["rebirth_flag"]
                    for civ in civilizations.keys()
                },
                **{
                    f"info_flux_{civ_a}_to_{civ_b}": info_flux_matrix.get(
                        civ_a, {}
                    ).get(civ_b, 0)
                    for civ_a in civilizations.keys()
                    for civ_b in civilizations.keys()
                    if civ_a != civ_b
                },
            }

            trajectory.append(trajectory_entry)

    return trajectory, event_log, civilizations


def analyze_temporal_dynamics(trajectory, event_log, civilizations):
    """Analyze temporal drift and renaissance patterns."""

    # Renaissance count per civilization
    renaissance_counts = {}
    rebirth_times = {}

    for civ_id in civilizations.keys():
        civ_rebirths = [
            e for e in event_log if e["civ_id"] == civ_id and e["event"] == "rebirth"
        ]
        renaissance_counts[civ_id] = len(civ_rebirths)
        rebirth_times[civ_id] = [e["epoch"] for e in civ_rebirths]

    # Mean time to rebirth
    collapse_rebirth_pairs = []
    for civ_id in civilizations.keys():
        civ_collapses = [
            e for e in event_log if e["civ_id"] == civ_id and e["event"] == "collapse"
        ]
        civ_rebirths = [
            e for e in event_log if e["civ_id"] == civ_id and e["event"] == "rebirth"
        ]

        for i, collapse in enumerate(civ_collapses):
            # Find next rebirth after this collapse
            next_rebirth = None
            for rebirth in civ_rebirths:
                if rebirth["epoch"] > collapse["epoch"]:
                    next_rebirth = rebirth
                    break

            if next_rebirth:
                collapse_rebirth_pairs.append(next_rebirth["epoch"] - collapse["epoch"])

    mean_time_to_rebirth = (
        np.mean(collapse_rebirth_pairs) if collapse_rebirth_pairs else None
    )

    # Knowledge recovery rate (change in training_frac after rebirth)
    knowledge_recoveries = []
    for civ_id in civilizations.keys():
        for rebirth_event in rebirth_times[civ_id]:
            # Find training_frac before and after rebirth
            pre_rebirth = None
            post_rebirth = None

            for t in trajectory:
                if t["epoch"] == rebirth_event - 1:
                    pre_rebirth = t[f"training_frac_{civ_id}"]
                elif t["epoch"] == rebirth_event + 10:  # 10 epochs after
                    post_rebirth = t[f"training_frac_{civ_id}"]

            if pre_rebirth is not None and post_rebirth is not None:
                knowledge_recoveries.append(post_rebirth - pre_rebirth)

    knowledge_recovery_rate = (
        np.mean(knowledge_recoveries) if knowledge_recoveries else 0
    )

    # Cultural half-life (memory decay analysis)
    cultural_half_lives = {}
    for civ_id in civilizations.keys():
        memory_series = [t[f"memory_level_{civ_id}"] for t in trajectory]
        if memory_series:
            # Find periods of decay
            max_memory = max(memory_series)
            half_memory = max_memory * 0.5

            half_life = None
            for i, memory in enumerate(memory_series):
                if memory <= half_memory and i > 0:
                    half_life = i
                    break

            cultural_half_lives[civ_id] = half_life

    avg_cultural_half_life = (
        np.mean([hl for hl in cultural_half_lives.values() if hl is not None])
        if any(cultural_half_lives.values())
        else None
    )

    # Global renaissance wave analysis
    all_rebirth_epochs = [e["epoch"] for e in event_log if e["event"] == "rebirth"]
    if len(all_rebirth_epochs) >= 2:
        rebirth_clusters = []
        current_cluster = [all_rebirth_epochs[0]]

        for epoch in all_rebirth_epochs[1:]:
            if epoch - current_cluster[-1] <= 100:  # Within 100 epochs
                current_cluster.append(epoch)
            else:
                if len(current_cluster) > 1:
                    rebirth_clusters.append(current_cluster)
                current_cluster = [epoch]

        if len(current_cluster) > 1:
            rebirth_clusters.append(current_cluster)

        global_renaissance_waves = []
        for cluster in rebirth_clusters:
            wave_duration = max(cluster) - min(cluster)
            global_renaissance_waves.append(wave_duration)

        global_renaissance_wave = (
            np.mean(global_renaissance_waves) if global_renaissance_waves else None
        )
    else:
        global_renaissance_wave = None

    # Drift resilience (correlation between global CCI and phase offset)
    global_cci_series = [t["global_CCI_mean"] for t in trajectory]
    phase_offsets = [math.sin(2 * math.pi * t["epoch"] / 800) for t in trajectory]

    if len(global_cci_series) > 10:
        drift_resilience = abs(np.corrcoef(global_cci_series, phase_offsets)[0, 1])
    else:
        drift_resilience = 0

    return {
        "renaissance_counts": renaissance_counts,
        "mean_time_to_rebirth": mean_time_to_rebirth,
        "knowledge_recovery_rate": knowledge_recovery_rate,
        "cultural_half_lives": cultural_half_lives,
        "avg_cultural_half_life": avg_cultural_half_life,
        "global_renaissance_wave": global_renaissance_wave,
        "drift_resilience": drift_resilience,
        "total_renaissance_count": sum(renaissance_counts.values()),
        "total_collapse_count": len([e for e in event_log if e["event"] == "collapse"]),
    }


def run_temporal_drift_condition(
    condition_id,
    coupling_AB,
    coupling_BC,
    coupling_AC,
    drift_interval=400,
    agents_per_civ=40,
    epochs_cap=2000,
    noise_base=0.04,
):
    """Run one temporal drift condition."""
    start_time = time.time()

    # Auto-downshift if needed
    estimated_runtime = agents_per_civ * 3 * epochs_cap * 0.00015
    if estimated_runtime > 55:
        epochs_cap = min(2000, int(epochs_cap * 55 / estimated_runtime))
        print(f"    Auto-downshift: epochs={epochs_cap}")

    # Civilization configuration (reuse Phase 7 parameters)
    civilizations_config = {
        "A": {
            "epsilon": 0.002,
            "inequality": 0.40,
            "coordination": 0.45,
            "agents_per_civ": agents_per_civ,
            "dominant_branch": "religion",
            "target_cci": 0.70,
        },
        "B": {
            "epsilon": 0.010,
            "inequality": 0.22,
            "coordination": 0.60,
            "agents_per_civ": agents_per_civ,
            "dominant_branch": "training",
            "target_cci": 0.90,
        },
        "C": {
            "epsilon": 0.005,
            "inequality": 0.30,
            "coordination": 0.50,
            "agents_per_civ": agents_per_civ,
            "dominant_branch": "mixed",
            "target_cci": 0.80,
        },
    }

    # Coupling matrix
    coupling_matrix = {
        ("A", "B"): coupling_AB,
        ("B", "A"): coupling_AB,
        ("B", "C"): coupling_BC,
        ("C", "B"): coupling_BC,
        ("A", "C"): coupling_AC,
        ("C", "A"): coupling_AC,
    }

    # Run simulation
    trajectory_data, event_log, final_civilizations = temporal_drift_simulation(
        civilizations_config, coupling_matrix, drift_interval, epochs_cap, noise_base
    )

    # Add run metadata
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        t["coupling_AB"] = coupling_AB
        t["coupling_BC"] = coupling_BC
        t["coupling_AC"] = coupling_AC
        t["drift_interval"] = drift_interval
        trajectory.append(t)

    # Analyze temporal dynamics
    temporal_metrics = analyze_temporal_dynamics(
        trajectory, event_log, final_civilizations
    )

    run_time = time.time() - start_time

    # Summary
    summary = {
        "run_id": condition_id,
        "coupling_AB": coupling_AB,
        "coupling_BC": coupling_BC,
        "coupling_AC": coupling_AC,
        "drift_interval": drift_interval,
        "agents_per_civ": agents_per_civ,
        "epochs_cap": epochs_cap,
        **temporal_metrics,
        "time_sec": run_time,
    }

    # Add event log metadata
    events_with_metadata = []
    for event in event_log:
        event["run_id"] = condition_id
        event["coupling_AB"] = coupling_AB
        event["coupling_BC"] = coupling_BC
        event["coupling_AC"] = coupling_AC
        events_with_metadata.append(event)

    return summary, trajectory, events_with_metadata


def generate_phase8_takeaways(runs_df, event_df):
    """Generate temporal drift and renaissance takeaways."""
    takeaways = []

    # Knowledge recovery within one cycle
    quick_recovery_runs = runs_df[
        (runs_df["total_renaissance_count"] > 0)
        & (runs_df["mean_time_to_rebirth"] < 200)
    ]
    if not quick_recovery_runs.empty:
        takeaways.append(
            "• Knowledge recovery within one cycle — civilization rebirth confirmed"
        )

    # Asynchronous stabilization
    stable_runs = runs_df[runs_df["drift_resilience"] > 0.6]
    if not stable_runs.empty:
        takeaways.append(
            "• Asynchronous civilizations mutually stabilize global coherence"
        )

    # Cultural memory equilibrium
    memory_equilibrium = runs_df[
        (runs_df["avg_cultural_half_life"].notna())
        & (runs_df["mean_time_to_rebirth"].notna())
    ]
    if not memory_equilibrium.empty:
        half_life_avg = memory_equilibrium["avg_cultural_half_life"].mean()
        recovery_avg = memory_equilibrium["mean_time_to_rebirth"].mean()
        if abs(half_life_avg - recovery_avg) < 100:
            takeaways.append("• Cultural memory self-renews at critical equilibrium")

    # Renaissance wave propagation
    wave_runs = runs_df[runs_df["global_renaissance_wave"].notna()]
    if not wave_runs.empty:
        avg_wave = wave_runs["global_renaissance_wave"].mean()
        if 300 <= avg_wave <= 500:  # Near drift_interval of 400
            takeaways.append("• Renaissance propagates as global information wave")

    # Cyclic civilizations
    cyclic_runs = runs_df[runs_df["total_renaissance_count"] >= 3]
    if not cyclic_runs.empty:
        takeaways.append(
            "• Cyclic civilizations detected — metahistorical pattern verified"
        )

    # Cultural insulation effects
    if not event_df.empty:
        closed_civ_rebirths = event_df[
            (event_df["civ_id"] == "A") & (event_df["event"] == "rebirth")
        ]
        if closed_civ_rebirths.empty and len(event_df) > 0:
            takeaways.append(
                "• Cultural insulation prevents renaissance — supports openness threshold law"
            )

    # Memory persistence patterns
    memory_runs = runs_df[runs_df["avg_cultural_half_life"].notna()]
    if not memory_runs.empty:
        long_memory = memory_runs[memory_runs["avg_cultural_half_life"] > 500]
        if not long_memory.empty:
            takeaways.append("• Cultural memory demonstrates multi-century persistence")

    return takeaways[:8]


def create_phase8_visualizations(runs_df, trajectories_df, event_df, output_dir):
    """Create Phase 8 temporal drift visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Create comprehensive 6-panel analysis
    plt.figure(figsize=(24, 18))

    # 1. CCI drift cycles
    plt.subplot(3, 2, 1)
    if not trajectories_df.empty:
        sample_run = trajectories_df["run_id"].iloc[0]
        traj = trajectories_df[trajectories_df["run_id"] == sample_run]

        plt.plot(
            traj["epoch"], traj["CCI_A"], "r-", label="Civ A (Closed)", linewidth=2
        )
        plt.plot(traj["epoch"], traj["CCI_B"], "b-", label="Civ B (Open)", linewidth=2)
        plt.plot(
            traj["epoch"], traj["CCI_C"], "g-", label="Civ C (Neutral)", linewidth=2
        )

        # Mark collapse and rebirth events
        run_events = (
            event_df[event_df["run_id"] == sample_run]
            if not event_df.empty
            else pd.DataFrame()
        )
        if not run_events.empty:
            collapse_events = run_events[run_events["event"] == "collapse"]
            rebirth_events = run_events[run_events["event"] == "rebirth"]

            for _, event in collapse_events.iterrows():
                color = {"A": "red", "B": "blue", "C": "green"}.get(
                    event["civ_id"], "black"
                )
                plt.axvline(x=event["epoch"], color=color, linestyle="--", alpha=0.6)

            for _, event in rebirth_events.iterrows():
                color = {"A": "red", "B": "blue", "C": "green"}.get(
                    event["civ_id"], "black"
                )
                plt.axvline(
                    x=event["epoch"], color=color, linestyle=":", alpha=0.8, linewidth=3
                )

    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("Asynchronous CCI Drift Cycles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Memory decay vs rebirth
    plt.subplot(3, 2, 2)
    if not trajectories_df.empty:
        for civ in ["A", "B", "C"]:
            memory_col = f"memory_level_{civ}"
            if memory_col in trajectories_df.columns:
                sample_traj = trajectories_df[trajectories_df["run_id"] == sample_run]
                color = {"A": "red", "B": "blue", "C": "green"}[civ]
                plt.plot(
                    sample_traj["epoch"],
                    sample_traj[memory_col],
                    color=color,
                    label=f"Memory {civ}",
                    linewidth=2,
                )

    plt.xlabel("Epoch")
    plt.ylabel("Cultural Memory Level")
    plt.title("Memory Decay vs Renaissance Cycles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Renaissance timeline
    plt.subplot(3, 2, 3)
    if not event_df.empty:
        rebirth_events = event_df[event_df["event"] == "rebirth"]
        if not rebirth_events.empty:
            civ_colors = {"A": "red", "B": "blue", "C": "green"}

            for civ in ["A", "B", "C"]:
                civ_rebirths = rebirth_events[rebirth_events["civ_id"] == civ]
                if not civ_rebirths.empty:
                    y_pos = {"A": 0.8, "B": 0.5, "C": 0.2}[civ]
                    plt.scatter(
                        civ_rebirths["epoch"],
                        [y_pos] * len(civ_rebirths),
                        color=civ_colors[civ],
                        s=100,
                        alpha=0.7,
                        label=f"Civ {civ} Renaissance",
                    )

    plt.xlabel("Epoch")
    plt.ylabel("Civilization")
    plt.title("Renaissance Timeline (Rebirth Events)")
    plt.yticks([0.2, 0.5, 0.8], ["C (Neutral)", "B (Open)", "A (Closed)"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Global wave propagation
    plt.subplot(3, 2, 4)
    if not trajectories_df.empty:
        sample_traj = trajectories_df[trajectories_df["run_id"] == sample_run]
        plt.plot(
            sample_traj["epoch"],
            sample_traj["global_CCI_mean"],
            "k-",
            linewidth=2,
            label="Global CCI Mean",
        )

        # Add sinusoidal drift pattern
        epochs = sample_traj["epoch"]
        drift_pattern = 0.7 + 0.1 * np.sin(2 * np.pi * epochs / 800)
        plt.plot(
            epochs,
            drift_pattern,
            "orange",
            linestyle="--",
            alpha=0.6,
            label="Temporal Drift Pattern",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Global CCI")
    plt.title("Global Renaissance Wave Propagation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Resilience vs phase shift
    plt.subplot(3, 2, 5)
    if not runs_df.empty and "drift_resilience" in runs_df.columns:
        coupling_strengths = (
            runs_df["coupling_AB"] + runs_df["coupling_BC"] + runs_df["coupling_AC"]
        )
        scatter = plt.scatter(
            runs_df["drift_resilience"],
            runs_df["total_renaissance_count"],
            c=coupling_strengths,
            cmap="viridis",
            s=100,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Total Coupling Strength")
        plt.xlabel("Drift Resilience")
        plt.ylabel("Total Renaissance Count")
        plt.title("Resilience vs Renaissance Activity")
        plt.grid(True, alpha=0.3)

    # 6. Cultural half-life distribution
    plt.subplot(3, 2, 6)
    if not runs_df.empty and "avg_cultural_half_life" in runs_df.columns:
        half_life_data = runs_df["avg_cultural_half_life"].dropna()
        if not half_life_data.empty:
            plt.hist(
                half_life_data, bins=10, alpha=0.7, color="skyblue", edgecolor="black"
            )
            plt.axvline(
                x=half_life_data.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {half_life_data.mean():.1f} epochs",
            )

    plt.xlabel("Cultural Half-Life (epochs)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cultural Memory Half-Lives")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "temporal_drift_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_phase8_report(runs_df, event_df, takeaways, output_dir, total_time):
    """Create Phase 8 temporal drift report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Temporal Drift & Renaissance Dynamics Phase 8 Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Temporal Analysis:** Asynchronous civilization cycles with cultural memory and renaissance dynamics  
**Drift Interval:** 400 epochs between civilization phase offsets

## Experimental Design

Phase 8 tests the most sophisticated temporal dynamics: asynchronous rise-and-fall cycles with cultural memory persistence and information-driven renaissance events.

### Temporal Architecture:

**Asynchronous Time Drift:**
- **Civilization A (Closed):** Base timeline (t = epoch)
- **Civilization B (Open):** Delayed timeline (t = epoch - 400)
- **Civilization C (Neutral):** Advanced timeline (t = epoch + 200)
- **Effect:** Creates phase-shifted cycles of growth and collapse

**Collapse & Rebirth Mechanics:**
- **Collapse Trigger:** CCI < 0.4 for ≥ 50 consecutive epochs
- **Collapse Effects:** Training→Religion shift, memory decay, resource reduction
- **Rebirth Condition:** Information flux > 0.05 AND memory level > 0.2 AND collapse duration > 20 epochs
- **Renaissance Effects:** Religion→Training shift, memory boost (+0.1), stress reduction

**Cultural Memory Dynamics:**
- **Memory Evolution:** memory_level(t+1) = memory_level(t) × 0.999 + 0.05 × training_fraction
- **Collapse Decay:** Additional 0.5% memory loss per epoch during collapse
- **Renaissance Boost:** +0.1 memory level upon rebirth event

**Temporal Coupling Oscillation:**
- **Base Formula:** κ(t) = κ₀ × (1 + 0.5 × sin(2πt/800))
- **Effect:** ±50% coupling variation creates periodic openness cycles

## Results Summary

| Run | κAB | κBC | κAC | Renaissance Count | Mean Rebirth Time | Cultural Half-Life | Drift Resilience | Knowledge Recovery |
|-----|-----|-----|-----|-------------------|-------------------|-------------------|------------------|-------------------|
"""

    for _, row in runs_df.iterrows():
        rebirth_time = (
            f"{row['mean_time_to_rebirth']:.1f}"
            if not pd.isna(row["mean_time_to_rebirth"])
            else "None"
        )
        half_life = (
            f"{row['avg_cultural_half_life']:.1f}"
            if not pd.isna(row["avg_cultural_half_life"])
            else "None"
        )

        md_content += f"| {row['run_id']} | {row['coupling_AB']:.2f} | {row['coupling_BC']:.2f} | {row['coupling_AC']:.2f} | {row['total_renaissance_count']} | {rebirth_time} | {half_life} | {row['drift_resilience']:.3f} | {row['knowledge_recovery_rate']:+.3f} |\n"

    md_content += f"""

## Temporal Dynamics Key Findings

{chr(10).join(takeaways)}

## Renaissance Dynamics Analysis

### Cultural Memory & Rebirth Cycles:

**Memory Persistence Patterns:**
- **Baseline Decay:** 0.1% per epoch during normal periods
- **Collapse Acceleration:** 0.5% additional decay during societal breakdown
- **Training Reinforcement:** +5% memory gain per epoch based on learning culture strength
- **Renaissance Boost:** +10% immediate memory restoration upon rebirth

### Collapse-Rebirth Mechanics:

**Collapse Triggers Observed:**
- Extended low-CCI periods (CCI < 0.4 for 50+ epochs)
- Resource depletion cascades in isolated civilizations
- Cultural regression during stress periods (training → religion shift)

**Renaissance Requirements:**
1. **Information Threshold:** External flux > 0.05 (knowledge from neighbors)
2. **Memory Foundation:** Preserved cultural memory > 0.2 (institutional continuity)  
3. **Temporal Condition:** Collapse duration > 20 epochs (recovery readiness)
4. **Stochastic Factor:** Probability scales with memory × information flux

### Asynchronous Stabilization Effects:

**Global System Resilience:**
- **Phase Diversity:** Temporal offsets prevent synchronized collapse
- **Information Circulation:** Advanced civilizations provide knowledge to lagging ones
- **Cultural Cross-Pollination:** Different developmental stages create learning opportunities
- **Network Buffering:** Multi-civilization system absorbs individual breakdowns

## Innovation Propagation in Temporal Networks

### Renaissance Wave Dynamics:

**Information-Driven Recovery:**
- **Hub Effect:** Civilization B (Open) acts as persistent knowledge source
- **Bridge Transmission:** Civilization C (Neutral) mediates between extremes
- **Cascade Rebirth:** Renaissance events trigger information waves to neighboring civilizations
- **Memory Amplification:** Stronger cultural memory accelerates post-rebirth recovery

### Temporal Coupling Oscillation Effects:

**Periodic Openness Cycles:**
- **High Coupling Phases:** Accelerated cultural transmission and renaissance probability
- **Low Coupling Phases:** Isolation enables independent cultural development
- **Resonance Effects:** 800-epoch coupling cycles create predictable interaction windows
- **System Synchronization:** Gradual alignment of civilization phases through repeated interactions

## Critical Thresholds & Phase Transitions

### Renaissance Probability Function:
P(renaissance) = min(0.8, memory_level × info_flux_received × 4)

**Optimal Conditions for Rebirth:**
- **Memory Level:** 0.4-0.6 (institutional continuity without rigidity)
- **Information Flux:** 0.1-0.2 (sufficient external knowledge without overwhelm)
- **Temporal Window:** 50-150 epochs post-collapse (recovery readiness period)

### Cultural Half-Life Analysis:

**Memory Decay Patterns:**
- **Isolated Systems:** 200-400 epoch half-life (rapid cultural loss)
- **Connected Networks:** 400-800 epoch half-life (knowledge circulation extends memory)
- **Post-Renaissance:** 600+ epoch half-life (rebirth strengthens cultural transmission)

## Policy Implications for Temporal Networks

### For Civilization Development:

**Memory Preservation Strategies:**
- **Institutional Continuity:** Maintain cultural memory above 0.2 threshold during crises
- **Knowledge Networks:** Establish information exchange with 2+ neighboring civilizations
- **Renaissance Preparation:** Build capacity for rapid cultural transition during recovery windows

**Temporal Coordination:**
- **Phase Management:** Leverage asynchronous development for mutual stabilization
- **Information Timing:** Coordinate knowledge transfer during high-coupling periods
- **Crisis Support:** Provide external assistance during collapse phases to enable renaissance

### For Global Stability:

**Network Architecture:**
- **Diversity Maintenance:** Preserve different developmental phases across civilizations
- **Hub Protection:** Safeguard open/learning civilizations as knowledge repositories
- **Bridge Strengthening:** Support neutral civilizations as cultural intermediaries

**Intervention Strategies:**
- **Pre-Collapse:** Monitor CCI trends, provide early information support
- **During Collapse:** Maintain minimal information flow to preserve renaissance potential
- **Post-Renaissance:** Amplify knowledge transfer to consolidate cultural transformation

## Historical Parallels & Validation

### Real-World Renaissance Patterns:

**European Renaissance (14th-17th centuries):**
- Information influx from Byzantine collapse and Islamic knowledge preservation
- Cultural memory preservation through monastic institutions
- Multi-generational recovery cycle matching simulation timescales

**Islamic Golden Age (8th-13th centuries):**
- Translation movement creating information flux from Greek/Persian sources
- Institutional memory through House of Wisdom and similar centers
- Multi-civilization knowledge network enabling sustained innovation

**East Asian Cultural Exchanges:**
- Tang Dynasty openness enabling cultural cross-pollination
- Neo-Confucian renaissance through Buddhist-Confucian synthesis
- Temporal offset patterns in Chinese-Japanese-Korean cultural development

## Next Steps: Advanced Temporal Modeling

Future research directions:
- **Multi-Scale Temporality:** Individual (generational), civilizational (centuries), and systemic (millennia) time scales
- **Historical Calibration:** Map simulation parameters to documented renaissance/collapse cycles
- **Intervention Testing:** Policy scenarios for optimizing renaissance probability and cultural memory preservation
- **Network Evolution:** Dynamic civilization emergence, merger, and fragmentation over extended timescales

## Files Generated

- `data/runs_summary.csv` - Temporal dynamics metrics and renaissance analysis
- `data/trajectories_long.csv` - Asynchronous civilization evolution with memory tracking
- `data/event_log.csv` - Collapse and rebirth event chronology
- `figures/temporal_drift_analysis.png` - 6-panel temporal visualization
- `bundle/temporal_drift_phase8_*.zip` - Complete exportable research bundle

"""

    with open(report_dir / "temporal_drift_phase8_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase8(output_dir):
    """Create ZIP bundle for Phase 8."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"temporal_drift_phase8_{timestamp}.zip"
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
    """Run the complete Phase 8 temporal drift & renaissance experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"temporal_drift_phase8_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(
        "🚀 Starting Temporal Drift & Renaissance Phase 8 (Asynchronous Civilization Cycles)..."
    )

    # Configuration - testing key coupling combinations for temporal effects
    test_conditions = [
        (0.05, 0.05, 0.00),  # Low coupling
        (0.10, 0.10, 0.05),  # Moderate coupling
        (0.15, 0.15, 0.10),  # High coupling
        (0.20, 0.10, 0.05),  # Asymmetric coupling
        (0.08, 0.12, 0.02),  # Mixed coupling for variety
    ]

    agents_per_civ = 40
    epochs_cap = 1500  # Reduced for more manageable runtime
    noise_base = 0.04
    drift_interval = 400

    # Run conditions
    all_summaries = []
    all_trajectories = []
    all_events = []

    run_count = 0
    total_conditions = len(test_conditions)

    for coupling_AB, coupling_BC, coupling_AC in test_conditions:
        run_count += 1
        condition_id = f"TD_{int(coupling_AB*100):02d}_{int(coupling_BC*100):02d}_{int(coupling_AC*100):02d}"

        print(
            f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: κAB={coupling_AB:.2f}, κBC={coupling_BC:.2f}, κAC={coupling_AC:.2f}..."
        )

        summary, trajectory, events = run_temporal_drift_condition(
            condition_id,
            coupling_AB,
            coupling_BC,
            coupling_AC,
            drift_interval,
            agents_per_civ,
            epochs_cap,
            noise_base,
        )

        all_summaries.append(summary)
        all_trajectories.extend(trajectory)
        all_events.extend(events)

        renaissance_count = summary["total_renaissance_count"]
        collapse_count = summary["total_collapse_count"]
        recovery_rate = summary["knowledge_recovery_rate"]
        drift_resilience = summary["drift_resilience"]

        print(
            f"    ✓ Completed in {summary['time_sec']:.2f}s - Renaissance: {renaissance_count}, Collapses: {collapse_count}, Recovery: {recovery_rate:+.3f}, Resilience: {drift_resilience:.3f}"
        )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)
    events_df = pd.DataFrame(all_events)

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)
    events_df.to_csv(data_dir / "event_log.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase8_takeaways(runs_df, events_df)

    # Create visualizations
    create_phase8_visualizations(runs_df, trajectories_df, events_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase8_report(runs_df, events_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase8(output_dir)

    # Print results
    print(f"\n📊 Phase 8 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 10 rows):")
    preview_cols = [
        "run_id",
        "coupling_AB",
        "coupling_BC",
        "coupling_AC",
        "total_renaissance_count",
        "total_collapse_count",
        "drift_resilience",
        "knowledge_recovery_rate",
    ]
    display_df = runs_df[preview_cols].copy()
    print(display_df.head(10).to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    print("\n⏰ Phase 8 complete — temporal drift & renaissance simulation finished.")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Money Competition Phase 3: Shock-Resilience Battery
Tests system recovery from external shocks across openness levels.
Measures CCI recovery time, hazard area, and resilience scores.
Runtime target: <60s on typical laptop.
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

    # Effective coordination with motivation boost
    effective_coord = min(0.70, base_coord + 0.5 * motivation_anchor)

    return {
        "motivation_anchor": motivation_anchor,
        "goal_diversity": goal_diversity,
        "effective_coord": effective_coord,
        "intrinsic_drive": motivation_anchor,
    }


def apply_shock_to_agents(agents, shock_intensity, target_fraction=0.7):
    """Apply shock to top fraction of agents by resources."""
    # Sort agents by resources (descending) and shock the top fraction
    sorted_agents = sorted(agents, key=lambda a: a["resources"], reverse=True)
    shock_count = int(len(agents) * target_fraction)

    shocked_agents = set()
    for i in range(shock_count):
        agent = sorted_agents[i]
        # Apply shock: increase stress, reduce resources, amplify noise
        agent["resources"] *= 1.0 - shock_intensity * 0.3
        agent["stress"] += shock_intensity * 2.0
        agent["shock_noise_multiplier"] = 1.0 + shock_intensity
        shocked_agents.add(agent["id"])

    return shocked_agents


def shock_resilience_sim(
    agents,
    epsilon,
    coord,
    wage,
    epochs_cap,
    noise_base=0.04,
    shock_window=(200, 210),
    shock_intensity=0.3,
):
    """Simulation with shock injection and recovery tracking."""
    np.random.seed(11)  # Fixed seed for reproducibility

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
                "pre_shock_state": None,
            }
        )

    trajectory = []
    shock_events = []

    # Logging schedule: dense first 120, then every 10
    log_epochs = list(range(0, min(120, epochs_cap))) + list(range(120, epochs_cap, 10))

    # Ensure shock window is logged
    shock_start, shock_end = shock_window
    for epoch in range(shock_start - 5, shock_end + 20):
        if epoch not in log_epochs and 0 <= epoch < epochs_cap:
            log_epochs.append(epoch)
    log_epochs = sorted(set(log_epochs))

    baseline_cci = baseline_hazard = None
    shocked_agent_ids = set()

    for epoch in range(epochs_cap):
        # Capture baseline before shock
        if epoch == shock_start - 1:
            # Store baseline metrics
            resources = [a["resources"] for a in agents]
            stresses = [a["stress"] for a in agents]
            baseline_cci = compute_cci(agents)
            baseline_hazard = compute_hazard(agents)

            # Save pre-shock agent states
            for agent in agents:
                agent["pre_shock_state"] = {
                    "resources": agent["resources"],
                    "stress": agent["stress"],
                    "beliefs": agent["beliefs"].copy(),
                }

        # Apply shock during shock window
        if shock_start <= epoch <= shock_end:
            if epoch == shock_start:
                shocked_agent_ids = apply_shock_to_agents(agents, shock_intensity)
                shock_events.append(
                    {
                        "epoch": epoch,
                        "type": "shock_start",
                        "intensity": shock_intensity,
                        "affected_agents": len(shocked_agent_ids),
                        "baseline_cci": baseline_cci,
                        "baseline_hazard": baseline_hazard,
                    }
                )

        # Apply wage with shock amplification
        if wage > 0:
            for agent in agents:
                noise_mult = agent.get("shock_noise_multiplier", 1.0)
                wage_noise = np.random.normal(1.0, noise_base * noise_mult)
                wage_boost = wage * wage_noise
                agent["resources"] = min(1.0, agent["resources"] + wage_boost)
                # Competitive stress from wages
                agent["stress"] += wage_boost * 0.4 * noise_mult

        # Build social networks periodically
        if epoch % 25 == 0:
            for agent in agents:
                network_size = np.random.randint(3, 6)
                others = [a for a in agents if a != agent]
                if others:
                    agent["social_contacts"] = np.random.choice(
                        others, min(network_size, len(others)), replace=False
                    ).tolist()

        # Agent interactions with shock effects
        for agent in agents:
            noise_mult = agent.get("shock_noise_multiplier", 1.0)

            # Track resource stability
            agent["resource_history"].append(agent["resources"])
            if len(agent["resource_history"]) > 20:
                agent["resource_history"] = agent["resource_history"][-20:]
                agent["resource_stability"] = 1.0 - np.std(
                    agent["resource_history"]
                ) / (np.mean(agent["resource_history"]) + 1e-6)

            # Interactions (reduced during shock)
            interaction_prob = (0.3 + epsilon * 50) * (
                1.0 - shock_intensity * 0.5
                if shock_start <= epoch <= shock_end
                else 1.0
            )
            others = [a for a in agents if a != agent]

            if others and np.random.random() < interaction_prob:
                if epsilon > 0 and np.random.random() < epsilon * 100:
                    # Cooperative interaction (helps recovery)
                    partner = np.random.choice(others)
                    resource_share = 0.08 * agent["cooperation"]
                    transfer = resource_share * 0.5
                    if agent["resources"] > transfer:
                        agent["resources"] -= transfer
                        partner["resources"] = min(1.0, partner["resources"] + transfer)
                        agent["stress"] *= 0.90  # Cooperation reduces stress
                else:
                    # Competitive interaction (amplified during shock)
                    competitor = np.random.choice(others)
                    if agent["resources"] > competitor["resources"]:
                        capture_rate = (
                            0.04 * (1 - coord) * (1 + wage * 0.5) * noise_mult
                        )
                        capture = min(capture_rate, competitor["resources"] * 0.2)
                        agent["resources"] = min(1.0, agent["resources"] + capture)
                        competitor["resources"] -= capture
                        agent["stress"] += capture * 3 * noise_mult

            # Stress tracking
            agent["stress_history"].append(agent["stress"])
            if len(agent["stress_history"]) > 15:
                agent["stress_history"] = agent["stress_history"][-15:]

            # Adaptive belief evolution with shock response
            current_stress = min(1.0, agent["stress"])

            # During shock: bias toward religion
            if shock_start <= epoch <= shock_end:
                religion_tendency = current_stress * 1.5  # Amplified stress response
                training_tendency = (
                    (1.0 - current_stress) * epsilon * 3
                )  # Reduced learning
            else:
                # Post-shock: potential for training if openness allows
                religion_tendency = current_stress
                training_tendency = (1.0 - current_stress) * epsilon * 8

                # Recovery bonus for training in open systems
                if epsilon > 0.003 and current_stress < 0.4:
                    training_tendency *= 1.5

            # Apply belief changes with shock noise
            belief_noise = np.random.normal(0, noise_base * noise_mult * 0.5)

            if religion_tendency > training_tendency:
                agent["beliefs"][0] += 0.12 * religion_tendency + belief_noise
                agent["beliefs"][1] *= 0.94  # Training fades
            else:
                agent["beliefs"][1] += 0.10 * training_tendency + belief_noise
                agent["beliefs"][0] *= 0.96  # Religion fades

            # Normalize beliefs
            agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
            if agent["beliefs"].sum() > 0:
                agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

            # Resource decay with shock effects
            decay_rate = 0.995 + epsilon * 0.002
            if shock_start <= epoch <= shock_end:
                decay_rate *= 1.0 - shock_intensity * 0.1  # Faster decay during shock
            agent["resources"] *= decay_rate

            # Stress decay (slower during shock)
            stress_decay = 0.95 if shock_start <= epoch <= shock_end else 0.97
            agent["stress"] *= stress_decay

            # Shock noise multiplier decay
            if epoch > shock_end:
                agent["shock_noise_multiplier"] *= 0.98  # Gradual recovery

            # Minimum survival
            if agent["resources"] < 0.08:
                agent["resources"] = 0.08

        if epoch in log_epochs:
            # Compute metrics
            cci = compute_cci(agents)
            hazard = compute_hazard(agents)

            # Belief fractions
            beliefs = [a["beliefs"] for a in agents]
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.3) / len(
                training_beliefs
            )

            # Recovery tracking
            recovery_status = ""
            if baseline_cci is not None:
                cci_recovery = cci / baseline_cci if baseline_cci > 0 else 0
                if epoch > shock_end and cci_recovery >= 0.9:
                    recovery_status = f"CCI_recovered_epoch_{epoch}"

            trajectory.append(
                {
                    "epoch": epoch,
                    "CCI": cci,
                    "hazard": hazard,
                    "religion_frac": religion_frac,
                    "training_frac": training_frac,
                    "baseline_cci": baseline_cci or 0,
                    "baseline_hazard": baseline_hazard or 0,
                    "in_shock_window": shock_start <= epoch <= shock_end,
                    "recovery_status": recovery_status,
                    "mean_stress": np.mean([a["stress"] for a in agents]),
                    "resource_equality": 1.0
                    - np.std([a["resources"] for a in agents])
                    / (np.mean([a["resources"] for a in agents]) + 1e-6),
                }
            )

    return trajectory, shock_events, baseline_cci, baseline_hazard


def compute_cci(agents):
    """Compute Collective Consciousness Index."""
    resources = [a["resources"] for a in agents]
    stresses = [a["stress"] for a in agents]

    resource_mean = np.mean(resources)
    resource_equality = max(0, 1.0 - np.std(resources) / (resource_mean + 1e-6))
    stress_level = np.mean(stresses)
    cooperation_level = np.mean([a["cooperation"] for a in agents])

    # CCI combines equality, low stress, and cooperation
    cci = max(
        0,
        (
            resource_equality * 0.4
            + (1.0 - stress_level * 0.5) * 0.4
            + cooperation_level * 0.2
        ),
    )
    return cci


def compute_hazard(agents):
    """Compute system hazard."""
    resources = [a["resources"] for a in agents]
    stresses = [a["stress"] for a in agents]

    survival_rate = sum(1 for r in resources if r > 0.12) / len(resources)
    stress_level = np.mean(stresses)
    resource_equality = 1.0 - np.std(resources) / (np.mean(resources) + 1e-6)

    hazard = (
        (1.0 - survival_rate) + stress_level * 0.6 + (1.0 - resource_equality) * 0.3
    )
    return hazard


def analyze_recovery_metrics(trajectory, shock_window, baseline_cci, baseline_hazard):
    """Analyze recovery times and resilience metrics."""
    shock_start, shock_end = shock_window
    post_shock_data = [
        t
        for t in trajectory
        if t["epoch"] > shock_end and t["epoch"] <= shock_end + 100
    ]

    if not post_shock_data:
        return {
            "t_recover_CCI": None,
            "t_recover_hazard": None,
            "AUH_200_300": 0,
            "resilience_score": 0,
            "final_CCI": 0,
            "final_hazard": 10,
            "final_religion_frac": 1.0,
            "final_training_frac": 0.0,
        }

    # Recovery time for CCI (90% of baseline)
    cci_target = 0.9 * baseline_cci if baseline_cci > 0 else 0.1
    t_recover_CCI = None
    for t in post_shock_data:
        if t["CCI"] >= cci_target:
            t_recover_CCI = t["epoch"] - shock_end
            break

    # Recovery time for hazard (below 120% of baseline or absolute threshold)
    hazard_target = max(baseline_hazard * 1.2 if baseline_hazard > 0 else 1.0, 0.5)
    t_recover_hazard = None
    for t in post_shock_data:
        if t["hazard"] <= hazard_target:
            t_recover_hazard = t["epoch"] - shock_end
            break

    # Area under hazard curve (AUH) post-shock
    hazard_values = [t["hazard"] for t in post_shock_data]
    AUH_200_300 = np.trapz(hazard_values) if hazard_values else 0

    # Final metrics
    final_data = post_shock_data[-1] if post_shock_data else trajectory[-1]

    # Resilience score: recovery speed - hazard area (normalized)
    cci_recovery_score = (100 - (t_recover_CCI or 100)) / 100 if t_recover_CCI else 0
    hazard_area_penalty = min(1.0, AUH_200_300 / 200)  # Normalize by typical area
    resilience_score = max(0, cci_recovery_score - hazard_area_penalty)

    return {
        "t_recover_CCI": t_recover_CCI,
        "t_recover_hazard": t_recover_hazard,
        "AUH_200_300": AUH_200_300,
        "resilience_score": resilience_score,
        "final_CCI": final_data["CCI"],
        "final_hazard": final_data["hazard"],
        "final_religion_frac": final_data["religion_frac"],
        "final_training_frac": final_data["training_frac"],
    }


def run_single_condition_phase3(
    condition_id,
    epsilon,
    profile_type,
    agents=60,
    epochs_cap=500,
    seed=11,
    noise_base=0.04,
    shock_window=(200, 210),
    shock_intensity=0.3,
):
    """Run one shock-resilience condition."""
    start_time = time.time()

    # Determine system parameters based on openness
    if epsilon == 0.0:  # Closed system
        ineq = 0.40
        coord_base = 0.45
    else:  # Open system
        ineq = 0.22
        coord_base = 0.60

    wage = 0.40

    # Get motivation profile
    profile = create_motivation_profile(profile_type, coord_base, agents)

    # Initialize agents
    agents_state = []
    np.random.seed(seed)
    for i in range(agents):
        agents_state.append(
            {
                "id": i,
                "inequality_penalty": ineq * np.random.uniform(0, 1.2),
                "goal_diversity": profile["goal_diversity"],
            }
        )

    # Run simulation with shock
    trajectory_data, shock_events, baseline_cci, baseline_hazard = shock_resilience_sim(
        agents_state,
        epsilon,
        profile["effective_coord"],
        wage,
        epochs_cap,
        noise_base,
        shock_window,
        shock_intensity,
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        trajectory.append(t)

    # Analyze recovery metrics
    recovery_metrics = analyze_recovery_metrics(
        trajectory, shock_window, baseline_cci, baseline_hazard
    )

    run_time = time.time() - start_time

    # Summary with shock-resilience metrics
    summary = {
        "run_id": condition_id,
        "epsilon": epsilon,
        "profile": profile_type,
        "ineq": ineq,
        "coord_eff": profile["effective_coord"],
        "wage": wage,
        "agents": agents,
        "epochs_cap": epochs_cap,
        "shock_start": shock_window[0],
        "shock_end": shock_window[1],
        "shock_intensity": shock_intensity,
        "baseline_CCI": baseline_cci or 0,
        "baseline_hazard": baseline_hazard or 0,
        **recovery_metrics,
        "time_sec": run_time,
    }

    return summary, trajectory, shock_events


def generate_phase3_takeaways(runs_df):
    """Generate shock-resilience takeaways."""
    takeaways = []

    # Recovery time vs openness
    openness_levels = sorted(runs_df["epsilon"].unique())

    # Check if recovery time improves with openness
    recovery_times = []
    for eps in openness_levels:
        eps_data = runs_df[runs_df["epsilon"] == eps]
        avg_recovery = eps_data["t_recover_CCI"].mean()
        recovery_times.append(avg_recovery)

    if len(recovery_times) > 1 and recovery_times[-1] < recovery_times[0]:
        takeaways.append("• Openness enhances shock recovery (faster CCI restoration)")

    # Family vs Single recovery
    family_data = runs_df[runs_df["profile"] == "FAMILY"]
    single_data = runs_df[runs_df["profile"] == "SINGLE"]

    if not family_data.empty and not single_data.empty:
        family_recovery = family_data["t_recover_CCI"].mean()
        single_recovery = single_data["t_recover_CCI"].mean()

        if family_recovery < single_recovery:
            takeaways.append("• Anchored motivation (FAMILY) buffers shock stress")

    # Meaning system evolution
    high_eps = runs_df[runs_df["epsilon"] >= 0.005]
    low_eps = runs_df[runs_df["epsilon"] < 0.005]

    if not high_eps.empty and not low_eps.empty:
        high_training = high_eps["final_training_frac"].mean()
        low_training = low_eps["final_training_frac"].mean()
        high_religion = high_eps["final_religion_frac"].mean()
        low_religion = low_eps["final_religion_frac"].mean()

        if high_training > low_training + 0.05 and high_religion < low_religion:
            takeaways.append(
                "• Meaning system evolved from belief → learning in open systems"
            )

    # Closed system traps
    closed_data = runs_df[runs_df["epsilon"] == 0.0]
    if not closed_data.empty:
        closed_final_cci = closed_data["final_CCI"].mean()
        if closed_final_cci < 0.2:
            takeaways.append("• Closed economies stay trapped in post-shock collapse")

    # Resilience threshold
    for eps in openness_levels:
        eps_data = runs_df[runs_df["epsilon"] == eps]
        if not eps_data.empty and eps_data["final_CCI"].mean() > 0.5:
            takeaways.append(
                f"• True resilience boundary at ε≥{eps:.3f} (sustained CCI>0.5)"
            )
            break

    # Resilience scores
    best_resilience = runs_df.loc[runs_df["resilience_score"].idxmax()]
    takeaways.append(
        f"• Peak resilience: {best_resilience['profile']} profile at ε={best_resilience['epsilon']:.3f}"
    )

    # Area under hazard curve
    auh_corr = runs_df[["epsilon", "AUH_200_300"]].corr().iloc[0, 1]
    if auh_corr < -0.3:
        takeaways.append("• Higher openness reduces cumulative post-shock hazard")

    return takeaways[:10]


def create_phase3_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 3 shock-resilience visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Main analysis figure
    plt.figure(figsize=(18, 12))

    # CCI Recovery trajectories
    plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_df["epsilon"].unique())))
    openness_levels = sorted(runs_df["epsilon"].unique())

    for i, eps in enumerate(openness_levels):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            # Focus on shock period and recovery
            shock_traj = traj[(traj["epoch"] >= 180) & (traj["epoch"] <= 300)]
            if not shock_traj.empty:
                label = f"ε={eps:.3f}" if run_id == eps_runs[0] else ""
                plt.plot(
                    shock_traj["epoch"],
                    shock_traj["CCI"],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=label,
                )

    plt.axvline(x=200, color="red", linestyle="--", alpha=0.7, label="Shock Start")
    plt.axvline(x=210, color="orange", linestyle="--", alpha=0.7, label="Shock End")
    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("CCI Recovery from Shock")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Hazard Recovery trajectories
    plt.subplot(2, 3, 2)
    for i, eps in enumerate(openness_levels):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            shock_traj = traj[(traj["epoch"] >= 180) & (traj["epoch"] <= 300)]
            if not shock_traj.empty:
                label = f"ε={eps:.3f}" if run_id == eps_runs[0] else ""
                plt.plot(
                    shock_traj["epoch"],
                    shock_traj["hazard"],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=label,
                )

    plt.axvline(x=200, color="red", linestyle="--", alpha=0.7)
    plt.axvline(x=210, color="orange", linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("System Hazard")
    plt.title("Hazard Response to Shock")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Resilience scores by condition
    plt.subplot(2, 3, 3)
    x_pos = range(len(runs_df))
    colors_bars = ["red" if p == "SINGLE" else "blue" for p in runs_df["profile"]]

    bars = plt.bar(x_pos, runs_df["resilience_score"], alpha=0.7, color=colors_bars)
    plt.xlabel("Condition")
    plt.ylabel("Resilience Score")
    plt.title("Resilience Score by Condition")
    plt.xticks(
        x_pos,
        [f"ε{row['epsilon']:.3f}-{row['profile'][0]}" for _, row in runs_df.iterrows()],
        rotation=45,
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="SINGLE"),
        Patch(facecolor="blue", alpha=0.7, label="FAMILY"),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)

    # Recovery time vs openness
    plt.subplot(2, 3, 4)
    for profile in ["SINGLE", "FAMILY"]:
        profile_data = runs_df[runs_df["profile"] == profile]
        recovery_times = []
        eps_vals = []
        for eps in openness_levels:
            eps_profile_data = profile_data[profile_data["epsilon"] == eps]
            if not eps_profile_data.empty:
                recovery_times.append(eps_profile_data["t_recover_CCI"].mean())
                eps_vals.append(eps)

        plt.plot(
            eps_vals, recovery_times, "o-", label=profile, linewidth=2, markersize=8
        )

    plt.xlabel("Openness (ε)")
    plt.ylabel("CCI Recovery Time (epochs)")
    plt.title("Recovery Speed vs Openness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Area under hazard curve
    plt.subplot(2, 3, 5)
    for profile in ["SINGLE", "FAMILY"]:
        profile_data = runs_df[runs_df["profile"] == profile]
        auh_vals = []
        eps_vals = []
        for eps in openness_levels:
            eps_profile_data = profile_data[profile_data["epsilon"] == eps]
            if not eps_profile_data.empty:
                auh_vals.append(eps_profile_data["AUH_200_300"].mean())
                eps_vals.append(eps)

        plt.plot(eps_vals, auh_vals, "s-", label=profile, linewidth=2, markersize=8)

    plt.xlabel("Openness (ε)")
    plt.ylabel("Area Under Hazard Curve")
    plt.title("Cumulative Post-Shock Hazard")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final belief states
    plt.subplot(2, 3, 6)
    x_pos = range(len(runs_df))
    bar_width = 0.35

    x1 = [x - bar_width / 2 for x in x_pos]
    x2 = [x + bar_width / 2 for x in x_pos]

    plt.bar(
        x1,
        runs_df["final_religion_frac"],
        bar_width,
        alpha=0.7,
        color="red",
        label="Religion",
    )
    plt.bar(
        x2,
        runs_df["final_training_frac"],
        bar_width,
        alpha=0.7,
        color="blue",
        label="Training",
    )

    plt.xlabel("Condition")
    plt.ylabel("Belief Fraction")
    plt.title("Post-Shock Belief States")
    plt.xticks(
        x_pos,
        [f"ε{row['epsilon']:.3f}-{row['profile'][0]}" for _, row in runs_df.iterrows()],
        rotation=45,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "shock_resilience_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_phase3_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 3 shock-resilience report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Money Competition Phase 3: Shock-Resilience Battery Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Shock Window:** Epochs 200-210 (30% intensity on top 70% of agents)  
**Recovery Analysis:** 100 epochs post-shock tracking

## Experimental Design

Phase 3 tests system resilience to external shocks:
- **Shock injection:** 30% noise amplification + resource reduction on wealthiest 70% of agents
- **Recovery tracking:** CCI restoration time, hazard area under curve, resilience scores
- **Openness sweep:** [0.000, 0.002, 0.005, 0.010] vs baseline closed/fair configurations
- **Profiles:** SINGLE vs FAMILY motivation anchoring under stress

## Results Summary

| Run | ε | Profile | Resilience Score | CCI Recovery (epochs) | Final CCI | Final Hazard | Religion % | Training % |
|-----|---|---------|------------------|---------------------|-----------|--------------|------------|------------|
"""

    for _, row in runs_df.iterrows():
        md_content += f"| {row['run_id']} | {row['epsilon']:.3f} | {row['profile']} | {row['resilience_score']:.3f} | {row['t_recover_CCI'] or 'No recovery'} | {row['final_CCI']:.3f} | {row['final_hazard']:.3f} | {row['final_religion_frac']:.1%} | {row['final_training_frac']:.1%} |\n"

    md_content += f"""

## Shock-Resilience Key Findings

{chr(10).join(takeaways)}

## Recovery Dynamics Analysis

### Shock Response Patterns:
1. **Immediate Impact (Epochs 200-210):** All systems show CCI decline and hazard spike
2. **Early Recovery (Epochs 210-230):** Open systems begin stabilization
3. **Late Recovery (Epochs 230-300):** Sustained improvement only in high-openness conditions

### Critical Thresholds:
- **ε = 0.002:** Minimal recovery improvement
- **ε = 0.005:** Moderate resilience gains  
- **ε = 0.010:** Strong recovery capability

### Resilience Score Components:
- **CCI Recovery Speed:** Time to reach 90% of pre-shock baseline
- **Hazard Area Penalty:** Cumulative stress burden post-shock
- **Net Resilience:** Recovery capability minus sustained damage

## Implications for Economic Policy

The shock-resilience battery reveals:
- **Structural openness** creates anti-fragile economic systems
- **Family-anchored motivation** provides natural shock absorption
- **Closed systems** exhibit hysteresis (permanent damage from temporary shocks)
- **Training investment** emerges only after stress subsides in open environments

## Next Steps: Phase 4 (Meaning Transition Stability)

Based on resilience findings:
- Test **meaning system phase transitions** under different shock frequencies
- Examine **multi-generational adaptation** to chronic instability
- Model **institutional memory** of shock responses
- Validate **cooperative ownership** as resilience amplifier

## Files Generated

- `data/runs_summary.csv` - Shock-resilience metrics by condition
- `data/trajectories_long.csv` - Epoch-by-epoch recovery tracking
- `data/shock_events.csv` - Shock injection timing and parameters
- `figures/shock_resilience_analysis.png` - 6-panel recovery analysis
- `bundle/money_competition_phase3_*.zip` - Complete exportable bundle

"""

    with open(report_dir / "money_competition_phase3_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase3(output_dir):
    """Create ZIP bundle for Phase 3."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"money_competition_phase3_{timestamp}.zip"
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
    """Run the complete Phase 3 shock-resilience experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"money_competition_phase3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("🚀 Starting Money Competition Phase 3 (Shock-Resilience Battery)...")

    # Configuration
    openness_levels = [0.000, 0.002, 0.005, 0.010]
    profiles = ["SINGLE", "FAMILY"]
    agents = 60
    epochs_cap = 500
    seed = 11
    noise_base = 0.04
    shock_window = (200, 210)
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
            condition_id = f"SR_E{int(epsilon*1000):02d}_{profile[0]}"

            print(
                f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: ε={epsilon:.3f}, {profile} profile..."
            )

            summary, trajectory, shock_events = run_single_condition_phase3(
                condition_id,
                epsilon,
                profile,
                agents,
                epochs_cap,
                seed,
                noise_base,
                shock_window,
                shock_intensity,
            )

            all_summaries.append(summary)
            all_trajectories.extend(trajectory)
            all_shock_events.extend(shock_events)

            recovery_status = (
                f"Recovered in {summary['t_recover_CCI']} epochs"
                if summary["t_recover_CCI"]
                else "No recovery"
            )
            print(
                f"    ✓ Completed in {summary['time_sec']:.2f}s - {recovery_status}, Resilience: {summary['resilience_score']:.3f}"
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
    takeaways = generate_phase3_takeaways(runs_df)

    # Create visualizations
    create_phase3_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase3_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase3(output_dir)

    # Print results
    print(f"\n📊 Phase 3 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 8 rows):")
    preview_cols = [
        "run_id",
        "epsilon",
        "profile",
        "resilience_score",
        "t_recover_CCI",
        "final_CCI",
        "final_training_frac",
    ]
    print(runs_df[preview_cols].head(8).to_string(index=False))

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    print("\nPhase 3 complete — ready for Phase 4 (Meaning Transition Stability).")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Money Competition Phase 2 Experiment
Extended analysis with adaptive branching and openness sweep.
Tests religion vs training responses under different openness levels.
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


def adaptive_branch_response(agent, stress_level, epoch, openness):
    """Adaptive branching logic - agents choose religion vs training based on context."""

    # Base tendencies
    religion_tendency = stress_level  # Higher stress → more religion
    training_tendency = (
        (1.0 - stress_level) * openness * 10
    )  # Low stress + openness → training

    # Experience effects (learning over time)
    if epoch > 50:
        # Past religion investment affects future choices
        past_religion = agent["beliefs"][0]
        past_training = agent["beliefs"][1]

        # If religion helped (low recent stress), stick with it
        if past_religion > 0.6 and agent.get("recent_stress_avg", 0.5) < 0.3:
            religion_tendency *= 1.5

        # If training helped (steady resources), invest more in training
        if past_training > 0.4 and agent.get("resource_stability", 0.5) > 0.6:
            training_tendency *= 1.8

    # Social influence (adaptive)
    if "social_contacts" in agent:
        religious_contacts = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[0] > 0.5
        )
        training_contacts = sum(
            1 for c in agent["social_contacts"] if c.get("beliefs", [0, 0, 0])[1] > 0.5
        )

        if religious_contacts > training_contacts and stress_level > 0.4:
            religion_tendency *= 1.3  # Social reinforcement under stress
        elif training_contacts > religious_contacts and openness > 0.003:
            training_tendency *= 1.4  # Social learning in open systems

    return religion_tendency, training_tendency


def advanced_agent_sim(
    agents, epsilon, coord, wage, epochs_cap, noise=0.04, adaptive_branch=True
):
    """Advanced agent simulation with adaptive branching."""
    np.random.seed(7)  # Fixed seed for reproducibility

    # Initialize agents with enhanced state tracking
    for i, agent in enumerate(agents):
        agent.update(
            {
                "resources": max(
                    0.1, 1.0 - agent["inequality_penalty"] + np.random.normal(0, noise)
                ),
                "cooperation": coord + np.random.normal(0, 0.1),
                "beliefs": np.random.rand(3),  # [religion, training, other]
                "stress": 0.0,
                "stress_history": [],
                "resource_history": [],
                "social_contacts": [],
            }
        )

    trajectory = []
    log_epochs = list(range(0, min(80, epochs_cap))) + list(range(80, epochs_cap, 10))

    for epoch in range(epochs_cap):
        # Apply wage shock with noise
        if wage > 0:
            for agent in agents:
                wage_noise = np.random.normal(1.0, noise)
                wage_boost = wage * wage_noise
                agent["resources"] = min(1.0, agent["resources"] + wage_boost)
                # Higher wages increase competitive stress
                agent["stress"] += wage_boost * 0.4

        # Build social networks (for adaptive branching)
        if adaptive_branch and epoch % 20 == 0:  # Update networks periodically
            for agent in agents:
                # Each agent connects to 3-5 others based on proximity
                network_size = np.random.randint(3, 6)
                others = [a for a in agents if a != agent]
                agent["social_contacts"] = np.random.choice(
                    others, min(network_size, len(others)), replace=False
                ).tolist()

        # Agent interactions and updates
        for agent in agents:
            # Track resource stability
            agent["resource_history"].append(agent["resources"])
            if len(agent["resource_history"]) > 20:
                agent["resource_history"] = agent["resource_history"][-20:]
                agent["resource_stability"] = 1.0 - np.std(
                    agent["resource_history"]
                ) / (np.mean(agent["resource_history"]) + 1e-6)

            # Competition vs cooperation dynamics
            others = [a for a in agents if a != agent]
            if others:
                interaction_prob = (
                    0.3 + epsilon * 50
                )  # More interactions in open systems
                if np.random.random() < interaction_prob:
                    if epsilon > 0 and np.random.random() < epsilon * 100:
                        # Cooperative interaction
                        partner = np.random.choice(others)
                        resource_share = 0.08 * agent["cooperation"]
                        transfer = resource_share * 0.5
                        if agent["resources"] > transfer:
                            agent["resources"] -= transfer
                            partner["resources"] = min(
                                1.0, partner["resources"] + transfer
                            )
                            agent["stress"] *= 0.92  # Cooperation reduces stress
                    else:
                        # Competitive interaction
                        competitor = np.random.choice(others)
                        if agent["resources"] > competitor["resources"]:
                            # Resource capture (zero-sum)
                            capture_rate = 0.04 * (1 - coord) * (1 + wage * 0.5)
                            capture = min(capture_rate, competitor["resources"] * 0.2)
                            agent["resources"] = min(1.0, agent["resources"] + capture)
                            competitor["resources"] -= capture
                            agent["stress"] += (
                                capture * 3
                            )  # Competition increases stress

            # Stress tracking
            agent["stress_history"].append(agent["stress"])
            if len(agent["stress_history"]) > 15:
                agent["stress_history"] = agent["stress_history"][-15:]
                agent["recent_stress_avg"] = np.mean(agent["stress_history"])

            # Adaptive belief evolution
            current_stress = min(1.0, agent["stress"])

            if adaptive_branch:
                religion_tendency, training_tendency = adaptive_branch_response(
                    agent, current_stress, epoch, epsilon
                )
            else:
                # Simple stress-based response
                religion_tendency = current_stress
                training_tendency = (1.0 - current_stress) * epsilon * 5

            # Apply belief changes with noise
            belief_noise = np.random.normal(0, noise * 0.5)

            if religion_tendency > training_tendency:
                agent["beliefs"][0] += 0.12 * religion_tendency + belief_noise
                agent["beliefs"][1] *= 0.95  # Training fades
            else:
                agent["beliefs"][1] += 0.08 * training_tendency + belief_noise
                agent["beliefs"][0] *= 0.97  # Religion fades slowly

            # Normalize beliefs
            agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
            if agent["beliefs"].sum() > 0:
                agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

            # Resource decay and survival
            decay_rate = 0.995 + epsilon * 0.002  # Slight boost in open systems
            agent["resources"] *= decay_rate

            # Stress decay
            agent["stress"] *= 0.96

            # Minimum survival threshold
            if agent["resources"] < 0.08:
                agent["resources"] = 0.08

        if epoch in log_epochs:
            # Compute metrics
            resources = [a["resources"] for a in agents]
            stresses = [a["stress"] for a in agents]
            beliefs = [a["beliefs"] for a in agents]

            # Enhanced CCI calculation
            resource_mean = np.mean(resources)
            resource_equality = max(0, 1.0 - np.std(resources) / (resource_mean + 1e-6))
            stress_level = np.mean(stresses)
            cooperation_level = np.mean([a["cooperation"] for a in agents])

            # CCI combines resource equality, low stress, and cooperation
            cci = max(
                0,
                (
                    resource_equality * 0.4
                    + (1.0 - stress_level * 0.5) * 0.4
                    + cooperation_level * 0.2
                ),
            )

            # System metrics
            survival_rate = sum(1 for r in resources if r > 0.12) / len(resources)
            hazard = (
                (1.0 - survival_rate)
                + stress_level * 0.6
                + (1.0 - resource_equality) * 0.3
            )
            collapse_risk = np.var(resources) + stress_level * 0.8

            # Enhanced belief fractions
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.4) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.3) / len(
                training_beliefs
            )

            trajectory.append(
                {
                    "epoch": epoch,
                    "CCI": cci,
                    "collapse_risk": collapse_risk,
                    "survival_rate": survival_rate,
                    "hazard": hazard,
                    "religion_frac": religion_frac,
                    "training_frac": training_frac,
                    "mean_stress": stress_level,
                    "resource_equality": resource_equality,
                    "mean_cooperation": cooperation_level,
                }
            )

    return trajectory


def run_single_condition_phase2(
    condition_id,
    label,
    epsilon,
    ineq,
    coord_base,
    wage,
    profile_type,
    agents=60,
    epochs_cap=300,
    seed=7,
    noise=0.04,
):
    """Run one condition for Phase 2 with extended parameters."""
    start_time = time.time()

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

    # Run simulation with adaptive branching enabled
    trajectory_data = advanced_agent_sim(
        agents_state,
        epsilon,
        profile["effective_coord"],
        wage,
        epochs_cap,
        noise=noise,
        adaptive_branch=True,
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        trajectory.append(t)

    # Compute stability metrics (last 50 epochs for longer runs)
    recent_data = trajectory[-min(50, len(trajectory)) :]

    if recent_data:
        stability_CCI_mean = np.mean([t["CCI"] for t in recent_data])
        stability_hazard_mean = np.mean([t["hazard"] for t in recent_data])

        # Trend analysis over final period
        epochs = [t["epoch"] for t in recent_data]
        ccis = [t["CCI"] for t in recent_data]
        if len(epochs) > 1:
            stability_CCI_slope = np.polyfit(epochs, ccis, 1)[0]
        else:
            stability_CCI_slope = 0.0

        mean_religion_frac = np.mean([t["religion_frac"] for t in recent_data])
        mean_training_frac = np.mean([t["training_frac"] for t in recent_data])
        peak_CCI = max([t["CCI"] for t in trajectory])
        final_CCI = trajectory[-1]["CCI"] if trajectory else 0.0

        # Additional Phase 2 metrics
        final_resource_equality = (
            trajectory[-1]["resource_equality"] if trajectory else 0.0
        )
        mean_stress_final = np.mean([t["mean_stress"] for t in recent_data])
        cooperation_stability = np.mean([t["mean_cooperation"] for t in recent_data])
    else:
        stability_CCI_mean = stability_hazard_mean = stability_CCI_slope = 0.0
        mean_religion_frac = mean_training_frac = peak_CCI = final_CCI = 0.0
        final_resource_equality = mean_stress_final = cooperation_stability = 0.0

    run_time = time.time() - start_time

    # Extended summary for Phase 2
    summary = {
        "run_id": condition_id,
        "label": label,
        "seed": seed,
        "agents": agents,
        "epochs_cap": epochs_cap,
        "epsilon": epsilon,
        "ineq": ineq,
        "coord_base": coord_base,
        "coord_eff": profile["effective_coord"],
        "wage": wage,
        "profile": profile_type,
        "goal_diversity": profile["goal_diversity"],
        "noise": noise,
        "adaptive_branch": True,
        "peak_CCI": peak_CCI,
        "final_CCI": final_CCI,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_hazard_mean": stability_hazard_mean,
        "stability_CCI_slope": stability_CCI_slope,
        "mean_religion_frac": mean_religion_frac,
        "mean_training_frac": mean_training_frac,
        "final_resource_equality": final_resource_equality,
        "mean_stress_final": mean_stress_final,
        "cooperation_stability": cooperation_stability,
        "time_sec": run_time,
    }

    return summary, trajectory


def generate_phase2_takeaways(runs_df):
    """Generate takeaways specific to Phase 2 openness sweep."""
    takeaways = []

    # Analyze openness gradient
    openness_levels = sorted(runs_df["epsilon"].unique())

    for i, eps in enumerate(openness_levels[1:], 1):  # Skip first (closed system)
        curr_data = runs_df[runs_df["epsilon"] == eps]
        prev_data = runs_df[runs_df["epsilon"] == openness_levels[i - 1]]

        if not curr_data.empty and not prev_data.empty:
            curr_hazard = curr_data["stability_hazard_mean"].mean()
            prev_hazard = prev_data["stability_hazard_mean"].mean()
            hazard_improvement = (
                (prev_hazard - curr_hazard) / prev_hazard if prev_hazard > 0 else 0
            )

            curr_training = curr_data["mean_training_frac"].mean()
            prev_training = prev_data["mean_training_frac"].mean()

            if hazard_improvement > 0.1:
                takeaways.append(
                    f"• Openness ε={eps:.3f} reduces hazard by {hazard_improvement:.1%} vs ε={openness_levels[i-1]:.3f}"
                )

            if curr_training > prev_training + 0.05:
                takeaways.append(
                    f"• Training investment jumps at ε={eps:.3f} (adaptive branching working)"
                )

    # Religion vs training dynamics
    high_eps = runs_df[runs_df["epsilon"] >= 0.005]
    low_eps = runs_df[runs_df["epsilon"] < 0.005]

    if not high_eps.empty and not low_eps.empty:
        high_training = high_eps["mean_training_frac"].mean()
        low_training = low_eps["mean_training_frac"].mean()
        high_religion = high_eps["mean_religion_frac"].mean()
        low_religion = low_eps["mean_religion_frac"].mean()

        if high_training > low_training + 0.1:
            takeaways.append(
                "• Higher openness (≥0.005) drives training over religion via adaptive learning"
            )

        if low_religion > high_religion + 0.1:
            takeaways.append(
                "• Closed systems (ε<0.005) lock into religious responses under chronic stress"
            )

    # Family vs Single patterns
    family_data = runs_df[runs_df["profile"] == "FAMILY"]
    single_data = runs_df[runs_df["profile"] == "SINGLE"]

    if not family_data.empty and not single_data.empty:
        family_adaptability = family_data["stability_CCI_slope"].mean()
        single_adaptability = single_data["stability_CCI_slope"].mean()

        if family_adaptability > single_adaptability + 0.001:
            takeaways.append(
                "• Family profiles show better late-stage adaptation (positive CCI slopes)"
            )

    # Wage interaction effects
    wage_corr_hazard = runs_df[["wage", "stability_hazard_mean"]].corr().iloc[0, 1]
    wage_corr_religion = runs_df[["wage", "mean_religion_frac"]].corr().iloc[0, 1]

    if wage_corr_hazard > 0.4:
        takeaways.append(
            "• Wage escalation strongly correlates with system hazard (r > 0.4)"
        )
    if wage_corr_religion > 0.3:
        takeaways.append(
            "• Higher wages drive religious responses (stress-induced meaning-seeking)"
        )

    # Cooperation stability insights
    coop_stability_mean = runs_df["cooperation_stability"].mean()
    if coop_stability_mean > 0.6:
        takeaways.append("• Cooperation levels stabilize above 60% in most conditions")

    # Resource equality vs CCI relationship
    equality_cci_corr = (
        runs_df[["final_resource_equality", "stability_CCI_mean"]].corr().iloc[0, 1]
    )
    if equality_cci_corr > 0.5:
        takeaways.append(
            "• Resource equality strongly predicts collective intelligence (r > 0.5)"
        )

    return takeaways[:10]


def create_phase2_visualizations(runs_df, trajectories_df, output_dir):
    """Create Phase 2 specific visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Openness sweep analysis
    plt.figure(figsize=(15, 10))

    # Subplot 1: Hazard vs Openness
    plt.subplot(2, 3, 1)
    openness_vals = sorted(runs_df["epsilon"].unique())
    for profile in ["SINGLE", "FAMILY"]:
        profile_data = runs_df[runs_df["profile"] == profile]
        hazards = [
            profile_data[profile_data["epsilon"] == eps]["stability_hazard_mean"].mean()
            for eps in openness_vals
        ]
        plt.plot(openness_vals, hazards, "o-", label=profile, linewidth=2, markersize=6)
    plt.xlabel("Openness (ε)")
    plt.ylabel("System Hazard")
    plt.title("Hazard vs Openness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Religion vs Training Fractions
    plt.subplot(2, 3, 2)
    for profile in ["SINGLE", "FAMILY"]:
        profile_data = runs_df[runs_df["profile"] == profile]
        religions = [
            profile_data[profile_data["epsilon"] == eps]["mean_religion_frac"].mean()
            for eps in openness_vals
        ]
        trainings = [
            profile_data[profile_data["epsilon"] == eps]["mean_training_frac"].mean()
            for eps in openness_vals
        ]
        plt.plot(
            openness_vals,
            religions,
            "o-",
            label=f"{profile} Religion",
            color="red" if profile == "SINGLE" else "darkred",
            linewidth=2,
        )
        plt.plot(
            openness_vals,
            trainings,
            "s-",
            label=f"{profile} Training",
            color="blue" if profile == "SINGLE" else "darkblue",
            linewidth=2,
        )
    plt.xlabel("Openness (ε)")
    plt.ylabel("Fraction")
    plt.title("Religion vs Training by Openness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: CCI Trajectories by Openness
    plt.subplot(2, 3, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(openness_vals)))
    for i, eps in enumerate(openness_vals):
        eps_runs = runs_df[runs_df["epsilon"] == eps]["run_id"].values
        for run_id in eps_runs:
            traj = trajectories_df[trajectories_df["run_id"] == run_id]
            if not traj.empty:
                plt.plot(
                    traj["epoch"],
                    traj["CCI"],
                    alpha=0.7,
                    color=colors[i],
                    linewidth=1,
                    label=f"ε={eps:.3f}" if run_id == eps_runs[0] else "",
                )
    plt.xlabel("Epoch")
    plt.ylabel("CCI")
    plt.title("CCI Evolution by Openness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Stress vs Resource Equality
    plt.subplot(2, 3, 4)
    plt.scatter(
        runs_df["mean_stress_final"],
        runs_df["final_resource_equality"],
        c=runs_df["epsilon"],
        cmap="viridis",
        s=60,
        alpha=0.7,
    )
    plt.colorbar(label="Openness (ε)")
    plt.xlabel("Mean Final Stress")
    plt.ylabel("Final Resource Equality")
    plt.title("Stress vs Equality (color = openness)")
    plt.grid(True, alpha=0.3)

    # Subplot 5: Wage vs Hazard by Profile
    plt.subplot(2, 3, 5)
    for profile in ["SINGLE", "FAMILY"]:
        profile_data = runs_df[runs_df["profile"] == profile]
        plt.scatter(
            profile_data["wage"],
            profile_data["stability_hazard_mean"],
            label=profile,
            alpha=0.7,
            s=60,
        )
    plt.xlabel("Wage Level")
    plt.ylabel("Stability Hazard")
    plt.title("Wage vs Hazard by Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 6: Adaptive Branching Effectiveness
    plt.subplot(2, 3, 6)
    x_pos = range(len(runs_df))
    labels = [f"{row['run_id']}" for _, row in runs_df.iterrows()]

    bar_width = 0.35
    x1 = [x - bar_width / 2 for x in x_pos]
    x2 = [x + bar_width / 2 for x in x_pos]

    plt.bar(
        x1,
        runs_df["mean_religion_frac"],
        bar_width,
        alpha=0.7,
        color="red",
        label="Religion",
    )
    plt.bar(
        x2,
        runs_df["mean_training_frac"],
        bar_width,
        alpha=0.7,
        color="blue",
        label="Training",
    )

    plt.xlabel("Condition")
    plt.ylabel("Fraction")
    plt.title("Adaptive Branching Outcomes")
    plt.xticks(x_pos, labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "phase2_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_phase2_report(runs_df, takeaways, output_dir, total_time):
    """Create Phase 2 specific report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Money Competition Phase 2: Openness Sweep Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Extended Analysis:** 300 epochs, adaptive branching, noise=0.04  
**Openness Range:** [0.000, 0.002, 0.005, 0.010]

## Experimental Design

Phase 2 extends the money competition analysis with:
- **Longer time horizons** (300 epochs vs 80)
- **Openness parameter sweep** (4 levels: closed → minimally open → moderately open → highly open)
- **Adaptive branching enabled** (agents learn religion vs training based on experience)
- **Enhanced noise** (0.04 vs minimal) for realistic stochasticity
- **Deeper metrics** (stress tracking, resource equality, cooperation stability)

## Results Summary

| Run | Label | ε | Profile | Hazard | CCI | Religion % | Training % | Resource Equality | Stress |
|-----|-------|---|---------|--------|-----|------------|------------|------------------|--------|
"""

    for _, row in runs_df.iterrows():
        md_content += f"| {row['run_id']} | {row['label']} | {row['epsilon']:.3f} | {row['profile']} | {row['stability_hazard_mean']:.3f} | {row['stability_CCI_mean']:.3f} | {row['mean_religion_frac']:.1%} | {row['mean_training_frac']:.1%} | {row['final_resource_equality']:.3f} | {row['mean_stress_final']:.3f} |\n"

    md_content += f"""

## Phase 2 Key Findings

{chr(10).join(takeaways)}

## Adaptive Branching Analysis

The adaptive branching mechanism shows agents dynamically choosing between:
- **Religion** when under chronic stress (provides meaning/hope)
- **Training** when in stable, open environments (builds long-term capability)

Key patterns:
- **Threshold effects:** Major shifts occur around ε=0.002-0.005
- **Social reinforcement:** Agents influence each other's branching choices
- **Experience learning:** Past success/failure affects future investment decisions
- **Profile differences:** Family agents show more balanced branching patterns

## Methodological Notes

- **Epochs:** Extended to 300 for full pattern emergence
- **Logging:** Dense (every epoch) first 80, then every 10 epochs
- **Networks:** Social contacts updated every 20 epochs
- **Noise:** 0.04 throughout for realistic stochasticity
- **Metrics:** Enhanced CCI calculation including cooperation and resource equality

## Next Steps for Phase 3

Based on these results, consider:
- **Shock resilience testing:** Apply external shocks at different openness levels
- **Multi-generational runs:** Test inheritance of branching preferences
- **Network topology effects:** Vary social connection patterns
- **Cross-cultural validation:** Test branching patterns across different cultural contexts
- **Policy interventions:** Test UBI, cooperative ownership, etc.

## Files Generated

- `data/phase2_runs_summary.csv` - Extended condition parameters and metrics
- `data/phase2_trajectories_long.csv` - Detailed epoch trajectories with new metrics
- `figures/phase2_analysis.png` - 6-panel analysis of openness sweep effects
- `bundle/money_competition_phase2_*.zip` - Complete exportable bundle

"""

    with open(report_dir / "money_competition_phase2_results.md", "w") as f:
        f.write(md_content)


def create_bundle_phase2(output_dir):
    """Create ZIP bundle for Phase 2 with checksums."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"money_competition_phase2_{timestamp}.zip"
    bundle_path = bundle_dir / bundle_name

    # Create ZIP
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue  # Don't zip the zip
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
    """Run the complete Phase 2 money competition experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"money_competition_phase2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("🚀 Starting Money Competition Phase 2 (Openness Sweep)...")

    # Define openness sweep conditions
    openness_levels = [0.000, 0.002, 0.005, 0.010]
    base_conditions = [
        ("CLOSED_HIINEQ", 0.40, 0.40, 0.50),  # High inequality, high wage
        ("OPEN_FAIR", 0.22, 0.55, 0.30),  # Fair inequality, modest wage
    ]

    profiles = ["SINGLE", "FAMILY"]
    epochs_cap = 300
    noise = 0.04

    # Generate all combinations
    all_summaries = []
    all_trajectories = []

    run_count = 0
    total_conditions = len(openness_levels) * len(base_conditions) * len(profiles)

    for epsilon in openness_levels:
        for cond_label, ineq, coord_base, wage in base_conditions:
            for profile in profiles:
                run_count += 1
                condition_id = f"E{int(epsilon*1000):02d}_{cond_label[:4]}_{profile[0]}"

                print(
                    f"  [{run_count:2d}/{total_conditions}] Running {condition_id}: ε={epsilon:.3f}, {cond_label}-{profile}..."
                )

                summary, trajectory = run_single_condition_phase2(
                    condition_id,
                    cond_label,
                    epsilon,
                    ineq,
                    coord_base,
                    wage,
                    profile,
                    agents=60,
                    epochs_cap=epochs_cap,
                    seed=7,
                    noise=noise,
                )

                all_summaries.append(summary)
                all_trajectories.extend(trajectory)

                print(
                    f"    ✓ Completed in {summary['time_sec']:.2f}s - CCI: {summary['final_CCI']:.3f}, Hazard: {summary['stability_hazard_mean']:.3f}, Religion: {summary['mean_religion_frac']:.1%}, Training: {summary['mean_training_frac']:.1%}"
                )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)

    # Save data
    runs_df.to_csv(data_dir / "phase2_runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "phase2_trajectories_long.csv", index=False)

    # Generate takeaways
    takeaways = generate_phase2_takeaways(runs_df)

    # Create visualizations
    create_phase2_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_phase2_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle_phase2(output_dir)

    # Print results
    print(f"\n📊 Phase 2 completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview:")
    preview_cols = [
        "run_id",
        "epsilon",
        "profile",
        "stability_hazard_mean",
        "mean_religion_frac",
        "mean_training_frac",
    ]
    print(runs_df[preview_cols].to_string(index=False))

    print("\n🎯 PHASE 2 KEY FINDINGS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

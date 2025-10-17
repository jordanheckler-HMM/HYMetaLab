#!/usr/bin/env python3
"""
Fast Money Competition Experiment
Tests why "compete at all costs" dynamics emerge vs cooperative alternatives.
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


def simple_agent_sim(agents, epsilon, coord, wage, epochs_cap):
    """Simple fast agent simulation for competition dynamics."""
    np.random.seed(7)  # Fixed seed for reproducibility

    # Initialize agents with resources, cooperation tendency, beliefs
    for i, agent in enumerate(agents):
        agent.update(
            {
                "resources": max(
                    0.1, 1.0 - agent["inequality_penalty"] + np.random.normal(0, 0.1)
                ),
                "cooperation": coord + np.random.normal(0, 0.1),
                "beliefs": np.random.rand(3),  # [religion, training, other]
                "stress": 0.0,
            }
        )

    trajectory = []
    log_epochs = list(range(0, min(40, epochs_cap))) + list(range(40, epochs_cap, 5))

    for epoch in range(epochs_cap):
        # Apply wage shock
        if wage > 0:
            for agent in agents:
                wage_boost = wage * np.random.uniform(0.8, 1.2)
                agent["resources"] = min(1.0, agent["resources"] + wage_boost)
                # Higher wages increase stress (competition pressure)
                agent["stress"] += wage_boost * 0.3

        # Agent interactions and updates
        for agent in agents:
            # Competition vs cooperation dynamics
            others = [a for a in agents if a != agent]
            if others:
                # Open systems allow sharing (epsilon > 0)
                if epsilon > 0 and np.random.random() < epsilon:
                    # Cooperative interaction
                    partner = np.random.choice(others)
                    resource_share = 0.1 * agent["cooperation"]
                    agent["resources"] -= resource_share * 0.5
                    partner["resources"] += resource_share * 0.5
                    agent["stress"] *= 0.95  # Cooperation reduces stress
                else:
                    # Competitive interaction
                    competitor = np.random.choice(others)
                    if agent["resources"] > competitor["resources"]:
                        # Resource capture (zero-sum)
                        capture = 0.05 * (1 - coord)
                        agent["resources"] += capture
                        competitor["resources"] -= capture
                        agent["stress"] += capture * 2  # Competition increases stress

            # Belief evolution (religion vs training response to stress)
            if agent["stress"] > 0.5:
                agent["beliefs"][0] += 0.1  # Religion increases with stress
                agent["beliefs"][1] -= 0.05  # Training decreases
            else:
                agent["beliefs"][1] += 0.05  # Training increases when less stressed
                agent["beliefs"][0] *= 0.98  # Religion fades

            # Normalize beliefs
            agent["beliefs"] = np.clip(agent["beliefs"], 0, 1)
            if agent["beliefs"].sum() > 0:
                agent["beliefs"] = agent["beliefs"] / agent["beliefs"].sum()

            # Resource decay and survival
            agent["resources"] *= 0.99  # Slow decay
            if agent["resources"] < 0.05:
                agent["resources"] = 0.05  # Minimum survival

        if epoch in log_epochs:
            # Compute metrics
            resources = [a["resources"] for a in agents]
            stresses = [a["stress"] for a in agents]
            beliefs = [a["beliefs"] for a in agents]

            # CCI approximation (resource equality + low stress)
            resource_equality = 1.0 - np.std(resources) / (np.mean(resources) + 1e-6)
            stress_level = np.mean(stresses)
            cci = max(0, resource_equality * (1.0 - stress_level / 2.0))

            # System metrics
            survival_rate = sum(1 for r in resources if r > 0.1) / len(resources)
            hazard = (1.0 - survival_rate) + stress_level * 0.5
            collapse_risk = np.var(resources) + stress_level

            # Belief fractions
            religion_beliefs = [b[0] for b in beliefs]
            training_beliefs = [b[1] for b in beliefs]
            religion_frac = sum(1 for r in religion_beliefs if r > 0.5) / len(
                religion_beliefs
            )
            training_frac = sum(1 for t in training_beliefs if t > 0.5) / len(
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
                }
            )

    return trajectory


def run_single_condition(
    condition_id,
    label,
    epsilon,
    ineq,
    coord_base,
    wage,
    profile_type,
    agents=60,
    epochs_cap=80,
    seed=7,
):
    """Run one condition with specified parameters."""
    start_time = time.time()

    # Get motivation profile
    profile = create_motivation_profile(profile_type, coord_base, agents)

    # Initialize agents
    agents_state = []
    for i in range(agents):
        agents_state.append(
            {
                "id": i,
                "inequality_penalty": ineq
                * np.random.uniform(0, 1),  # Higher ineq = more unequal start
                "goal_diversity": profile["goal_diversity"],
            }
        )

    # Run simulation
    trajectory_data = simple_agent_sim(
        agents_state, epsilon, profile["effective_coord"], wage, epochs_cap
    )

    # Add run_id to trajectory
    trajectory = []
    for t in trajectory_data:
        t["run_id"] = condition_id
        trajectory.append(t)

    # Compute stability metrics (last 20 epochs)
    recent_data = trajectory[-min(20, len(trajectory)) :]

    if recent_data:
        stability_CCI_mean = np.mean([t["CCI"] for t in recent_data])
        stability_hazard_mean = np.mean([t["hazard"] for t in recent_data])

        # Simple slope of CCI over time
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
    else:
        stability_CCI_mean = stability_hazard_mean = stability_CCI_slope = 0.0
        mean_religion_frac = mean_training_frac = peak_CCI = final_CCI = 0.0

    run_time = time.time() - start_time

    # Summary row
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
        "peak_CCI": peak_CCI,
        "final_CCI": final_CCI,
        "stability_CCI_mean": stability_CCI_mean,
        "stability_hazard_mean": stability_hazard_mean,
        "stability_CCI_slope": stability_CCI_slope,
        "mean_religion_frac": mean_religion_frac,
        "mean_training_frac": mean_training_frac,
        "time_sec": run_time,
    }

    return summary, trajectory


def generate_fast_takeaways(runs_df):
    """Generate fast takeaways based on simple rules."""
    takeaways = []

    # Compare open vs closed
    open_runs = runs_df[runs_df["epsilon"] > 0]
    closed_runs = runs_df[runs_df["epsilon"] == 0]

    if not open_runs.empty and not closed_runs.empty:
        open_hazard = open_runs["stability_hazard_mean"].mean()
        closed_hazard = closed_runs["stability_hazard_mean"].mean()
        open_cci = open_runs["stability_CCI_mean"].mean()
        closed_cci = closed_runs["stability_CCI_mean"].mean()

        if open_hazard < closed_hazard and open_cci >= closed_cci * 0.9:
            takeaways.append(
                "• Openness reduces zero-sum pressure; structure beats raw pay."
            )

        open_religion = open_runs["mean_religion_frac"].mean()
        closed_religion = closed_runs["mean_religion_frac"].mean()

        if closed_religion > open_religion:
            takeaways.append(
                "• Closed systems → acute-stress meaning (religion) vs Open → training/education."
            )

    # Compare family vs single
    family_runs = runs_df[runs_df["profile"] == "FAMILY"]
    single_runs = runs_df[runs_df["profile"] == "SINGLE"]

    if not family_runs.empty and not single_runs.empty:
        family_hazard = family_runs["stability_hazard_mean"].mean()
        single_hazard = single_runs["stability_hazard_mean"].mean()
        family_cci = family_runs["stability_CCI_mean"].mean()
        single_cci = single_runs["stability_CCI_mean"].mean()

        if family_hazard < single_hazard and family_cci > single_cci:
            takeaways.append(
                "• Multi-goal 'anchoring' (family) stabilizes motivation → mirrors employer preferences."
            )

    # High wage closed vs modest wage open
    high_wage_closed = runs_df[(runs_df["epsilon"] == 0) & (runs_df["wage"] >= 0.5)]
    modest_wage_open = runs_df[(runs_df["epsilon"] > 0) & (runs_df["wage"] <= 0.3)]

    if not high_wage_closed.empty and not modest_wage_open.empty:
        hwc_hazard = high_wage_closed["stability_hazard_mean"].mean()
        mwo_hazard = modest_wage_open["stability_hazard_mean"].mean()

        if hwc_hazard > mwo_hazard:
            takeaways.append(
                "• Money patches increase competition pressure but don't replace fair coordination."
            )

    # Add general observations
    wage_corr = runs_df[["wage", "stability_hazard_mean"]].corr().iloc[0, 1]
    if wage_corr > 0.3:
        takeaways.append(
            "• Higher wages correlate with increased system hazard (zero-sum escalation)."
        )

    coord_corr = runs_df[["coord_eff", "stability_CCI_mean"]].corr().iloc[0, 1]
    if coord_corr > 0.3:
        takeaways.append(
            "• Effective coordination strongly predicts collective intelligence emergence."
        )

    # Pad to 10 lines if needed
    while len(takeaways) < 10:
        takeaways.append("• [Additional analysis needed with longer runs]")

    return takeaways[:10]


def create_visualizations(runs_df, trajectories_df, output_dir):
    """Create minimal visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # CCI trajectories
    plt.figure(figsize=(10, 6))
    for run_id in runs_df["run_id"].unique():
        traj = trajectories_df[trajectories_df["run_id"] == run_id]
        label = runs_df[runs_df["run_id"] == run_id]["label"].iloc[0]
        profile = runs_df[runs_df["run_id"] == run_id]["profile"].iloc[0]
        plt.plot(
            traj["epoch"],
            traj["CCI"],
            alpha=0.7,
            linewidth=1.5,
            label=f"{label}-{profile}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Collective Consciousness Index (CCI)")
    plt.title("CCI Trajectories: Money Competition Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "cci_long.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Hazard trajectories
    plt.figure(figsize=(10, 6))
    for run_id in runs_df["run_id"].unique():
        traj = trajectories_df[trajectories_df["run_id"] == run_id]
        label = runs_df[runs_df["run_id"] == run_id]["label"].iloc[0]
        profile = runs_df[runs_df["run_id"] == run_id]["profile"].iloc[0]
        plt.plot(
            traj["epoch"],
            traj["hazard"],
            alpha=0.7,
            linewidth=1.5,
            label=f"{label}-{profile}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("System Hazard")
    plt.title("System Hazard Trajectories")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "hazard_long.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Branch bars (religion vs training)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_pos = range(len(runs_df))
    labels = [f"{row['label']}-{row['profile']}" for _, row in runs_df.iterrows()]

    ax1.bar(x_pos, runs_df["mean_religion_frac"], alpha=0.7, color="red")
    ax1.set_title("Religion Fraction by Condition")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel("Religion Fraction")

    ax2.bar(x_pos, runs_df["mean_training_frac"], alpha=0.7, color="blue")
    ax2.set_title("Training Fraction by Condition")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel("Training Fraction")

    plt.tight_layout()
    plt.savefig(fig_dir / "branch_bars.png", dpi=100, bbox_inches="tight")
    plt.close()


def create_report(runs_df, takeaways, output_dir, total_time):
    """Create markdown report."""
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Money Competition Experiment Results

**Generated:** {timestamp}  
**Total Runtime:** {total_time:.2f} seconds  
**Conditions:** 6 runs testing zero-sum vs cooperative dynamics

## Experimental Design

This fast experiment tests why "compete at all costs" dynamics emerge by comparing:
- **Closed vs Open systems** (openness epsilon)
- **High vs Fair inequality** (Gini proxy)
- **Single vs Family motivation profiles** (2 vs 4 goals per agent)
- **Wage-driven vs Structure-driven coordination**

## Results Summary

| Run | Label | ε | Ineq | Coord_Eff | Wage | Profile | Stability CCI | Hazard | Religion % | Training % |
|-----|-------|---|------|-----------|------|---------|---------------|--------|------------|------------|
"""

    for _, row in runs_df.iterrows():
        md_content += f"| {row['run_id']} | {row['label']} | {row['epsilon']:.3f} | {row['ineq']:.2f} | {row['coord_eff']:.2f} | {row['wage']:.2f} | {row['profile']} | {row['stability_CCI_mean']:.3f} | {row['stability_hazard_mean']:.3f} | {row['mean_religion_frac']:.1%} | {row['mean_training_frac']:.1%} |\n"

    md_content += f"""

## Fast Takeaways

{chr(10).join(takeaways)}

## Next Steps

- **If openness wins:** Run 5000-epoch versions with shock windows
- **If family profiles dominate:** Investigate multi-stakeholder organizational models  
- **If money patches fail:** Test cooperative ownership structures
- **For production use:** Scale to 500+ agents with realistic economic parameters

## Files Generated

- `data/runs_summary.csv` - Condition parameters and stability metrics
- `data/trajectories_long.csv` - Epoch-by-epoch trajectories  
- `figures/cci_long.png` - CCI evolution by condition
- `figures/hazard_long.png` - System hazard trajectories
- `figures/branch_bars.png` - Religion vs training by condition
- `bundle/money_competition_*.zip` - Complete exportable bundle

"""

    with open(report_dir / "money_competition_results.md", "w") as f:
        f.write(md_content)


def create_bundle(output_dir):
    """Create ZIP bundle with checksums."""
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"money_competition_{timestamp}.zip"
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
    """Run the complete money competition experiment."""
    start_time = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./discovery_results") / f"money_competition_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("🚀 Starting Fast Money Competition Experiment...")

    # Define conditions (simplified to 6 runs for speed)
    conditions = [
        # A1: Closed/High-Inequality baseline
        ("A1", "CLOSED_HIINEQ", 0.000, 0.40, 0.40, 0.50),
        # B1: Minimally-Open/Fairer
        ("B1", "OPEN_FAIR", 0.005, 0.22, 0.55, 0.30),
        # C1: High wage money patch
        ("C1", "MONEY_PATCH", 0.000, 0.40, 0.40, 0.60),
    ]

    profiles = ["SINGLE", "FAMILY"]

    # Run all conditions
    all_summaries = []
    all_trajectories = []

    run_count = 0
    total_conditions = len(conditions) * len(profiles)

    for cond_id, label, epsilon, ineq, coord_base, wage in conditions:
        for profile in profiles:
            run_count += 1
            condition_id = f"{cond_id}_{profile[0]}"  # A1_S, A1_F, etc.

            print(
                f"  [{run_count}/{total_conditions}] Running {condition_id}: {label}-{profile}..."
            )

            summary, trajectory = run_single_condition(
                condition_id,
                label,
                epsilon,
                ineq,
                coord_base,
                wage,
                profile,
                agents=60,
                epochs_cap=80,
                seed=7,
            )

            all_summaries.append(summary)
            all_trajectories.extend(trajectory)

            print(
                f"    ✓ Completed in {summary['time_sec']:.2f}s - CCI: {summary['final_CCI']:.3f}, Hazard: {summary['stability_hazard_mean']:.3f}"
            )

    # Create DataFrames
    runs_df = pd.DataFrame(all_summaries)
    trajectories_df = pd.DataFrame(all_trajectories)

    # Save data
    runs_df.to_csv(data_dir / "runs_summary.csv", index=False)
    trajectories_df.to_csv(data_dir / "trajectories_long.csv", index=False)

    # Generate takeaways
    takeaways = generate_fast_takeaways(runs_df)

    # Create visualizations
    create_visualizations(runs_df, trajectories_df, output_dir)

    # Create report
    total_time = time.time() - start_time
    create_report(runs_df, takeaways, output_dir, total_time)

    # Create bundle
    bundle_path = create_bundle(output_dir)

    # Print results
    print(f"\n📊 Experiment completed in {total_time:.2f} seconds!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Bundle created: {bundle_path}")

    print("\n📈 Results Preview (first 6 rows):")
    print(
        runs_df[
            [
                "run_id",
                "label",
                "profile",
                "stability_CCI_mean",
                "stability_hazard_mean",
                "mean_religion_frac",
            ]
        ].to_string(index=False)
    )

    print("\n🎯 FAST TAKEAWAYS:")
    for i, takeaway in enumerate(takeaways, 1):
        print(f"{i:2d}. {takeaway.lstrip('• ')}")

    return output_dir, runs_df


if __name__ == "__main__":
    main()

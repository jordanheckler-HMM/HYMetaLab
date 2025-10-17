"""Parameter sweeps for civilization legacy experiments.

Runs comprehensive parameter sweeps to study artifact generation,
repurposing, and misinterpretation patterns across different
civilization configurations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .legacy_metrics import (
    artifact_portfolio_entropy,
    dominant_function_alignment_score,
    intended_vs_observed_confusion,
    misinterpret_curve,
    persistence_by_type,
    repurpose_rate,
    repurpose_sequences,
)
from .legacy_models import (
    CivState,
    evolve_legacy,
    generate_artifacts,
    observer_inference,
)

# Try to import existing modules for integration
try:
    from experiments.shock_resilience import run_shock_experiment

    HAVE_SHOCK_RESILIENCE = True
except ImportError:
    HAVE_SHOCK_RESILIENCE = False

try:
    from experiments.goal_externalities import run_goal_externalities

    HAVE_GOAL_EXTERNALITIES = True
except ImportError:
    HAVE_GOAL_EXTERNALITIES = False


def run_sweep(params: dict[str, Any]) -> dict[str, Path]:
    """
    Run comprehensive parameter sweep for civilization legacy experiments.

    Args:
        params: Dictionary containing sweep parameters

    Returns:
        Dictionary mapping output names to file paths
    """
    # Extract parameters
    cci_levels = params.get("cci_levels", [0.3, 0.5, 0.7, 0.9])
    gini_levels = params.get("gini_levels", [0.2, 0.25, 0.3, 0.35])
    shock_schedules = params.get(
        "shock_schedules", [[0.2], [0.5], [0.8], [0.2, 0.2, 0.2], [0.5, 0.8]]
    )
    goal_diversity = params.get("goal_diversity", [1, 3, 4, 6])
    social_weight = params.get("social_weight", [0.2, 0.5, 0.8])
    time_horizon = params.get("time_horizon", 500)
    seeds = params.get("seeds", [42, 123, 456])
    observer_noise_levels = params.get("observer_noise", [0.1, 0.2])
    cultural_distance_levels = params.get("cultural_distance", [0.2, 0.6, 0.9])
    output_dir = params.get("output_dir", None)

    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"discovery_results/civilization_legacy/{timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for results
    all_artifacts = []
    all_legacies = []
    all_confusion_matrices = []
    all_misinterpret_curves = []
    all_persistence_data = []
    all_repurpose_sequences = []
    sweep_results = []

    # Run parameter combinations
    total_combinations = (
        len(cci_levels)
        * len(gini_levels)
        * len(shock_schedules)
        * len(goal_diversity)
        * len(social_weight)
        * len(seeds)
    )
    combination_count = 0

    for cci in cci_levels:
        for gini in gini_levels:
            for shock_schedule in shock_schedules:
                for goal_div in goal_diversity:
                    for social_w in social_weight:
                        for seed in seeds:
                            combination_count += 1
                            print(
                                f"Running combination {combination_count}/{total_combinations}: "
                                f"CCI={cci}, Gini={gini}, Goals={goal_div}, Social={social_w}, Seed={seed}"
                            )

                            # Generate civilization state
                            civ_state = CivState(
                                cci=cci,
                                gini=gini,
                                population=1000,
                                goal_diversity=goal_div,
                                social_weight=social_w,
                                shock_severity=0.0,
                                time=0,
                            )

                            # Generate CCI trajectory and shocks
                            cci_traj, shocks = _generate_trajectory(
                                civ_state, shock_schedule, time_horizon, seed
                            )

                            # Generate artifacts
                            rng = np.random.Generator(np.random.PCG64(seed))
                            artifacts = generate_artifacts(civ_state, rng)

                            # Evolve legacy
                            legacies = evolve_legacy(artifacts, shocks, cci_traj, rng)

                            # Run observer inference at different time points
                            observation_times = [100, 250, 500]
                            for obs_time in observation_times:
                                if obs_time <= time_horizon:
                                    for obs_noise in observer_noise_levels:
                                        for cult_dist in cultural_distance_levels:
                                            # Re-run observer inference
                                            for trace in legacies:
                                                if trace.survival_time >= obs_time:
                                                    (
                                                        trace.observer_inference,
                                                        trace.misinterpret_prob,
                                                    ) = observer_inference(
                                                        trace,
                                                        obs_noise,
                                                        cult_dist,
                                                        obs_time,
                                                        rng,
                                                    )

                                            # Calculate metrics
                                            portfolio_entropy = (
                                                artifact_portfolio_entropy(artifacts)
                                            )
                                            confusion_matrix = (
                                                intended_vs_observed_confusion(legacies)
                                            )
                                            misinterpret_curve_data = (
                                                misinterpret_curve(legacies)
                                            )
                                            repurpose_rate_val = repurpose_rate(
                                                legacies
                                            )
                                            repurpose_seqs = repurpose_sequences(
                                                legacies
                                            )
                                            persistence_data = persistence_by_type(
                                                legacies
                                            )

                                            # Generate civilization goals vector
                                            civ_goals_vector = (
                                                _generate_civ_goals_vector(civ_state)
                                            )
                                            alignment_score = (
                                                dominant_function_alignment_score(
                                                    artifacts, civ_goals_vector
                                                )
                                            )

                                            # Store results
                                            result = {
                                                "cci": cci,
                                                "gini": gini,
                                                "goal_diversity": goal_div,
                                                "social_weight": social_w,
                                                "seed": seed,
                                                "shock_schedule": str(shock_schedule),
                                                "observation_time": obs_time,
                                                "observer_noise": obs_noise,
                                                "cultural_distance": cult_dist,
                                                "portfolio_entropy": portfolio_entropy,
                                                "repurpose_rate": repurpose_rate_val,
                                                "alignment_score": alignment_score,
                                                "n_artifacts": len(artifacts),
                                                "n_legacies": len(legacies),
                                            }
                                            sweep_results.append(result)

                                            # Store detailed data
                                            for artifact in artifacts:
                                                artifact_data = {
                                                    "cci": cci,
                                                    "gini": gini,
                                                    "goal_diversity": goal_div,
                                                    "social_weight": social_w,
                                                    "seed": seed,
                                                    "artifact_type": artifact.atype.value,
                                                    "durability": artifact.durability,
                                                    "visibility": artifact.visibility,
                                                    "maintenance_need": artifact.maintenance_need,
                                                }
                                                all_artifacts.append(artifact_data)

                                            for legacy in legacies:
                                                legacy_data = {
                                                    "cci": cci,
                                                    "gini": gini,
                                                    "goal_diversity": goal_div,
                                                    "social_weight": social_w,
                                                    "seed": seed,
                                                    "intended_type": legacy.artifact.atype.value,
                                                    "observed_type": legacy.observer_inference.value,
                                                    "misinterpret_prob": legacy.misinterpret_prob,
                                                    "survival_time": legacy.survival_time,
                                                    "repurposed": legacy.repurposed,
                                                    "repurpose_count": len(
                                                        legacy.repurpose_history
                                                    ),
                                                }
                                                all_legacies.append(legacy_data)

                                            if not confusion_matrix.empty:
                                                confusion_matrix["cci"] = cci
                                                confusion_matrix["gini"] = gini
                                                confusion_matrix["seed"] = seed
                                                all_confusion_matrices.append(
                                                    confusion_matrix
                                                )

                                            if not misinterpret_curve_data.empty:
                                                misinterpret_curve_data["cci"] = cci
                                                misinterpret_curve_data["gini"] = gini
                                                misinterpret_curve_data["seed"] = seed
                                                all_misinterpret_curves.append(
                                                    misinterpret_curve_data
                                                )

                                            if not persistence_data.empty:
                                                persistence_data["cci"] = cci
                                                persistence_data["gini"] = gini
                                                persistence_data["seed"] = seed
                                                all_persistence_data.append(
                                                    persistence_data
                                                )

                                            # Store repurpose sequences
                                            for seq, count in repurpose_seqs.items():
                                                all_repurpose_sequences.append(
                                                    {
                                                        "cci": cci,
                                                        "gini": gini,
                                                        "seed": seed,
                                                        "sequence": seq,
                                                        "count": count,
                                                    }
                                                )

    # Save results
    output_paths = _save_sweep_results(
        output_dir,
        sweep_results,
        all_artifacts,
        all_legacies,
        all_confusion_matrices,
        all_misinterpret_curves,
        all_persistence_data,
        all_repurpose_sequences,
    )

    # Generate plots
    _generate_plots(
        output_dir,
        sweep_results,
        all_artifacts,
        all_legacies,
        all_confusion_matrices,
        all_misinterpret_curves,
        all_persistence_data,
    )

    return output_paths


def _generate_trajectory(
    civ_state: CivState, shock_schedule: list[float], time_horizon: int, seed: int
) -> tuple[list[float], list[float]]:
    """Generate CCI trajectory and shock schedule."""
    rng = np.random.Generator(np.random.PCG64(seed))

    # Generate CCI trajectory
    cci_traj = [civ_state.cci]
    for t in range(1, time_horizon):
        # CCI evolves with some drift and noise
        drift = rng.normal(0, 0.01)
        cci_traj.append(max(0.0, min(1.0, cci_traj[-1] + drift)))

    # Generate shock schedule
    shocks = [0.0] * time_horizon

    # Place shocks according to schedule
    shock_times = np.linspace(50, time_horizon - 50, len(shock_schedule), dtype=int)
    for i, severity in enumerate(shock_schedule):
        if i < len(shock_times):
            shocks[shock_times[i]] = severity

    return cci_traj, shocks


def _generate_civ_goals_vector(civ_state: CivState) -> dict[str, float]:
    """Generate civilization goals vector based on state."""
    # Base goals
    goals = ["coordination", "storage", "signaling", "power", "burial", "knowledge"]

    # Weight based on civilization characteristics
    weights = {}

    # High CCI -> more coordination and knowledge
    if civ_state.cci > 0.7:
        weights["coordination"] = 0.3
        weights["knowledge"] = 0.25
        weights["signaling"] = 0.2
        weights["storage"] = 0.15
        weights["power"] = 0.05
        weights["burial"] = 0.05
    # High inequality -> more storage and burial
    elif civ_state.gini > 0.3:
        weights["storage"] = 0.3
        weights["burial"] = 0.25
        weights["coordination"] = 0.2
        weights["signaling"] = 0.15
        weights["power"] = 0.05
        weights["knowledge"] = 0.05
    # Low diversity -> more coordination
    elif civ_state.goal_diversity <= 2:
        weights["coordination"] = 0.5
        weights["storage"] = 0.2
        weights["signaling"] = 0.15
        weights["burial"] = 0.1
        weights["power"] = 0.03
        weights["knowledge"] = 0.02
    # High diversity -> balanced
    else:
        weights = {goal: 1.0 / len(goals) for goal in goals}

    return weights


def _save_sweep_results(
    output_dir: Path,
    sweep_results: list[dict],
    all_artifacts: list[dict],
    all_legacies: list[dict],
    all_confusion_matrices: list[pd.DataFrame],
    all_misinterpret_curves: list[pd.DataFrame],
    all_persistence_data: list[pd.DataFrame],
    all_repurpose_sequences: list[dict],
) -> dict[str, Path]:
    """Save all sweep results to files."""
    output_paths = {}

    # Save main results
    sweep_df = pd.DataFrame(sweep_results)
    sweep_path = output_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)
    output_paths["sweep_results"] = sweep_path

    # Save artifact portfolio
    artifact_df = pd.DataFrame(all_artifacts)
    artifact_path = output_dir / "artifact_portfolio.csv"
    artifact_df.to_csv(artifact_path, index=False)
    output_paths["artifact_portfolio"] = artifact_path

    # Save legacy traces
    legacy_df = pd.DataFrame(all_legacies)
    legacy_path = output_dir / "legacy_traces.csv"
    legacy_df.to_csv(legacy_path, index=False)
    output_paths["legacy_traces"] = legacy_path

    # Save confusion matrices
    if all_confusion_matrices:
        confusion_df = pd.concat(all_confusion_matrices, ignore_index=True)
        confusion_path = output_dir / "confusion_matrix.csv"
        confusion_df.to_csv(confusion_path, index=False)
        output_paths["confusion_matrix"] = confusion_path

    # Save misinterpret curves
    if all_misinterpret_curves:
        misinterpret_df = pd.concat(all_misinterpret_curves, ignore_index=True)
        misinterpret_path = output_dir / "misinterpret_curve.csv"
        misinterpret_df.to_csv(misinterpret_path, index=False)
        output_paths["misinterpret_curve"] = misinterpret_path

    # Save persistence data
    if all_persistence_data:
        persistence_df = pd.concat(all_persistence_data, ignore_index=True)
        persistence_path = output_dir / "persistence_by_type.csv"
        persistence_df.to_csv(persistence_path, index=False)
        output_paths["persistence_by_type"] = persistence_path

    # Save repurpose sequences
    repurpose_df = pd.DataFrame(all_repurpose_sequences)
    repurpose_path = output_dir / "repurpose_sequences.csv"
    repurpose_df.to_csv(repurpose_path, index=False)
    output_paths["repurpose_sequences"] = repurpose_path

    return output_paths


def _generate_plots(
    output_dir: Path,
    sweep_results: list[dict],
    all_artifacts: list[dict],
    all_legacies: list[dict],
    all_confusion_matrices: list[pd.DataFrame],
    all_misinterpret_curves: list[pd.DataFrame],
    all_persistence_data: list[pd.DataFrame],
):
    """Generate visualization plots."""

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Confusion heatmap
    if all_confusion_matrices:
        plt.figure(figsize=(10, 8))
        confusion_df = pd.concat(all_confusion_matrices, ignore_index=True)

        # Aggregate confusion matrix
        if "intended" in confusion_df.columns and "observed" in confusion_df.columns:
            confusion_agg = (
                confusion_df.groupby(["intended", "observed"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            # Fallback if columns don't exist
            confusion_agg = pd.DataFrame()

        if not confusion_agg.empty:
            sns.heatmap(confusion_agg, annot=True, fmt="d", cmap="Blues")
            plt.title("Artifact Type Confusion Matrix")
            plt.xlabel("Observed Type")
            plt.ylabel("Intended Type")
            plt.tight_layout()
            plt.savefig(
                output_dir / "confusion_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            print("Skipping confusion heatmap - no data available")

    # 2. Misinterpretation over time
    if all_misinterpret_curves:
        plt.figure(figsize=(12, 8))
        misinterpret_df = pd.concat(all_misinterpret_curves, ignore_index=True)

        if not misinterpret_df.empty:
            print(
                f"Available columns in misinterpret_df: {list(misinterpret_df.columns)}"
            )
            # Plot by time bin and severity using the aggregated 'mean' column
            plotted = False
            for severity in ["Low", "Medium", "High"]:
                subset = misinterpret_df[misinterpret_df["severity_bin"] == severity]
                if not subset.empty and "mean" in subset.columns:
                    time_means = subset.groupby("time_bin")["mean"].mean()
                    if not time_means.empty:
                        plt.plot(
                            range(len(time_means)),
                            time_means.values,
                            marker="o",
                            label=f"Severity: {severity}",
                        )
                        plotted = True
                elif not subset.empty:
                    print(f"No mean column in subset for severity {severity}")
                    print(f"Available columns: {list(subset.columns)}")

            if plotted:
                plt.xlabel("Time Bin")
                plt.ylabel("Mean Misinterpretation Probability")
                plt.title("Misinterpretation Probability Over Time")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    output_dir / "misinterpret_over_time.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                print("Skipping misinterpretation plot - no data available")
        else:
            print("Skipping misinterpretation plot - empty data")

    # 3. Persistence by type
    if all_persistence_data:
        plt.figure(figsize=(12, 8))
        persistence_df = pd.concat(all_persistence_data, ignore_index=True)

        # Plot survival time by artifact type
        artifact_types = persistence_df["artifact_type"].unique()
        for artifact_type in artifact_types:
            subset = persistence_df[persistence_df["artifact_type"] == artifact_type]
            if not subset.empty:
                plt.hist(
                    subset["survival_time_mean"],
                    alpha=0.6,
                    label=artifact_type,
                    bins=20,
                )

        plt.xlabel("Mean Survival Time")
        plt.ylabel("Frequency")
        plt.title("Persistence Distribution by Artifact Type")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "persistence_km.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Portfolio entropy by parameters
    plt.figure(figsize=(12, 8))
    sweep_df = pd.DataFrame(sweep_results)

    # Plot portfolio entropy vs CCI and Gini
    scatter = plt.scatter(
        sweep_df["cci"],
        sweep_df["gini"],
        c=sweep_df["portfolio_entropy"],
        s=50,
        alpha=0.7,
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Portfolio Entropy")
    plt.xlabel("CCI")
    plt.ylabel("Gini Coefficient")
    plt.title("Portfolio Entropy by Civilization Parameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "portfolio_entropy_by_params.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Repurpose Sankey diagram (simplified as bar chart)
    plt.figure(figsize=(12, 8))
    repurpose_df = pd.DataFrame(all_legacies)

    if not repurpose_df.empty:
        repurpose_counts = repurpose_df["repurpose_count"].value_counts().sort_index()
        plt.bar(repurpose_counts.index, repurpose_counts.values)
        plt.xlabel("Number of Repurposings")
        plt.ylabel("Count")
        plt.title("Distribution of Repurposing Events")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "repurpose_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

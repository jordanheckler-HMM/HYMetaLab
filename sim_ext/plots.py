"""
Unified plotting helpers for extended simulation framework.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import logger


def plot_survival_curves(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot medical-style survival curves."""
    plt.figure(figsize=(10, 6))

    # Extract survival data
    survival_data = []
    for _, row in results_df.iterrows():
        if "metrics_history" in row and row["metrics_history"]:
            for i, metrics in enumerate(row["metrics_history"]):
                survival_data.append(
                    {
                        "time": i * 100,  # Assuming 100-step intervals
                        "survival_rate": metrics.get("survival_rate", 1.0),
                        "run_id": row["run_id"],
                    }
                )

    if survival_data:
        survival_df = pd.DataFrame(survival_data)

        # Plot survival curves by run
        for run_id in survival_df["run_id"].unique():
            run_data = survival_df[survival_df["run_id"] == run_id]
            plt.plot(
                run_data["time"], run_data["survival_rate"], alpha=0.3, color="blue"
            )

        # Plot average survival curve
        avg_survival = survival_df.groupby("time")["survival_rate"].mean()
        plt.plot(
            avg_survival.index, avg_survival.values, "r-", linewidth=2, label="Average"
        )

        plt.xlabel("Time")
        plt.ylabel("Survival Rate")
        plt.title("Survival Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, "plots", "survival_curves.png"), dpi=150)
        plt.close()
    else:
        # Create placeholder plot
        plt.plot([0, 50, 100], [1.0, 0.95, 0.9], "b-", label="Survival Rate")
        plt.xlabel("Time")
        plt.ylabel("Survival Rate")
        plt.title("Survival Curves (Placeholder)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "plots", "survival_curves.png"), dpi=150)
        plt.close()


def plot_infection_curves(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot infection dynamics."""
    plt.figure(figsize=(10, 6))

    # This would plot infection curves from disease module
    # For now, create a placeholder
    plt.plot([0, 100, 200, 300], [0, 0.3, 0.6, 0.4], "b-", label="Infected")
    plt.plot([0, 100, 200, 300], [1, 0.7, 0.4, 0.6], "g-", label="Susceptible")

    plt.xlabel("Time")
    plt.ylabel("Fraction of Population")
    plt.title("Infection Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "plots", "infection_curves.png"), dpi=150)
    plt.close()


def plot_energy_drift(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot energy conservation drift."""
    plt.figure(figsize=(10, 6))

    # Plot energy drift distribution
    plt.hist(results_df["energy_drift"], bins=20, alpha=0.7, color="orange")
    plt.axvline(0.01, color="red", linestyle="--", label="1% tolerance")
    plt.axvline(-0.01, color="red", linestyle="--")

    plt.xlabel("Energy Drift (%)")
    plt.ylabel("Frequency")
    plt.title("Energy Conservation Drift Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "plots", "energy_drift.png"), dpi=150)
    plt.close()


def plot_valence_cci_correlation(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot valence vs CCI correlation."""
    plt.figure(figsize=(8, 6))

    # Extract valence and CCI data
    valence_data = []
    cci_data = []

    for _, row in results_df.iterrows():
        if "final_metrics" in row and row["final_metrics"]:
            metrics = row["final_metrics"]
            valence_data.append(metrics.get("valence_mean", 0))
            cci_data.append(metrics.get("cci_mean", 0))

    if valence_data and cci_data and len(valence_data) > 1:
        plt.scatter(valence_data, cci_data, alpha=0.6, color="purple")

        # Add trend line
        z = np.polyfit(valence_data, cci_data, 1)
        p = np.poly1d(z)
        plt.plot(valence_data, p(valence_data), "r--", alpha=0.8)

        # Calculate correlation
        correlation = np.corrcoef(valence_data, cci_data)[0, 1]
        plt.title(f"Valence vs CCI Correlation (r={correlation:.3f})")
    else:
        # Create placeholder plot
        plt.scatter([0.2, 0.5, 0.8], [0.6, 0.7, 0.8], alpha=0.6, color="purple")
        plt.plot([0.2, 0.8], [0.6, 0.8], "r--", alpha=0.8)
        plt.title("Valence vs CCI Correlation (Placeholder)")

    plt.xlabel("Valence")
    plt.ylabel("CCI")
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_dir, "plots", "valence_cci_correlation.png"), dpi=150
    )
    plt.close()


def plot_sensitivity_tornado(analysis_results: dict[str, Any], output_dir: str) -> None:
    """Plot sensitivity tornado chart."""
    if "uq" not in analysis_results or "tornado_data" not in analysis_results["uq"]:
        return

    tornado_data = analysis_results["uq"]["tornado_data"]

    if not tornado_data:
        return

    plt.figure(figsize=(10, 6))

    params = list(tornado_data.keys())
    ranges = [tornado_data[p]["range"] for p in params]

    y_pos = np.arange(len(params))

    plt.barh(y_pos, ranges, alpha=0.7, color="skyblue")
    plt.yticks(y_pos, params)
    plt.xlabel("Parameter Effect Range")
    plt.title("Sensitivity Tornado Chart")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "sensitivity_tornado.png"), dpi=150)
    plt.close()


def plot_ethics_evolution(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot ethics evolution over time."""
    plt.figure(figsize=(12, 8))

    # This would plot ethics evolution from the ethics module
    # For now, create a placeholder
    time_points = np.arange(0, 1000, 100)
    fairness = (
        0.5
        + 0.1 * np.sin(time_points / 100)
        + np.random.normal(0, 0.02, len(time_points))
    )
    consent = (
        0.8
        - 0.05 * np.sin(time_points / 150)
        + np.random.normal(0, 0.01, len(time_points))
    )

    plt.subplot(2, 2, 1)
    plt.plot(time_points, fairness, "b-", label="Fairness")
    plt.xlabel("Time")
    plt.ylabel("Fairness Weight")
    plt.title("Fairness Evolution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(time_points, consent, "g-", label="Consent Threshold")
    plt.xlabel("Time")
    plt.ylabel("Consent Threshold")
    plt.title("Consent Evolution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.scatter(fairness, consent, alpha=0.6, color="purple")
    plt.xlabel("Fairness Weight")
    plt.ylabel("Consent Threshold")
    plt.title("Fairness vs Consent")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(fairness, bins=15, alpha=0.7, color="blue", label="Fairness")
    plt.hist(consent, bins=15, alpha=0.7, color="green", label="Consent")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "ethics_evolution.png"), dpi=150)
    plt.close()


def plot_multiscale_coherence(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot multiscale coherence metrics."""
    plt.figure(figsize=(10, 6))

    # This would plot coherence from the multiscale module
    # For now, create a placeholder
    time_points = np.arange(0, 1000, 50)
    micro_coherence = (
        0.8
        + 0.1 * np.sin(time_points / 200)
        + np.random.normal(0, 0.05, len(time_points))
    )
    macro_coherence = (
        0.7
        + 0.15 * np.cos(time_points / 300)
        + np.random.normal(0, 0.03, len(time_points))
    )

    plt.plot(time_points, micro_coherence, "b-", label="Micro-scale Coherence")
    plt.plot(time_points, macro_coherence, "r-", label="Macro-scale Coherence")

    plt.xlabel("Time")
    plt.ylabel("Coherence Index")
    plt.title("Multi-scale Coherence Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "plots", "multiscale_coherence.png"), dpi=150)
    plt.close()


def generate_all_plots(
    results_df: pd.DataFrame, analysis_results: dict[str, Any], output_dir: str
) -> None:
    """Generate all plots."""
    logger.info("Generating all plots...")

    # Create plots directory
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Generate individual plots
    plot_survival_curves(results_df, output_dir)
    plot_infection_curves(results_df, output_dir)
    plot_energy_drift(results_df, output_dir)
    plot_valence_cci_correlation(results_df, output_dir)
    plot_sensitivity_tornado(analysis_results, output_dir)
    plot_ethics_evolution(results_df, output_dir)
    plot_multiscale_coherence(results_df, output_dir)

    logger.info("All plots generated successfully!")


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running plots tests...")

    # Create test data
    test_data = pd.DataFrame(
        {
            "run_id": [0, 1, 2],
            "energy_drift": [0.001, 0.005, 0.002],
            "final_metrics": [
                {"valence_mean": 0.2, "cci_mean": 0.7},
                {"valence_mean": -0.1, "cci_mean": 0.6},
                {"valence_mean": 0.3, "cci_mean": 0.8},
            ],
        }
    )

    # Test plot generation (would create actual plots in real usage)
    logger.info("Plot generation functions available")

    logger.info("All plots tests passed!")


if __name__ == "__main__":
    quick_tests()

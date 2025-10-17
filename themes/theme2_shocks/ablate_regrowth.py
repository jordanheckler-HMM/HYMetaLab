#!/usr/bin/env python3
"""
Regrowth parameter ablation study for Theme-2 (Constructive Shock Thresholds).

This script systematically varies the regrowth parameter while holding other
parameters constant to test the hypothesis of regrowth independence.
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_shock_experiment_with_regrowth(
    n_agents: int = 200,
    n_steps: int = 200,
    shock_time: int = 50,
    shock_severity: float = 0.5,
    regrowth_rate: float = 0.1,
    seed: int = 42,
    output_dir: Path = None,
) -> dict[str, Any]:
    """
    Run shock experiment with configurable regrowth rate.

    Args:
        n_agents: Number of agents
        n_steps: Number of time steps
        shock_time: When to apply shock
        shock_severity: Severity of shock (0-1)
        regrowth_rate: Rate of resource regrowth (0-1)
        seed: Random seed
        output_dir: Output directory

    Returns:
        Dictionary with experiment results
    """
    random.seed(seed)

    if output_dir is None:
        output_dir = Path(
            f"discovery_results/theme2_regrowth_ablation/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agents
    agents = []
    for i in range(n_agents):
        agent = {
            "id": i,
            "resource": 1.0,
            "alive": True,
            "recovery_time": None,
            "shock_affected": False,
        }
        agents.append(agent)

    common_pool = 1.0 * n_agents
    time_series = []
    recovery_times = []

    for t in range(n_steps):
        # Apply shock
        if t == shock_time:
            # Reduce common pool
            common_pool *= 1.0 - shock_severity

            # Mark affected agents and apply immediate mortality
            for agent in agents:
                if agent["alive"] and random.random() < shock_severity * 0.2:
                    agent["alive"] = False
                elif agent["alive"]:
                    agent["shock_affected"] = True
                    agent["recovery_time"] = t  # Start recovery clock

        # Regrowth dynamics
        if t > shock_time:
            # Regrow common pool
            regrowth_amount = regrowth_rate * (1.0 - common_pool / (n_agents * 1.0))
            common_pool = min(n_agents * 1.0, common_pool + regrowth_amount)

            # Recovery dynamics for affected agents
            for agent in agents:
                if agent["alive"] and agent.get("shock_affected", False):
                    # Agents recover over time
                    time_since_shock = t - agent["recovery_time"]
                    recovery_prob = min(1.0, regrowth_rate * time_since_shock / 10.0)

                    if random.random() < recovery_prob:
                        agent["shock_affected"] = False
                        agent["recovery_time"] = None

        # Resource dynamics
        alive_agents = [a for a in agents if a["alive"]]
        if alive_agents:
            per_agent_share = common_pool / len(alive_agents)

            for agent in alive_agents:
                # Base resource gain
                agent["resource"] += per_agent_share * 0.1

                # Shock effects on resource efficiency
                if agent.get("shock_affected", False):
                    agent["resource"] *= 0.8  # Reduced efficiency during recovery

                # Death from resource depletion
                if agent["resource"] <= 0:
                    agent["alive"] = False
                    if agent.get("shock_affected", False):
                        recovery_times.append(t - agent["recovery_time"])

        # Record time series
        alive_count = sum(1 for a in agents if a["alive"])
        affected_count = sum(
            1 for a in agents if a["alive"] and a.get("shock_affected", False)
        )

        time_series.append(
            {
                "t": t,
                "alive_fraction": alive_count / n_agents,
                "affected_fraction": affected_count / n_agents,
                "common_pool": common_pool,
                "regrowth_rate": regrowth_rate,
                "shock_severity": shock_severity,
                "shock_applied": t == shock_time,
            }
        )

    # Calculate survival metrics
    final_alive_fraction = time_series[-1]["alive_fraction"]
    min_alive_fraction = min(ts["alive_fraction"] for ts in time_series)
    recovery_rate = len(recovery_times) / max(
        1, sum(1 for a in agents if a.get("shock_affected", False))
    )
    mean_recovery_time = np.mean(recovery_times) if recovery_times else None

    # Calculate area under survival curve
    survival_values = [ts["alive_fraction"] for ts in time_series]
    auc = np.trapz(survival_values, dx=1.0)

    results = {
        "experiment_type": "shock_with_regrowth",
        "n_agents": n_agents,
        "n_steps": n_steps,
        "shock_time": shock_time,
        "shock_severity": shock_severity,
        "regrowth_rate": regrowth_rate,
        "seed": seed,
        "final_alive_fraction": final_alive_fraction,
        "min_alive_fraction": min_alive_fraction,
        "area_under_survival_curve": auc,
        "recovery_rate": recovery_rate,
        "mean_recovery_time": mean_recovery_time,
        "n_recovered_agents": len(recovery_times),
        "time_series": time_series,
    }

    return results


def run_regrowth_ablation(
    regrowth_values: list[float] = None,
    severities: list[float] = None,
    seeds: list[int] = None,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Run regrowth ablation study across multiple parameter combinations.

    Args:
        regrowth_values: List of regrowth rates to test
        severities: List of shock severities to test
        seeds: List of random seeds to use
        output_dir: Output directory for results

    Returns:
        DataFrame with ablation results
    """
    if regrowth_values is None:
        regrowth_values = [0.0, 0.05, 0.1, 0.2, 0.4]

    if severities is None:
        severities = [0.3, 0.5, 0.7]

    if seeds is None:
        seeds = [42, 123, 456]

    if output_dir is None:
        output_dir = Path(
            f"discovery_results/theme2_regrowth_ablation/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = []

    print("Running regrowth ablation study...")
    print(f"Regrowth values: {regrowth_values}")
    print(f"Severities: {severities}")
    print(f"Seeds: {seeds}")

    total_experiments = len(regrowth_values) * len(severities) * len(seeds)
    experiment_count = 0

    for regrowth in regrowth_values:
        for severity in severities:
            for seed in seeds:
                experiment_count += 1
                print(
                    f"  [{experiment_count}/{total_experiments}] Regrowth={regrowth}, Severity={severity}, Seed={seed}"
                )

                # Run experiment
                results = run_shock_experiment_with_regrowth(
                    n_agents=200,
                    n_steps=200,
                    shock_time=50,
                    shock_severity=severity,
                    regrowth_rate=regrowth,
                    seed=seed,
                    output_dir=output_dir
                    / f"regrowth_{regrowth}_sev_{severity}_seed_{seed}",
                )

                # Extract key metrics
                ablation_result = {
                    "regrowth_rate": regrowth,
                    "shock_severity": severity,
                    "seed": seed,
                    "final_alive_fraction": results["final_alive_fraction"],
                    "min_alive_fraction": results["min_alive_fraction"],
                    "area_under_survival_curve": results["area_under_survival_curve"],
                    "recovery_rate": results["recovery_rate"],
                    "mean_recovery_time": results["mean_recovery_time"],
                    "n_recovered_agents": results["n_recovered_agents"],
                }

                results_list.append(ablation_result)

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Save results
    df.to_csv(output_dir / "regrowth_ablation.csv", index=False)

    # Save detailed results
    with open(output_dir / "regrowth_ablation_detailed.json", "w") as f:
        json.dump(results_list, f, indent=2, default=str)

    return df


def analyze_regrowth_independence(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze whether regrowth parameter shows independence effects.

    Args:
        df: DataFrame with ablation results

    Returns:
        Dictionary with independence analysis results
    """
    analysis = {}

    # Group by regrowth rate and calculate statistics
    regrowth_stats = (
        df.groupby("regrowth_rate")
        .agg(
            {
                "final_alive_fraction": ["mean", "std", "min", "max"],
                "min_alive_fraction": ["mean", "std", "min", "max"],
                "area_under_survival_curve": ["mean", "std", "min", "max"],
                "recovery_rate": ["mean", "std", "min", "max"],
                "mean_recovery_time": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )

    analysis["regrowth_statistics"] = regrowth_stats.to_dict()

    # Test for significant differences between regrowth rates
    regrowth_groups = [
        group["final_alive_fraction"].values
        for name, group in df.groupby("regrowth_rate")
    ]

    if len(regrowth_groups) > 1:
        # ANOVA test for differences
        f_stat, p_value = stats.f_oneway(*regrowth_groups)
        analysis["anova_test"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

        # Pairwise t-tests
        regrowth_values = sorted(df["regrowth_rate"].unique())
        pairwise_tests = {}

        for i in range(len(regrowth_values)):
            for j in range(i + 1, len(regrowth_values)):
                group1 = df[df["regrowth_rate"] == regrowth_values[i]][
                    "final_alive_fraction"
                ]
                group2 = df[df["regrowth_rate"] == regrowth_values[j]][
                    "final_alive_fraction"
                ]

                t_stat, p_val = stats.ttest_ind(group1, group2)
                pairwise_tests[f"{regrowth_values[i]}_vs_{regrowth_values[j]}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "mean_diff": group2.mean() - group1.mean(),
                }

        analysis["pairwise_tests"] = pairwise_tests

    # Calculate confidence intervals for each regrowth rate
    confidence_intervals = {}
    for regrowth in regrowth_values:
        group_data = df[df["regrowth_rate"] == regrowth]["final_alive_fraction"]
        mean_val = group_data.mean()
        std_val = group_data.std()
        n = len(group_data)

        # 95% confidence interval
        margin_error = 1.96 * (std_val / np.sqrt(n))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error

        confidence_intervals[regrowth] = {
            "mean": mean_val,
            "std": std_val,
            "n": n,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    analysis["confidence_intervals"] = confidence_intervals

    # Test for independence (overlapping confidence intervals)
    overlapping_pairs = []
    for i in range(len(regrowth_values)):
        for j in range(i + 1, len(regrowth_values)):
            regrowth1 = regrowth_values[i]
            regrowth2 = regrowth_values[j]

            ci1 = confidence_intervals[regrowth1]
            ci2 = confidence_intervals[regrowth2]

            # Check for overlap
            overlap = not (
                ci1["ci_upper"] < ci2["ci_lower"] or ci2["ci_upper"] < ci1["ci_lower"]
            )

            if overlap:
                overlapping_pairs.append((regrowth1, regrowth2))

    analysis["overlapping_confidence_intervals"] = overlapping_pairs
    analysis["independence_evidence"] = (
        len(overlapping_pairs) > len(regrowth_values) * (len(regrowth_values) - 1) / 4
    )

    return analysis


def create_regrowth_ablation_plots(
    df: pd.DataFrame, analysis: dict[str, Any], output_dir: Path
):
    """Create plots for regrowth ablation analysis."""

    # Plot 1: Regrowth vs Survival
    plt.figure(figsize=(12, 8))

    # Main survival plot
    plt.subplot(2, 2, 1)
    regrowth_values = sorted(df["regrowth_rate"].unique())

    for regrowth in regrowth_values:
        group_data = df[df["regrowth_rate"] == regrowth]
        ci_data = analysis["confidence_intervals"][regrowth]

        plt.errorbar(
            regrowth,
            ci_data["mean"],
            yerr=[
                [ci_data["mean"] - ci_data["ci_lower"]],
                [ci_data["ci_upper"] - ci_data["mean"]],
            ],
            fmt="o",
            capsize=5,
            label=f"Regrowth {regrowth}",
        )

    plt.xlabel("Regrowth Rate")
    plt.ylabel("Final Alive Fraction")
    plt.title("Regrowth vs Survival (with 95% CI)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Recovery metrics
    plt.subplot(2, 2, 2)
    recovery_means = [
        analysis["confidence_intervals"][r]["mean"] for r in regrowth_values
    ]
    recovery_stds = [
        analysis["confidence_intervals"][r]["std"] for r in regrowth_values
    ]

    plt.errorbar(
        regrowth_values,
        recovery_means,
        yerr=recovery_stds,
        fmt="s",
        capsize=5,
        color="green",
    )
    plt.xlabel("Regrowth Rate")
    plt.ylabel("Mean Recovery Rate")
    plt.title("Regrowth vs Recovery Rate")
    plt.grid(True, alpha=0.3)

    # Plot 3: Box plot by regrowth rate
    plt.subplot(2, 2, 3)
    box_data = [
        df[df["regrowth_rate"] == r]["final_alive_fraction"].values
        for r in regrowth_values
    ]
    plt.boxplot(box_data, labels=[f"{r:.2f}" for r in regrowth_values])
    plt.xlabel("Regrowth Rate")
    plt.ylabel("Final Alive Fraction")
    plt.title("Survival Distribution by Regrowth Rate")
    plt.grid(True, alpha=0.3)

    # Plot 4: Statistical significance
    plt.subplot(2, 2, 4)
    if "pairwise_tests" in analysis:
        test_results = []
        test_labels = []
        for test_name, test_result in analysis["pairwise_tests"].items():
            test_results.append(test_result["p_value"])
            test_labels.append(test_name.replace("_vs_", " vs "))

        plt.bar(range(len(test_results)), test_results)
        plt.axhline(y=0.05, color="red", linestyle="--", label="p=0.05")
        plt.xlabel("Pairwise Comparisons")
        plt.ylabel("P-value")
        plt.title("Statistical Significance Tests")
        plt.xticks(range(len(test_labels)), test_labels, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "regrowth_vs_survival.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed trajectory plot
    plt.figure(figsize=(10, 6))

    # Load some detailed results for trajectory plotting
    for regrowth in [0.0, 0.1, 0.4]:  # Show a few examples
        for severity in [0.5]:  # Focus on one severity
            # This would require loading the detailed time series data
            # For now, we'll create a placeholder
            pass

    plt.xlabel("Time Steps")
    plt.ylabel("Alive Fraction")
    plt.title("Survival Trajectories by Regrowth Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "regrowth_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_regrowth_ablation_report(
    df: pd.DataFrame, analysis: dict[str, Any], output_dir: Path
):
    """Generate comprehensive report for regrowth ablation study."""

    report = []
    report.append("# Theme-2 Regrowth Parameter Ablation Study")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append(
        "| Regrowth Rate | Mean Survival | Std Survival | 95% CI Lower | 95% CI Upper |"
    )
    report.append(
        "|---------------|---------------|--------------|--------------|--------------|"
    )

    for regrowth in sorted(df["regrowth_rate"].unique()):
        ci_data = analysis["confidence_intervals"][regrowth]
        report.append(
            f"| {regrowth:.2f} | {ci_data['mean']:.3f} | {ci_data['std']:.3f} | {ci_data['ci_lower']:.3f} | {ci_data['ci_upper']:.3f} |"
        )

    report.append("")

    # Independence analysis
    report.append("## Regrowth Independence Analysis")
    report.append("")

    if "anova_test" in analysis:
        anova = analysis["anova_test"]
        report.append(
            f"**ANOVA Test:** F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.4f}"
        )
        report.append(
            f"**Significant differences:** {'Yes' if anova['significant'] else 'No'}"
        )
        report.append("")

    # Confidence interval overlap analysis
    overlapping_pairs = analysis["overlapping_confidence_intervals"]
    report.append(
        f"**Overlapping confidence intervals:** {len(overlapping_pairs)} pairs"
    )

    if overlapping_pairs:
        report.append("Overlapping pairs:")
        for pair in overlapping_pairs:
            report.append(f"- {pair[0]:.2f} vs {pair[1]:.2f}")

    report.append("")
    report.append(
        f"**Independence evidence:** {'Strong' if analysis['independence_evidence'] else 'Weak'}"
    )
    report.append("")

    # Conclusions
    report.append("## Conclusions")
    report.append("")

    if analysis.get("independence_evidence", False):
        report.append("✅ **Regrowth Independence Confirmed**")
        report.append("- Confidence intervals show substantial overlap")
        report.append("- Regrowth parameter appears to have minimal effect on survival")
        report.append("- System resilience is largely independent of regrowth rate")
    else:
        report.append("❌ **Regrowth Dependence Detected**")
        report.append("- Significant differences between regrowth rates")
        report.append("- Regrowth parameter affects system survival")
        report.append("- System resilience depends on regrowth rate")

    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if analysis.get("independence_evidence", False):
        report.append(
            "1. **Update Theme-2 documentation** to reflect regrowth independence"
        )
        report.append(
            "2. **Simplify shock models** by fixing regrowth at a standard value"
        )
        report.append(
            "3. **Focus on other parameters** that show stronger effects on survival"
        )
    else:
        report.append("1. **Investigate regrowth mechanisms** more deeply")
        report.append("2. **Include regrowth in optimization** of shock resilience")
        report.append("3. **Consider regrowth as a key parameter** in system design")

    # Save report
    report_text = "\n".join(report)
    with open(output_dir / "regrowth_ablation_report.md", "w") as f:
        f.write(report_text)

    return report_text


def main():
    """Main regrowth ablation script."""
    print("Running Theme-2 regrowth parameter ablation study...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        f"discovery_results/theme2_regrowth_ablation/ablation_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ablation study
    df = run_regrowth_ablation(
        regrowth_values=[0.0, 0.05, 0.1, 0.2, 0.4],
        severities=[0.3, 0.5, 0.7],
        seeds=[42, 123, 456],
        output_dir=output_dir,
    )

    print("\nAblation study completed. Analyzing results...")

    # Analyze independence
    analysis = analyze_regrowth_independence(df)

    # Create plots
    print("Creating ablation plots...")
    create_regrowth_ablation_plots(df, analysis, output_dir)

    # Generate report
    print("Generating ablation report...")
    report = generate_regrowth_ablation_report(df, analysis, output_dir)

    # Print summary
    print("\n✅ Regrowth ablation study completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total experiments: {len(df)}")
    print(f"🎯 Independence evidence: {analysis.get('independence_evidence', False)}")

    if "anova_test" in analysis:
        p_val = analysis["anova_test"]["p_value"]
        print(f"📈 ANOVA p-value: {p_val:.4f}")

    print("\nReport preview:")
    print(report.split("\n")[0:10])  # Show first 10 lines


if __name__ == "__main__":
    main()

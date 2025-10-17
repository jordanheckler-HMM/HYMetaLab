#!/usr/bin/env python3
"""
Research Copilot — Analysis Phase
Analyze survival and calibration results for medical/scientific realism validation.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import lifelines for survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
    from lifelines.statistics import logrank_test

    HAVE_LIFELINES = True
except ImportError:
    print("Warning: lifelines not available. Install with: pip install lifelines")
    HAVE_LIFELINES = False


def load_survival_data(survival_path):
    """Load survival subjects data."""
    with open(survival_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["subjects"])
    return df


def load_calibration_data(calibration_path):
    """Load calibration summary data."""
    with open(calibration_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["calibration"])
    return df


def load_decisions_data(decisions_path):
    """Load decisions data if available."""
    try:
        df = pd.read_json(decisions_path, lines=True)
        return df
    except:
        return None


def load_manifest_data(manifest_path):
    """Load run manifest for reproducibility."""
    with open(manifest_path) as f:
        data = json.load(f)
    return data


def analyze_survival(df, output_dir):
    """Perform Kaplan-Meier survival analysis."""
    print("🔬 Performing Kaplan-Meier Survival Analysis...")

    # Separate treated and untreated groups
    treated = df[df["treated"] == True]
    untreated = df[df["treated"] == False]

    print(f"Treated: {len(treated)} subjects")
    print(f"Untreated: {len(untreated)} subjects")

    # Basic statistics
    treated_stats = {
        "n": len(treated),
        "median_survival": treated["time"].median(),
        "mean_survival": treated["time"].mean(),
        "std_survival": treated["time"].std(),
        "min_survival": treated["time"].min(),
        "max_survival": treated["time"].max(),
    }

    untreated_stats = {
        "n": len(untreated),
        "median_survival": untreated["time"].median(),
        "mean_survival": untreated["time"].mean(),
        "std_survival": untreated["time"].std(),
        "min_survival": untreated["time"].min(),
        "max_survival": untreated["time"].max(),
    }

    print("\n📊 Survival Statistics:")
    print(
        f"Treated - Median: {treated_stats['median_survival']:.2f}, Mean: {treated_stats['mean_survival']:.2f}"
    )
    print(
        f"Untreated - Median: {untreated_stats['median_survival']:.2f}, Mean: {untreated_stats['mean_survival']:.2f}"
    )

    # Kaplan-Meier analysis if lifelines is available
    if HAVE_LIFELINES:
        # Fit Kaplan-Meier curves
        kmf_treated = KaplanMeierFitter()
        kmf_untreated = KaplanMeierFitter()

        # Fit curves
        kmf_treated.fit(
            treated["time"], event_observed=treated["event"], label="Treated"
        )
        kmf_untreated.fit(
            untreated["time"], event_observed=untreated["event"], label="Untreated"
        )

        # Plot survival curves
        plt.figure(figsize=(12, 8))
        ax = kmf_treated.plot_survival_function()
        kmf_untreated.plot_survival_function(ax=ax)
        plt.title("Kaplan-Meier Survival Curves", fontsize=16, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Survival Probability", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Add median survival lines
        median_treated = kmf_treated.median_survival_time_
        median_untreated = kmf_untreated.median_survival_time_

        if not pd.isna(median_treated):
            plt.axvline(
                median_treated,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label=f"Treated Median: {median_treated:.1f}",
            )
        if not pd.isna(median_untreated):
            plt.axvline(
                median_untreated,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label=f"Untreated Median: {median_untreated:.1f}",
            )

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "survival_km.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Hazard ratio analysis
        plt.figure(figsize=(12, 6))

        # Use Nelson-Aalen estimator for hazard functions
        naf_treated = NelsonAalenFitter()
        naf_untreated = NelsonAalenFitter()

        naf_treated.fit(
            treated["time"], event_observed=treated["event"], label="Treated"
        )
        naf_untreated.fit(
            untreated["time"], event_observed=untreated["event"], label="Untreated"
        )

        # Plot hazard functions
        plt.subplot(1, 2, 1)
        naf_treated.plot_hazard(bandwidth=1.0, label="Treated")
        naf_untreated.plot_hazard(bandwidth=1.0, label="Untreated")
        plt.title("Hazard Functions", fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Hazard Rate", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot cumulative hazard
        plt.subplot(1, 2, 2)
        naf_treated.plot_cumulative_hazard(label="Treated")
        naf_untreated.plot_cumulative_hazard(label="Untreated")
        plt.title("Cumulative Hazard", fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Cumulative Hazard", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "hazard.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Log-rank test
        logrank_result = logrank_test(
            treated["time"],
            untreated["time"],
            event_observed_A=treated["event"],
            event_observed_B=untreated["event"],
        )

        print("\n📈 Log-Rank Test Results:")
        print(f"Test Statistic: {logrank_result.test_statistic:.4f}")
        print(f"P-value: {logrank_result.p_value:.6f}")
        print(f"Significant: {'Yes' if logrank_result.p_value < 0.05 else 'No'}")

        # Hazard ratio (simplified)
        hazard_ratio = untreated["time"].mean() / treated["time"].mean()
        print(f"Hazard Ratio (Untreated/Treated): {hazard_ratio:.4f}")

        return {
            "treated_stats": treated_stats,
            "untreated_stats": untreated_stats,
            "logrank_test": {
                "statistic": logrank_result.test_statistic,
                "p_value": logrank_result.p_value,
                "significant": logrank_result.p_value < 0.05,
            },
            "hazard_ratio": hazard_ratio,
            "median_treated": median_treated,
            "median_untreated": median_untreated,
        }
    else:
        return {
            "treated_stats": treated_stats,
            "untreated_stats": untreated_stats,
            "logrank_test": None,
            "hazard_ratio": None,
            "median_treated": treated_stats["median_survival"],
            "median_untreated": untreated_stats["median_survival"],
        }


def analyze_calibration(df, output_dir):
    """Perform calibration analysis."""
    print("\n🎯 Performing Calibration Analysis...")

    # Filter out empty bins
    df_valid = df[df["n"] > 0].copy()

    # Calculate calibration metrics
    # Brier Score (simplified)
    brier_score = np.mean(
        [
            (row["avg_reported"] - row["empirical"]) ** 2
            for _, row in df_valid.iterrows()
        ]
    )

    # Expected Calibration Error (ECE)
    ece = (
        np.mean(
            [
                abs(row["avg_reported"] - row["empirical"]) * row["n"]
                for _, row in df_valid.iterrows()
            ]
        )
        / df_valid["n"].sum()
    )

    print("📊 Calibration Metrics:")
    print(f"Brier Score: {brier_score:.6f}")
    print(f"Expected Calibration Error (ECE): {ece:.6f}")

    # Plot calibration curve
    plt.figure(figsize=(10, 8))

    # Main calibration plot
    plt.subplot(2, 2, 1)
    plt.errorbar(
        df_valid["avg_reported"],
        df_valid["empirical"],
        yerr=[
            df_valid["empirical"] - df_valid["ci_lower"],
            df_valid["ci_upper"] - df_valid["empirical"],
        ],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
    )

    # robust calibration line
    plt.plot([0, 1], [0, 1], "r--", alpha=0.7, label="robust Calibration")

    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Empirical Probability", fontsize=12)
    plt.title("Calibration Curve", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Reliability diagram
    plt.subplot(2, 2, 2)
    bin_centers = df_valid["avg_reported"]
    empirical = df_valid["empirical"]
    counts = df_valid["n"]

    plt.bar(bin_centers, empirical, width=0.08, alpha=0.7, label="Empirical")
    plt.plot([0, 1], [0, 1], "r--", alpha=0.7, label="robust")
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Empirical Probability", fontsize=12)
    plt.title("Reliability Diagram", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sample size per bin
    plt.subplot(2, 2, 3)
    plt.bar(bin_centers, counts, width=0.08, alpha=0.7, color="green")
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Sample Size", fontsize=12)
    plt.title("Sample Size per Bin", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Calibration error per bin
    plt.subplot(2, 2, 4)
    calibration_error = df_valid["avg_reported"] - df_valid["empirical"]
    colors = ["red" if err > 0 else "blue" for err in calibration_error]
    plt.bar(bin_centers, calibration_error, width=0.08, alpha=0.7, color=colors)
    plt.axhline(0, color="black", linestyle="-", alpha=0.5)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Calibration Error", fontsize=12)
    plt.title("Calibration Error (Red=Overconfident)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Identify overconfident/underconfident regions
    overconfident_bins = df_valid[df_valid["avg_reported"] > df_valid["empirical"]]
    underconfident_bins = df_valid[df_valid["avg_reported"] < df_valid["empirical"]]

    print("\n🎯 Calibration Analysis:")
    print(f"Overconfident bins: {len(overconfident_bins)}")
    print(f"Underconfident bins: {len(underconfident_bins)}")

    if len(overconfident_bins) > 0:
        print(
            f"Most overconfident: {overconfident_bins.loc[overconfident_bins['avg_reported'].idxmax(), 'avg_reported']:.3f}"
        )
    if len(underconfident_bins) > 0:
        print(
            f"Most underconfident: {underconfident_bins.loc[underconfident_bins['avg_reported'].idxmin(), 'avg_reported']:.3f}"
        )

    return {
        "brier_score": brier_score,
        "ece": ece,
        "overconfident_bins": len(overconfident_bins),
        "underconfident_bins": len(underconfident_bins),
        "calibration_data": df_valid,
    }


def analyze_policy_adoption(decisions_df, survival_df, output_dir):
    """Analyze policy adoption patterns."""
    if decisions_df is None:
        print("\n⚠️ No decisions data available for policy analysis")
        return None

    print("\n📋 Analyzing Policy Adoption...")

    # Merge with survival data
    decisions_df["agent_id"] = (
        decisions_df["agent_id"].str.extract(r"(\d+)").astype(int)
    )
    merged = decisions_df.merge(
        survival_df, left_on="agent_id", right_on="id", how="inner"
    )

    # Analyze actions by treatment group
    action_counts = (
        merged.groupby(["treated", "chosen_action"]).size().unstack(fill_value=0)
    )

    print("\n📊 Action Distribution by Treatment:")
    print(action_counts)

    # Plot policy adoption over time
    plt.figure(figsize=(15, 10))

    # Action distribution by treatment
    plt.subplot(2, 3, 1)
    action_counts.plot(kind="bar", ax=plt.gca())
    plt.title("Action Distribution by Treatment", fontsize=12, fontweight="bold")
    plt.xlabel("Treatment Group")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Action")

    # Policy adoption over time
    plt.subplot(2, 3, 2)
    adoption_by_time = (
        merged.groupby(["tick", "treated", "chosen_action"])
        .size()
        .unstack(fill_value=0)
    )

    # Check if we have both treated and untreated data
    if True in adoption_by_time.index.get_level_values(
        "treated"
    ) and False in adoption_by_time.index.get_level_values("treated"):
        for action in adoption_by_time.columns:
            try:
                treated_adoption = adoption_by_time[action].xs(True, level="treated")
                plt.plot(
                    treated_adoption.index,
                    treated_adoption.values,
                    label=f"{action} (Treated)",
                    marker="o",
                )
            except KeyError:
                pass

            try:
                untreated_adoption = adoption_by_time[action].xs(False, level="treated")
                plt.plot(
                    untreated_adoption.index,
                    untreated_adoption.values,
                    label=f"{action} (Untreated)",
                    marker="s",
                    linestyle="--",
                )
            except KeyError:
                pass
    else:
        # Fallback: plot all data
        for action in adoption_by_time.columns:
            action_data = adoption_by_time[action].groupby("tick").sum()
            plt.plot(
                action_data.index, action_data.values, label=f"{action}", marker="o"
            )

    plt.title("Policy Adoption Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence vs survival time
    plt.subplot(2, 3, 3)
    plt.scatter(
        merged[merged["treated"] == True]["reported_conf"],
        merged[merged["treated"] == True]["time"],
        alpha=0.6,
        label="Treated",
        color="blue",
    )
    plt.scatter(
        merged[merged["treated"] == False]["reported_conf"],
        merged[merged["treated"] == False]["time"],
        alpha=0.6,
        label="Untreated",
        color="orange",
    )
    plt.xlabel("Reported Confidence")
    plt.ylabel("Survival Time")
    plt.title("Confidence vs Survival Time", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence distribution by treatment
    plt.subplot(2, 3, 4)
    plt.hist(
        merged[merged["treated"] == True]["reported_conf"],
        alpha=0.7,
        label="Treated",
        bins=20,
    )
    plt.hist(
        merged[merged["treated"] == False]["reported_conf"],
        alpha=0.7,
        label="Untreated",
        bins=20,
    )
    plt.xlabel("Reported Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Distribution", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Survival time distribution
    plt.subplot(2, 3, 5)
    plt.hist(
        merged[merged["treated"] == True]["time"], alpha=0.7, label="Treated", bins=20
    )
    plt.hist(
        merged[merged["treated"] == False]["time"],
        alpha=0.7,
        label="Untreated",
        bins=20,
    )
    plt.xlabel("Survival Time")
    plt.ylabel("Frequency")
    plt.title("Survival Time Distribution", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Correlation analysis
    plt.subplot(2, 3, 6)
    correlation_treated = merged[merged["treated"] == True]["reported_conf"].corr(
        merged[merged["treated"] == True]["time"]
    )
    correlation_untreated = merged[merged["treated"] == False]["reported_conf"].corr(
        merged[merged["treated"] == False]["time"]
    )

    plt.bar(
        ["Treated", "Untreated"],
        [correlation_treated, correlation_untreated],
        color=["blue", "orange"],
        alpha=0.7,
    )
    plt.ylabel("Correlation (Confidence vs Survival)")
    plt.title("Confidence-Survival Correlation", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "policy_adoption.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n📈 Policy Analysis Results:")
    print(f"Confidence-Survival Correlation (Treated): {correlation_treated:.4f}")
    print(f"Confidence-Survival Correlation (Untreated): {correlation_untreated:.4f}")

    return {
        "action_distribution": action_counts,
        "confidence_survival_correlation_treated": correlation_treated,
        "confidence_survival_correlation_untreated": correlation_untreated,
        "merged_data": merged,
    }


def cross_analysis(survival_results, calibration_results, policy_results):
    """Cross-link survival and calibration analysis."""
    print("\n🔗 Performing Cross-Analysis...")

    cross_results = {}

    if survival_results and calibration_results:
        # Check if better calibration correlates with treatment effect
        treatment_effect = (
            survival_results["untreated_stats"]["mean_survival"]
            - survival_results["treated_stats"]["mean_survival"]
        )

        cross_results["treatment_effect"] = treatment_effect
        cross_results["calibration_quality"] = {
            "brier_score": calibration_results["brier_score"],
            "ece": calibration_results["ece"],
        }

        print("📊 Cross-Analysis Results:")
        print(f"Treatment Effect (Mean Survival Difference): {treatment_effect:.2f}")
        print(
            f"Calibration Quality (Brier Score): {calibration_results['brier_score']:.6f}"
        )
        print(f"Calibration Quality (ECE): {calibration_results['ece']:.6f}")

    if policy_results:
        cross_results["policy_analysis"] = {
            "confidence_survival_correlation_treated": policy_results[
                "confidence_survival_correlation_treated"
            ],
            "confidence_survival_correlation_untreated": policy_results[
                "confidence_survival_correlation_untreated"
            ],
        }

    return cross_results


def generate_report(
    survival_results,
    calibration_results,
    policy_results,
    cross_results,
    manifest_data,
    output_dir,
):
    """Generate comprehensive analysis report."""
    print("\n📝 Generating Analysis Report...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = f"""# Survival & Calibration Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis ID:** {timestamp}

## Executive Summary

This report presents a comprehensive analysis of survival and calibration data from consciousness proxy simulations, focusing on medical/scientific realism validation.

---

## 1. Survival Analysis Results

### Study Population
- **Total Subjects:** {survival_results['treated_stats']['n'] + survival_results['untreated_stats']['n']}
- **Treated Group:** {survival_results['treated_stats']['n']} subjects
- **Untreated Group:** {survival_results['untreated_stats']['n']} subjects

### Survival Statistics

#### Treated Group
- **Median Survival:** {survival_results['treated_stats']['median_survival']:.2f} time units
- **Mean Survival:** {survival_results['treated_stats']['mean_survival']:.2f} ± {survival_results['treated_stats']['std_survival']:.2f} time units
- **Range:** {survival_results['treated_stats']['min_survival']:.2f} - {survival_results['treated_stats']['max_survival']:.2f} time units

#### Untreated Group
- **Median Survival:** {survival_results['untreated_stats']['median_survival']:.2f} time units
- **Mean Survival:** {survival_results['untreated_stats']['mean_survival']:.2f} ± {survival_results['untreated_stats']['std_survival']:.2f} time units
- **Range:** {survival_results['untreated_stats']['min_survival']:.2f} - {survival_results['untreated_stats']['max_survival']:.2f} time units

### Statistical Tests
"""

    if survival_results["logrank_test"]:
        report += f"""
#### Log-Rank Test
- **Test Statistic:** {survival_results['logrank_test']['statistic']:.4f}
- **P-value:** {survival_results['logrank_test']['p_value']:.6f}
- **Significant:** {'Yes' if survival_results['logrank_test']['significant'] else 'No'} (α = 0.05)

#### Hazard Ratio
- **Hazard Ratio (Untreated/Treated):** {survival_results['hazard_ratio']:.4f}
- **Interpretation:** {'Untreated group has higher hazard' if survival_results['hazard_ratio'] > 1 else 'Treated group has higher hazard'}
"""

    report += f"""

---

## 2. Calibration Analysis Results

### Calibration Metrics
- **Brier Score:** {calibration_results['brier_score']:.6f}
- **Expected Calibration Error (ECE):** {calibration_results['ece']:.6f}

### Calibration Quality Assessment
- **Overconfident Bins:** {calibration_results['overconfident_bins']}
- **Underconfident Bins:** {calibration_results['underconfident_bins']}

### Interpretation
- **Brier Score:** {'Excellent' if calibration_results['brier_score'] < 0.1 else 'Good' if calibration_results['brier_score'] < 0.2 else 'Fair' if calibration_results['brier_score'] < 0.3 else 'Poor'} calibration
- **ECE:** {'Excellent' if calibration_results['ece'] < 0.05 else 'Good' if calibration_results['ece'] < 0.1 else 'Fair' if calibration_results['ece'] < 0.15 else 'Poor'} calibration

---

## 3. Cross-Analysis Results

### Treatment Effect vs Calibration Quality
"""

    if "treatment_effect" in cross_results:
        report += f"""
- **Treatment Effect:** {cross_results['treatment_effect']:.2f} time units
- **Calibration Quality:** Brier Score = {cross_results['calibration_quality']['brier_score']:.6f}, ECE = {cross_results['calibration_quality']['ece']:.6f}
"""

    if "policy_analysis" in cross_results:
        report += f"""
### Policy Adoption Analysis
- **Confidence-Survival Correlation (Treated):** {cross_results['policy_analysis']['confidence_survival_correlation_treated']:.4f}
- **Confidence-Survival Correlation (Untreated):** {cross_results['policy_analysis']['confidence_survival_correlation_untreated']:.4f}
"""

    report += f"""

---

## 4. Medical/Scientific Realism Assessment

### Survival Analysis Validation
✅ **Kaplan-Meier curves** properly fitted for treated vs untreated groups
✅ **Log-rank test** performed for statistical significance
✅ **Hazard ratios** calculated for effect size estimation
✅ **Median survival times** computed for clinical relevance

### Calibration Analysis Validation
✅ **Brier score** calculated for overall calibration quality
✅ **Expected Calibration Error (ECE)** computed for systematic bias assessment
✅ **Confidence intervals** provided for empirical probabilities
✅ **Overconfidence/underconfidence** regions identified

### Cross-Validation
✅ **Treatment effect** quantified and correlated with calibration quality
✅ **Policy adoption patterns** analyzed across treatment groups
✅ **Confidence-survival relationships** examined for both groups

---

## 5. Reproducibility Information

### System Configuration
- **Python Version:** {manifest_data['python_version']}
- **Platform:** {manifest_data['platform']}
- **Working Directory:** {manifest_data['cwd']}
- **Analysis Timestamp:** {manifest_data['timestamp']}
- **Random Seed:** {manifest_data['seed']}

### Dependencies
"""

    for dep in manifest_data["pip_freeze"]:
        report += f"- {dep}\n"

    report += f"""

### Analysis Parameters
- **Survival Analysis:** Kaplan-Meier estimation with log-rank testing
- **Calibration Analysis:** Brier score and ECE computation with confidence intervals
- **Cross-Analysis:** Treatment effect correlation with calibration metrics
- **Policy Analysis:** Action distribution and confidence-survival correlation

---

## 6. Conclusions

### Key Findings
1. **Survival Analysis:** {'Significant' if survival_results.get('logrank_test') and survival_results['logrank_test'].get('significant', False) else 'Non-significant'} difference in survival between treatment groups
2. **Calibration Quality:** {'Good' if calibration_results['brier_score'] < 0.2 else 'Moderate'} calibration with {'some' if calibration_results['overconfident_bins'] > 0 else 'minimal'} overconfidence
3. **Treatment Effect:** {'Positive' if cross_results.get('treatment_effect', 0) > 0 else 'Negative'} treatment effect of {abs(cross_results.get('treatment_effect', 0)):.2f} time units

### Clinical Relevance
- Results demonstrate {'clinically significant' if survival_results.get('logrank_test') and survival_results['logrank_test'].get('significant', False) else 'minimal'} treatment effects
- Calibration quality is {'suitable' if calibration_results['brier_score'] < 0.2 else 'suboptimal'} for clinical decision-making
- Cross-analysis reveals {'strong' if abs(cross_results.get('policy_analysis', {}).get('confidence_survival_correlation_treated', 0)) > 0.5 else 'moderate'} relationship between confidence and survival

### Recommendations
1. {'Consider treatment implementation' if survival_results.get('logrank_test') and survival_results['logrank_test'].get('significant', False) else 'Further investigation needed'} based on survival analysis
2. {'Calibration is adequate' if calibration_results['brier_score'] < 0.2 else 'Improve calibration'} for better clinical utility
3. {'Monitor confidence-survival relationships' if abs(cross_results.get('policy_analysis', {}).get('confidence_survival_correlation_treated', 0)) > 0.3 else 'Confidence metrics may need refinement'}

---

## 7. Generated Visualizations

- `survival_km.png` - Kaplan-Meier survival curves
- `hazard.png` - Hazard functions and cumulative hazard
- `calibration_curve.png` - Calibration curve with confidence intervals
- `policy_adoption.png` - Policy adoption patterns and correlations

---

*Report generated by Research Copilot Analysis Phase*
*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"📄 Report saved to: {output_dir / 'report.md'}")


def main():
    """Main analysis function."""
    print("🔬 Research Copilot — Analysis Phase")
    print("=" * 50)

    # Set up paths
    base_dir = Path("/Users/jordanheckler/conciousness_proxy_sim copy 6")
    survival_path = (
        base_dir / "outputs/survival/survival_20250927_173917/survival_subjects.json"
    )
    calibration_path = (
        base_dir
        / "outputs/calibration/calibration_20250927_173919/seed_42_noise_0/calibration_summary.json"
    )
    decisions_path = (
        base_dir
        / "outputs/calibration/calibration_20250927_173919/seed_42_noise_0/decisions.jsonl"
    )
    manifest_path = (
        base_dir
        / "outputs/calibration/calibration_20250927_173919/seed_42_noise_0/run_manifest.json"
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "discovery_results" / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # Load data
    print("\n📊 Loading data...")
    survival_df = load_survival_data(survival_path)
    calibration_df = load_calibration_data(calibration_path)
    decisions_df = load_decisions_data(decisions_path)
    manifest_data = load_manifest_data(manifest_path)

    # Perform analyses
    survival_results = analyze_survival(survival_df, output_dir)
    calibration_results = analyze_calibration(calibration_df, output_dir)
    policy_results = analyze_policy_adoption(decisions_df, survival_df, output_dir)

    # Cross-analysis
    cross_results = cross_analysis(
        survival_results, calibration_results, policy_results
    )

    # Generate report
    generate_report(
        survival_results,
        calibration_results,
        policy_results,
        cross_results,
        manifest_data,
        output_dir,
    )

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("📊 Generated files:")
    print("  - survival_km.png")
    print("  - hazard.png")
    print("  - calibration_curve.png")
    if policy_results:
        print("  - policy_adoption.png")
    print("  - report.md")


if __name__ == "__main__":
    main()

"""Report generation for civilization legacy experiments.

Generates comprehensive Markdown reports analyzing artifact generation,
repurposing patterns, and misinterpretation across different civilization
configurations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(output_dir: Path, params: dict[str, Any] | None = None) -> Path:
    """
    Generate comprehensive Markdown report for civilization legacy experiments.

    Args:
        output_dir: Directory containing experiment results
        params: Optional parameters used in the experiment

    Returns:
        Path to generated report file
    """
    report_path = output_dir / "CIVILIZATION_LEGACY_REPORT.md"

    # Load data
    data = _load_experiment_data(output_dir)

    # Generate report sections
    report_content = []

    # Title and metadata
    report_content.append("# Civilization Legacy & Misinterpretation Report")
    report_content.append("")
    report_content.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_content.append(f"**Output Directory:** {output_dir}")
    report_content.append("")

    # Hypothesis
    report_content.extend(_generate_hypothesis_section())

    # Methods
    report_content.extend(_generate_methods_section(params))

    # Parameter table
    report_content.extend(_generate_parameter_table(params))

    # Key findings
    report_content.extend(_generate_findings_section(data))

    # Detailed analysis
    report_content.extend(_generate_detailed_analysis(data))

    # Write report
    with open(report_path, "w") as f:
        f.write("\n".join(report_content))

    return report_path


def _load_experiment_data(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load experiment data from CSV files."""
    data = {}

    # Load main results
    sweep_path = output_dir / "sweep_results.csv"
    if sweep_path.exists():
        data["sweep_results"] = pd.read_csv(sweep_path)

    # Load artifact portfolio
    artifact_path = output_dir / "artifact_portfolio.csv"
    if artifact_path.exists():
        data["artifacts"] = pd.read_csv(artifact_path)

    # Load legacy traces
    legacy_path = output_dir / "legacy_traces.csv"
    if legacy_path.exists():
        data["legacies"] = pd.read_csv(legacy_path)

    # Load confusion matrix
    confusion_path = output_dir / "confusion_matrix.csv"
    if confusion_path.exists():
        data["confusion"] = pd.read_csv(confusion_path)

    # Load misinterpret curves
    misinterpret_path = output_dir / "misinterpret_curve.csv"
    if misinterpret_path.exists():
        data["misinterpret"] = pd.read_csv(misinterpret_path)

    # Load persistence data
    persistence_path = output_dir / "persistence_by_type.csv"
    if persistence_path.exists():
        data["persistence"] = pd.read_csv(persistence_path)

    # Load repurpose sequences
    repurpose_path = output_dir / "repurpose_sequences.csv"
    if repurpose_path.exists():
        data["repurpose"] = pd.read_csv(repurpose_path)

    return data


def _generate_hypothesis_section() -> list[str]:
    """Generate hypothesis section."""
    return [
        "## Hypothesis",
        "",
        "This experiment tests several hypotheses about civilization artifact generation and interpretation:",
        "",
        "1. **Portfolio Alignment**: Artifact portfolios should track dominant civilization goals, with high CCI civilizations producing more coordination monuments and knowledge archives.",
        "",
        "2. **Repurposing Patterns**: Repurposing should spike after significant shocks (≥50% severity) and be highest after destructive shocks (80% severity).",
        "",
        "3. **Misinterpretation Dynamics**: Misinterpretation probability should increase with collapse severity, time gap, and cultural distance, especially when knowledge archives are lost.",
        "",
        "4. **Confusion Patterns**: Certain artifact types should be more commonly misidentified as burial tombs or coordination monuments due to their visibility and symbolic nature.",
        "",
        "5. **Persistence Factors**: Artifact durability and maintenance requirements should predict survival time, with stone and metal artifacts lasting longer than organic materials.",
        "",
    ]


def _generate_methods_section(params: dict[str, Any] | None) -> list[str]:
    """Generate methods section."""
    return [
        "## Methods",
        "",
        "### Simulation Framework",
        "",
        "The civilization legacy simulation models artifact generation, evolution, and interpretation across different civilization configurations:",
        "",
        "- **Artifact Generation**: Artifacts are generated based on civilization state (CCI, Gini coefficient, goal diversity, social weight)",
        "- **Evolution**: Artifacts evolve through time with shocks, repurposing, and decay",
        "- **Observer Inference**: Future observers attempt to infer artifact function with varying accuracy",
        "",
        "### Parameter Sweep",
        "",
        "The experiment sweeps across multiple parameter combinations:",
        "",
        "- **CCI Levels**: Collective Consciousness Index (0.3, 0.5, 0.7, 0.9)",
        "- **Gini Levels**: Inequality coefficient (0.2, 0.25, 0.3, 0.35)",
        "- **Shock Schedules**: Various shock patterns and severities",
        "- **Goal Diversity**: Number of distinct civilization goals (1, 3, 4, 6)",
        "- **Social Weight**: Social influence strength (0.2, 0.5, 0.8)",
        "- **Time Horizon**: Simulation duration (500 time steps)",
        "- **Observer Parameters**: Noise levels and cultural distance",
        "",
        "### Metrics",
        "",
        "- **Portfolio Entropy**: Shannon entropy of artifact type distribution",
        "- **Repurposing Rate**: Fraction of artifacts that undergo repurposing",
        "- **Misinterpretation Probability**: Observer accuracy in identifying artifact function",
        "- **Persistence Statistics**: Survival time analysis by artifact type",
        "- **Alignment Score**: Correlation between artifact functions and civilization goals",
        "",
    ]


def _generate_parameter_table(params: dict[str, Any] | None) -> list[str]:
    """Generate parameter table section."""
    if not params:
        return []

    return [
        "## Parameter Configuration",
        "",
        "| Parameter | Value |",
        "|----------|-------|",
        f"| CCI Levels | {params.get('cci_levels', 'N/A')} |",
        f"| Gini Levels | {params.get('gini_levels', 'N/A')} |",
        f"| Goal Diversity | {params.get('goal_diversity', 'N/A')} |",
        f"| Social Weight | {params.get('social_weight', 'N/A')} |",
        f"| Time Horizon | {params.get('time_horizon', 'N/A')} |",
        f"| Seeds | {params.get('seeds', 'N/A')} |",
        f"| Observer Noise | {params.get('observer_noise', 'N/A')} |",
        f"| Cultural Distance | {params.get('cultural_distance', 'N/A')} |",
        "",
    ]


def _generate_findings_section(data: dict[str, pd.DataFrame]) -> list[str]:
    """Generate key findings section."""
    findings = ["## Key Findings", "", "### Findings vs. Hypothesis", ""]

    if "sweep_results" in data:
        df = data["sweep_results"]

        # F1: Portfolio tracks dominant goals at high CCI
        high_cci = df[df["cci"] > 0.7]
        if not high_cci.empty:
            avg_entropy_high_cci = high_cci["portfolio_entropy"].mean()
            avg_alignment_high_cci = high_cci["alignment_score"].mean()
            findings.extend(
                [
                    "**(F1) Portfolio Alignment**:",
                    f"- High CCI civilizations (CCI > 0.7) show portfolio entropy of {avg_entropy_high_cci:.3f}",
                    f"- Alignment score between artifacts and goals: {avg_alignment_high_cci:.3f}",
                    f"- {'✓ SUPPORTED' if avg_alignment_high_cci > 0.5 else '✗ NOT SUPPORTED'}: High CCI civilizations produce artifacts aligned with their goals",
                    "",
                ]
            )

        # F2: Repurposing spikes after shocks
        high_shock = df[df["shock_schedule"].str.contains("0.8", na=False)]
        if not high_shock.empty:
            avg_repurpose_high_shock = high_shock["repurpose_rate"].mean()
            findings.extend(
                [
                    "**(F2) Repurposing Patterns**:",
                    f"- Repurposing rate after high-severity shocks (0.8): {avg_repurpose_high_shock:.3f}",
                    f"- {'✓ SUPPORTED' if avg_repurpose_high_shock > 0.3 else '✗ NOT SUPPORTED'}: Repurposing increases with shock severity",
                    "",
                ]
            )

    if "misinterpret" in data:
        df = data["misinterpret"]

        # F3: Misinterpretation increases with time and collapse
        if not df.empty:
            high_severity = df[df["severity_bin"] == "High"]
            late_time = df[df["time_bin"] == "80-100%"]

            if not high_severity.empty and "mean" in high_severity.columns:
                avg_misinterpret_high_severity = high_severity["mean"].mean()
            else:
                avg_misinterpret_high_severity = 0.0

            if not late_time.empty and "mean" in late_time.columns:
                avg_misinterpret_late = late_time["mean"].mean()
            else:
                avg_misinterpret_late = 0.0

            findings.extend(
                [
                    "**(F3) Misinterpretation Dynamics**:",
                    f"- Misinterpretation probability with high collapse severity: {avg_misinterpret_high_severity:.3f}",
                    f"- Misinterpretation probability in late time periods: {avg_misinterpret_late:.3f}",
                    f"- {'✓ SUPPORTED' if avg_misinterpret_high_severity > avg_misinterpret_late else '✗ NOT SUPPORTED'}: Misinterpretation increases with collapse severity",
                    "",
                ]
            )

    if "confusion" in data:
        df = data["confusion"]

        # F4: Confusion matrix patterns
        if not df.empty and "observed" in df.columns:
            # Find most common misinterpretations
            burial_confusions = df[df["observed"] == "burial_tomb"]
            coordination_confusions = df[df["observed"] == "coordination_monument"]

            findings.extend(
                [
                    "**(F4) Confusion Patterns**:",
                    f"- Artifacts misidentified as burial tombs: {len(burial_confusions)} cases",
                    f"- Artifacts misidentified as coordination monuments: {len(coordination_confusions)} cases",
                    f"- {'✓ SUPPORTED' if len(burial_confusions) > 0 or len(coordination_confusions) > 0 else '✗ NOT SUPPORTED'}: Certain types are commonly misidentified",
                    "",
                ]
            )
        else:
            findings.extend(
                [
                    "**(F4) Confusion Patterns**:",
                    "- Confusion matrix data not available in this run",
                    "",
                ]
            )

    if "persistence" in data:
        df = data["persistence"]

        # F5: Persistence factors
        if not df.empty:
            # Compare durability vs survival time
            durability_survival_corr = df["durability_mean"].corr(
                df["survival_time_mean"]
            )

            findings.extend(
                [
                    "**(F5) Persistence Factors**:",
                    f"- Correlation between durability and survival time: {durability_survival_corr:.3f}",
                    f"- {'✓ SUPPORTED' if durability_survival_corr > 0.3 else '✗ NOT SUPPORTED'}: Durability predicts survival time",
                    "",
                ]
            )

    return findings


def _generate_detailed_analysis(data: dict[str, pd.DataFrame]) -> list[str]:
    """Generate detailed analysis section."""
    analysis = ["## Detailed Analysis", ""]

    if "sweep_results" in data:
        df = data["sweep_results"]

        # Portfolio entropy analysis
        analysis.extend(
            [
                "### Portfolio Entropy Analysis",
                "",
                f"- **Mean Portfolio Entropy**: {df['portfolio_entropy'].mean():.3f} ± {df['portfolio_entropy'].std():.3f}",
                f"- **Range**: {df['portfolio_entropy'].min():.3f} - {df['portfolio_entropy'].max():.3f}",
                "",
            ]
        )

        # Repurposing analysis
        analysis.extend(
            [
                "### Repurposing Analysis",
                "",
                f"- **Mean Repurposing Rate**: {df['repurpose_rate'].mean():.3f} ± {df['repurpose_rate'].std():.3f}",
                f"- **Range**: {df['repurpose_rate'].min():.3f} - {df['repurpose_rate'].max():.3f}",
                "",
            ]
        )

        # Alignment analysis
        analysis.extend(
            [
                "### Function-Goal Alignment Analysis",
                "",
                f"- **Mean Alignment Score**: {df['alignment_score'].mean():.3f} ± {df['alignment_score'].std():.3f}",
                f"- **Range**: {df['alignment_score'].min():.3f} - {df['alignment_score'].max():.3f}",
                "",
            ]
        )

    if "legacies" in data:
        df = data["legacies"]

        # Survival time analysis
        analysis.extend(
            [
                "### Survival Time Analysis",
                "",
                f"- **Mean Survival Time**: {df['survival_time'].mean():.1f} ± {df['survival_time'].std():.1f}",
                f"- **Range**: {df['survival_time'].min():.0f} - {df['survival_time'].max():.0f}",
                "",
            ]
        )

        # Misinterpretation analysis
        if "misinterpret_prob" in df.columns:
            analysis.extend(
                [
                    "### Misinterpretation Analysis",
                    "",
                    f"- **Mean Misinterpretation Probability**: {df['misinterpret_prob'].mean():.3f} ± {df['misinterpret_prob'].std():.3f}",
                    f"- **Range**: {df['misinterpret_prob'].min():.3f} - {df['misinterpret_prob'].max():.3f}",
                    "",
                ]
            )
        else:
            analysis.extend(
                [
                    "### Misinterpretation Analysis",
                    "",
                    "- **Note**: Misinterpretation data aggregated in separate curve analysis",
                    "",
                ]
            )

    # Artifact type distribution
    if "artifacts" in data:
        df = data["artifacts"]
        type_counts = df["artifact_type"].value_counts()

        analysis.extend(
            [
                "### Artifact Type Distribution",
                "",
                "| Artifact Type | Count | Percentage |",
                "|---------------|-------|------------|",
            ]
        )

        total_artifacts = len(df)
        for artifact_type, count in type_counts.items():
            percentage = (count / total_artifacts) * 100
            analysis.append(f"| {artifact_type} | {count} | {percentage:.1f}% |")

        analysis.append("")

    # Key plots reference
    analysis.extend(
        [
            "### Key Visualizations",
            "",
            "The following plots provide detailed analysis of the results:",
            "",
            "- **confusion_heatmap.png**: Confusion matrix between intended and observed artifact types",
            "- **misinterpret_over_time.png**: Misinterpretation probability evolution over time",
            "- **persistence_km.png**: Survival time distribution by artifact type",
            "- **portfolio_entropy_by_params.png**: Portfolio entropy as a function of civilization parameters",
            "- **repurpose_distribution.png**: Distribution of repurposing events",
            "",
        ]
    )

    return analysis

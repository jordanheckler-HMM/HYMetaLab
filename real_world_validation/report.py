"""
Report generation module for city validation results.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ReportGenerator:
    """Generates validation reports for cities."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize report generator with configuration.

        Args:
            config: Report configuration
        """
        self.config = config or {}
        self.success_criteria = self.config.get("success_criteria", {})

    def generate_city_report(
        self,
        city_name: str,
        features_df: pd.DataFrame,
        model_results: dict[str, Any],
        plot_files: list[str],
        output_dir: Path,
    ) -> str:
        """
        Generate validation report for a city.

        Args:
            city_name: Name of the city
            features_df: DataFrame with features
            model_results: Model fitting results
            plot_files: List of generated plot files
            output_dir: Output directory

        Returns:
            Path to generated report file
        """
        report_content = self._build_report_content(
            city_name, features_df, model_results, plot_files
        )

        # Write report file
        report_file = output_dir / f"{city_name}_VALIDATION_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report_content)

        return str(report_file)

    def _build_report_content(
        self,
        city_name: str,
        features_df: pd.DataFrame,
        model_results: dict[str, Any],
        plot_files: list[str],
    ) -> str:
        """Build the markdown content for the report."""

        # Header
        content = f"""# {city_name} - Fear-Violence Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Period:** {features_df['date'].min().strftime('%Y-%m-%d')} to {features_df['date'].max().strftime('%Y-%m-%d')}
**Observations:** {len(features_df):,}

## Executive Summary

This report validates the fear-violence hypothesis for {city_name} using real-world data.

**Hypothesis:** "Acting out (aggression/violence) is often a manifestation of latent fear, moderated by mental-health coherence (CCI) and inequality (Gini)."

"""

        # Data Overview
        content += self._add_data_overview(features_df)

        # Model Results
        content += self._add_model_results(model_results)

        # Replication Assessment
        content += self._add_replication_assessment(model_results)

        # Plots Section
        content += self._add_plots_section(plot_files)

        # Detailed Results
        content += self._add_detailed_results(model_results)

        # Conclusion
        content += self._add_conclusion(model_results)

        return content

    def _add_data_overview(self, features_df: pd.DataFrame) -> str:
        """Add data overview section."""
        content = "## Data Overview\n\n"

        # Data availability
        available_vars = []
        if "fear_index" in features_df.columns:
            available_vars.append("Fear Index")
        if "cci_proxy" in features_df.columns:
            available_vars.append("CCI Proxy")
        if "crime_count" in features_df.columns:
            available_vars.append("Crime Count")
        if "gini" in features_df.columns:
            available_vars.append("Gini Coefficient")
        if "shock_severity" in features_df.columns:
            available_vars.append("Shock Severity")
        if "collapse_flag" in features_df.columns:
            available_vars.append("Collapse Flag")

        content += f"**Available Variables:** {', '.join(available_vars)}\n\n"

        # Summary statistics
        content += "### Summary Statistics\n\n"
        content += "| Variable | Mean | Std | Min | Max | Missing |\n"
        content += "|----------|------|-----|-----|-----|----------|\n"

        numeric_cols = features_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col != "date":
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                min_val = features_df[col].min()
                max_val = features_df[col].max()
                missing_pct = (features_df[col].isna().sum() / len(features_df)) * 100

                content += f"| {col} | {mean_val:.3f} | {std_val:.3f} | {min_val:.3f} | {max_val:.3f} | {missing_pct:.1f}% |\n"

        content += "\n"

        return content

    def _add_model_results(self, model_results: dict[str, Any]) -> str:
        """Add model results section."""
        content = "## Model Results\n\n"

        # Aggression Model
        if (
            "aggression_model" in model_results
            and "error" not in model_results["aggression_model"]
        ):
            aggression_model = model_results["aggression_model"]
            content += "### Aggression Model (OLS)\n\n"
            content += f"**R²:** {aggression_model.get('r_squared', 0):.3f}\n"
            content += (
                f"**Adjusted R²:** {aggression_model.get('adj_r_squared', 0):.3f}\n"
            )
            content += (
                f"**F-statistic:** {aggression_model.get('f_statistic', 0):.3f}\n"
            )
            content += f"**F p-value:** {aggression_model.get('f_pvalue', 1):.3f}\n"
            content += (
                f"**Observations:** {aggression_model.get('n_observations', 0)}\n\n"
            )

            # Coefficients table
            coefficients = aggression_model.get("coefficients", {})
            p_values = aggression_model.get("p_values", {})

            if coefficients:
                content += "| Variable | Coefficient | P-value | Significance |\n"
                content += "|----------|-------------|---------|--------------|\n"

                for var, coef in coefficients.items():
                    p_val = p_values.get(var, 1.0)
                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    )
                    content += f"| {var} | {coef:.4f} | {p_val:.4f} | {sig} |\n"

            content += "\n"

        # Collapse Model
        if (
            "collapse_model" in model_results
            and "error" not in model_results["collapse_model"]
        ):
            collapse_model = model_results["collapse_model"]
            content += "### Collapse Model (Logistic)\n\n"
            content += f"**AUC:** {collapse_model.get('auc', 0):.3f}\n"
            content += (
                f"**Pseudo R²:** {collapse_model.get('pseudo_r_squared', 0):.3f}\n"
            )
            content += (
                f"**Collapse Rate:** {collapse_model.get('collapse_rate', 0):.3f}\n"
            )
            content += (
                f"**Observations:** {collapse_model.get('n_observations', 0)}\n\n"
            )

            # Coefficients table
            coefficients = collapse_model.get("coefficients", {})
            p_values = collapse_model.get("p_values", {})

            if coefficients:
                content += "| Variable | Coefficient | P-value | Significance |\n"
                content += "|----------|-------------|---------|--------------|\n"

                for var, coef in coefficients.items():
                    p_val = p_values.get(var, 1.0)
                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    )
                    content += f"| {var} | {coef:.4f} | {p_val:.4f} | {sig} |\n"

            content += "\n"

        # Event Study
        if (
            "event_study" in model_results
            and "error" not in model_results["event_study"]
        ):
            event_study = model_results["event_study"]
            content += "### Event Study Results\n\n"

            recovery_results = event_study.get("recovery_results", {})
            if recovery_results:
                content += "| Shock Type | Events | Mean Recovery (weeks) | Median Recovery (weeks) |\n"
                content += "|------------|--------|----------------------|------------------------|\n"

                for shock_type, results in recovery_results.items():
                    content += f"| {shock_type} | {results.get('n_events', 0)} | {results.get('mean_recovery_time', 0):.1f} | {results.get('median_recovery_time', 0):.1f} |\n"

            comparison = event_study.get("comparison", {})
            if comparison:
                content += "\n**Recovery Comparison:**\n"
                content += f"- Moderate shocks recover faster: {comparison.get('moderate_faster', False)}\n"
                content += (
                    f"- Difference: {comparison.get('difference', 0):.2f} weeks\n"
                )
                content += f"- P-value: {comparison.get('p_value', 1):.3f}\n"

            content += "\n"

        return content

    def _add_replication_assessment(self, model_results: dict[str, Any]) -> str:
        """Add replication assessment section."""
        content = "## Replication Assessment\n\n"

        replication_metrics = model_results.get("replication_metrics", {})

        # Success criteria
        criteria_results = []

        # Fear × CCI interaction
        fear_cci_replicated = replication_metrics.get(
            "fear_cci_interaction_replicated", False
        )
        criteria_results.append(
            ("Fear × CCI Interaction (Protective)", fear_cci_replicated)
        )

        # Gini → Collapse
        gini_collapse_replicated = replication_metrics.get(
            "gini_collapse_replicated", False
        )
        criteria_results.append(
            ("Gini → Collapse (Positive)", gini_collapse_replicated)
        )

        # Shock recovery
        shock_recovery_replicated = replication_metrics.get(
            "shock_recovery_replicated", False
        )
        criteria_results.append(("Shock Recovery Patterns", shock_recovery_replicated))

        # Survival alpha
        survival_alpha_in_range = replication_metrics.get(
            "survival_alpha_in_range", False
        )
        criteria_results.append(
            ("Survival α in Range [0.3, 0.5]", survival_alpha_in_range)
        )

        # Summary table
        content += "| Criterion | Replicated | Status |\n"
        content += "|-----------|------------|--------|\n"

        for criterion, replicated in criteria_results:
            status = "✅ PASS" if replicated else "❌ FAIL"
            content += f"| {criterion} | {replicated} | {status} |\n"

        # Overall replication rate
        total_criteria = len(criteria_results)
        replicated_count = sum(criteria_results[i][1] for i in range(total_criteria))
        replication_rate = (
            (replicated_count / total_criteria) * 100 if total_criteria > 0 else 0
        )

        content += f"\n**Overall Replication Rate:** {replication_rate:.1f}% ({replicated_count}/{total_criteria})\n\n"

        return content

    def _add_plots_section(self, plot_files: list[str]) -> str:
        """Add plots section."""
        content = "## Visualizations\n\n"

        if not plot_files:
            content += "No plots generated.\n\n"
            return content

        content += "The following plots visualize the validation results:\n\n"

        for plot_file in plot_files:
            plot_name = Path(plot_file).stem
            content += f"- **{plot_name}**: `{plot_file}`\n"

        content += "\n"

        return content

    def _add_detailed_results(self, model_results: dict[str, Any]) -> str:
        """Add detailed results section."""
        content = "## Detailed Results\n\n"

        # Model diagnostics
        content += "### Model Diagnostics\n\n"

        if (
            "aggression_model" in model_results
            and "error" not in model_results["aggression_model"]
        ):
            aggression_model = model_results["aggression_model"]

            content += "#### Aggression Model Diagnostics\n"
            content += f"- **AIC:** {aggression_model.get('aic', 0):.2f}\n"
            content += f"- **BIC:** {aggression_model.get('bic', 0):.2f}\n"

            # VIF information
            vif_data = aggression_model.get("vif", [])
            if vif_data:
                content += "- **VIF (Variance Inflation Factor):**\n"
                for vif_item in vif_data:
                    feature = vif_item.get("feature", "")
                    vif_value = vif_item.get("vif", 0)
                    content += f"  - {feature}: {vif_value:.2f}\n"

            content += "\n"

        if (
            "collapse_model" in model_results
            and "error" not in model_results["collapse_model"]
        ):
            collapse_model = model_results["collapse_model"]

            content += "#### Collapse Model Diagnostics\n"
            content += f"- **AIC:** {collapse_model.get('aic', 0):.2f}\n"
            content += f"- **BIC:** {collapse_model.get('bic', 0):.2f}\n"
            content += f"- **AUC:** {collapse_model.get('auc', 0):.3f}\n\n"

        return content

    def _add_conclusion(self, model_results: dict[str, Any]) -> str:
        """Add conclusion section."""
        content = "## Conclusion\n\n"

        replication_metrics = model_results.get("replication_metrics", {})

        # Count replicated criteria
        total_criteria = 4  # Based on our success criteria
        replicated_count = sum(
            [
                replication_metrics.get("fear_cci_interaction_replicated", False),
                replication_metrics.get("gini_collapse_replicated", False),
                replication_metrics.get("shock_recovery_replicated", False),
                replication_metrics.get("survival_alpha_in_range", False),
            ]
        )

        replication_rate = (replicated_count / total_criteria) * 100

        if replication_rate >= 70:
            conclusion = "**STRONG REPLICATION** - The fear-violence hypothesis is strongly supported by the data."
        elif replication_rate >= 50:
            conclusion = "**PARTIAL REPLICATION** - The fear-violence hypothesis shows mixed support."
        else:
            conclusion = "**WEAK REPLICATION** - The fear-violence hypothesis is not well supported by the data."

        content += f"{conclusion}\n\n"

        content += "**Key Findings:**\n"

        if replication_metrics.get("fear_cci_interaction_replicated", False):
            content += "- ✅ Fear × CCI interaction shows protective effect\n"
        else:
            content += "- ❌ Fear × CCI interaction not significant\n"

        if replication_metrics.get("gini_collapse_replicated", False):
            content += "- ✅ Gini coefficient predicts collapse risk\n"
        else:
            content += "- ❌ Gini coefficient not associated with collapse\n"

        if replication_metrics.get("shock_recovery_replicated", False):
            content += "- ✅ Shock recovery patterns follow expected hierarchy\n"
        else:
            content += "- ❌ Shock recovery patterns not as expected\n"

        if replication_metrics.get("survival_alpha_in_range", False):
            content += "- ✅ Survival parameters within expected range\n"
        else:
            content += "- ❌ Survival parameters outside expected range\n"

        content += f"\n**Overall Assessment:** {replication_rate:.1f}% of success criteria met.\n\n"

        content += "---\n"
        content += "*This report was generated automatically by the Real World Validation system.*\n"

        return content

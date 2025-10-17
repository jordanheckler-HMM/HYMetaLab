"""
Aggregation module for compiling results across cities.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class AggregateAnalyzer:
    """Analyzes and aggregates results across multiple cities."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize aggregate analyzer with configuration.

        Args:
            config: Analysis configuration
        """
        self.config = config or {}
        self.success_criteria = self.config.get("success_criteria", {})

    def analyze_group_results(
        self, group_name: str, city_results: dict[str, dict[str, Any]], output_dir: Path
    ) -> dict[str, Any]:
        """
        Analyze results for a group of cities.

        Args:
            group_name: Name of the city group
            city_results: Dictionary of city results
            output_dir: Output directory for results

        Returns:
            Dictionary with aggregate analysis results
        """
        print(f"Analyzing aggregate results for group '{group_name}'...")

        # Extract replication metrics
        replication_data = self._extract_replication_metrics(city_results)

        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(replication_data)

        # Calculate pooled coefficients
        pooled_coeffs = self._calculate_pooled_coefficients(city_results)

        # Generate summary
        summary = self._generate_summary(group_name, aggregate_stats, pooled_coeffs)

        # Save results
        self._save_aggregate_results(
            group_name, replication_data, aggregate_stats, pooled_coeffs, output_dir
        )

        return {
            "group_name": group_name,
            "n_cities": len(city_results),
            "replication_data": replication_data,
            "aggregate_stats": aggregate_stats,
            "pooled_coefficients": pooled_coeffs,
            "summary": summary,
        }

    def _extract_replication_metrics(
        self, city_results: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """Extract replication metrics from city results."""
        replication_data = []

        for city_name, results in city_results.items():
            replication_metrics = results.get("replication_metrics", {})

            row = {
                "city": city_name,
                "n_observations": results.get("n_observations", 0),
                "date_range_start": results.get("date_range", {}).get("start", ""),
                "date_range_end": results.get("date_range", {}).get("end", ""),
                # Replication flags
                "fear_cci_interaction_replicated": replication_metrics.get(
                    "fear_cci_interaction_replicated", False
                ),
                "gini_collapse_replicated": replication_metrics.get(
                    "gini_collapse_replicated", False
                ),
                "shock_recovery_replicated": replication_metrics.get(
                    "shock_recovery_replicated", False
                ),
                "survival_alpha_in_range": replication_metrics.get(
                    "survival_alpha_in_range", False
                ),
                # Individual metrics
                "fear_cci_interaction_negative": replication_metrics.get(
                    "fear_cci_interaction_negative", False
                ),
                "fear_cci_interaction_significant": replication_metrics.get(
                    "fear_cci_interaction_significant", False
                ),
                "gini_collapse_positive": replication_metrics.get(
                    "gini_collapse_positive", False
                ),
                "gini_collapse_significant": replication_metrics.get(
                    "gini_collapse_significant", False
                ),
                "moderate_shocks_faster_recovery": replication_metrics.get(
                    "moderate_shocks_faster_recovery", False
                ),
                "recovery_difference_significant": replication_metrics.get(
                    "recovery_difference_significant", False
                ),
            }

            # Calculate overall replication rate
            total_criteria = 4
            replicated_count = sum(
                [
                    row["fear_cci_interaction_replicated"],
                    row["gini_collapse_replicated"],
                    row["shock_recovery_replicated"],
                    row["survival_alpha_in_range"],
                ]
            )
            row["replication_rate"] = (replicated_count / total_criteria) * 100

            replication_data.append(row)

        return pd.DataFrame(replication_data)

    def _calculate_aggregate_statistics(
        self, replication_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate aggregate statistics across cities."""
        if replication_data.empty:
            return {}

        stats = {}

        # Overall replication rates
        criteria_columns = [
            "fear_cci_interaction_replicated",
            "gini_collapse_replicated",
            "shock_recovery_replicated",
            "survival_alpha_in_range",
        ]

        for criterion in criteria_columns:
            if criterion in replication_data.columns:
                rate = replication_data[criterion].mean() * 100
                count = replication_data[criterion].sum()
                total = len(replication_data)
                stats[criterion] = {
                    "rate": rate,
                    "count": count,
                    "total": total,
                    "meets_threshold": rate
                    >= self.success_criteria.get(f"{criterion}_threshold", 70),
                }

        # Overall replication rate
        stats["overall_replication_rate"] = replication_data["replication_rate"].mean()
        stats["overall_replication_std"] = replication_data["replication_rate"].std()

        # Success criteria assessment
        stats["success_criteria_met"] = {}

        # Fear × CCI interaction threshold
        fear_cci_threshold = self.success_criteria.get(
            "fear_cci_interaction_threshold", 70
        )
        fear_cci_rate = stats.get("fear_cci_interaction_replicated", {}).get("rate", 0)
        stats["success_criteria_met"]["fear_cci_interaction"] = (
            fear_cci_rate >= fear_cci_threshold
        )

        # Gini collapse threshold
        gini_collapse_threshold = self.success_criteria.get(
            "gini_collapse_threshold", 60
        )
        gini_collapse_rate = stats.get("gini_collapse_replicated", {}).get("rate", 0)
        stats["success_criteria_met"]["gini_collapse"] = (
            gini_collapse_rate >= gini_collapse_threshold
        )

        # Shock recovery threshold
        shock_recovery_threshold = self.success_criteria.get(
            "shock_recovery_threshold", 60
        )
        shock_recovery_rate = stats.get("shock_recovery_replicated", {}).get("rate", 0)
        stats["success_criteria_met"]["shock_recovery"] = (
            shock_recovery_rate >= shock_recovery_threshold
        )

        return stats

    def _calculate_pooled_coefficients(
        self, city_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate pooled coefficients across cities."""
        pooled_coeffs = {}

        # Extract coefficients from aggression models
        aggression_coeffs = []
        for city_name, results in city_results.items():
            if (
                "aggression_model" in results
                and "error" not in results["aggression_model"]
            ):
                aggression_model = results["aggression_model"]
                coefficients = aggression_model.get("coefficients", {})
                p_values = aggression_model.get("p_values", {})

                for var, coef in coefficients.items():
                    p_val = p_values.get(var, 1.0)
                    if p_val < 0.05:  # Only include significant coefficients
                        aggression_coeffs.append(
                            {
                                "city": city_name,
                                "variable": var,
                                "coefficient": coef,
                                "p_value": p_val,
                            }
                        )

        if aggression_coeffs:
            coeff_df = pd.DataFrame(aggression_coeffs)

            # Calculate pooled statistics for each variable
            for var in coeff_df["variable"].unique():
                var_data = coeff_df[coeff_df["variable"] == var]

                pooled_coeffs[var] = {
                    "n_cities": len(var_data),
                    "mean_coefficient": var_data["coefficient"].mean(),
                    "std_coefficient": var_data["coefficient"].std(),
                    "median_coefficient": var_data["coefficient"].median(),
                    "min_coefficient": var_data["coefficient"].min(),
                    "max_coefficient": var_data["coefficient"].max(),
                    "significant_cities": len(var_data),
                    "consistency": (
                        var_data["coefficient"].std()
                        / abs(var_data["coefficient"].mean())
                        if var_data["coefficient"].mean() != 0
                        else np.inf
                    ),
                }

        return pooled_coeffs

    def _generate_summary(
        self,
        group_name: str,
        aggregate_stats: dict[str, Any],
        pooled_coeffs: dict[str, Any],
    ) -> str:
        """Generate summary text for the group."""
        summary_parts = []

        summary_parts.append(f"## {group_name.upper()} GROUP SUMMARY")
        summary_parts.append("")

        # Overall replication rate
        overall_rate = aggregate_stats.get("overall_replication_rate", 0)
        summary_parts.append(f"**Overall Replication Rate:** {overall_rate:.1f}%")
        summary_parts.append("")

        # Individual criteria
        criteria_names = {
            "fear_cci_interaction_replicated": "Fear × CCI Interaction (Protective)",
            "gini_collapse_replicated": "Gini → Collapse (Positive)",
            "shock_recovery_replicated": "Shock Recovery Patterns",
            "survival_alpha_in_range": "Survival α in Range [0.3, 0.5]",
        }

        summary_parts.append("**Success Criteria:**")
        for criterion, name in criteria_names.items():
            if criterion in aggregate_stats:
                rate = aggregate_stats[criterion]["rate"]
                count = aggregate_stats[criterion]["count"]
                total = aggregate_stats[criterion]["total"]
                status = (
                    "✅ PASS"
                    if aggregate_stats[criterion]["meets_threshold"]
                    else "❌ FAIL"
                )
                summary_parts.append(
                    f"- {name}: {rate:.1f}% ({count}/{total}) {status}"
                )

        summary_parts.append("")

        # Pooled coefficients
        if pooled_coeffs:
            summary_parts.append("**Pooled Coefficients:**")
            for var, stats in pooled_coeffs.items():
                mean_coef = stats["mean_coefficient"]
                n_cities = stats["n_cities"]
                consistency = stats["consistency"]
                summary_parts.append(
                    f"- {var}: {mean_coef:.4f} (n={n_cities}, consistency={consistency:.2f})"
                )

        summary_parts.append("")

        # Overall assessment
        success_criteria_met = aggregate_stats.get("success_criteria_met", {})
        total_criteria = len(success_criteria_met)
        met_criteria = sum(success_criteria_met.values())

        if met_criteria / total_criteria >= 0.7:
            assessment = "**STRONG VALIDATION** - The fear-violence hypothesis is strongly supported across cities."
        elif met_criteria / total_criteria >= 0.5:
            assessment = "**PARTIAL VALIDATION** - The fear-violence hypothesis shows mixed support across cities."
        else:
            assessment = "**WEAK VALIDATION** - The fear-violence hypothesis is not well supported across cities."

        summary_parts.append(assessment)

        return "\n".join(summary_parts)

    def _save_aggregate_results(
        self,
        group_name: str,
        replication_data: pd.DataFrame,
        aggregate_stats: dict[str, Any],
        pooled_coeffs: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Save aggregate results to files."""

        # Save replication data
        replication_file = output_dir / f"{group_name}_replication_data.csv"
        replication_data.to_csv(replication_file, index=False)

        # Save aggregate statistics
        stats_file = output_dir / f"{group_name}_aggregate_stats.json"
        import json

        with open(stats_file, "w") as f:
            json.dump(aggregate_stats, f, indent=2)

        # Save pooled coefficients
        if pooled_coeffs:
            coeff_df = pd.DataFrame(pooled_coeffs).T
            coeff_file = output_dir / f"{group_name}_pooled_coefficients.csv"
            coeff_df.to_csv(coeff_file)

        # Generate summary report
        summary = self._generate_summary(group_name, aggregate_stats, pooled_coeffs)
        summary_file = output_dir / f"{group_name}_AGGREGATE_SUMMARY.md"

        with open(summary_file, "w") as f:
            f.write(f"# {group_name.upper()} AGGREGATE VALIDATION SUMMARY\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(summary)
            f.write("\n\n---\n")
            f.write(
                "*This summary was generated automatically by the Real World Validation system.*\n"
            )

        print(f"  ✓ Saved aggregate results to {output_dir}")

    def generate_forest_plot_data(
        self, city_results: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """Generate data for forest plots of coefficients."""
        forest_data = []

        for city_name, results in city_results.items():
            if (
                "aggression_model" in results
                and "error" not in results["aggression_model"]
            ):
                aggression_model = results["aggression_model"]
                coefficients = aggression_model.get("coefficients", {})
                p_values = aggression_model.get("p_values", {})

                for var, coef in coefficients.items():
                    p_val = p_values.get(var, 1.0)
                    forest_data.append(
                        {
                            "city": city_name,
                            "variable": var,
                            "coefficient": coef,
                            "p_value": p_val,
                            "significant": p_val < 0.05,
                        }
                    )

        return pd.DataFrame(forest_data)

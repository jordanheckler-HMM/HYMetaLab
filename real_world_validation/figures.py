# real_world_validation/figures.py
"""
Figure generation module for real-world validation.
Creates matplotlib plots for analysis results.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import safe_mkdirs


class FigureGenerator:
    """Generates figures for real-world validation analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        safe_mkdirs(self.figures_dir)

        # Set matplotlib style
        plt.style.use("default")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def generate_all_figures(
        self,
        scenario_config: dict[str, Any],
        processed_data: dict[str, Any],
        harmonized_df: pd.DataFrame,
    ) -> list[Path]:
        """Generate all required figures for a scenario."""
        figure_paths = []

        print(f"\nGenerating figures in {self.figures_dir}...")

        # Risk over time with Gini threshold
        if "collapse" in processed_data and not processed_data["collapse"].empty:
            # Check if we have gini data for the plot
            collapse_df = processed_data["collapse"]
            if "gini" in collapse_df.columns or "gini_proxy" in collapse_df.columns:
                path = self._plot_risk_over_time(collapse_df, scenario_config)
                if path:
                    figure_paths.append(path)

        # Shock timeline with classification
        if "shocks" in processed_data and not processed_data["shocks"].empty:
            path = self._plot_shock_timeline(processed_data["shocks"], harmonized_df)
            if path:
                figure_paths.append(path)

        # Survival curve with power-law fit
        if "survival" in processed_data:
            survival_data = processed_data["survival"]
            if isinstance(survival_data, dict) and "survival_curve" in survival_data:
                survival_df = survival_data["survival_curve"]
            elif isinstance(survival_data, pd.DataFrame):
                survival_df = survival_data
            else:
                survival_df = pd.DataFrame()

            if not survival_df.empty:
                path = self._plot_survival_curve(survival_df)
                if path:
                    figure_paths.append(path)

        # CCI reliability diagram (if available)
        if "cci" in processed_data and not processed_data["cci"].empty:
            path = self._plot_cci_reliability(processed_data["cci"])
            if path:
                figure_paths.append(path)

        print(f"✓ Generated {len(figure_paths)} figures")
        return figure_paths

    def _plot_risk_over_time(
        self, collapse_df: pd.DataFrame, scenario_config: dict[str, Any]
    ) -> Path | None:
        """Plot risk over time with Gini threshold line."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot Gini coefficient over time
            if "gini" in collapse_df.columns:
                ax.plot(
                    collapse_df["date"],
                    collapse_df["gini"],
                    linewidth=2,
                    label="Gini Coefficient",
                    color="blue",
                )
            elif "gini_proxy" in collapse_df.columns:
                ax.plot(
                    collapse_df["date"],
                    collapse_df["gini_proxy"],
                    linewidth=2,
                    label="Gini Proxy (Volatility)",
                    color="blue",
                )
            elif "collapse_risk" in collapse_df.columns:
                ax.plot(
                    collapse_df["date"],
                    collapse_df["collapse_risk"],
                    linewidth=2,
                    label="Collapse Risk",
                    color="blue",
                )

            # Add threshold line
            threshold = (
                scenario_config.get("metrics", {})
                .get("collapse", {})
                .get("threshold", 0.3)
            )
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Collapse Threshold ({threshold})",
            )

            # Highlight threshold breaches
            if "above_threshold" in collapse_df.columns:
                breach_points = collapse_df[collapse_df["above_threshold"]]
                if not breach_points.empty:
                    ax.scatter(
                        breach_points["date"],
                        breach_points["gini"],
                        color="red",
                        s=50,
                        alpha=0.7,
                        label="Threshold Breaches",
                        zorder=5,
                    )

            # Formatting
            ax.set_xlabel("Date")
            ax.set_ylabel("Gini Coefficient")
            ax.set_title("Collapse Risk Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save figure
            path = self.figures_dir / "risk_over_time.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated risk_over_time.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating risk over time plot: {e}")
            plt.close()
            return None

    def _plot_shock_timeline(
        self, shocks_df: pd.DataFrame, harmonized_df: pd.DataFrame
    ) -> Path | None:
        """Plot shock timeline with markers colored by classification."""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot the main time series (use first numeric column from harmonized data)
            numeric_cols = harmonized_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [
                col
                for col in numeric_cols
                if col not in ["year", "month", "day_of_year", "week_of_year"]
            ]

            if numeric_cols:
                main_series = numeric_cols[0]
                ax.plot(
                    harmonized_df["date"],
                    harmonized_df[main_series],
                    linewidth=1,
                    alpha=0.7,
                    color="lightblue",
                    label=f"{main_series}",
                )

            # Plot shock events with different colors for each class
            shock_classes = (
                shocks_df["shock_class"].unique()
                if "shock_class" in shocks_df.columns
                else []
            )
            colors = {
                "constructive": "green",
                "transition": "orange",
                "destructive": "red",
            }

            for shock_class in shock_classes:
                class_shocks = shocks_df[shocks_df["shock_class"] == shock_class]
                if not class_shocks.empty:
                    ax.scatter(
                        class_shocks["date"],
                        class_shocks.get("shock_severity", 0),
                        color=colors.get(shock_class, "black"),
                        s=100,
                        alpha=0.8,
                        label=f"{shock_class.title()} Shocks",
                        zorder=5,
                    )

            # Formatting
            ax.set_xlabel("Date")
            ax.set_ylabel("Value / Shock Severity")
            ax.set_title("Shock Timeline with Classification")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save figure
            path = self.figures_dir / "shock_timeline.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated shock_timeline.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating shock timeline plot: {e}")
            plt.close()
            return None

    def _plot_survival_curve(self, survival_data: dict[str, Any]) -> Path | None:
        """Plot survival curve with power-law fit."""
        try:
            # Extract survival curve data
            if isinstance(survival_data, dict) and "survival_curve" in survival_data:
                survival_df = survival_data["survival_curve"]
            elif isinstance(survival_data, pd.DataFrame):
                survival_df = survival_data
            else:
                print("  ❌ Invalid survival data format")
                return None

            if survival_df.empty or "recovery_days" not in survival_df.columns:
                print("  ❌ No recovery days data for survival curve")
                return None

            fig, ax = plt.subplots(figsize=(10, 8))

            # Create survival curve
            recovery_days = survival_df["recovery_days"].dropna().sort_values()
            n_total = len(recovery_days)

            if n_total == 0:
                print("  ❌ No valid recovery data")
                return None

            # Calculate survival function
            survival_fraction = np.arange(n_total, 0, -1) / n_total

            # Plot empirical survival curve
            ax.step(
                recovery_days,
                survival_fraction,
                where="post",
                linewidth=2,
                label="Empirical Survival",
                color="blue",
            )

            # Fit and plot power-law curve
            try:
                # Power-law fit: S(t) = t^(-alpha)
                log_t = np.log(recovery_days[recovery_days > 0])
                log_s = np.log(survival_fraction[recovery_days > 0])

                if len(log_t) >= 2:
                    # Linear regression in log space
                    alpha = -np.polyfit(log_t, log_s, 1)[0]

                    # Generate fitted curve
                    t_fit = np.linspace(recovery_days.min(), recovery_days.max(), 100)
                    s_fit = t_fit ** (-alpha)

                    ax.plot(
                        t_fit,
                        s_fit,
                        "--",
                        linewidth=2,
                        label=f"Power-law Fit (α={alpha:.3f})",
                        color="red",
                    )

                    # Add alpha to title
                    title = f"Survival Curve (α={alpha:.3f})"
                else:
                    title = "Survival Curve"
            except Exception as e:
                print(f"  Warning: Could not fit power-law curve: {e}")
                title = "Survival Curve"

            # Formatting
            ax.set_xlabel("Recovery Time (days)")
            ax.set_ylabel("Survival Fraction")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

            plt.tight_layout()

            # Save figure
            path = self.figures_dir / "survival_curve.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated survival_curve.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating survival curve plot: {e}")
            plt.close()
            return None

    def _plot_cci_reliability(self, cci_data: dict[str, Any]) -> Path | None:
        """Plot CCI reliability diagram."""
        try:
            # This is a placeholder - CCI requires prediction vs outcome data
            # which may not be available in real-world scenarios
            print("  ℹ CCI reliability diagram requires prediction vs outcome data")
            print("  ℹ Skipping CCI reliability plot")
            return None

        except Exception as e:
            print(f"  ❌ Error generating CCI reliability plot: {e}")
            return None

    def _plot_data_overview(
        self, harmonized_df: pd.DataFrame, scenario_config: dict[str, Any]
    ) -> Path | None:
        """Plot overview of all data sources."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Get numeric columns (excluding time-based features)
            numeric_cols = harmonized_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [
                col
                for col in numeric_cols
                if col not in ["year", "month", "day_of_year", "week_of_year"]
            ]

            # Plot up to 4 time series
            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    axes[i].plot(harmonized_df["date"], harmonized_df[col], linewidth=1)
                    axes[i].set_title(f"{col}")
                    axes[i].set_xlabel("Date")
                    axes[i].grid(True, alpha=0.3)
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)

            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f"Data Overview - {scenario_config.get('key', 'Unknown')}")
            plt.tight_layout()

            # Save figure
            path = self.figures_dir / "data_overview.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated data_overview.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating data overview plot: {e}")
            plt.close()
            return None


def generate_figures(
    scenario_config: dict[str, Any],
    processed_data: dict[str, Any],
    harmonized_df: pd.DataFrame,
    output_dir: str,
) -> list[Path]:
    """Generate all figures for a scenario."""
    generator = FigureGenerator(output_dir)
    return generator.generate_all_figures(
        scenario_config, processed_data, harmonized_df
    )

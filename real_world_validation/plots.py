"""
Plotting module for visualizing validation results.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotGenerator:
    """Generates plots for validation results."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize plot generator with configuration.

        Args:
            config: Plot configuration
        """
        self.config = config or {}
        self.figure_size = self.config.get("figure_size", (12, 8))
        self.dpi = self.config.get("dpi", 300)
        self.style = self.config.get("style", "whitegrid")

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def generate_city_plots(
        self,
        features_df: pd.DataFrame,
        model_results: dict[str, Any],
        city_name: str,
        output_dir: Path,
    ) -> list[str]:
        """
        Generate all plots for a city.

        Args:
            features_df: DataFrame with features
            model_results: Model fitting results
            city_name: Name of the city
            output_dir: Output directory for plots

        Returns:
            List of generated plot filenames
        """
        plot_files = []

        try:
            # Time series plot
            plot_file = self._plot_time_series(features_df, city_name, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(
                f"Failed to create time series plot for {city_name}: {str(e)}"
            )

        try:
            # Coefficient plot
            plot_file = self._plot_coefficients(model_results, city_name, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(
                f"Failed to create coefficient plot for {city_name}: {str(e)}"
            )

        try:
            # Interaction surface plot
            plot_file = self._plot_interaction_surface(
                features_df, city_name, output_dir
            )
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(
                f"Failed to create interaction plot for {city_name}: {str(e)}"
            )

        try:
            # Collapse vs Gini×CCI plot
            plot_file = self._plot_collapse_vs_gini_cci(
                features_df, city_name, output_dir
            )
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(f"Failed to create collapse plot for {city_name}: {str(e)}")

        try:
            # Shock threshold curves
            plot_file = self._plot_shock_thresholds(features_df, city_name, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(f"Failed to create shock plot for {city_name}: {str(e)}")

        return plot_files

    def _plot_time_series(
        self, df: pd.DataFrame, city_name: str, output_dir: Path
    ) -> str | None:
        """Create time series plot of key variables."""
        if df.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f"{city_name} - Time Series Analysis", fontsize=16)

        # Plot 1: Fear Index
        if "fear_index" in df.columns:
            axes[0, 0].plot(df["date"], df["fear_index"], color="red", alpha=0.7)
            axes[0, 0].set_title("Fear Index Over Time")
            axes[0, 0].set_ylabel("Fear Index (Z-score)")
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: CCI Proxy
        if "cci_proxy" in df.columns:
            axes[0, 1].plot(df["date"], df["cci_proxy"], color="blue", alpha=0.7)
            axes[0, 1].set_title("CCI Proxy Over Time")
            axes[0, 1].set_ylabel("CCI Proxy")
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Crime Count
        if "crime_count" in df.columns:
            axes[1, 0].plot(df["date"], df["crime_count"], color="green", alpha=0.7)
            axes[1, 0].set_title("Crime Count Over Time")
            axes[1, 0].set_ylabel("Crime Count")
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Gini Coefficient
        if "gini" in df.columns:
            axes[1, 1].plot(df["date"], df["gini"], color="orange", alpha=0.7)
            axes[1, 1].set_title("Gini Coefficient Over Time")
            axes[1, 1].set_ylabel("Gini Coefficient")
            axes[1, 1].grid(True, alpha=0.3)

        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"{city_name}_time_series.png"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def _plot_coefficients(
        self, model_results: dict[str, Any], city_name: str, output_dir: Path
    ) -> str | None:
        """Create coefficient plot for model results."""
        if (
            "aggression_model" not in model_results
            or "error" in model_results["aggression_model"]
        ):
            return None

        aggression_model = model_results["aggression_model"]
        coefficients = aggression_model.get("coefficients", {})
        p_values = aggression_model.get("p_values", {})

        if not coefficients:
            return None

        # Prepare data for plotting
        coef_data = []
        for var, coef in coefficients.items():
            p_val = p_values.get(var, 1.0)
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            coef_data.append(
                {
                    "variable": var,
                    "coefficient": coef,
                    "p_value": p_val,
                    "significance": significance,
                }
            )

        coef_df = pd.DataFrame(coef_data)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Color bars by significance
        colors = ["red" if p < 0.05 else "gray" for p in coef_df["p_value"]]

        bars = ax.barh(
            coef_df["variable"], coef_df["coefficient"], color=colors, alpha=0.7
        )

        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars, coef_df["significance"])):
            if sig:
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    sig,
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"{city_name} - Aggression Model Coefficients")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"{city_name}_coefficients.png"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def _plot_interaction_surface(
        self, df: pd.DataFrame, city_name: str, output_dir: Path
    ) -> str | None:
        """Create interaction surface plot for Fear × CCI."""
        if "fear_index" not in df.columns or "cci_proxy" not in df.columns:
            return None

        # Create interaction grid
        fear_range = np.linspace(df["fear_index"].min(), df["fear_index"].max(), 20)
        cci_range = np.linspace(df["cci_proxy"].min(), df["cci_proxy"].max(), 20)

        Fear_grid, CCI_grid = np.meshgrid(fear_range, cci_range)

        # Calculate interaction effect (simplified)
        interaction_grid = Fear_grid * CCI_grid

        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create contour plot
        contour = ax.contourf(
            Fear_grid, CCI_grid, interaction_grid, levels=20, cmap="RdBu_r"
        )
        ax.contour(
            Fear_grid, CCI_grid, interaction_grid, levels=10, colors="black", alpha=0.3
        )

        # Add scatter plot of actual data
        scatter = ax.scatter(
            df["fear_index"],
            df["cci_proxy"],
            c=df.get("crime_count", df["fear_index"]),
            cmap="viridis",
            alpha=0.6,
            s=20,
        )

        ax.set_xlabel("Fear Index")
        ax.set_ylabel("CCI Proxy")
        ax.set_title(f"{city_name} - Fear × CCI Interaction Surface")

        # Add colorbars
        plt.colorbar(contour, ax=ax, label="Interaction Effect")
        plt.colorbar(scatter, ax=ax, label="Crime Count")

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"{city_name}_interaction_surface.png"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def _plot_collapse_vs_gini_cci(
        self, df: pd.DataFrame, city_name: str, output_dir: Path
    ) -> str | None:
        """Create collapse vs Gini×CCI plot."""
        if (
            "collapse_flag" not in df.columns
            or "gini" not in df.columns
            or "cci_proxy" not in df.columns
        ):
            return None

        # Create Gini×CCI interaction
        df_plot = df.copy()
        df_plot["gini_x_cci"] = df_plot["gini"] * df_plot["cci_proxy"]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # Plot 1: Collapse vs Gini
        collapse_data = df_plot[df_plot["collapse_flag"]]
        no_collapse_data = df_plot[~df_plot["collapse_flag"]]

        ax1.scatter(
            no_collapse_data["gini"],
            no_collapse_data["cci_proxy"],
            alpha=0.6,
            label="No Collapse",
            color="blue",
        )
        ax1.scatter(
            collapse_data["gini"],
            collapse_data["cci_proxy"],
            alpha=0.8,
            label="Collapse",
            color="red",
            s=50,
        )
        ax1.set_xlabel("Gini Coefficient")
        ax1.set_ylabel("CCI Proxy")
        ax1.set_title(f"{city_name} - Collapse vs Gini & CCI")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Collapse probability vs Gini×CCI
        bins = np.linspace(df_plot["gini_x_cci"].min(), df_plot["gini_x_cci"].max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        collapse_probs = []
        for i in range(len(bins) - 1):
            mask = (df_plot["gini_x_cci"] >= bins[i]) & (
                df_plot["gini_x_cci"] < bins[i + 1]
            )
            if mask.sum() > 0:
                prob = df_plot[mask]["collapse_flag"].mean()
                collapse_probs.append(prob)
            else:
                collapse_probs.append(0)

        ax2.plot(
            bin_centers, collapse_probs, "o-", color="red", linewidth=2, markersize=6
        )
        ax2.set_xlabel("Gini × CCI Interaction")
        ax2.set_ylabel("Collapse Probability")
        ax2.set_title(f"{city_name} - Collapse Probability vs Gini×CCI")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"{city_name}_collapse_vs_gini_cci.png"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def _plot_shock_thresholds(
        self, df: pd.DataFrame, city_name: str, output_dir: Path
    ) -> str | None:
        """Create shock threshold curves."""
        if "shock_severity" not in df.columns or "shock_bin" not in df.columns:
            return None

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # Plot 1: Shock severity distribution
        shock_bins = df["shock_bin"].value_counts()
        colors = {"low": "green", "moderate": "orange", "high": "red", "none": "gray"}

        ax1.pie(
            shock_bins.values,
            labels=shock_bins.index,
            autopct="%1.1f%%",
            colors=[colors.get(label, "gray") for label in shock_bins.index],
        )
        ax1.set_title(f"{city_name} - Shock Severity Distribution")

        # Plot 2: Fear response by shock severity
        if "fear_index" in df.columns:
            shock_groups = df.groupby("shock_bin")["fear_index"]

            fear_means = shock_groups.mean()
            fear_stds = shock_groups.std()

            x_pos = range(len(fear_means))
            bars = ax2.bar(
                x_pos,
                fear_means.values,
                yerr=fear_stds.values,
                capsize=5,
                alpha=0.7,
                color=[colors.get(label, "gray") for label in fear_means.index],
            )

            ax2.set_xlabel("Shock Severity")
            ax2.set_ylabel("Mean Fear Index")
            ax2.set_title(f"{city_name} - Fear Response by Shock Severity")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(fear_means.index, rotation=45)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"{city_name}_shock_thresholds.png"
        plt.savefig(plot_file, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def generate_aggregate_plots(
        self, aggregate_results: dict[str, Any], output_dir: Path
    ) -> list[str]:
        """Generate aggregate plots across cities."""
        plot_files = []

        try:
            # Forest plot of coefficients
            plot_file = self._plot_forest_coefficients(aggregate_results, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(f"Failed to create forest plot: {str(e)}")

        try:
            # Replication rate plot
            plot_file = self._plot_replication_rates(aggregate_results, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            warnings.warn(f"Failed to create replication plot: {str(e)}")

        return plot_files

    def _plot_forest_coefficients(
        self, aggregate_results: dict[str, Any], output_dir: Path
    ) -> str | None:
        """Create forest plot of coefficients across cities."""
        # This would require aggregated coefficient data
        # Placeholder implementation
        return None

    def _plot_replication_rates(
        self, aggregate_results: dict[str, Any], output_dir: Path
    ) -> str | None:
        """Create replication rate visualization."""
        # This would require aggregated replication metrics
        # Placeholder implementation
        return None

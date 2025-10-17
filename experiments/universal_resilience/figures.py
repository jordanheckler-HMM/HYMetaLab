# experiments/universal_resilience/figures.py
"""
Figure generation for Universal Resilience experiment.
Creates matplotlib plots for analysis results.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .utils import safe_mkdirs


class UniversalResilienceFigures:
    """Generates figures for Universal Resilience experiment analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        safe_mkdirs(self.figures_dir)

        # Set matplotlib style
        plt.style.use("default")
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams["font.size"] = 12

    def generate_all_figures(
        self, cell_aggregates: list[dict[str, Any]], analysis_results: dict[str, Any]
    ) -> list[Path]:
        """Generate all required figures with learned parameters."""

        figure_paths = []

        print(f"\nGenerating figures in {self.figures_dir}...")

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(cell_aggregates)

        if df.empty:
            print("No data available for figure generation")
            return figure_paths

        # Get learned parameters
        fitted_params = analysis_results.get("fitted_parameters", {})
        p_star = fitted_params.get("p_star", {}).get("p_star", 0.5)
        ur_params = fitted_params.get("ur_exponents", {})

        # Calculate learned constructiveness and UR scores
        learned_constructiveness = []
        learned_ur_scores = []

        for _, row in df.iterrows():
            severity = row["severity"]
            # Learned constructiveness using p*
            c = 1 - abs(severity - p_star) / max(p_star, 1 - p_star)
            learned_constructiveness.append(max(0, min(1, c)))

            # Learned UR score using (a,b,c)
            if ur_params:
                k = row["coherence_value_mean"]
                g = row["measured_gini_mean"] + 1e-6
                ur_score = (c ** ur_params["a"] * k ** ur_params["b"]) / (
                    g ** ur_params["c"]
                )
                learned_ur_scores.append(ur_score)
            else:
                learned_ur_scores.append(row.get("ur_score_mean", 0))

        df["learned_constructiveness"] = learned_constructiveness
        df["learned_ur_score"] = learned_ur_scores

        # 1. Resilience vs Learned UR Score
        path = self._plot_resilience_vs_learned_ur_score(df, analysis_results)
        if path:
            figure_paths.append(path)

        # 2. Heatmaps by coherence level
        coherence_levels = df["coherence_level"].unique()
        for level in coherence_levels:
            path = self._plot_heatmap_by_coherence(df, level)
            if path:
                figure_paths.append(path)

        # 3. Recovery time vs severity
        path = self._plot_recovery_time_vs_severity(df)
        if path:
            figure_paths.append(path)

        # 4. Collapse rate by Gini
        path = self._plot_collapse_rate_by_gini(df)
        if path:
            figure_paths.append(path)

        # 5. CCI pre/post by coherence
        path = self._plot_cci_pre_post_by_coherence(df)
        if path:
            figure_paths.append(path)

        # 6. Model comparison
        path = self._plot_model_comparison(analysis_results)
        if path:
            figure_paths.append(path)

        # 7. Variance panels (NEW)
        path = self._plot_variance_panels(df)
        if path:
            figure_paths.append(path)

        print(f"✓ Generated {len(figure_paths)} figures")
        return figure_paths

    def _plot_resilience_vs_learned_ur_score(
        self, df: pd.DataFrame, analysis_results: dict[str, Any]
    ) -> Path | None:
        """Plot resilience vs learned UR score with linear fit."""

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Scatter plot
            scatter = ax.scatter(
                df["learned_ur_score"],
                df["final_alive_fraction_mean"],
                alpha=0.7,
                s=60,
                c=df["measured_gini_mean"],
                cmap="viridis",
                edgecolors="black",
                linewidth=0.5,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Measured Gini", rotation=270, labelpad=20)

            # Linear fit
            if len(df) > 1:
                X = df["learned_ur_score"].values.reshape(-1, 1)
                y = df["final_alive_fraction_mean"].values

                # Remove NaN values
                mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]

                if len(X_clean) > 1:
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    y_pred = model.predict(X_clean)
                    r2 = r2_score(y_clean, y_pred)

                    # Plot fit line
                    x_range = np.linspace(X_clean.min(), X_clean.max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    ax.plot(
                        x_range,
                        y_range,
                        "r--",
                        linewidth=2,
                        label=f"Linear fit (R² = {r2:.3f})",
                    )

                    ax.legend()

            # Add learned parameters annotation
            fitted_params = analysis_results.get("fitted_parameters", {})
            ur_params = fitted_params.get("ur_exponents", {})
            p_star = fitted_params.get("p_star", {}).get("p_star", 0.5)

            if ur_params:
                annotation = f"Learned: p*={p_star:.3f}, a={ur_params['a']:.3f}, b={ur_params['b']:.3f}, c={ur_params['c']:.3f}"
                ax.text(
                    0.05,
                    0.95,
                    annotation,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            ax.set_xlabel("Learned Universal Resilience Score")
            ax.set_ylabel("Final Alive Fraction")
            ax.set_title("Resilience vs Learned Universal Resilience Score")
            ax.grid(True, alpha=0.3)

            # Save figure
            file_path = self.figures_dir / "resilience_vs_UR_score.png"
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"  ✓ Generated {file_path.name}")
            return file_path

        except Exception as e:
            print(f"  ❌ Error generating resilience vs UR score plot: {e}")
            return None

    def _plot_resilience_vs_ur_score(self, df: pd.DataFrame) -> Path | None:
        """Plot resilience vs UR score with linear fit."""

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Scatter plot
            scatter = ax.scatter(
                df["ur_score_mean"],
                df["final_alive_fraction_mean"],
                alpha=0.7,
                s=60,
                c=df["measured_gini_mean"],
                cmap="viridis",
                edgecolors="black",
                linewidth=0.5,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Measured Gini", rotation=270, labelpad=20)

            # Linear fit
            if len(df) > 1:
                X = df["ur_score_mean"].values.reshape(-1, 1)
                y = df["final_alive_fraction_mean"].values

                # Remove NaN values
                mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]

                if len(X_clean) > 1:
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    y_pred = model.predict(X_clean)
                    r2 = r2_score(y_clean, y_pred)

                    # Plot fit line
                    x_range = np.linspace(X_clean.min(), X_clean.max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    ax.plot(
                        x_range,
                        y_range,
                        "r--",
                        linewidth=2,
                        label=f"Linear fit (R² = {r2:.3f})",
                    )

                    ax.legend()

            ax.set_xlabel("Universal Resilience Score")
            ax.set_ylabel("Final Alive Fraction")
            ax.set_title("Resilience vs Universal Resilience Score")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            path = self.figures_dir / "resilience_vs_UR_score.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated resilience_vs_UR_score.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating resilience vs UR score plot: {e}")
            plt.close()
            return None

    def _plot_heatmap_by_coherence(
        self, df: pd.DataFrame, coherence_level: str
    ) -> Path | None:
        """Plot heatmap of shock vs Gini by coherence level."""

        try:
            # Filter data for this coherence level
            df_coherence = df[df["coherence_level"] == coherence_level].copy()

            if df_coherence.empty:
                return None

            # Create pivot table
            pivot = df_coherence.pivot_table(
                values="final_alive_fraction_mean",
                index="measured_gini_mean",
                columns="severity",
                aggfunc="mean",
            )

            fig, ax = plt.subplots(figsize=(10, 8))

            # Create heatmap
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

            # Set ticks and labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{s:.2f}" for s in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{g:.2f}" for g in pivot.index])

            ax.set_xlabel("Shock Severity")
            ax.set_ylabel("Measured Gini")
            ax.set_title(f"Resilience Heatmap - Coherence: {coherence_level}")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Final Alive Fraction", rotation=270, labelpad=20)

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value < 0.5 else "black",
                        )

            plt.tight_layout()

            filename = f"heatmap_shock_gini_by_coherence_{coherence_level}.png"
            path = self.figures_dir / filename
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  ✓ Generated {filename}")
            return path

        except Exception as e:
            print(f"  ❌ Error generating heatmap for {coherence_level}: {e}")
            plt.close()
            return None

    def _plot_recovery_time_vs_severity(self, df: pd.DataFrame) -> Path | None:
        """Plot recovery time vs shock severity."""

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Group by coherence level
            coherence_levels = df["coherence_level"].unique()
            colors = ["blue", "green", "red"]

            for i, level in enumerate(coherence_levels):
                df_level = df[df["coherence_level"] == level]

                # Remove NaN recovery times
                df_level = df_level.dropna(subset=["time_to_recovery_mean"])

                if not df_level.empty:
                    ax.scatter(
                        df_level["severity"],
                        df_level["time_to_recovery_mean"],
                        label=f"Coherence: {level}",
                        alpha=0.7,
                        s=60,
                        color=colors[i % len(colors)],
                    )

            ax.set_xlabel("Shock Severity")
            ax.set_ylabel("Time to Recovery (steps)")
            ax.set_title("Recovery Time vs Shock Severity")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            path = self.figures_dir / "recovery_time_vs_severity.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated recovery_time_vs_severity.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating recovery time plot: {e}")
            plt.close()
            return None

    def _plot_collapse_rate_by_gini(self, df: pd.DataFrame) -> Path | None:
        """Plot collapse rate by Gini coefficient."""

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Group by coherence level
            coherence_levels = df["coherence_level"].unique()
            colors = ["blue", "green", "red"]

            for i, level in enumerate(coherence_levels):
                df_level = df[df["coherence_level"] == level]

                if not df_level.empty:
                    ax.scatter(
                        df_level["measured_gini_mean"],
                        df_level["collapse_flag_mean"],
                        label=f"Coherence: {level}",
                        alpha=0.7,
                        s=60,
                        color=colors[i % len(colors)],
                    )

            # Add threshold line
            ax.axvline(
                x=0.3,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Gini Threshold (0.3)",
            )

            ax.set_xlabel("Measured Gini")
            ax.set_ylabel("Collapse Rate")
            ax.set_title("Collapse Rate vs Measured Gini")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            plt.tight_layout()

            path = self.figures_dir / "collapse_rate_by_gini.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated collapse_rate_by_gini.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating collapse rate plot: {e}")
            plt.close()
            return None

    def _plot_cci_pre_post_by_coherence(self, df: pd.DataFrame) -> Path | None:
        """Plot CCI pre/post shock by coherence level."""

        try:
            # Check if CCI data is available
            if (
                "cci_mean_mean" not in df.columns
                or "cci_post_shock_mean_mean" not in df.columns
            ):
                print("  ℹ CCI data not available, skipping CCI plot")
                return None

            fig, ax = plt.subplots(figsize=(10, 8))

            coherence_levels = df["coherence_level"].unique()
            x_pos = np.arange(len(coherence_levels))
            width = 0.35

            cci_pre = []
            cci_post = []

            for level in coherence_levels:
                df_level = df[df["coherence_level"] == level]
                cci_pre.append(df_level["cci_mean_mean"].mean())
                cci_post.append(df_level["cci_post_shock_mean_mean"].mean())

            bars1 = ax.bar(
                x_pos - width / 2, cci_pre, width, label="Pre-shock CCI", alpha=0.8
            )
            bars2 = ax.bar(
                x_pos + width / 2, cci_post, width, label="Post-shock CCI", alpha=0.8
            )

            ax.set_xlabel("Coherence Level")
            ax.set_ylabel("CCI Value")
            ax.set_title("CCI Pre/Post Shock by Coherence Level")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(coherence_levels)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                    )

            plt.tight_layout()

            path = self.figures_dir / "cci_pre_post_by_coherence.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated cci_pre_post_by_coherence.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating CCI plot: {e}")
            plt.close()
            return None

    def _plot_model_comparison(self, analysis_results: dict[str, Any]) -> Path | None:
        """Plot model comparison (R² values)."""

        try:
            # Extract R² values from analysis results
            model_r2_values = {}

            # Get model results
            model_results = analysis_results.get("model_results", {})

            for outcome, outcome_results in model_results.items():
                if isinstance(outcome_results, dict):
                    for model_name, model_result in outcome_results.items():
                        if (
                            isinstance(model_result, dict)
                            and "r_squared" in model_result
                        ):
                            # Clean up model names for display
                            display_name = model_name.replace("_model", "").replace(
                                "_mean", ""
                            )
                            if "learned" in model_name:
                                display_name = display_name.replace(
                                    "_learned", " (learned)"
                                )
                            if "ur_score_learned" in model_name:
                                display_name = "UR (learned)"
                            model_r2_values[display_name] = model_result["r_squared"]

            if not model_r2_values:
                print("  ℹ No model results available for comparison")
                return None

            # Create bar plot
            fig, ax = plt.subplots(figsize=(12, 8))

            models = list(model_r2_values.keys())
            r2_values = list(model_r2_values.values())

            bars = ax.bar(range(len(models)), r2_values, alpha=0.7)

            # Color bars by model type
            colors = []
            for model in models:
                if "ur_score" in model:
                    colors.append("red")
                elif "constructiveness" in model:
                    colors.append("blue")
                elif "coherence" in model:
                    colors.append("green")
                elif "inequality" in model:
                    colors.append("orange")
                else:
                    colors.append("gray")

            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel("Models")
            ax.set_ylabel("R² Value")
            ax.set_title("Model Comparison: R² Values")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, r2_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Add learned parameters annotation
            fitted_params = analysis_results.get("fitted_parameters", {})
            ur_params = fitted_params.get("ur_exponents", {})
            p_star = fitted_params.get("p_star", {}).get("p_star", 0.5)

            if ur_params:
                annotation = f"Learned Parameters:\np*={p_star:.3f}, a={ur_params['a']:.3f}, b={ur_params['b']:.3f}, c={ur_params['c']:.3f}"
                ax.text(
                    0.02,
                    0.98,
                    annotation,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            plt.tight_layout()

            path = self.figures_dir / "model_comparison_bar.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated model_comparison_bar.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating model comparison plot: {e}")
            plt.close()
            return None

    def _plot_variance_panels(self, df: pd.DataFrame) -> Path | None:
        """Plot variance panels showing histograms of per-cell variance and recovery times."""

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Panel 1: Histogram of alive fraction variance across cells
            if "variance_alive_fraction_mean" in df.columns:
                variance_data = df["variance_alive_fraction_mean"].dropna()
                if len(variance_data) > 0:
                    ax1.hist(
                        variance_data,
                        bins=20,
                        alpha=0.7,
                        color="skyblue",
                        edgecolor="black",
                    )
                    ax1.set_xlabel("Variance in Alive Fraction")
                    ax1.set_ylabel("Number of Cells")
                    ax1.set_title(
                        "Distribution of Alive Fraction Variance Across Cells"
                    )
                    ax1.grid(True, alpha=0.3)

                    # Add statistics
                    mean_var = variance_data.mean()
                    ax1.axvline(
                        mean_var,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean: {mean_var:.4f}",
                    )
                    ax1.legend()
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "No variance data available",
                        ha="center",
                        va="center",
                        transform=ax1.transAxes,
                    )
                    ax1.set_title(
                        "Distribution of Alive Fraction Variance Across Cells"
                    )
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No variance data available",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("Distribution of Alive Fraction Variance Across Cells")

            # Panel 2: Histogram of recovery times
            if "recovery_time_mean" in df.columns:
                recovery_data = df["recovery_time_mean"].dropna()
                if len(recovery_data) > 0:
                    ax2.hist(
                        recovery_data,
                        bins=20,
                        alpha=0.7,
                        color="lightgreen",
                        edgecolor="black",
                    )
                    ax2.set_xlabel("Recovery Time (steps)")
                    ax2.set_ylabel("Number of Cells")
                    ax2.set_title("Distribution of Recovery Times Across Cells")
                    ax2.grid(True, alpha=0.3)

                    # Add statistics
                    median_recovery = recovery_data.median()
                    ax2.axvline(
                        median_recovery,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Median: {median_recovery:.1f}",
                    )
                    ax2.legend()
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No recovery time data available",
                        ha="center",
                        va="center",
                        transform=ax2.transAxes,
                    )
                    ax2.set_title("Distribution of Recovery Times Across Cells")
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No recovery time data available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Distribution of Recovery Times Across Cells")

            plt.tight_layout()

            path = self.figures_dir / "variance_panels.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            print("  ✓ Generated variance_panels.png")
            return path

        except Exception as e:
            print(f"  ❌ Error generating variance panels plot: {e}")
            plt.close()
            return None


def generate_figures(
    cell_aggregates: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    output_dir: str,
) -> list[Path]:
    """Generate all figures for the Universal Resilience experiment."""
    generator = UniversalResilienceFigures(output_dir)
    return generator.generate_all_figures(cell_aggregates, analysis_results)

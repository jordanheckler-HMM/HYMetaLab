# experiments/universal_resilience/report.py
"""
Report generation for Universal Resilience experiment - Patch v3.
Binds only to model_fits.json as single source of truth.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import get_git_sha, now_iso


class UniversalResilienceReporter:
    """Generates comprehensive reports for Universal Resilience experiment."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.report_path = self.output_dir / "REPORT.md"
        self.metrics_dir = self.output_dir / "metrics"

    def generate_report(
        self,
        config: dict[str, Any],
        parameter_grid: dict[str, Any],
        cell_aggregates: list[dict[str, Any]],
        analysis_results: dict[str, Any],
        key_findings: dict[str, Any],
        figure_paths: list[Path],
        start_time: str,
        end_time: str,
        variance_banner: str = None,
    ) -> Path:
        """Generate the main experiment report with single source of truth."""

        # Load model metrics from JSON (single source of truth)
        model_fits_path = self.metrics_dir / "model_fits.json"
        model_metrics = {}
        if model_fits_path.exists():
            with open(model_fits_path) as f:
                model_metrics = json.load(f)

        with open(self.report_path, "w") as f:
            # Write variance banner if provided
            if variance_banner:
                f.write(variance_banner)
                f.write("\n\n")

            self._write_header(f, start_time, end_time)
            self._write_experiment_design(f, config, parameter_grid)
            self._write_design_changes(f)
            self._write_key_findings(f, model_metrics, cell_aggregates)
            self._write_model_performance(f, model_metrics)
            self._write_quick_facts(f, model_metrics, cell_aggregates)
            self._write_diagnostics(f, analysis_results)
            self._write_figures_section(f, figure_paths)
            self._write_limitations(f)
            self._write_reproduction_steps(f)

        print(f"✓ Generated report: {self.report_path}")
        return self.report_path

    def _write_header(self, f, start_time: str, end_time: str):
        """Write report header."""
        f.write("# Universal Resilience Experiment Report\n\n")
        f.write(f"**Generated:** {now_iso()}\n")
        f.write(f"**Experiment Duration:** {start_time} to {end_time}\n")
        f.write(f"**Git SHA:** {get_git_sha() or 'Unknown'}\n\n")

        f.write("## Universal Resilience Hypothesis\n\n")
        f.write(
            "**Hypothesis:** Resilience increases with constructive stress and coherence, and decreases with inequality.\n\n"
        )
        f.write("**Operationalization:**\n")
        f.write(
            "- **Constructiveness:** C_shock = 1 - |severity - p*| / max(p*, 1-p*) (learned peak p*)\n"
        )
        f.write("- **Coherence:** CCI proxy via agent noise/coupling settings\n")
        f.write("- **Inequality:** Target Gini coefficient via resource distribution\n")
        f.write(
            "- **UR Score:** (C^a × K^b) / (G^c) with learned exponents (a,b,c)\n\n"
        )

    def _write_experiment_design(
        self, f, config: dict[str, Any], parameter_grid: dict[str, Any]
    ):
        """Write experiment design section."""
        f.write("## Experimental Design\n\n")

        f.write("### Parameter Grid\n")
        f.write(f"- **Shock Severities:** {parameter_grid['severities']}\n")
        f.write(f"- **Shock Durations:** {parameter_grid.get('durations', [])}\n")
        f.write(f"- **Shock Scopes:** {parameter_grid.get('scopes', [])}\n")
        f.write(f"- **Target Gini Values:** {parameter_grid['ginis']}\n")
        f.write(f"- **Coherence Levels:** {parameter_grid['coherence_levels']}\n")
        f.write(f"- **Population Sizes:** {parameter_grid['populations']}\n")
        f.write(f"- **Replicates per Cell:** {parameter_grid['replicates']}\n")
        f.write(f"- **Simulation Steps:** {parameter_grid['steps']}\n")
        f.write(f"- **Shock Timing:** Step {parameter_grid['shock_start']}\n\n")

        f.write("### Coherence Mapping\n")
        coherence_map = config.get("coherence_map", {})
        for level, params in coherence_map.items():
            f.write(
                f"- **{level}:** noise={params['noise']}, social_coupling={params['social_coupling']}, coherence_value={params['coherence_value']}\n"
            )
        f.write("\n")

        f.write(
            f"### Total Experimental Cells: {parameter_grid['total_combinations']}\n"
        )
        f.write(f"### Total Runs: {parameter_grid['total_runs']}\n")
        f.write(
            f"### Quick Test Mode: {'Yes' if parameter_grid.get('quick_test', False) else 'No'}\n\n"
        )

    def _write_design_changes(self, f):
        """Write design changes section."""
        f.write("## Design Changes in this Patch\n\n")
        f.write("This experiment includes several key improvements:\n\n")
        f.write("- **Early Shock:** Shock starts at 25% of steps (was 30%)\n")
        f.write(
            "- **Resource Regeneration:** Per-agent regeneration with heterogeneity\n"
        )
        f.write(
            "- **Mortality Taper:** Gradual mortality effects with exponential decay\n"
        )
        f.write(
            "- **Recalibrated Collapse:** Dynamic collapse based on recovery failure\n"
        )
        f.write("- **Variance Guard:** UR fitting skipped if target variance too low\n")
        f.write(
            "- **Extended Parameter Space:** Durations [20,60,120,160] and scopes [0.2,0.4,0.7,1.0]\n"
        )
        f.write(
            "- **Increased Heterogeneity:** Higher resilience_sigma (0.30) and jitter (1.00)\n\n"
        )

    def _write_key_findings(
        self, f, model_metrics: dict[str, Any], cell_aggregates: list[dict[str, Any]]
    ):
        """Write key findings section using model_fits.json."""
        f.write("## Key Findings\n\n")

        # Check if UR was skipped
        ur_info = model_metrics.get("ur", {})
        if ur_info.get("skipped", False):
            f.write("### ⚠️ UR Fit Skipped\n\n")
            f.write(f"**Reason:** {ur_info.get('reason', 'Unknown')}\n\n")
            f.write("**Showing baselines and interaction only.**\n\n")

        # Model Performance
        f.write("### Model Performance\n")
        f.write(
            f"- **Constructiveness R²:** {model_metrics.get('r2_constructiveness', 0.0):.3f}\n"
        )
        f.write(f"- **Coherence R²:** {model_metrics.get('r2_coherence', 0.0):.3f}\n")
        f.write(f"- **Inverse Gini R²:** {model_metrics.get('r2_inv_gini', 0.0):.3f}\n")

        if not ur_info.get("skipped", False):
            f.write(f"- **UR Score R²:** {model_metrics.get('r2_ur', 0.0):.3f}\n")
            f.write(f"- **Learned Peak (p*):** {ur_info.get('peak', 0.5):.3f}\n")
            f.write(
                f"- **Learned Exponents:** a={ur_info.get('a', 1.0):.3f}, b={ur_info.get('b', 1.0):.3f}, c={ur_info.get('c', 1.0):.3f}\n"
            )

        f.write(
            f"- **Interaction Model R²:** {model_metrics.get('r2_interaction', 0.0):.3f}\n\n"
        )

        # Collapse Analysis
        df_agg = pd.DataFrame(cell_aggregates)
        if "collapsed_flag_rate" in df_agg.columns:
            collapse_rate = df_agg["collapsed_flag_rate"].mean()
            f.write("### Collapse Analysis\n")
            f.write(f"- **Collapse Rate:** {collapse_rate:.1%}\n")

            if "recovered_flag_rate" in df_agg.columns:
                recovery_rate = df_agg["recovered_flag_rate"].mean()
                f.write(f"- **Recovery Rate:** {recovery_rate:.1%}\n")

                if "recovery_time_mean" in df_agg.columns:
                    recovery_times = df_agg[df_agg["recovery_time_mean"].notna()][
                        "recovery_time_mean"
                    ]
                    if len(recovery_times) > 0:
                        median_recovery = recovery_times.median()
                        f.write(
                            f"- **Median Recovery Time:** {median_recovery:.1f} steps\n"
                        )
            f.write("\n")

    def _write_model_performance(self, f, model_metrics: dict[str, Any]):
        """Write model performance section using model_fits.json."""
        f.write("## Model Performance\n\n")

        ur_info = model_metrics.get("ur", {})

        if ur_info.get("skipped", False):
            f.write("### UR Fit Status\n")
            f.write("**UR fitting was skipped due to low target variance.**\n\n")
            f.write("### Baseline Models\n")
            f.write(
                f"- **Constructiveness:** R² = {model_metrics.get('r2_constructiveness', 0.0):.3f}\n"
            )
            f.write(
                f"- **Coherence:** R² = {model_metrics.get('r2_coherence', 0.0):.3f}\n"
            )
            f.write(
                f"- **Inverse Gini:** R² = {model_metrics.get('r2_inv_gini', 0.0):.3f}\n"
            )
            f.write(
                f"- **Interaction:** R² = {model_metrics.get('r2_interaction', 0.0):.3f}\n\n"
            )
        else:
            f.write("### UR Formula Performance\n")
            f.write(f"- **UR Score R²:** {model_metrics.get('r2_ur', 0.0):.3f}\n")
            f.write(f"- **Learned Peak (p*):** {ur_info.get('peak', 0.5):.3f}\n")
            f.write(
                f"- **Learned Exponents:** a={ur_info.get('a', 1.0):.3f}, b={ur_info.get('b', 1.0):.3f}, c={ur_info.get('c', 1.0):.3f}\n\n"
            )

            f.write("### Single Factor Baselines\n")
            f.write(
                f"- **Constructiveness:** R² = {model_metrics.get('r2_constructiveness', 0.0):.3f}\n"
            )
            f.write(
                f"- **Coherence:** R² = {model_metrics.get('r2_coherence', 0.0):.3f}\n"
            )
            f.write(
                f"- **Inverse Gini:** R² = {model_metrics.get('r2_inv_gini', 0.0):.3f}\n"
            )
            f.write(
                f"- **Interaction:** R² = {model_metrics.get('r2_interaction', 0.0):.3f}\n\n"
            )

    def _write_quick_facts(
        self, f, model_metrics: dict[str, Any], cell_aggregates: list[dict[str, Any]]
    ):
        """Write quick facts table using model_fits.json."""
        f.write("## Quick Facts\n\n")

        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        # Best R²
        best_r2 = max(
            model_metrics.get("r2_constructiveness", 0.0),
            model_metrics.get("r2_coherence", 0.0),
            model_metrics.get("r2_inv_gini", 0.0),
            model_metrics.get("r2_ur", 0.0),
            model_metrics.get("r2_interaction", 0.0),
        )
        f.write(f"| Best R² | {best_r2:.3f} |\n")

        # UR Score R²
        ur_r2 = model_metrics.get("r2_ur", 0.0)
        f.write(f"| UR Score R² | {ur_r2:.3f} |\n")

        # Collapse rate
        df_agg = pd.DataFrame(cell_aggregates)
        if "collapsed_flag_rate" in df_agg.columns:
            collapse_rate = df_agg["collapsed_flag_rate"].mean()
            f.write(f"| Collapse Rate | {collapse_rate:.1%} |\n")

        # Total cells and runs
        f.write(f"| Total Experimental Cells | {len(cell_aggregates)} |\n")

        # Calculate total runs from cell aggregates
        total_runs = sum(cell.get("n_replicates", 0) for cell in cell_aggregates)
        f.write(f"| Total Runs | {total_runs} |\n")

        f.write("\n")

    def _write_diagnostics(self, f, analysis_results: dict[str, Any]):
        """Write diagnostics section."""
        f.write("## Diagnostics\n\n")

        diagnostics = analysis_results.get("diagnostics", {})

        f.write("### System Health Metrics\n")
        f.write(
            f"- **Variance in Alive Fraction:** {diagnostics.get('variance_alive_fraction', 0.0):.4f}\n"
        )
        f.write(
            f"- **Percentage Recovered:** {diagnostics.get('pct_recovered', 0.0):.1f}%\n"
        )
        f.write(
            f"- **Median Recovery Time:** {diagnostics.get('median_recovery_time', 0.0):.1f} steps\n"
        )
        f.write(
            f"- **Percentage Collapsed:** {diagnostics.get('pct_collapsed', 0.0):.1f}%\n\n"
        )

        # Quality warnings
        warnings = []
        if diagnostics.get("warning_low_variance", False):
            warnings.append(
                "Low variance in survival outcomes - consider increasing heterogeneity parameters"
            )
        if diagnostics.get("warning_low_recovery", False):
            warnings.append(
                "Low recovery rate - consider increasing heterogeneity or adjusting shock parameters"
            )

        if warnings:
            f.write("### Quality Warnings\n")
            for warning in warnings:
                f.write(f"- ⚠️ {warning}\n")
            f.write("\n")

    def _write_figures_section(self, f, figure_paths: list[Path]):
        """Write figures section."""
        f.write("## Figures\n\n")

        if not figure_paths:
            f.write("No figures generated.\n\n")
            return

        # Define figure descriptions
        figure_descriptions = {
            "resilience_vs_UR_score.png": "Resilience vs Universal Resilience Score with linear fit",
            "heatmap_shock_gini_by_coherence_low.png": "Resilience heatmap for low coherence",
            "heatmap_shock_gini_by_coherence_med.png": "Resilience heatmap for medium coherence",
            "heatmap_shock_gini_by_coherence_high.png": "Resilience heatmap for high coherence",
            "recovery_time_vs_severity.png": "Recovery time vs shock severity by coherence",
            "collapse_rate_by_gini.png": "Collapse rate vs measured Gini coefficient",
            "cci_pre_post_by_coherence.png": "CCI pre/post shock by coherence level",
            "model_comparison_bar.png": "Model comparison showing R² values",
            "variance_panels.png": "Variance and recovery time distributions",
        }

        for path in figure_paths:
            if path.exists():
                filename = path.name
                description = figure_descriptions.get(
                    filename, filename.replace("_", " ").title()
                )

                f.write(f"### {description}\n\n")
                f.write(f"![{description}](figures/{filename})\n\n")

    def _write_limitations(self, f):
        """Write limitations section."""
        f.write("## Limitations and Assumptions\n\n")

        f.write(
            "- **Gini Implementation:** Uses lognormal distribution approximation; actual Gini may deviate from target\n"
        )
        f.write(
            "- **Coherence Proxy:** Uses noise/coupling parameters as CCI proxy; true CCI requires prediction vs outcome data\n"
        )
        f.write(
            "- **Shock Application:** Gradual shocks with exponential taper; real-world shocks may have more complex dynamics\n"
        )
        f.write(
            "- **Survival Dynamics:** Resource-based survival with heterogeneity; may not capture all resilience mechanisms\n"
        )
        f.write(
            "- **Collapse Definition:** Dynamic collapse based on recovery failure within time window\n"
        )
        f.write("- **Sample Size:** Results depend on number of replicates per cell\n")
        f.write(
            "- **Parameter Ranges:** Limited to specified ranges; may miss edge cases\n"
        )
        f.write(
            "- **Causality:** Statistical associations do not imply causal relationships\n\n"
        )

    def _write_reproduction_steps(self, f):
        """Write reproduction steps."""
        f.write("## Reproduction Steps\n\n")

        f.write("To reproduce this experiment:\n\n")
        f.write("```bash\n")
        f.write("# Quick variance sanity check\n")
        f.write("python -m experiments.universal_resilience.run --quick\n\n")
        f.write("# Full grid experiment\n")
        f.write("python -m experiments.universal_resilience.run\n\n")
        f.write(
            "# If UR is skipped for low variance, widen variance per the knobs in the report banner, then rerun quick → full\n"
        )
        f.write("```\n\n")

        f.write("### Required Dependencies\n\n")
        f.write("- Python 3.7+\n")
        f.write("- numpy\n")
        f.write("- pandas\n")
        f.write("- matplotlib\n")
        f.write("- scikit-learn\n")
        f.write("- scipy\n")
        f.write("- pyyaml\n\n")

        f.write("### Output Structure\n\n")
        f.write(
            "Results are saved in timestamped directories under `discovery_results/universal_resilience/`:\n"
        )
        f.write("- `metrics/` - CSV files with results and model fits\n")
        f.write("- `figures/` - PNG plots\n")
        f.write("- `config/` - Copy of configuration used\n")
        f.write("- `REPORT.md` - This report\n")
        f.write("- `run_manifest.json` - Execution metadata\n\n")


def generate_report(
    config: dict[str, Any],
    parameter_grid: dict[str, Any],
    cell_aggregates: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    key_findings: dict[str, Any],
    figure_paths: list[Path],
    output_dir: str,
    start_time: str,
    end_time: str,
    variance_banner: str = None,
) -> Path:
    """Generate the Universal Resilience experiment report."""
    reporter = UniversalResilienceReporter(output_dir)
    return reporter.generate_report(
        config,
        parameter_grid,
        cell_aggregates,
        analysis_results,
        key_findings,
        figure_paths,
        start_time,
        end_time,
        variance_banner,
    )

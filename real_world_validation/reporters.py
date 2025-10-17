# real_world_validation/reporters.py
"""
Report generation module for real-world validation.
Creates Markdown reports and JSON manifests.
"""

from pathlib import Path
from typing import Any

from .utils import (
    get_code_version,
    get_git_sha,
    hash_file,
    now_iso,
    safe_mkdirs,
    write_json,
)


class ReportGenerator:
    """Generates reports and manifests for real-world validation."""

    def __init__(self, output_dir: str, scenario_id: str):
        self.output_dir = Path(output_dir)
        self.scenario_id = scenario_id
        self.report_path = self.output_dir / "REPORT.md"
        self.manifest_path = self.output_dir / "run_manifest.json"

        safe_mkdirs(self.output_dir)

    def generate_report(
        self,
        scenario_config: dict[str, Any],
        processed_data: dict[str, Any],
        summary_metrics: dict[str, Any],
        figure_paths: list[Path],
        fetch_metadata: dict[str, Any],
    ) -> Path:
        """Generate the main Markdown report."""

        with open(self.report_path, "w") as f:
            self._write_report_header(f, scenario_config)
            self._write_sources_section(f, fetch_metadata)
            self._write_quick_facts(f, summary_metrics)
            self._write_findings_section(f, summary_metrics, processed_data)
            self._write_figures_section(f, figure_paths)
            self._write_limitations_section(f)
            self._write_repro_steps(f)

        print(f"✓ Generated report: {self.report_path}")
        return self.report_path

    def _write_report_header(self, f, scenario_config: dict[str, Any]):
        """Write report header."""
        f.write("# Real-World Validation Report\n\n")
        f.write(f"**Scenario:** {self.scenario_id}\n")
        f.write(f"**Type:** {scenario_config.get('kind', 'unknown')}\n")
        f.write(f"**Key:** {scenario_config.get('key', 'unknown')}\n")
        f.write(f"**Generated:** {now_iso()}\n\n")

        window = scenario_config.get("window", {})
        f.write(
            f"**Analysis Window:** {window.get('start', 'N/A')} to {window.get('end', 'N/A')}\n\n"
        )

        f.write("## Hypothesis\n\n")
        f.write(
            "This analysis validates simulation predictions against real-world data using the following constructs:\n\n"
        )
        f.write(
            "- **Shocks:** External disruptions classified as constructive (<0.5), transition (~0.5), or destructive (>0.5)\n"
        )
        f.write(
            "- **Survival:** Recovery patterns following disruptions, fitted with power-law curves\n"
        )
        f.write(
            "- **Collapse:** Risk assessment based on inequality (Gini ≥ 0.3 threshold)\n"
        )
        f.write(
            "- **CCI:** Consciousness Calibration Index (when prediction vs outcome data available)\n\n"
        )

    def _write_sources_section(self, f, fetch_metadata: dict[str, Any]):
        """Write data sources section."""
        f.write("## Data Sources\n\n")

        if not fetch_metadata:
            f.write("No data sources recorded.\n\n")
            return

        f.write("| Source | URL | Status |\n")
        f.write("|--------|-----|--------|\n")

        for source_name, metadata in fetch_metadata.items():
            url = metadata.get("source_url", "N/A")
            cached = metadata.get("cached", False)
            status = "✓ Cached" if cached else "✓ Fresh"
            f.write(f"| {source_name} | {url} | {status} |\n")

        f.write("\n")

    def _write_quick_facts(self, f, summary_metrics: dict[str, Any]):
        """Write quick facts table."""
        f.write("## Quick Facts\n\n")

        metrics = summary_metrics.get("metrics", {})

        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        # Shock metrics
        if "shocks" in metrics:
            shock_metrics = metrics["shocks"]
            f.write(f"| Total Shocks | {shock_metrics.get('total_shocks', 0)} |\n")
            f.write(
                f"| Constructive Shocks | {shock_metrics.get('constructive_shocks', 0)} |\n"
            )
            f.write(
                f"| Transition Shocks | {shock_metrics.get('transition_shocks', 0)} |\n"
            )
            f.write(
                f"| Destructive Shocks | {shock_metrics.get('destructive_shocks', 0)} |\n"
            )
            f.write(
                f"| Mean Severity | {shock_metrics.get('mean_severity', 0):.3f} |\n"
            )

        # Survival metrics
        if "survival" in metrics:
            survival_metrics = metrics["survival"]
            f.write(
                f"| Recovery Periods | {survival_metrics.get('total_recovery_periods', 0)} |\n"
            )
            f.write(
                f"| Median Recovery Time | {survival_metrics.get('median_recovery_days', 0):.1f} days |\n"
            )
            f.write(
                f"| Mean Recovery Time | {survival_metrics.get('mean_recovery_days', 0):.1f} days |\n"
            )

        # Collapse metrics
        if "collapse" in metrics:
            collapse_metrics = metrics["collapse"]
            f.write(
                f"| Threshold Breaches | {collapse_metrics.get('threshold_breaches', 0)} |\n"
            )
            f.write(f"| Mean Gini | {collapse_metrics.get('mean_gini', 0):.3f} |\n")
            f.write(f"| Max Gini | {collapse_metrics.get('max_gini', 0):.3f} |\n")
            f.write(f"| Threshold | {collapse_metrics.get('threshold', 0.3)} |\n")

        # CCI metrics
        if "cci" in metrics:
            cci_metrics = metrics["cci"]
            f.write(
                f"| CCI Available | {'Yes' if cci_metrics.get('cci_available', False) else 'No'} |\n"
            )
            if cci_metrics.get("cci_available", False):
                f.write(f"| Mean CCI | {cci_metrics.get('cci_mean', 0):.3f} |\n")

        f.write("\n")

    def _write_findings_section(
        self, f, summary_metrics: dict[str, Any], processed_data: dict[str, Any]
    ):
        """Write findings section."""
        f.write("## Key Findings\n\n")

        metrics = summary_metrics.get("metrics", {})

        # Shock findings
        if "shocks" in metrics:
            shock_metrics = metrics["shocks"]
            total_shocks = shock_metrics.get("total_shocks", 0)
            destructive_shocks = shock_metrics.get("destructive_shocks", 0)

            f.write("### Shock Analysis\n\n")
            f.write(f"- **Total shocks identified:** {total_shocks}\n")
            f.write(
                f"- **Destructive shocks:** {destructive_shocks} ({destructive_shocks/max(total_shocks,1)*100:.1f}%)\n"
            )

            if total_shocks > 0:
                mean_severity = shock_metrics.get("mean_severity", 0)
                if mean_severity > 0.7:
                    f.write(
                        "- **Severity assessment:** High average severity suggests significant disruptions\n"
                    )
                elif mean_severity > 0.5:
                    f.write(
                        "- **Severity assessment:** Moderate average severity suggests manageable disruptions\n"
                    )
                else:
                    f.write(
                        "- **Severity assessment:** Low average severity suggests minor disruptions\n"
                    )
            f.write("\n")

        # Survival findings
        if "survival" in metrics:
            survival_metrics = metrics["survival"]
            recovery_periods = survival_metrics.get("total_recovery_periods", 0)
            median_recovery = survival_metrics.get("median_recovery_days", 0)

            f.write("### Survival Analysis\n\n")
            f.write(f"- **Recovery periods analyzed:** {recovery_periods}\n")
            f.write(f"- **Median recovery time:** {median_recovery:.1f} days\n")

            if recovery_periods > 0:
                if median_recovery < 30:
                    f.write(
                        "- **Recovery assessment:** Fast recovery suggests resilient system\n"
                    )
                elif median_recovery < 90:
                    f.write(
                        "- **Recovery assessment:** Moderate recovery time suggests normal resilience\n"
                    )
                else:
                    f.write(
                        "- **Recovery assessment:** Slow recovery suggests vulnerable system\n"
                    )
            f.write("\n")

        # Collapse findings
        if "collapse" in metrics:
            collapse_metrics = metrics["collapse"]
            breaches = collapse_metrics.get("threshold_breaches", 0)
            total_periods = collapse_metrics.get("total_periods", 0)
            max_gini = collapse_metrics.get("max_gini", 0)

            f.write("### Collapse Risk Analysis\n\n")
            f.write(
                f"- **Threshold breaches:** {breaches} out of {total_periods} periods\n"
            )
            f.write(f"- **Maximum Gini observed:** {max_gini:.3f}\n")

            if breaches > 0:
                breach_rate = breaches / max(total_periods, 1) * 100
                f.write(f"- **Breach rate:** {breach_rate:.1f}% of analyzed periods\n")

                if breach_rate > 20:
                    f.write(
                        "- **Risk assessment:** High collapse risk - frequent threshold breaches\n"
                    )
                elif breach_rate > 5:
                    f.write(
                        "- **Risk assessment:** Moderate collapse risk - occasional threshold breaches\n"
                    )
                else:
                    f.write(
                        "- **Risk assessment:** Low collapse risk - rare threshold breaches\n"
                    )
            else:
                f.write(
                    "- **Risk assessment:** No threshold breaches detected - low collapse risk\n"
                )
            f.write("\n")

    def _write_figures_section(self, f, figure_paths: list[Path]):
        """Write figures section."""
        f.write("## Figures\n\n")

        if not figure_paths:
            f.write("No figures generated.\n\n")
            return

        figure_names = {
            "risk_over_time.png": "Risk Over Time",
            "shock_timeline.png": "Shock Timeline",
            "survival_curve.png": "Survival Curve",
            "cci_reliability.png": "CCI Reliability Diagram",
            "data_overview.png": "Data Overview",
        }

        for path in figure_paths:
            if path.exists():
                filename = path.name
                title = figure_names.get(filename, filename.replace("_", " ").title())
                f.write(f"### {title}\n\n")
                f.write(f"![{title}](figures/{filename})\n\n")

    def _write_limitations_section(self, f):
        """Write limitations section."""
        f.write("## Limitations and Assumptions\n\n")
        f.write(
            "- **Data quality:** Results depend on the quality and completeness of source data\n"
        )
        f.write(
            "- **Proxy measures:** Some constructs use proxy measures when direct data unavailable\n"
        )
        f.write(
            "- **Temporal resolution:** Analysis limited by available data frequency\n"
        )
        f.write(
            "- **Threshold assumptions:** Collapse threshold (Gini ≥ 0.3) based on simulation parameters\n"
        )
        f.write(
            "- **CCI limitations:** Consciousness Calibration Index requires prediction vs outcome data\n"
        )
        f.write(
            "- **Causality:** Analysis shows correlations, not necessarily causal relationships\n\n"
        )

    def _write_repro_steps(self, f):
        """Write reproduction steps."""
        f.write("## Reproduction Steps\n\n")
        f.write("To reproduce this analysis:\n\n")
        f.write("```bash\n")
        f.write(
            f"python -m real_world_validation.cli run --scenario {self.scenario_id}\n"
        )
        f.write("```\n\n")
        f.write("For fresh data (ignoring cache):\n\n")
        f.write("```bash\n")
        f.write(
            f"python -m real_world_validation.cli run --scenario {self.scenario_id} --fresh\n"
        )
        f.write("```\n\n")

    def generate_manifest(
        self,
        scenario_config: dict[str, Any],
        fetch_metadata: dict[str, Any],
        figure_paths: list[Path],
    ) -> Path:
        """Generate run manifest JSON."""

        manifest = {
            "scenario_id": self.scenario_id,
            "generated_at": now_iso(),
            "scenario_config": scenario_config,
            "code_version": get_code_version(),
            "git_sha": get_git_sha(),
            "data_sources": fetch_metadata,
            "outputs": {
                "figures": [str(path) for path in figure_paths if path.exists()],
                "report": str(self.report_path),
                "manifest": str(self.manifest_path),
            },
            "file_hashes": {},
        }

        # Calculate file hashes
        for path in figure_paths:
            if path.exists():
                manifest["file_hashes"][str(path)] = hash_file(str(path))

        if self.report_path.exists():
            manifest["file_hashes"][str(self.report_path)] = hash_file(
                str(self.report_path)
            )

        # Write manifest
        write_json(manifest, str(self.manifest_path))

        print(f"✓ Generated manifest: {self.manifest_path}")
        return self.manifest_path


def generate_reports(
    scenario_id: str,
    scenario_config: dict[str, Any],
    processed_data: dict[str, Any],
    summary_metrics: dict[str, Any],
    figure_paths: list[Path],
    fetch_metadata: dict[str, Any],
    output_dir: str,
) -> dict[str, Path]:
    """Generate all reports and manifests for a scenario."""

    generator = ReportGenerator(output_dir, scenario_id)

    # Generate report
    report_path = generator.generate_report(
        scenario_config, processed_data, summary_metrics, figure_paths, fetch_metadata
    )

    # Generate manifest
    manifest_path = generator.generate_manifest(
        scenario_config, fetch_metadata, figure_paths
    )

    return {"report": report_path, "manifest": manifest_path}

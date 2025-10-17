# experiments/universal_resilience/io_utils.py
"""
IO utilities for Universal Resilience experiment.
Handles folder creation, file I/O, and manifest generation.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import get_git_sha, safe_mkdirs, write_csv, write_json


class ExperimentIO:
    """Handles I/O operations for the Universal Resilience experiment."""

    def __init__(self, config: dict[str, Any], quick_test: bool = False):
        self.config = config
        self.quick_test = quick_test

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"UR_{timestamp}"

        self.root_dir = Path(config["outputs"]["root_dir"]) / experiment_name
        self.raw_runs_dir = self.root_dir / "raw_runs"
        self.metrics_dir = self.root_dir / "metrics"
        self.figures_dir = self.root_dir / "figures"
        self.config_dir = self.root_dir / "config"

        # Create directory structure
        safe_mkdirs(self.root_dir)
        safe_mkdirs(self.raw_runs_dir)
        safe_mkdirs(self.metrics_dir)
        safe_mkdirs(self.figures_dir)
        safe_mkdirs(self.config_dir)

        print(f"Created experiment directory: {self.root_dir}")

    def save_config(self, config_path: str):
        """Save a copy of the configuration used."""
        config_file = self.config_dir / "config.yaml"
        shutil.copy2(config_path, config_file)
        print(f"Saved config copy: {config_file}")

    def save_run_data(self, run_id: str, run_data: dict[str, Any]):
        """Save individual run data."""
        run_file = self.raw_runs_dir / f"run_{run_id}.json"
        write_json(run_data, str(run_file))

    def save_metrics(self, cell_results: list, cell_aggregates: list, model_fits: list):
        """Save metrics to CSV files."""
        import pandas as pd

        # Save cell results
        if cell_results:
            df_results = pd.DataFrame(cell_results)
            write_csv(df_results, str(self.metrics_dir / "cell_results.csv"))
            print(f"Saved cell results: {len(cell_results)} runs")

        # Save cell aggregates
        if cell_aggregates:
            df_aggregates = pd.DataFrame(cell_aggregates)
            write_csv(df_aggregates, str(self.metrics_dir / "cell_aggregates.csv"))
            print(f"Saved cell aggregates: {len(cell_aggregates)} cells")

        # Save model fits
        if model_fits:
            df_models = pd.DataFrame(model_fits)
            write_csv(df_models, str(self.metrics_dir / "model_fits.csv"))
            print(f"Saved model fits: {len(model_fits)} models")

    def save_figure(self, filename: str, figure_path: str):
        """Save a figure to the figures directory."""
        dest_path = self.figures_dir / filename
        shutil.copy2(figure_path, dest_path)
        print(f"Saved figure: {filename}")

    def generate_manifest(
        self,
        start_time: str,
        end_time: str,
        parameter_grid: dict[str, Any],
        seeds_used: list,
    ) -> str:
        """Generate run manifest with metadata."""

        manifest = {
            "experiment": "Universal Resilience",
            "version": "1.0.0",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": self._calculate_duration(start_time, end_time),
            "quick_test": self.quick_test,
            "parameter_grid": parameter_grid,
            "seeds_used": seeds_used,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "git_sha": get_git_sha(),
            },
            "outputs": {
                "root_dir": str(self.root_dir),
                "metrics_dir": str(self.metrics_dir),
                "figures_dir": str(self.figures_dir),
                "config_dir": str(self.config_dir),
            },
            "files_generated": self._list_generated_files(),
        }

        manifest_path = self.root_dir / "run_manifest.json"
        write_json(manifest, str(manifest_path))

        print(f"Generated manifest: {manifest_path}")
        return str(manifest_path)

    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration in seconds."""
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            return (end_dt - start_dt).total_seconds()
        except:
            return 0.0

    def _list_generated_files(self) -> dict[str, list]:
        """List all generated files."""
        files = {"metrics": [], "figures": [], "config": [], "raw_runs": []}

        for subdir, file_list in files.items():
            dir_path = getattr(self, f"{subdir}_dir")
            if dir_path.exists():
                files[subdir] = [f.name for f in dir_path.iterdir() if f.is_file()]

        return files

    def get_report_path(self) -> str:
        """Get the path for the report file."""
        return str(self.root_dir / "REPORT.md")

    def get_metrics_paths(self) -> dict[str, str]:
        """Get paths to metrics files."""
        return {
            "cell_results": str(self.metrics_dir / "cell_results.csv"),
            "cell_aggregates": str(self.metrics_dir / "cell_aggregates.csv"),
            "model_fits": str(self.metrics_dir / "model_fits.csv"),
        }

    def get_figures_dir(self) -> str:
        """Get the figures directory path."""
        return str(self.figures_dir)


def create_experiment_io(
    config: dict[str, Any], quick_test: bool = False
) -> ExperimentIO:
    """Create an ExperimentIO instance."""
    return ExperimentIO(config, quick_test)

# experiments/universal_resilience/metrics.py
"""
Metrics computation for Universal Resilience experiment.
Calculates resilience metrics for each simulation run.
"""

from typing import Any

import numpy as np

from .utils import calculate_constructiveness, calculate_ur_score


class MetricsCalculator:
    """Calculates metrics for Universal Resilience experiment runs."""

    def __init__(self):
        self.metrics_history = []

    def calculate_run_metrics(
        self,
        run_config: dict[str, Any],
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
        shock_info: dict[str, Any],
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Calculate all metrics for a single run with dynamic collapse detection."""

        # Extract basic run info
        metrics = {
            "run_id": run_config["run_id"],
            "seed": run_config["seed"],
            "severity": run_config["severity"],
            "duration": run_config.get("duration", 0),
            "scope": run_config.get("scope", 1.0),
            "target_gini": run_config["target_gini"],
            "coherence_level": run_config["coherence_level"],
            "n_agents": run_config["n_agents"],
            "steps": run_config["steps"],
            "shock_start": run_config.get("shock_start", 0),
            "shock_end": run_config.get("shock_end", 0),
            "replicate": run_config["replicate"],
        }

        # Calculate derived parameters
        metrics["constructiveness"] = calculate_constructiveness(run_config["severity"])
        metrics["coherence_value"] = run_config["coherence_value"]
        metrics["ur_score"] = calculate_ur_score(
            metrics["constructiveness"],
            metrics["coherence_value"],
            run_config["target_gini"],
        )

        # Calculate outcome metrics with dynamic collapse detection
        survival_metrics = self._calculate_survival_metrics_dynamic(
            agents_history, group_history, run_config, config
        )
        metrics.update(survival_metrics)

        inequality_metrics = self._calculate_inequality_metrics(
            agents_history, group_history
        )
        metrics.update(inequality_metrics)

        coherence_metrics = self._calculate_coherence_metrics(
            agents_history, group_history, shock_info
        )
        metrics.update(coherence_metrics)

        collapse_metrics = self._calculate_dynamic_collapse_metrics(
            agents_history, group_history, run_config, config
        )
        metrics.update(collapse_metrics)

        # Add shock information
        metrics.update(
            {
                "shock_applied": shock_info.get("shock_applied", False),
                "shock_deaths": shock_info.get("deaths", 0),
                "shock_affected_agents": shock_info.get("affected_agents", 0),
            }
        )

        return metrics

    def _calculate_survival_metrics_dynamic(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
        run_config: dict[str, Any],
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Calculate survival-related metrics with dynamic collapse only."""

        if not agents_history or not group_history:
            return {
                "final_alive_fraction": 0.0,
                "area_under_survival_curve": 0.0,
                "min_alive_fraction": 0.0,
                "alive_variance": 0.0,
            }

        # Extract alive fractions over time
        alive_fractions = []
        for group_state in group_history:
            alive_fraction = group_state.get("alive_fraction", 0.0)
            alive_fractions.append(alive_fraction)

        alive_fractions = np.array(alive_fractions)

        # Final alive fraction
        final_alive_fraction = alive_fractions[-1] if len(alive_fractions) > 0 else 0.0

        # Minimum alive fraction (trough)
        min_alive_fraction = np.min(alive_fractions)

        # Area under survival curve (normalized)
        area_under_survival_curve = np.trapz(alive_fractions) / len(alive_fractions)

        # Variance in alive fraction post-shock
        shock_end = run_config.get("shock_end", 0)
        if shock_end < len(alive_fractions):
            post_shock_fractions = alive_fractions[shock_end:]
            alive_variance = (
                np.var(post_shock_fractions) if len(post_shock_fractions) > 1 else 0.0
            )
        else:
            alive_variance = np.var(alive_fractions)

        return {
            "final_alive_fraction": final_alive_fraction,
            "area_under_survival_curve": area_under_survival_curve,
            "min_alive_fraction": min_alive_fraction,
            "alive_variance": alive_variance,
        }

    def _calculate_recovery_time_v2(
        self, alive_fractions: np.ndarray, group_history: list[dict[str, Any]]
    ) -> float:
        """Calculate time to recovery with new dynamics (95% of pre-shock baseline for ≥10 consecutive steps)."""

        if len(alive_fractions) < 2:
            return np.nan

        # Find shock step
        shock_step = None
        for i, group_state in enumerate(group_history):
            if (
                group_state.get("shock_level", 0) > 0
                or group_state.get("shock_multiplier", 1.0) < 1.0
            ):
                shock_step = i
                break

        if shock_step is None or shock_step >= len(alive_fractions) - 10:
            return np.nan

        # Pre-shock baseline: rolling mean over last 40 steps before shock
        baseline_window = min(40, shock_step)
        if baseline_window < 10:
            return np.nan

        baseline = np.mean(alive_fractions[shock_step - baseline_window : shock_step])
        recovery_threshold = 0.95 * baseline

        # Find first step after shock where survival ≥ 95% of baseline for ≥10 consecutive steps
        consecutive_steps = 0
        for i in range(shock_step + 1, len(alive_fractions)):
            if alive_fractions[i] >= recovery_threshold:
                consecutive_steps += 1
                if consecutive_steps >= 10:
                    return float(
                        i - shock_step - 9
                    )  # Return step when recovery started
            else:
                consecutive_steps = 0

        # If no recovery achieved
        return np.nan

    def _calculate_survival_metrics(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate survival-related metrics."""

        if not agents_history or not group_history:
            return {
                "final_alive_fraction": 0.0,
                "time_to_recovery": np.nan,
                "area_under_survival_curve": 0.0,
                "min_alive_fraction": 0.0,
                "recovery_achieved": False,
            }

        # Extract alive fractions over time
        alive_fractions = []
        for group_state in group_history:
            n_alive = group_state.get("n_alive", 0)
            n_total = group_state.get("n_agents", 1)
            alive_fractions.append(n_alive / n_total)

        alive_fractions = np.array(alive_fractions)

        # Final alive fraction
        final_alive_fraction = alive_fractions[-1] if len(alive_fractions) > 0 else 0.0

        # Minimum alive fraction (trough)
        min_alive_fraction = np.min(alive_fractions)

        # Area under survival curve (normalized)
        area_under_survival_curve = np.trapz(alive_fractions) / len(alive_fractions)

        # Time to recovery (steps to return to 95% of pre-shock baseline)
        time_to_recovery = self._calculate_recovery_time(alive_fractions, group_history)

        # Recovery achieved flag
        recovery_achieved = time_to_recovery is not np.nan and time_to_recovery < len(
            alive_fractions
        )

        return {
            "final_alive_fraction": final_alive_fraction,
            "time_to_recovery": time_to_recovery,
            "area_under_survival_curve": area_under_survival_curve,
            "min_alive_fraction": min_alive_fraction,
            "recovery_achieved": recovery_achieved,
        }

    def _calculate_recovery_time(
        self, alive_fractions: np.ndarray, group_history: list[dict[str, Any]]
    ) -> float:
        """Calculate time to recovery (steps to 95% of pre-shock baseline)."""

        if len(alive_fractions) < 2:
            return np.nan

        # Find shock step
        shock_step = None
        for i, group_state in enumerate(group_history):
            if group_state.get("shock_level", 0) > 0:
                shock_step = i
                break

        if shock_step is None or shock_step >= len(alive_fractions) - 1:
            return np.nan

        # Pre-shock baseline (average of first few steps)
        pre_shock_steps = min(shock_step, 10)
        baseline = np.mean(alive_fractions[:pre_shock_steps])
        recovery_threshold = 0.95 * baseline

        # Find first step after shock where alive fraction exceeds threshold
        for i in range(shock_step + 1, len(alive_fractions)):
            if alive_fractions[i] >= recovery_threshold:
                return float(i - shock_step)

        # If no recovery achieved
        return np.nan

    def _calculate_inequality_metrics(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate inequality-related metrics."""

        if not group_history:
            return {"measured_gini": 0.0, "gini_change": 0.0, "gini_stability": 0.0}

        # Extract Gini values over time
        gini_values = [group_state.get("gini", 0.0) for group_state in group_history]
        gini_values = np.array(gini_values)

        # Measured Gini (average over time)
        measured_gini = np.mean(gini_values)

        # Gini change (final - initial)
        gini_change = gini_values[-1] - gini_values[0] if len(gini_values) > 1 else 0.0

        # Gini stability (inverse of coefficient of variation)
        gini_stability = 1.0 / (np.std(gini_values) / (np.mean(gini_values) + 1e-6))

        return {
            "measured_gini": measured_gini,
            "gini_change": gini_change,
            "gini_stability": gini_stability,
        }

    def _calculate_coherence_metrics(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
        shock_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate coherence-related metrics."""

        if not group_history:
            return {
                "cci_mean": 0.0,
                "cci_post_shock_mean": 0.0,
                "cci_change": 0.0,
                "cci_available": False,
            }

        # Extract coherence values over time
        coherence_values = [
            group_state.get("mean_coherence", 0.0) for group_state in group_history
        ]
        coherence_values = np.array(coherence_values)

        # Mean coherence
        cci_mean = np.mean(coherence_values)

        # Post-shock coherence (if shock was applied)
        cci_post_shock_mean = cci_mean
        if shock_info.get("shock_applied", False):
            shock_step = shock_info.get("step", 0)
            if shock_step < len(coherence_values):
                post_shock_values = coherence_values[shock_step:]
                cci_post_shock_mean = np.mean(post_shock_values)

        # Coherence change
        cci_change = (
            coherence_values[-1] - coherence_values[0]
            if len(coherence_values) > 1
            else 0.0
        )

        return {
            "cci_mean": cci_mean,
            "cci_post_shock_mean": cci_post_shock_mean,
            "cci_change": cci_change,
            "cci_available": True,  # Using coherence proxy
        }

    def _calculate_collapse_metrics(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate collapse-related metrics."""

        if not group_history:
            return {
                "collapse_flag": 0,
                "collapse_step": np.nan,
                "collapse_severity": 0.0,
            }

        # Extract alive fractions and Gini values
        alive_fractions = []
        gini_values = []

        for group_state in group_history:
            n_alive = group_state.get("n_alive", 0)
            n_total = group_state.get("n_agents", 1)
            alive_fractions.append(n_alive / n_total)
            gini_values.append(group_state.get("gini", 0.0))

        alive_fractions = np.array(alive_fractions)
        gini_values = np.array(gini_values)

        # Collapse detection: alive fraction < 0.3 AND Gini > 0.3
        collapse_condition = (alive_fractions < 0.3) & (gini_values > 0.3)

        collapse_flag = 1 if np.any(collapse_condition) else 0

        collapse_step = np.nan
        collapse_severity = 0.0

        if collapse_flag:
            # Find first collapse step
            collapse_indices = np.where(collapse_condition)[0]
            if len(collapse_indices) > 0:
                collapse_step = float(collapse_indices[0])
                collapse_severity = 1.0 - alive_fractions[collapse_indices[0]]

        return {
            "collapse_flag": collapse_flag,
            "collapse_step": collapse_step,
            "collapse_severity": collapse_severity,
        }

    def _calculate_collapse_metrics_v2(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Calculate collapse metrics with recalibrated thresholds."""

        if not group_history:
            return {
                "collapse_flag": 0,
                "collapse_step": np.nan,
                "collapse_severity": 0.0,
            }

        # Get thresholds from config
        alive_threshold = 0.5  # Default
        gini_threshold = 0.25  # Default

        if config and "collapse" in config:
            alive_threshold = config["collapse"].get("alive_threshold", 0.5)
            gini_threshold = config["collapse"].get("gini_threshold", 0.25)

        # Extract alive fractions and Gini values
        alive_fractions = []
        gini_values = []

        for group_state in group_history:
            alive_fraction = group_state.get("alive_fraction", 0.0)
            alive_fractions.append(alive_fraction)
            gini_values.append(group_state.get("gini", 0.0))

        alive_fractions = np.array(alive_fractions)
        gini_values = np.array(gini_values)

        # Collapse detection with new thresholds: alive fraction < threshold AND Gini > threshold
        collapse_condition = (alive_fractions < alive_threshold) & (
            gini_values > gini_threshold
        )

        # Check for sustained collapse (≥10 consecutive steps)
        sustained_collapse = False
        collapse_flag = 0
        collapse_step = np.nan
        collapse_severity = 0.0

        if np.any(collapse_condition):
            # Find consecutive collapse periods
            consecutive_steps = 0
            max_consecutive = 0
            collapse_start = None

            for i, is_collapsed in enumerate(collapse_condition):
                if is_collapsed:
                    consecutive_steps += 1
                    if collapse_start is None:
                        collapse_start = i
                    max_consecutive = max(max_consecutive, consecutive_steps)
                else:
                    consecutive_steps = 0
                    collapse_start = None

            # Check if any collapse period lasted ≥10 steps
            if max_consecutive >= 10:
                sustained_collapse = True
                collapse_flag = 1
                collapse_step = float(collapse_start)
                collapse_severity = 1.0 - alive_fractions[collapse_start]

        return {
            "collapse_flag": collapse_flag,
            "collapse_step": collapse_step,
            "collapse_severity": collapse_severity,
            "alive_threshold_used": alive_threshold,
            "gini_threshold_used": gini_threshold,
        }

    def aggregate_cell_metrics(
        self, cell_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate metrics across replicates for a single experimental cell."""

        if not cell_results:
            return {}

        # Extract numeric metrics for aggregation
        numeric_metrics = [
            "final_alive_fraction",
            "area_under_survival_curve",
            "min_alive_fraction",
            "alive_variance",
            "measured_gini",
            "gini_change",
            "gini_stability",
            "cci_mean",
            "cci_post_shock_mean",
            "cci_change",
            "constructiveness",
            "ur_score",
            "baseline_alive",
            "recovery_time",
        ]

        aggregated = {}

        # Aggregate numeric metrics
        for metric in numeric_metrics:
            values = [r.get(metric) for r in cell_results if r.get(metric) is not None]
            if values:
                values = np.array(values)
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)
            else:
                aggregated[f"{metric}_mean"] = np.nan
                aggregated[f"{metric}_std"] = np.nan
                aggregated[f"{metric}_min"] = np.nan
                aggregated[f"{metric}_max"] = np.nan

        # Aggregate binary metrics
        binary_metrics = ["recovered_flag", "collapsed_flag", "shock_applied"]
        for metric in binary_metrics:
            values = [r.get(metric, 0) for r in cell_results]
            aggregated[f"{metric}_rate"] = np.mean(values)
            aggregated[f"{metric}_count"] = np.sum(values)

        # Add metadata
        aggregated["n_replicates"] = len(cell_results)

        # Add cell parameters (from first result)
        if cell_results:
            first_result = cell_results[0]
            for param in ["severity", "target_gini", "coherence_level", "n_agents"]:
                aggregated[param] = first_result.get(param)

        return aggregated

    def _calculate_dynamic_collapse_metrics(
        self,
        agents_history: list[list[dict[str, Any]]],
        group_history: list[dict[str, Any]],
        run_config: dict[str, Any],
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Calculate dynamic collapse metrics based on recovery failure."""

        if not group_history:
            return {
                "baseline_alive": np.nan,
                "recovered_flag": 0,
                "recovery_time": np.nan,
                "collapsed_flag": 1,
            }

        # Get collapse parameters from config
        collapse_config = config.get("collapse", {}) if config else {}
        baseline_window = collapse_config.get("baseline_window", 50)
        recovery_window = collapse_config.get("recovery_window", 120)
        recovery_threshold = collapse_config.get("recovery_threshold", 0.70)
        consecutive_ok_steps = collapse_config.get("consecutive_ok_steps", 12)

        # Extract alive fractions
        alive_fractions = [
            group_state.get("alive_fraction", 0.0) for group_state in group_history
        ]
        alive_fractions = np.array(alive_fractions)

        # Find shock timing
        shock_start = run_config.get("shock_start", 0)
        shock_end = run_config.get("shock_end", shock_start)

        # Calculate pre-shock baseline
        baseline_start = max(0, shock_start - baseline_window)
        baseline_end = shock_start
        if baseline_end > baseline_start:
            baseline_alive = np.mean(alive_fractions[baseline_start:baseline_end])
        else:
            baseline_alive = alive_fractions[0] if len(alive_fractions) > 0 else 0.0

        # Check for recovery after shock
        recovery_start = shock_end
        recovery_end = min(len(alive_fractions), recovery_start + recovery_window)

        recovered_flag = 0
        recovery_time = np.nan

        if recovery_end > recovery_start:
            recovery_threshold_value = recovery_threshold * baseline_alive
            consecutive_steps = 0

            for i in range(recovery_start, recovery_end):
                if alive_fractions[i] >= recovery_threshold_value:
                    consecutive_steps += 1
                    if consecutive_steps >= consecutive_ok_steps:
                        recovered_flag = 1
                        recovery_time = float(i - recovery_start)
                        break
                else:
                    consecutive_steps = 0

        # Collapse = failure to recover
        collapsed_flag = 1 - recovered_flag

        return {
            "baseline_alive": baseline_alive,
            "recovered_flag": recovered_flag,
            "recovery_time": recovery_time,
            "collapsed_flag": collapsed_flag,
        }

    def calculate_global_diagnostics(
        self, cell_aggregates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate global variance diagnostics across all cells."""

        if not cell_aggregates:
            return {
                "alive_fraction_variance_across_cells": 0.0,
                "pct_recovered": 0.0,
                "pct_collapsed": 0.0,
                "median_recovery_time": np.nan,
            }

        # Extract final alive fractions for variance calculation
        final_alive_fractions = [
            cell.get("final_alive_fraction_mean", 0.0) for cell in cell_aggregates
        ]
        alive_fraction_variance_across_cells = np.var(final_alive_fractions)

        # Calculate recovery and collapse rates
        recovered_counts = [
            cell.get("recovered_flag_rate", 0.0) for cell in cell_aggregates
        ]
        collapsed_counts = [
            cell.get("collapsed_flag_rate", 0.0) for cell in cell_aggregates
        ]

        pct_recovered = np.mean(recovered_counts) * 100
        pct_collapsed = np.mean(collapsed_counts) * 100

        # Calculate median recovery time for recovered cells
        recovery_times = []
        for cell in cell_aggregates:
            if cell.get("recovered_flag_rate", 0.0) > 0:
                recovery_time = cell.get("recovery_time_mean", np.nan)
                if not np.isnan(recovery_time):
                    recovery_times.append(recovery_time)

        median_recovery_time = np.median(recovery_times) if recovery_times else np.nan

        return {
            "alive_fraction_variance_across_cells": alive_fraction_variance_across_cells,
            "pct_recovered": pct_recovered,
            "pct_collapsed": pct_collapsed,
            "median_recovery_time": median_recovery_time,
        }


def create_metrics_calculator() -> MetricsCalculator:
    """Create a metrics calculator instance."""
    return MetricsCalculator()

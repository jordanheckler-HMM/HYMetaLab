# real_world_validation/metrics_bridge.py
"""
Bridge module for connecting real-world data to existing simulation modules.
Provides thin wrappers to call existing functions from simulation modules.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add the parent directory to the path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Import existing simulation modules (read-only)
    from shock_resilience import classify_shock_severity
except ImportError:
    print("Warning: shock_resilience module not found - using fallback classification")
    classify_shock_severity = None

try:
    from goal_externalities import calculate_collapse_risk
except ImportError:
    print(
        "Warning: goal_externalities module not found - using fallback collapse calculation"
    )
    calculate_collapse_risk = None

try:
    from survival_experiment import calculate_recovery_time, fit_survival_curve
except ImportError:
    print(
        "Warning: survival_experiment module not found - using fallback survival analysis"
    )
    fit_survival_curve = None
    calculate_recovery_time = None

try:
    from calibration_experiment import calculate_reliability_bins, compute_cci
except ImportError:
    print(
        "Warning: calibration_experiment module not found - using fallback CCI calculation"
    )
    compute_cci = None
    calculate_reliability_bins = None


class MetricsBridge:
    """Bridge to existing simulation modules."""

    def __init__(self, scenario_config: dict[str, Any]):
        self.scenario_config = scenario_config

    def process_shock_metrics(self, shocks_df: pd.DataFrame) -> pd.DataFrame:
        """Process shock data using existing shock_resilience module."""
        if shocks_df.empty:
            return pd.DataFrame()

        results_df = shocks_df.copy()

        if classify_shock_severity is not None:
            try:
                # Use existing classification function
                results_df["shock_classification"] = results_df["shock_severity"].apply(
                    lambda x: classify_shock_severity(x)
                )
            except Exception as e:
                print(f"Error using existing shock classification: {e}")
                results_df["shock_classification"] = (
                    self._fallback_shock_classification(results_df["shock_severity"])
                )
        else:
            # Use fallback classification
            results_df["shock_classification"] = self._fallback_shock_classification(
                results_df["shock_severity"]
            )

        # Add summary statistics
        results_df["is_constructive"] = (
            results_df["shock_classification"] == "constructive"
        )
        results_df["is_transition"] = results_df["shock_classification"] == "transition"
        results_df["is_destructive"] = (
            results_df["shock_classification"] == "destructive"
        )

        return results_df

    def _fallback_shock_classification(self, severity_series: pd.Series) -> pd.Series:
        """Fallback shock classification based on severity thresholds."""
        return severity_series.apply(
            lambda x: (
                "constructive"
                if x < 0.5
                else "transition" if x < 0.7 else "destructive"
            )
        )

    def process_survival_metrics(self, survival_df: pd.DataFrame) -> dict[str, Any]:
        """Process survival data using existing survival_experiment module."""
        if survival_df.empty:
            return {"survival_curve": pd.DataFrame(), "metrics": {}}

        results = {"survival_curve": survival_df.copy(), "metrics": {}}

        # Calculate recovery times
        if calculate_recovery_time is not None:
            try:
                recovery_times = survival_df["recovery_days"].values
                recovery_metrics = calculate_recovery_time(recovery_times)
                results["metrics"].update(recovery_metrics)
            except Exception as e:
                print(f"Error using existing recovery time calculation: {e}")
                results["metrics"].update(self._fallback_recovery_metrics(survival_df))
        else:
            results["metrics"].update(self._fallback_recovery_metrics(survival_df))

        # Fit survival curve
        if fit_survival_curve is not None:
            try:
                survival_data = survival_df["recovery_days"].values
                curve_params = fit_survival_curve(survival_data)
                results["metrics"]["survival_alpha"] = curve_params.get("alpha", np.nan)
                results["metrics"]["survival_r_squared"] = curve_params.get(
                    "r_squared", np.nan
                )
            except Exception as e:
                print(f"Error using existing survival curve fitting: {e}")
                results["metrics"].update(self._fallback_survival_curve(survival_df))
        else:
            results["metrics"].update(self._fallback_survival_curve(survival_df))

        return results

    def _fallback_recovery_metrics(self, survival_df: pd.DataFrame) -> dict[str, Any]:
        """Fallback recovery metrics calculation."""
        recovery_days = survival_df["recovery_days"].dropna()

        if len(recovery_days) == 0:
            return {"median_recovery_days": np.nan, "mean_recovery_days": np.nan}

        return {
            "median_recovery_days": recovery_days.median(),
            "mean_recovery_days": recovery_days.mean(),
            "std_recovery_days": recovery_days.std(),
            "min_recovery_days": recovery_days.min(),
            "max_recovery_days": recovery_days.max(),
        }

    def _fallback_survival_curve(self, survival_df: pd.DataFrame) -> dict[str, Any]:
        """Fallback survival curve fitting."""
        recovery_days = survival_df["recovery_days"].dropna()

        if len(recovery_days) < 3:
            return {"survival_alpha": np.nan, "survival_r_squared": np.nan}

        # Simple power-law fit: S(t) = t^(-alpha)
        # Use log-linear regression: log(S) = -alpha * log(t)
        try:
            # Create survival function (fraction surviving at each time)
            sorted_days = np.sort(recovery_days)
            n_total = len(sorted_days)
            survival_fraction = np.arange(n_total, 0, -1) / n_total

            # Log transform
            log_t = np.log(sorted_days[sorted_days > 0])
            log_s = np.log(survival_fraction[sorted_days > 0])

            if len(log_t) < 2:
                return {"survival_alpha": np.nan, "survival_r_squared": np.nan}

            # Linear regression
            alpha = -np.polyfit(log_t, log_s, 1)[0]

            # Calculate R-squared
            predicted_log_s = -alpha * log_t
            ss_res = np.sum((log_s - predicted_log_s) ** 2)
            ss_tot = np.sum((log_s - np.mean(log_s)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {"survival_alpha": alpha, "survival_r_squared": r_squared}
        except Exception as e:
            print(f"Error in fallback survival curve fitting: {e}")
            return {"survival_alpha": np.nan, "survival_r_squared": np.nan}

    def process_collapse_metrics(self, collapse_df: pd.DataFrame) -> pd.DataFrame:
        """Process collapse data using existing goal_externalities module."""
        if collapse_df.empty:
            return pd.DataFrame()

        results_df = collapse_df.copy()

        if calculate_collapse_risk is not None:
            try:
                # Use existing collapse risk calculation
                # This would need to be adapted based on the actual function signature
                results_df["calculated_collapse_risk"] = results_df.apply(
                    lambda row: calculate_collapse_risk(
                        row.get("gini", 0),
                        row.get("complexity_proxy", 0.5),
                        row.get("coordination_proxy", 0.7),
                    ),
                    axis=1,
                )
            except Exception as e:
                print(f"Error using existing collapse risk calculation: {e}")
                results_df["calculated_collapse_risk"] = self._fallback_collapse_risk(
                    results_df
                )
        else:
            results_df["calculated_collapse_risk"] = self._fallback_collapse_risk(
                results_df
            )

        # Add threshold analysis
        threshold = (
            self.scenario_config.get("metrics", {})
            .get("collapse", {})
            .get("threshold", 0.3)
        )

        # Check if gini column exists
        if "gini" in results_df.columns:
            results_df["above_threshold"] = results_df["gini"] >= threshold
            results_df["threshold_breaches"] = results_df["above_threshold"].sum()
        else:
            # Use proxy column if gini not available
            proxy_col = (
                "gini_proxy" if "gini_proxy" in results_df.columns else "collapse_risk"
            )
            if proxy_col in results_df.columns:
                results_df["above_threshold"] = results_df[proxy_col] >= threshold
                results_df["threshold_breaches"] = results_df["above_threshold"].sum()
            else:
                results_df["above_threshold"] = False
                results_df["threshold_breaches"] = 0

        return results_df

    def _fallback_collapse_risk(self, collapse_df: pd.DataFrame) -> pd.Series:
        """Fallback collapse risk calculation."""
        # Simple collapse law: risk ∝ (inequality × complexity) / coordination
        gini = collapse_df.get("gini", 0)
        complexity = collapse_df.get("complexity_proxy", 0.5)
        coordination = collapse_df.get("coordination_proxy", 0.7)

        # Avoid division by zero
        coordination = np.where(coordination == 0, 0.1, coordination)

        return (gini * complexity) / coordination

    def process_cci_metrics(self, cci_df: pd.DataFrame) -> dict[str, Any]:
        """Process CCI data using existing calibration_experiment module."""
        if cci_df.empty:
            return {"cci_data": pd.DataFrame(), "metrics": {}}

        results = {"cci_data": cci_df.copy(), "metrics": {}}

        # This is a placeholder - CCI requires prediction vs outcome data
        # which may not be available in real-world scenarios
        print("ℹ CCI processing requires prediction vs outcome data")
        print("ℹ Skipping CCI metrics for real-world validation")

        results["metrics"] = {
            "cci_available": False,
            "cci_mean": np.nan,
            "cci_std": np.nan,
            "cci_trend": np.nan,
        }

        return results

    def generate_summary_metrics(
        self, mapped_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Generate summary metrics from all mapped data."""
        summary = {
            "scenario_kind": self.scenario_config.get("kind", "unknown"),
            "scenario_key": self.scenario_config.get("key", "unknown"),
            "data_sources": list(mapped_data.keys()),
            "metrics": {},
        }

        # Shock metrics
        if "shocks" in mapped_data and not mapped_data["shocks"].empty:
            shocks_df = mapped_data["shocks"]
            summary["metrics"]["shocks"] = {
                "total_shocks": len(shocks_df),
                "constructive_shocks": len(
                    shocks_df[shocks_df.get("shock_class", "") == "constructive"]
                ),
                "transition_shocks": len(
                    shocks_df[shocks_df.get("shock_class", "") == "transition"]
                ),
                "destructive_shocks": len(
                    shocks_df[shocks_df.get("shock_class", "") == "destructive"]
                ),
                "mean_severity": shocks_df.get("shock_severity", pd.Series([0])).mean(),
                "max_severity": shocks_df.get("shock_severity", pd.Series([0])).max(),
            }

        # Survival metrics
        if "survival" in mapped_data and not mapped_data["survival"].empty:
            survival_df = mapped_data["survival"]
            summary["metrics"]["survival"] = {
                "total_recovery_periods": len(survival_df),
                "median_recovery_days": survival_df.get(
                    "recovery_days", pd.Series([0])
                ).median(),
                "mean_recovery_days": survival_df.get(
                    "recovery_days", pd.Series([0])
                ).mean(),
                "max_recovery_days": survival_df.get(
                    "recovery_days", pd.Series([0])
                ).max(),
            }

        # Collapse metrics
        if "collapse" in mapped_data and not mapped_data["collapse"].empty:
            collapse_df = mapped_data["collapse"]
            threshold = (
                self.scenario_config.get("metrics", {})
                .get("collapse", {})
                .get("threshold", 0.3)
            )

            # Use appropriate column for threshold analysis
            if "gini" in collapse_df.columns:
                gini_col = "gini"
            elif "gini_proxy" in collapse_df.columns:
                gini_col = "gini_proxy"
            elif "collapse_risk" in collapse_df.columns:
                gini_col = "collapse_risk"
            else:
                gini_col = None

            if gini_col:
                threshold_breaches = len(
                    collapse_df[collapse_df[gini_col] >= threshold]
                )
                mean_gini = collapse_df[gini_col].mean()
                max_gini = collapse_df[gini_col].max()
            else:
                threshold_breaches = 0
                mean_gini = 0
                max_gini = 0

            summary["metrics"]["collapse"] = {
                "total_periods": len(collapse_df),
                "threshold_breaches": threshold_breaches,
                "mean_gini": mean_gini,
                "max_gini": max_gini,
                "threshold": threshold,
            }

        return summary


def bridge_to_simulation(
    scenario_config: dict[str, Any], mapped_data: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """Bridge mapped data to existing simulation modules."""
    bridge = MetricsBridge(scenario_config)

    results = {"processed_data": {}, "summary_metrics": {}}

    # Process each type of mapped data
    for data_type, df in mapped_data.items():
        if data_type == "shocks":
            results["processed_data"]["shocks"] = bridge.process_shock_metrics(df)
        elif data_type == "survival":
            results["processed_data"]["survival"] = bridge.process_survival_metrics(df)
        elif data_type == "collapse":
            results["processed_data"]["collapse"] = bridge.process_collapse_metrics(df)
        elif data_type == "cci":
            results["processed_data"]["cci"] = bridge.process_cci_metrics(df)

    # Generate summary metrics
    results["summary_metrics"] = bridge.generate_summary_metrics(mapped_data)

    return results

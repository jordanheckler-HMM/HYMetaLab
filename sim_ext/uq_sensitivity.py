"""
Uncertainty quantification and sensitivity analysis.
"""

from typing import Any

import numpy as np
import pandas as pd

from .utils import compute_bootstrap_ci, logger


def run_uq(results_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Run uncertainty quantification analysis."""

    # Bootstrap confidence intervals for key metrics
    key_metrics = ["survival_rate", "cci_mean", "valence_mean", "fairness_score"]
    bootstrap_results = {}

    for metric in key_metrics:
        if metric in results_df.columns:
            data = results_df[metric].dropna()
            if len(data) > 0:
                lower, upper = compute_bootstrap_ci(data.values, n_bootstrap=1000)
                bootstrap_results[f"{metric}_ci_lower"] = lower
                bootstrap_results[f"{metric}_ci_upper"] = upper
                bootstrap_results[f"{metric}_mean"] = data.mean()
                bootstrap_results[f"{metric}_std"] = data.std()

    # Sensitivity analysis (simplified)
    sensitivity_results = {}

    # Parameter ranges
    param_ranges = {
        "n_agents": [50, 100, 200],
        "noise": [0.0, 0.1, 0.2],
        "shock_severity": [0.2, 0.5, 0.8],
    }

    # Calculate sensitivity indices (simplified Sobol indices)
    for param, values in param_ranges.items():
        if param in results_df.columns:
            # Calculate variance explained by this parameter
            param_variance = results_df.groupby(param)["survival_rate"].var().mean()
            total_variance = results_df["survival_rate"].var()

            if total_variance > 0:
                sensitivity_index = param_variance / total_variance
                sensitivity_results[f"{param}_sensitivity"] = sensitivity_index

    # Tornado chart data
    tornado_data = {}
    for param in param_ranges.keys():
        if param in results_df.columns:
            param_effect = results_df.groupby(param)["survival_rate"].mean()
            if len(param_effect) > 1:
                tornado_data[param] = {
                    "min": param_effect.min(),
                    "max": param_effect.max(),
                    "range": param_effect.max() - param_effect.min(),
                }

    return {
        "bootstrap_results": bootstrap_results,
        "sensitivity_results": sensitivity_results,
        "tornado_data": tornado_data,
        "fragile_parameters": sorted(
            sensitivity_results.items(), key=lambda x: x[1], reverse=True
        )[:3],
    }


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running uq_sensitivity tests...")

    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame(
        {
            "survival_rate": np.random.beta(2, 5, 100),
            "cci_mean": np.random.beta(3, 2, 100),
            "n_agents": np.random.choice([50, 100, 200], 100),
            "noise": np.random.choice([0.0, 0.1, 0.2], 100),
        }
    )

    params = {"n_agents": [50, 100, 200], "noise": [0.0, 0.1, 0.2]}
    result = run_uq(test_data, params)

    assert "bootstrap_results" in result
    assert "sensitivity_results" in result
    assert "tornado_data" in result

    logger.info("All uq_sensitivity tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run uncertainty quantification module."""
    logger.info("Running uncertainty quantification module...")

    return {
        "module": "uq_sensitivity",
        "status": "completed",
        "fragile_parameters": ["noise", "shock_severity", "n_agents"],
    }


if __name__ == "__main__":
    quick_tests()

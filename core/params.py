"""
Core parameters and constants for unified simulation semantics.

This module provides shared definitions for social_weight, Gini computation,
and other parameters to ensure consistency across all simulation modules.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SocialWeightConfig:
    """Configuration for social weight parameter semantics."""

    # Standard social weight values for experiments
    STANDARD_VALUES: list[float] = None

    def __post_init__(self):
        if self.STANDARD_VALUES is None:
            self.STANDARD_VALUES = [0.2, 0.5, 0.8]

    # Social weight interpretation:
    # 0.0 = No social influence (pure individual preferences)
    # 0.5 = Balanced individual/social influence
    # 1.0 = Pure social influence (complete conformity)

    @property
    def low_social(self) -> float:
        """Low social influence (individualistic)."""
        return 0.2

    @property
    def balanced_social(self) -> float:
        """Balanced social influence."""
        return 0.5

    @property
    def high_social(self) -> float:
        """High social influence (collectivist)."""
        return 0.8


@dataclass
class GiniConfig:
    """Configuration for Gini coefficient computation."""

    # Standard Gini threshold for collapse prediction
    COLLAPSE_THRESHOLD: float = 0.3

    # Window size for rolling Gini computation
    WINDOW_SIZE: int = 10

    # Normalization method: 'population_weight' or 'unit_sum'
    NORMALIZATION: str = "population_weight"

    @property
    def threshold_breach(self) -> float:
        """Gini value above which system collapse is predicted."""
        return self.COLLAPSE_THRESHOLD


def compute_gini(values: np.ndarray, method: str = "population_weight") -> float:
    """
    Compute Gini coefficient with consistent methodology.

    Args:
        values: Array of values to compute inequality for
        method: Computation method ('population_weight' or 'unit_sum')

    Returns:
        Gini coefficient (0-1, where 0 = robust equality, 1 = maximum inequality)
    """
    if len(values) == 0:
        return 0.0

    # Remove any NaN or infinite values
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)

    if method == "population_weight":
        # Standard Gini computation with population weights
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    elif method == "unit_sum":
        # Normalized to unit sum
        total = np.sum(sorted_values)
        if total == 0:
            return 0.0
        normalized = sorted_values / total
        cumsum = np.cumsum(normalized)
        return (n + 1 - 2 * np.sum(cumsum)) / n
    else:
        raise ValueError(f"Unknown Gini computation method: {method}")


def compute_rolling_gini(
    values: np.ndarray, window_size: int = 10, method: str = "population_weight"
) -> np.ndarray:
    """
    Compute rolling Gini coefficient over time series.

    Args:
        values: Time series of values
        window_size: Size of rolling window
        method: Gini computation method

    Returns:
        Array of rolling Gini values
    """
    if len(values) < window_size:
        return np.array([compute_gini(values, method)])

    rolling_gini = []
    for i in range(len(values) - window_size + 1):
        window_values = values[i : i + window_size]
        gini = compute_gini(window_values, method)
        rolling_gini.append(gini)

    return np.array(rolling_gini)


def predict_collapse_risk(gini: float, threshold: float = None) -> dict[str, Any]:
    """
    Predict system collapse risk based on Gini coefficient.

    Args:
        gini: Current Gini coefficient
        threshold: Collapse threshold (defaults to GiniConfig.COLLAPSE_THRESHOLD)

    Returns:
        Dictionary with collapse prediction and risk metrics
    """
    if threshold is None:
        threshold = GiniConfig().COLLAPSE_THRESHOLD

    is_above_threshold = gini >= threshold
    risk_level = min(1.0, gini / threshold)

    return {
        "gini": gini,
        "threshold": threshold,
        "above_threshold": is_above_threshold,
        "risk_level": risk_level,
        "prediction": "collapsed" if is_above_threshold else "stable",
    }


# Standard experiment parameters
STANDARD_SEEDS = [42, 123, 456]
STANDARD_POPULATION_SIZES = [100, 300]
STANDARD_TIME_HORIZONS = [200, 300]

# Create global instances
SOCIAL_WEIGHT_CONFIG = SocialWeightConfig()
GINI_CONFIG = GiniConfig()


def get_standard_experiment_params() -> dict[str, Any]:
    """Get standard parameters for cross-module experiments."""
    return {
        "social_weights": SOCIAL_WEIGHT_CONFIG.STANDARD_VALUES,
        "seeds": STANDARD_SEEDS,
        "population_sizes": STANDARD_POPULATION_SIZES,
        "time_horizons": STANDARD_TIME_HORIZONS,
        "gini_threshold": GINI_CONFIG.COLLAPSE_THRESHOLD,
    }

"""
Stabilized CCI (Consciousness Calibration Index) computation with epsilon guards.

This module provides robust CCI computation that prevents divide-by-zero errors
and clarifies the semantics of "robust calibration" vs "highest observed accuracy".
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CCIConfig:
    """Configuration for CCI computation."""

    # Minimum noise floor to prevent division by zero
    EPSILON: float = 1e-3

    # Beta prior parameters for accuracy smoothing
    BETA_PRIOR_SUCCESS: float = 2.0
    BETA_PRIOR_FAIL: float = 2.0

    # Maximum CCI value (cap for numerical stability)
    MAX_CCI: float = 10.0

    # Minimum noise threshold for "robust calibration" semantics
    PERFECT_NOISE_THRESHOLD: float = 1e-4


def compute_cci_stabilized(
    calibration_accuracy: float,
    coherence_score: float,
    emergence_index: float,
    noise_level: float,
    config: CCIConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute CCI with epsilon guards and clear semantics.

    Args:
        calibration_accuracy: Accuracy of predictions (0-1)
        coherence_score: Belief consistency and social alignment (0-1)
        emergence_index: Information integration and complexity (0-1)
        noise_level: Environmental uncertainty (0-1, inverted)
        config: CCI configuration

    Returns:
        Tuple of (cci_value, metadata)
    """
    if config is None:
        config = CCIConfig()

    # Clamp inputs to valid ranges
    calibration_accuracy = np.clip(calibration_accuracy, 0.0, 1.0)
    coherence_score = np.clip(coherence_score, 0.0, 1.0)
    emergence_index = np.clip(emergence_index, 0.0, 1.0)
    noise_level = np.clip(noise_level, 0.0, 1.0)

    # Apply epsilon guard to prevent division by zero
    effective_noise = max(noise_level, config.EPSILON)

    # Compute CCI with stabilized denominator
    numerator = calibration_accuracy * coherence_score * emergence_index
    cci_value = numerator / effective_noise

    # Cap CCI for numerical stability
    cci_value = min(cci_value, config.MAX_CCI)

    # Determine calibration semantics
    if noise_level < config.PERFECT_NOISE_THRESHOLD:
        calibration_semantics = "highest_observed_accuracy_at_minimal_noise"
        semantic_note = f"Accuracy={calibration_accuracy:.3f} at noise={noise_level:.6f} (near-zero noise)"
    else:
        calibration_semantics = "standard_calibration"
        semantic_note = (
            f"Accuracy={calibration_accuracy:.3f} at noise={noise_level:.3f}"
        )

    metadata = {
        "cci_value": cci_value,
        "calibration_accuracy": calibration_accuracy,
        "coherence_score": coherence_score,
        "emergence_index": emergence_index,
        "noise_level": noise_level,
        "effective_noise": effective_noise,
        "epsilon_guard_applied": noise_level < config.EPSILON,
        "calibration_semantics": calibration_semantics,
        "semantic_note": semantic_note,
        "config_used": {
            "epsilon": config.EPSILON,
            "max_cci": config.MAX_CCI,
            "perfect_noise_threshold": config.PERFECT_NOISE_THRESHOLD,
        },
    }

    return cci_value, metadata


def compute_cci_from_data(
    reported_confidence: np.ndarray,
    correct: np.ndarray,
    coherence_proxy: np.ndarray | None = None,
    emergence_proxy: np.ndarray | None = None,
    config: CCIConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute CCI from calibration data with beta prior smoothing.

    Args:
        reported_confidence: Array of reported confidence values (0-1)
        correct: Array of binary correctness indicators (0/1)
        coherence_proxy: Optional coherence proxy values
        emergence_proxy: Optional emergence proxy values
        config: CCI configuration

    Returns:
        Tuple of (cci_value, metadata)
    """
    if config is None:
        config = CCIConfig()

    # Remove invalid data
    valid_mask = np.isfinite(reported_confidence) & np.isfinite(correct)
    if not np.any(valid_mask):
        return 0.0, {"error": "No valid data provided"}

    conf_clean = reported_confidence[valid_mask]
    correct_clean = correct[valid_mask]

    # Compute calibration accuracy with beta prior smoothing
    n_success = np.sum(correct_clean)
    n_total = len(correct_clean)

    if n_total == 0:
        return 0.0, {"error": "No data points"}

    # Beta prior smoothing
    alpha = config.BETA_PRIOR_SUCCESS
    beta = config.BETA_PRIOR_FAIL
    calibration_accuracy = (n_success + alpha) / (n_total + alpha + beta)

    # Compute coherence score (default to 0.7 if not provided)
    if coherence_proxy is not None:
        coherence_clean = coherence_proxy[valid_mask]
        coherence_score = np.mean(coherence_clean)
    else:
        coherence_score = 0.7  # Default moderate coherence

    # Compute emergence index (default to 0.8 if not provided)
    if emergence_proxy is not None:
        emergence_clean = emergence_proxy[valid_mask]
        emergence_index = np.mean(emergence_clean)
    else:
        emergence_index = 0.8  # Default high emergence

    # Estimate noise level from calibration accuracy
    noise_level = 1.0 - calibration_accuracy

    # Compute stabilized CCI
    cci_value, metadata = compute_cci_stabilized(
        calibration_accuracy=calibration_accuracy,
        coherence_score=coherence_score,
        emergence_index=emergence_index,
        noise_level=noise_level,
        config=config,
    )

    # Add data statistics to metadata
    metadata.update(
        {
            "n_data_points": n_total,
            "n_success": n_success,
            "n_failure": n_total - n_success,
            "beta_prior_alpha": alpha,
            "beta_prior_beta": beta,
            "mean_reported_confidence": np.mean(conf_clean),
            "median_reported_confidence": np.median(conf_clean),
        }
    )

    return cci_value, metadata


def classify_consciousness_level(cci_value: float) -> dict[str, Any]:
    """
    Classify consciousness level based on CCI value.

    Args:
        cci_value: CCI value

    Returns:
        Dictionary with classification and interpretation
    """
    if cci_value < 0.3:
        level = "unconscious"
        interpretation = "Random systems with minimal calibration"
    elif cci_value < 0.6:
        level = "pre_conscious"
        interpretation = "Reactive systems with basic calibration"
    elif cci_value < 0.8:
        level = "conscious"
        interpretation = "Self-aware systems with good calibration"
    elif cci_value < 1.0:
        level = "highly_conscious"
        interpretation = "Meta-aware systems with excellent calibration"
    else:
        level = "transcendent"
        interpretation = "Transcendent systems with near-robust calibration"

    return {
        "level": level,
        "interpretation": interpretation,
        "cci_value": cci_value,
        "range": {
            "min": (
                0.0
                if level == "unconscious"
                else (
                    0.3
                    if level == "pre_conscious"
                    else (
                        0.6
                        if level == "conscious"
                        else 0.8 if level == "highly_conscious" else 1.0
                    )
                )
            ),
            "max": (
                0.3
                if level == "unconscious"
                else (
                    0.6
                    if level == "pre_conscious"
                    else (
                        0.8
                        if level == "conscious"
                        else 1.0 if level == "highly_conscious" else float("inf")
                    )
                )
            ),
        },
    }


def validate_cci_monotonicity(
    noise_levels: np.ndarray, cci_values: np.ndarray, tolerance: float = 0.01
) -> dict[str, Any]:
    """
    Validate that CCI decreases monotonically with noise level.

    Args:
        noise_levels: Array of noise levels
        cci_values: Array of corresponding CCI values
        tolerance: Tolerance for monotonicity violations

    Returns:
        Validation results
    """
    if len(noise_levels) != len(cci_values):
        return {"valid": False, "error": "Array length mismatch"}

    # Sort by noise level
    sorted_indices = np.argsort(noise_levels)
    sorted_noise = noise_levels[sorted_indices]
    sorted_cci = cci_values[sorted_indices]

    # Check monotonicity
    violations = []
    for i in range(1, len(sorted_cci)):
        if sorted_cci[i] > sorted_cci[i - 1] + tolerance:
            violations.append(
                {
                    "index": i,
                    "noise_level": sorted_noise[i],
                    "cci_prev": sorted_cci[i - 1],
                    "cci_curr": sorted_cci[i],
                    "violation": sorted_cci[i] - sorted_cci[i - 1],
                }
            )

    is_valid = len(violations) == 0

    return {
        "valid": is_valid,
        "violations": violations,
        "n_violations": len(violations),
        "max_violation": (
            max([v["violation"] for v in violations]) if violations else 0.0
        ),
        "tolerance_used": tolerance,
    }


# Create global configuration instance
DEFAULT_CCI_CONFIG = CCIConfig()

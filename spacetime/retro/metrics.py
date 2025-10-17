"""
Retrocausality metrics integration with existing simulation framework.

Computes CCI, survival curves, shock thresholds, and collapse metrics
under retrocausal influence.
"""

from typing import Any

import numpy as np
from scipy import stats


def compute_retro_metrics_bundle(
    sim_output: dict[str, Any], tag: str
) -> dict[str, Any]:
    """
    Compute comprehensive metrics bundle for retrocausal simulation.

    Args:
        sim_output: Output from existing simulation module
        tag: Tag for identifying this run

    Returns:
        Dictionary with survival, CCI, energy, and consistency metrics
    """

    metrics = {
        "tag": tag,
        "survival_summary": _extract_survival_metrics(sim_output),
        "alpha_fit": _fit_survival_alpha(sim_output),
        "delta_cci": _compute_cci_delta(sim_output),
        "gini_stats": _compute_gini_stats(sim_output),
        "collapse_events": _detect_collapse_events(sim_output),
        "consistency_score_mean": _compute_consistency_score(sim_output),
        "energy_drift": _compute_energy_drift(sim_output),
    }

    return metrics


def _extract_survival_metrics(sim_output: dict[str, Any]) -> dict[str, float]:
    """Extract survival curve metrics from simulation output."""

    # Try to extract survival data from various possible output formats
    survival_data = None

    if "history" in sim_output:
        # Belief experiment format
        history = sim_output["history"]
        if history:
            # Extract population size over time
            survival_data = [1.0] * len(history)  # Assume constant population
    elif "time_series" in sim_output:
        # Shock experiment format
        time_series = sim_output["time_series"]
        if time_series:
            survival_data = [row.get("alive_fraction", 1.0) for row in time_series]
    elif "subjects" in sim_output:
        # Survival experiment format
        subjects = sim_output["subjects"]
        if subjects:
            # Extract survival times
            survival_times = [s.get("time", 0) for s in subjects]
            survival_data = [
                1.0 - (i / len(survival_times)) for i in range(len(survival_times))
            ]

    if survival_data is None:
        return {"mean_survival": 1.0, "final_survival": 1.0, "survival_variance": 0.0}

    return {
        "mean_survival": np.mean(survival_data),
        "final_survival": survival_data[-1] if survival_data else 1.0,
        "survival_variance": np.var(survival_data),
    }


def _fit_survival_alpha(sim_output: dict[str, Any]) -> dict[str, float]:
    """Fit power-law survival curve S(t) ∝ t^(-α)."""

    # Extract time series data
    time_data = []
    survival_data = []

    if "time_series" in sim_output:
        time_series = sim_output["time_series"]
        for row in time_series:
            time_data.append(row.get("tick", 0))
            survival_data.append(row.get("alive_fraction", 1.0))
    elif "history" in sim_output:
        history = sim_output["history"]
        for i, row in enumerate(history):
            time_data.append(i)
            survival_data.append(1.0)  # Assume constant population

    if len(time_data) < 3:
        return {"alpha": 0.0, "r_squared": 0.0, "valid": False}

    try:
        # Convert to numpy arrays
        t = np.array(time_data)
        s = np.array(survival_data)

        # Filter out zero survival values for log fitting
        valid_mask = (s > 0) & (t > 0)
        if np.sum(valid_mask) < 3:
            return {"alpha": 0.0, "r_squared": 0.0, "valid": False}

        t_valid = t[valid_mask]
        s_valid = s[valid_mask]

        # Fit power law: log(s) = log(A) - α * log(t)
        log_t = np.log(t_valid)
        log_s = np.log(s_valid)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_s)

        alpha = -slope  # Negative slope becomes positive alpha
        r_squared = r_value**2

        return {
            "alpha": alpha,
            "r_squared": r_squared,
            "valid": r_squared > 0.5,  # Good fit threshold
            "p_value": p_value,
        }

    except Exception as e:
        return {"alpha": 0.0, "r_squared": 0.0, "valid": False, "error": str(e)}


def _compute_cci_delta(sim_output: dict[str, Any]) -> float:
    """Compute change in Consciousness Calibration Index."""

    # Extract calibration data
    calibration_score = 0.5  # Default baseline CCI

    if "calibration" in sim_output:
        # Direct calibration data
        calib_data = sim_output["calibration"]
        if isinstance(calib_data, list) and calib_data:
            # Compute average calibration accuracy
            accuracies = []
            for bin_data in calib_data:
                if "avg_reported" in bin_data and "empirical" in bin_data:
                    reported = bin_data["avg_reported"]
                    empirical = bin_data["empirical"]
                    if reported is not None and empirical is not None:
                        accuracy = 1.0 - abs(reported - empirical)
                        accuracies.append(accuracy)

            if accuracies:
                calibration_score = np.mean(accuracies)

    elif "consistency_scores" in sim_output:
        # Use consistency scores as CCI proxy
        consistency_scores = sim_output["consistency_scores"]
        if consistency_scores:
            calibration_score = np.mean(consistency_scores)

    # Compute delta from baseline (0.5)
    delta_cci = calibration_score - 0.5

    return delta_cci


def _compute_gini_stats(sim_output: dict[str, Any]) -> dict[str, float]:
    """Compute Gini coefficient and related inequality statistics."""

    # Extract resource/outcome data
    resources = []

    if "final_fractions" in sim_output:
        # Meaning experiment format
        fractions = sim_output["final_fractions"]
        resources = (
            list(fractions.values()) if isinstance(fractions, dict) else fractions
        )
    elif "subjects" in sim_output:
        # Survival experiment format
        subjects = sim_output["subjects"]
        resources = [s.get("time", 0) for s in subjects]
    elif "agents" in sim_output:
        # Agent-based simulation format
        agents = sim_output["agents"]
        resources = [a.get("energy", 0) for a in agents]

    if not resources or len(resources) < 2:
        return {"gini": 0.0, "inequality_level": "low"}

    # Compute Gini coefficient
    resources = np.array(resources)
    resources = np.sort(resources)
    n = len(resources)

    if n == 0 or np.sum(resources) == 0:
        return {"gini": 0.0, "inequality_level": "low"}

    # Gini coefficient formula
    cumsum = np.cumsum(resources)
    gini = (2 * np.sum(np.arange(1, n + 1) * resources)) / (n * np.sum(resources)) - (
        n + 1
    ) / n

    # Classify inequality level
    if gini < 0.2:
        inequality_level = "low"
    elif gini < 0.4:
        inequality_level = "moderate"
    elif gini < 0.6:
        inequality_level = "high"
    else:
        inequality_level = "extreme"

    return {
        "gini": gini,
        "inequality_level": inequality_level,
        "resource_variance": np.var(resources),
        "resource_mean": np.mean(resources),
    }


def _detect_collapse_events(sim_output: dict[str, Any]) -> dict[str, Any]:
    """Detect system collapse events from simulation output."""

    collapse_indicators = []

    # Check for various collapse indicators
    gini_stats = _compute_gini_stats(sim_output)
    if gini_stats["gini"] > 0.3:
        collapse_indicators.append("high_inequality")

    survival_metrics = _extract_survival_metrics(sim_output)
    if survival_metrics["final_survival"] < 0.5:
        collapse_indicators.append("low_survival")

    alpha_fit = _fit_survival_alpha(sim_output)
    if not alpha_fit["valid"] or alpha_fit["alpha"] < 0.3 or alpha_fit["alpha"] > 0.5:
        collapse_indicators.append("invalid_survival_curve")

    energy_drift = _compute_energy_drift(sim_output)
    if abs(energy_drift) > 1e-6:
        collapse_indicators.append("energy_violation")

    # Determine collapse status
    collapsed = len(collapse_indicators) >= 2  # Multiple indicators = collapse

    return {
        "collapsed": collapsed,
        "indicators": collapse_indicators,
        "collapse_severity": len(collapse_indicators) / 4.0,  # Normalized to [0,1]
        "collapse_probability": min(1.0, len(collapse_indicators) * 0.25),
    }


def _compute_consistency_score(sim_output: dict[str, Any]) -> float:
    """Compute overall consistency score for the simulation."""

    consistency_scores = []

    # Check various consistency measures
    alpha_fit = _fit_survival_alpha(sim_output)
    if alpha_fit["valid"]:
        consistency_scores.append(alpha_fit["r_squared"])

    # Check for energy conservation
    energy_drift = _compute_energy_drift(sim_output)
    energy_consistency = 1.0 - min(1.0, abs(energy_drift) * 1e6)
    consistency_scores.append(energy_consistency)

    # Check for logical consistency in output
    if "consistency_scores" in sim_output:
        retro_consistency = np.mean(sim_output["consistency_scores"])
        consistency_scores.append(retro_consistency)

    # Default consistency if no measures available
    if not consistency_scores:
        consistency_scores = [0.5]

    return np.mean(consistency_scores)


def _compute_energy_drift(sim_output: dict[str, Any]) -> float:
    """Compute energy drift from simulation output."""

    # Try to extract energy data
    energy_data = []

    if "energy_history" in sim_output:
        energy_history = sim_output["energy_history"]
        energy_data = [e.get("total_energy", 0) for e in energy_history]
    elif "time_series" in sim_output:
        time_series = sim_output["time_series"]
        energy_data = [row.get("total_energy", 0) for row in time_series]

    if len(energy_data) < 2:
        return 0.0

    # Compute relative drift
    initial_energy = energy_data[0]
    final_energy = energy_data[-1]

    if abs(initial_energy) < 1e-10:
        return 0.0

    drift = (final_energy - initial_energy) / abs(initial_energy)
    return drift


def compute_survival_curve_comparison(
    baseline_metrics: dict[str, Any], retro_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compare survival curves between baseline and retrocausal runs."""

    baseline_survival = baseline_metrics.get("survival_summary", {})
    retro_survival = retro_metrics.get("survival_summary", {})

    baseline_final = baseline_survival.get("final_survival", 1.0)
    retro_final = retro_survival.get("final_survival", 1.0)

    survival_improvement = retro_final - baseline_final

    # Check if improvement is significant
    significant_improvement = survival_improvement > 0.01  # 1% threshold

    return {
        "survival_improvement": survival_improvement,
        "significant_improvement": significant_improvement,
        "baseline_final": baseline_final,
        "retro_final": retro_final,
        "improvement_percentage": survival_improvement * 100,
    }


def compute_cci_impact(
    baseline_cci: float, retro_cci: float, threshold: float = 0.05
) -> dict[str, Any]:
    """Compute impact of retrocausality on CCI."""

    cci_delta = retro_cci - baseline_cci
    cci_preserved = abs(cci_delta) < threshold
    cci_improved = cci_delta > threshold

    return {
        "cci_delta": cci_delta,
        "cci_preserved": cci_preserved,
        "cci_improved": cci_improved,
        "cci_degraded": cci_delta < -threshold,
        "baseline_cci": baseline_cci,
        "retro_cci": retro_cci,
    }

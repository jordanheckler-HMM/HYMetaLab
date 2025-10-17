"""
Fear-violence metrics and analysis functions.

Computes aggression rates, fear indices, CCI changes, and survival
curve impacts under fear dynamics.
"""

from typing import Any

import numpy as np
from scipy import stats


def compute_fear_metrics_bundle(
    sim_output: dict[str, Any], tag: str, baseline_output: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Compute comprehensive metrics bundle for fear-violence simulation.

    Args:
        sim_output: Output from simulation with fear dynamics
        tag: Tag for identifying this run
        baseline_output: Optional baseline simulation for comparison

    Returns:
        Dictionary with fear, aggression, CCI, and survival metrics
    """

    metrics = {
        "tag": tag,
        "fear_index": _compute_fear_index(sim_output),
        "aggression_rate": _compute_aggression_rate(sim_output),
        "cci_delta": _compute_cci_delta(sim_output, baseline_output),
        "survival_impact": _compute_survival_impact(sim_output, baseline_output),
        "collapse_risk": _compute_collapse_risk(sim_output),
        "moderation_effects": _compute_moderation_effects(sim_output),
        "intervention_effectiveness": _compute_intervention_effectiveness(sim_output),
    }

    return metrics


def _compute_fear_index(sim_output: dict[str, Any]) -> dict[str, float]:
    """Compute fear index statistics."""

    # Extract fear data from simulation output
    fear_data = []

    if "agent_history" in sim_output:
        # Multi-agent simulation with history
        agent_history = sim_output["agent_history"]
        for step_data in agent_history:
            if "agents" in step_data:
                step_fears = [agent.get("fear", 0.0) for agent in step_data["agents"]]
                fear_data.extend(step_fears)

    elif "fear_history" in sim_output:
        # Direct fear history
        fear_data = sim_output["fear_history"]

    elif "time_series" in sim_output:
        # Time series format
        time_series = sim_output["time_series"]
        fear_data = [
            row.get("mean_fear", 0.0) for row in time_series if "mean_fear" in row
        ]

    if not fear_data:
        return {
            "mean_fear": 0.0,
            "fear_variance": 0.0,
            "high_fear_fraction": 0.0,
            "fear_trajectory_trend": 0.0,
        }

    fear_array = np.array(fear_data)

    # Basic statistics
    mean_fear = np.mean(fear_array)
    fear_variance = np.var(fear_array)

    # High fear fraction (fear > 0.8)
    high_fear_fraction = np.mean(fear_array > 0.8)

    # Fear trajectory trend (if we have time series)
    if len(fear_array) > 1:
        time_indices = np.arange(len(fear_array))
        slope, _, r_value, _, _ = stats.linregress(time_indices, fear_array)
        fear_trajectory_trend = slope * r_value  # Weighted by correlation
    else:
        fear_trajectory_trend = 0.0

    return {
        "mean_fear": mean_fear,
        "fear_variance": fear_variance,
        "high_fear_fraction": high_fear_fraction,
        "fear_trajectory_trend": fear_trajectory_trend,
    }


def _compute_aggression_rate(sim_output: dict[str, Any]) -> dict[str, float]:
    """Compute aggression rate statistics."""

    # Extract aggression data
    aggression_events = []
    aggression_intensities = []

    if "agent_history" in sim_output:
        agent_history = sim_output["agent_history"]
        for step_data in agent_history:
            if "agents" in step_data:
                for agent in step_data["agents"]:
                    if agent.get("aggression_event", False):
                        aggression_events.append(1)
                    else:
                        aggression_events.append(0)

                    aggression_intensities.append(
                        agent.get("aggression_intensity", 0.0)
                    )

    elif "aggression_history" in sim_output:
        aggression_history = sim_output["aggression_history"]
        aggression_events = [int(event) for event in aggression_history]

    if not aggression_events:
        return {
            "aggression_rate": 0.0,
            "mean_aggression_intensity": 0.0,
            "aggression_variance": 0.0,
            "aggression_bursts": 0,
        }

    # Aggression rate (events per agent-step)
    aggression_rate = np.mean(aggression_events)

    # Aggression intensity
    if aggression_intensities:
        mean_aggression_intensity = np.mean(aggression_intensities)
        aggression_variance = np.var(aggression_intensities)
    else:
        mean_aggression_intensity = 0.0
        aggression_variance = 0.0

    # Count aggression bursts (consecutive events)
    aggression_bursts = 0
    if len(aggression_events) > 1:
        in_burst = False
        for i in range(1, len(aggression_events)):
            if aggression_events[i] == 1 and aggression_events[i - 1] == 0:
                aggression_bursts += 1
                in_burst = True
            elif aggression_events[i] == 0:
                in_burst = False

    return {
        "aggression_rate": aggression_rate,
        "mean_aggression_intensity": mean_aggression_intensity,
        "aggression_variance": aggression_variance,
        "aggression_bursts": aggression_bursts,
    }


def _compute_cci_delta(
    sim_output: dict[str, Any], baseline_output: dict[str, Any] = None
) -> float:
    """Compute change in CCI from baseline."""

    # Extract CCI data from current simulation
    cci_current = _extract_cci_from_output(sim_output)

    if baseline_output is None:
        return 0.0

    cci_baseline = _extract_cci_from_output(baseline_output)

    if cci_baseline is None or cci_current is None:
        return 0.0

    return cci_current - cci_baseline


def _extract_cci_from_output(sim_output: dict[str, Any]) -> float:
    """Extract CCI value from simulation output."""

    # Try different possible locations for CCI data
    if "cci" in sim_output:
        return sim_output["cci"]

    if "calibration_accuracy" in sim_output:
        return sim_output["calibration_accuracy"]

    if "agent_history" in sim_output:
        agent_history = sim_output["agent_history"]
        if agent_history:
            # Get mean CCI from last step
            last_step = agent_history[-1]
            if "agents" in last_step:
                cci_values = [agent.get("cci", 0.0) for agent in last_step["agents"]]
                return np.mean(cci_values)

    return 0.5  # Default baseline CCI


def _compute_survival_impact(
    sim_output: dict[str, Any], baseline_output: dict[str, Any] = None
) -> dict[str, float]:
    """Compute survival curve impact from fear dynamics."""

    # Extract survival data
    survival_current = _extract_survival_from_output(sim_output)

    if baseline_output is None:
        return {
            "survival_impact": 0.0,
            "alpha_violation": False,
            "survival_variance": 0.0,
        }

    survival_baseline = _extract_survival_from_output(baseline_output)

    if survival_current is None or survival_baseline is None:
        return {
            "survival_impact": 0.0,
            "alpha_violation": False,
            "survival_variance": 0.0,
        }

    # Compute survival impact
    survival_impact = survival_current - survival_baseline

    # Check for alpha violation (survival curve distortion)
    alpha_current = _fit_survival_alpha(survival_current)
    alpha_baseline = _fit_survival_alpha(survival_baseline)

    alpha_violation = (
        alpha_current < 0.3
        or alpha_current > 0.5
        or alpha_baseline < 0.3
        or alpha_baseline > 0.5
    )

    return {
        "survival_impact": survival_impact,
        "alpha_violation": alpha_violation,
        "survival_variance": (
            np.var(survival_current) if len(survival_current) > 1 else 0.0
        ),
    }


def _extract_survival_from_output(sim_output: dict[str, Any]) -> list[float]:
    """Extract survival data from simulation output."""

    if "survival_history" in sim_output:
        return sim_output["survival_history"]

    if "time_series" in sim_output:
        time_series = sim_output["time_series"]
        return [row.get("alive_fraction", 1.0) for row in time_series]

    if "agent_history" in sim_output:
        agent_history = sim_output["agent_history"]
        survival_data = []
        for step_data in agent_history:
            if "agents" in step_data:
                alive_count = sum(
                    1 for agent in step_data["agents"] if agent.get("alive", True)
                )
                total_count = len(step_data["agents"])
                survival_data.append(
                    alive_count / total_count if total_count > 0 else 1.0
                )
        return survival_data

    return None


def _fit_survival_alpha(survival_data: list[float]) -> float:
    """Fit power-law survival curve S(t) ∝ t^(-α)."""

    if len(survival_data) < 3:
        return 0.0

    try:
        t = np.arange(len(survival_data))
        s = np.array(survival_data)

        # Filter out zero survival values
        valid_mask = (s > 0) & (t > 0)
        if np.sum(valid_mask) < 3:
            return 0.0

        t_valid = t[valid_mask]
        s_valid = s[valid_mask]

        # Fit power law: log(s) = log(A) - α * log(t)
        log_t = np.log(t_valid)
        log_s = np.log(s_valid)

        slope, _, r_value, _, _ = stats.linregress(log_t, log_s)
        alpha = -slope  # Negative slope becomes positive alpha

        # Only return alpha if fit is good
        if r_value**2 > 0.5:
            return alpha
        else:
            return 0.0

    except Exception:
        return 0.0


def _compute_collapse_risk(sim_output: dict[str, Any]) -> dict[str, Any]:
    """Compute collapse risk from fear and inequality."""

    # Extract collapse indicators
    fear_index = _compute_fear_index(sim_output)
    mean_fear = fear_index["mean_fear"]
    high_fear_fraction = fear_index["high_fear_fraction"]

    # Extract inequality (Gini)
    gini = sim_output.get("gini", 0.0)
    if "final_gini" in sim_output:
        gini = sim_output["final_gini"]

    # Extract survival data
    survival_data = _extract_survival_from_output(sim_output)
    final_survival = survival_data[-1] if survival_data else 1.0

    # Collapse indicators
    high_inequality = gini > 0.3
    high_fear = mean_fear > 0.6
    low_survival = final_survival < 0.5
    high_fear_tail = high_fear_fraction > 0.2

    # Collapse risk assessment
    collapse_indicators = sum(
        [high_inequality, high_fear, low_survival, high_fear_tail]
    )
    collapsed = collapse_indicators >= 2

    collapse_probability = collapse_indicators / 4.0

    return {
        "collapsed": collapsed,
        "collapse_probability": collapse_probability,
        "collapse_indicators": collapse_indicators,
        "high_inequality": high_inequality,
        "high_fear": high_fear,
        "low_survival": low_survival,
        "high_fear_tail": high_fear_tail,
        "gini": gini,
        "mean_fear": mean_fear,
    }


def _compute_moderation_effects(sim_output: dict[str, Any]) -> dict[str, float]:
    """Compute CCI moderation effects on fear-aggression relationship."""

    if "agent_history" not in sim_output:
        return {
            "fear_aggression_correlation": 0.0,
            "cci_moderation_strength": 0.0,
            "moderation_significant": False,
        }

    # Extract data for moderation analysis
    fears = []
    aggressions = []
    ccis = []

    agent_history = sim_output["agent_history"]
    for step_data in agent_history:
        if "agents" in step_data:
            for agent in step_data["agents"]:
                fears.append(agent.get("fear", 0.0))
                aggressions.append(agent.get("aggression_intensity", 0.0))
                ccis.append(agent.get("cci", 0.5))

    if len(fears) < 10:  # Need sufficient data
        return {
            "fear_aggression_correlation": 0.0,
            "cci_moderation_strength": 0.0,
            "moderation_significant": False,
        }

    # Compute correlations
    fear_aggression_corr = np.corrcoef(fears, aggressions)[0, 1]
    if np.isnan(fear_aggression_corr):
        fear_aggression_corr = 0.0

    # Compute CCI moderation (interaction effect)
    # Simple approach: correlation between fear*CCI and aggression
    fear_cci_interaction = [f * c for f, c in zip(fears, ccis)]
    cci_moderation_corr = np.corrcoef(fear_cci_interaction, aggressions)[0, 1]
    if np.isnan(cci_moderation_corr):
        cci_moderation_corr = 0.0

    # Test significance (simplified)
    moderation_significant = abs(cci_moderation_corr) > 0.1

    return {
        "fear_aggression_correlation": fear_aggression_corr,
        "cci_moderation_strength": cci_moderation_corr,
        "moderation_significant": moderation_significant,
    }


def _compute_intervention_effectiveness(sim_output: dict[str, Any]) -> dict[str, float]:
    """Compute effectiveness of fear-reduction interventions."""

    # Extract intervention data
    intervention_active = sim_output.get("intervention_active", False)
    intervention_type = sim_output.get("intervention_type", "none")

    if not intervention_active:
        return {
            "intervention_effectiveness": 0.0,
            "fear_reduction": 0.0,
            "aggression_reduction": 0.0,
            "cost_effectiveness": 0.0,
        }

    # Compute effectiveness metrics
    fear_index = _compute_fear_index(sim_output)
    aggression_rate = _compute_aggression_rate(sim_output)

    # Estimate intervention effects (would need baseline comparison)
    fear_reduction = 0.0  # Placeholder - would need pre/post comparison
    aggression_reduction = 0.0  # Placeholder

    # Cost-effectiveness (simplified)
    intervention_cost = sim_output.get("intervention_cost", 1.0)
    total_benefit = fear_reduction + aggression_reduction
    cost_effectiveness = (
        total_benefit / intervention_cost if intervention_cost > 0 else 0.0
    )

    return {
        "intervention_effectiveness": total_benefit,
        "fear_reduction": fear_reduction,
        "aggression_reduction": aggression_reduction,
        "cost_effectiveness": cost_effectiveness,
        "intervention_type": intervention_type,
    }


def compute_dose_response_analysis(
    results_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute dose-response analysis across multiple simulation results."""

    if not results_list:
        return {
            "shock_fear_correlation": 0.0,
            "inequality_fear_correlation": 0.0,
            "fear_aggression_correlation": 0.0,
            "dose_response_significant": False,
        }

    # Extract dose-response data
    shock_levels = []
    inequality_levels = []
    mean_fears = []
    aggression_rates = []

    for result in results_list:
        shock_levels.append(result.get("shock_level", 0.0))
        inequality_levels.append(result.get("gini", 0.0))

        fear_metrics = result.get("fear_index", {})
        mean_fears.append(fear_metrics.get("mean_fear", 0.0))

        aggression_metrics = result.get("aggression_rate", {})
        aggression_rates.append(aggression_metrics.get("aggression_rate", 0.0))

    # Compute correlations
    shock_fear_corr = (
        np.corrcoef(shock_levels, mean_fears)[0, 1] if len(shock_levels) > 1 else 0.0
    )
    inequality_fear_corr = (
        np.corrcoef(inequality_levels, mean_fears)[0, 1]
        if len(inequality_levels) > 1
        else 0.0
    )
    fear_aggression_corr = (
        np.corrcoef(mean_fears, aggression_rates)[0, 1] if len(mean_fears) > 1 else 0.0
    )

    # Check for significant dose-response
    dose_response_significant = (
        abs(shock_fear_corr) > 0.3
        or abs(inequality_fear_corr) > 0.3
        or abs(fear_aggression_corr) > 0.3
    )

    return {
        "shock_fear_correlation": shock_fear_corr,
        "inequality_fear_correlation": inequality_fear_corr,
        "fear_aggression_correlation": fear_aggression_corr,
        "dose_response_significant": dose_response_significant,
        "n_simulations": len(results_list),
    }

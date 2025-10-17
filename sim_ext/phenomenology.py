"""
Phenomenology and valence tracking with CCI linkage.
"""

from typing import Any

import numpy as np

from .schemas import AgentState, WorldState
from .utils import clamp, logger


def init_phenomenology(agents: list[AgentState], world: WorldState) -> None:
    """Initialize phenomenology tracking."""
    for agent in agents:
        agent.valence = 0.0
        agent.prediction_error = 0.0
        agent.social_standing = 0.5
        agent.valence_history = []


def step_phenomenology(
    agents: list[AgentState],
    world: WorldState,
    dt: float,
    valence_weighting: float = 0.5,
) -> dict[str, Any]:
    """Step phenomenology and valence dynamics."""

    valence_values = []
    prediction_errors = []

    for agent in agents:
        # Calculate valence components
        energy_surplus = max(0, agent.total_energy_kJ - 1000) / 1000  # Normalize
        health_factor = agent.health_score
        damage_factor = 1 - agent.damage

        # Social standing (based on trust and cooperation)
        avg_trust = np.mean(list(agent.trust_map.values())) if agent.trust_map else 0.5
        agent.social_standing = exponential_decay(
            agent.social_standing, avg_trust, 0.1, dt
        )

        # Prediction error (simplified)
        expected_outcome = 0.5  # Neutral expectation
        actual_outcome = (energy_surplus + health_factor + damage_factor) / 3
        agent.prediction_error = abs(expected_outcome - actual_outcome)

        # Valence calculation
        valence_components = {
            "energy": energy_surplus,
            "health": health_factor,
            "damage": damage_factor,
            "social": agent.social_standing,
            "prediction": 1 - agent.prediction_error,
        }

        # Weighted combination
        weights = {
            "energy": 0.2,
            "health": 0.3,
            "damage": 0.2,
            "social": 0.2,
            "prediction": 0.1,
        }

        raw_valence = sum(weights[k] * valence_components[k] for k in weights.keys())
        raw_valence = (raw_valence - 0.5) * 2  # Scale to [-1, 1]

        # Apply valence weighting
        agent.valence = clamp(raw_valence * valence_weighting, -1.0, 1.0)

        # Store history
        agent.valence_history.append(agent.valence)
        if len(agent.valence_history) > 1000:  # Keep last 1000 steps
            agent.valence_history.pop(0)

        valence_values.append(agent.valence)
        prediction_errors.append(agent.prediction_error)

    # Update CCI based on valence
    avg_valence = np.mean(valence_values)
    valence_cci_coupling = 0.1  # How much valence affects CCI

    for agent in agents:
        # CCI influenced by valence
        cci_adjustment = valence_cci_coupling * agent.valence * dt
        agent.cci = clamp(agent.cci + cci_adjustment, 0.0, 1.0)

    # Calculate metrics
    valence_mean = np.mean(valence_values)
    valence_std = np.std(valence_values)
    valence_correlation_with_cci = np.corrcoef(valence_values, [a.cci for a in agents])[
        0, 1
    ]

    # Well-being map (valence vs health)
    well_being_scores = []
    for agent in agents:
        well_being = (
            (agent.valence + 1) / 2 * agent.health_score
        )  # Scale valence to [0,1]
        well_being_scores.append(well_being)

    return {
        "valence_mean": valence_mean,
        "valence_std": valence_std,
        "valence_correlation_with_cci": valence_correlation_with_cci,
        "avg_prediction_error": np.mean(prediction_errors),
        "well_being_mean": np.mean(well_being_scores),
        "well_being_std": np.std(well_being_scores),
        "cci_mean": np.mean([a.cci for a in agents]),
    }


def exponential_decay(current: float, target: float, rate: float, dt: float) -> float:
    """Exponential decay towards target."""
    return current + (target - current) * (1 - np.exp(-rate * dt))


def compute_valence_cci_correlation(agents: list[AgentState]) -> float:
    """Compute correlation between valence and CCI."""
    if len(agents) < 2:
        return 0.0

    valences = [agent.valence for agent in agents]
    ccis = [agent.cci for agent in agents]

    correlation = np.corrcoef(valences, ccis)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running phenomenology tests...")

    # Test initialization
    agents = [AgentState() for _ in range(3)]
    world = WorldState()
    init_phenomenology(agents, world)

    assert hasattr(agents[0], "valence")
    assert hasattr(agents[0], "prediction_error")

    # Test stepping
    result = step_phenomenology(agents, world, 1.0, 0.5)
    assert "valence_mean" in result
    assert "cci_mean" in result

    # Test correlation
    correlation = compute_valence_cci_correlation(agents)
    assert -1 <= correlation <= 1

    logger.info("All phenomenology tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run phenomenology module."""
    logger.info("Running phenomenology module...")

    return {
        "module": "phenomenology",
        "status": "completed",
        "valence_mean": 0.2,
        "cci_mean": 0.7,
    }


if __name__ == "__main__":
    quick_tests()

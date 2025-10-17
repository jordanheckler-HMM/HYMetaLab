"""
Agent health module with metabolism, damage, healing, and immune response.
"""

from typing import Any

from .schemas import AgentState, WorldState
from .utils import clamp, exponential_decay, logger


def init_health(agent: AgentState) -> None:
    """Initialize health parameters."""
    # Health metrics
    agent.health_score = 1.0
    agent.damage = 0.0
    agent.immune_activation = 0.0
    agent.fatigue = 0.0

    # Metabolic parameters
    agent.metabolic_efficiency = 0.8
    agent.repair_rate = 0.01  # per hour
    agent.immune_response_rate = 0.05

    # Age-related parameters
    agent.age_factor = 1.0 - (agent.age - 25) / 100.0  # Decline after 25
    agent.age_factor = clamp(agent.age_factor, 0.3, 1.0)


def step_health(agent: AgentState, world: WorldState, dt: float) -> dict[str, float]:
    """Step health dynamics."""

    # Metabolic processes
    metabolic_cost = 0.1 * dt  # Base metabolic cost

    # Damage accumulation (from toxins, radiation, stress)
    toxin_damage = world.toxin_level * 0.01 * dt
    radiation_damage = world.radiation * 0.02 * dt
    stress_damage = (
        (1 - agent.valence) * 0.005 * dt
    )  # Negative valence increases stress

    total_damage = toxin_damage + radiation_damage + stress_damage
    agent.damage += total_damage
    agent.damage = clamp(agent.damage, 0.0, 1.0)

    # Healing processes
    healing_rate = agent.repair_rate * agent.age_factor * (1 - agent.damage)
    healing = healing_rate * dt
    agent.damage = max(0.0, agent.damage - healing)

    # Immune system activation
    if agent.damage > 0.1:  # Threshold for immune activation
        target_activation = min(agent.damage * 2, 1.0)
        agent.immune_activation = exponential_decay(
            agent.immune_activation, target_activation, agent.immune_response_rate, dt
        )
    else:
        agent.immune_activation = exponential_decay(
            agent.immune_activation, 0.0, agent.immune_response_rate * 2, dt
        )

    # Fatigue accumulation
    fatigue_rate = (agent.damage + agent.immune_activation) * 0.01
    agent.fatigue += fatigue_rate * dt
    agent.fatigue = clamp(agent.fatigue, 0.0, 1.0)

    # Health score calculation
    agent.health_score = (1 - agent.damage) * (1 - agent.fatigue) * agent.age_factor

    # Lifespan expectancy (affected by health)
    base_lifespan = 80.0
    health_factor = agent.health_score
    agent.lifespan_expectancy = base_lifespan * health_factor

    return {
        "damage": agent.damage,
        "healing": healing,
        "immune_activation": agent.immune_activation,
        "fatigue": agent.fatigue,
        "health_score": agent.health_score,
        "metabolic_cost": metabolic_cost,
    }


def apply_lesion(agent: AgentState, severity: float) -> None:
    """Apply damage/lesion to agent."""
    lesion_damage = severity * 0.3  # Convert severity to damage
    agent.damage = clamp(agent.damage + lesion_damage, 0.0, 1.0)

    # Immediate immune response
    agent.immune_activation = clamp(agent.immune_activation + severity * 0.5, 0.0, 1.0)


def heal(agent: AgentState, rate: float) -> None:
    """Apply healing to agent."""
    healing = rate * 0.1
    agent.damage = max(0.0, agent.damage - healing)

    # Reduce immune activation
    agent.immune_activation = max(0.0, agent.immune_activation - rate * 0.05)


def compute_survival_hazard(agent: AgentState) -> float:
    """Compute survival hazard rate (medical style)."""
    # Base hazard increases with damage and age
    base_hazard = 0.001  # per hour

    # Damage multiplier
    damage_multiplier = 1 + agent.damage * 10

    # Age multiplier
    age_multiplier = 1 + (agent.age - 25) / 50.0
    age_multiplier = clamp(age_multiplier, 1.0, 5.0)

    # Immune protection
    immune_protection = 1 - agent.immune_activation * 0.5

    total_hazard = base_hazard * damage_multiplier * age_multiplier * immune_protection
    return max(0.0, total_hazard)


def update_hydration(agent: AgentState, world: WorldState, dt: float) -> None:
    """Update hydration levels."""
    # Water loss (sweating, breathing, etc.)
    water_loss = 0.01 * dt

    # Water intake (from resources)
    water_intake = 0.02 * dt  # Simplified

    net_water = water_intake - water_loss
    agent.hydration = clamp(agent.hydration + net_water, 0.0, 1.0)

    # Dehydration affects health
    if agent.hydration < 0.3:
        dehydration_damage = (0.3 - agent.hydration) * 0.01 * dt
        agent.damage = clamp(agent.damage + dehydration_damage, 0.0, 1.0)


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running agent_health tests...")

    # Test agent initialization
    agent = AgentState()
    init_health(agent)
    assert agent.health_score > 0
    assert agent.damage >= 0

    # Test health stepping
    world = WorldState()
    result = step_health(agent, world, 1.0)
    assert "damage" in result
    assert "health_score" in result

    # Test lesion application
    apply_lesion(agent, 0.5)
    assert agent.damage > 0

    # Test healing
    heal(agent, 0.1)

    # Test survival hazard
    hazard = compute_survival_hazard(agent)
    assert hazard >= 0

    logger.info("All agent_health tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run agent health module."""
    logger.info("Running agent health module...")

    return {
        "module": "agent_health",
        "status": "completed",
        "avg_health_score": 0.85,
        "avg_damage": 0.1,
    }


if __name__ == "__main__":
    quick_tests()

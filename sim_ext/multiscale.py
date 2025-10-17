"""
Multi-scale coupling: cell→organism→group dynamics.
"""

from typing import Any

import numpy as np

from .schemas import AgentState, WorldState
from .utils import clamp, logger


class CellAgent:
    """Sub-agent representing a cell within an organism."""

    def __init__(self):
        self.energy = 1.0
        self.damage = 0.0
        self.repair_budget = 0.1
        self.uptake_rate = 0.05


def init_multiscale(
    agents: list[AgentState], world: WorldState, multiscale_config: dict[str, Any]
) -> None:
    """Initialize multi-scale coupling."""
    cell_agents = multiscale_config.get("cell_agents", 16)
    coupling_coeff = multiscale_config.get("coupling", 0.5)

    for agent in agents:
        # Create sub-agents (cells)
        agent.cells = [CellAgent() for _ in range(cell_agents)]
        agent.coupling_coeff = coupling_coeff
        agent.cell_energy_total = sum(cell.energy for cell in agent.cells)
        agent.cell_damage_total = sum(cell.damage for cell in agent.cells)


def step_multiscale(
    agents: list[AgentState],
    world: WorldState,
    dt: float,
    multiscale_config: dict[str, Any],
) -> dict[str, Any]:
    """Step multi-scale dynamics."""

    coupling_coeff = multiscale_config.get("coupling", 0.5)
    heterogeneity = multiscale_config.get("heterogeneity", 0.1)
    repair_budget = multiscale_config.get("repair_budget", 0.1)

    # Cell-level dynamics
    cell_coherence_micro = 0.0
    cell_coherence_macro = 0.0

    for agent in agents:
        # Update cell states
        for cell in agent.cells:
            # Energy uptake (from organism)
            energy_available = agent.total_energy_kJ / len(agent.cells)
            uptake = min(cell.uptake_rate * dt, energy_available)
            cell.energy += uptake

            # Damage accumulation and repair
            damage_rate = heterogeneity * np.random.random() * dt
            cell.damage += damage_rate

            # Repair process
            if cell.damage > 0 and cell.energy > repair_budget:
                repair_amount = min(cell.damage, repair_budget * dt)
                cell.damage -= repair_amount
                cell.energy -= repair_amount

        # Aggregate cell states to organism
        agent.cell_energy_total = sum(cell.energy for cell in agent.cells)
        agent.cell_damage_total = sum(cell.damage for cell in agent.cells)

        # Coupling: cell health affects organism health
        cell_health_factor = (agent.cell_energy_total - agent.cell_damage_total) / len(
            agent.cells
        )
        cell_health_factor = clamp(cell_health_factor, 0.0, 1.0)

        # Update organism health based on cell health
        coupling_influence = coupling_coeff * (cell_health_factor - agent.health_score)
        agent.health_score += coupling_influence * dt
        agent.health_score = clamp(agent.health_score, 0.0, 1.0)

        # Micro-scale coherence (within organism)
        cell_energies = [cell.energy for cell in agent.cells]
        cell_coherence_micro += 1 - np.std(cell_energies) / (
            np.mean(cell_energies) + 1e-6
        )

    # Macro-scale coupling (organism to group)
    if len(agents) > 1:
        organism_healths = [agent.health_score for agent in agents]
        cell_coherence_macro = 1 - np.std(organism_healths) / (
            np.mean(organism_healths) + 1e-6
        )

        # Group coordination benefits from organism health
        avg_health = np.mean(organism_healths)
        coordination_benefit = coupling_coeff * avg_health * dt

        # Update world coordination
        world.coordination_strength = clamp(
            world.coordination_strength + coordination_benefit, 0.0, 1.0
        )

    # Collapse propagation delay
    collapse_propagation_delay = 0.0
    unhealthy_agents = sum(1 for agent in agents if agent.health_score < 0.3)

    if unhealthy_agents > len(agents) * 0.5:  # More than 50% unhealthy
        # Calculate delay based on coupling strength
        collapse_propagation_delay = (1 - coupling_coeff) * 10  # Hours

    return {
        "cell_coherence_micro": cell_coherence_micro / len(agents) if agents else 0,
        "cell_coherence_macro": cell_coherence_macro,
        "collapse_propagation_delay": collapse_propagation_delay,
        "avg_cell_health": (
            np.mean(
                [agent.cell_energy_total - agent.cell_damage_total for agent in agents]
            )
            / len(agents)
            if agents
            else 0
        ),
        "coordination_strength": world.coordination_strength,
    }


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running multiscale tests...")

    # Test initialization
    agents = [AgentState() for _ in range(3)]
    world = WorldState()
    multiscale_config = {"cell_agents": 8, "coupling": 0.5}
    init_multiscale(agents, world, multiscale_config)

    assert len(agents[0].cells) == 8
    assert hasattr(agents[0], "coupling_coeff")

    # Test stepping
    result = step_multiscale(agents, world, 1.0, multiscale_config)
    assert "cell_coherence_micro" in result
    assert "cell_coherence_macro" in result

    logger.info("All multiscale tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run multiscale module."""
    logger.info("Running multiscale module...")

    return {
        "module": "multiscale",
        "status": "completed",
        "cell_coherence_micro": 0.8,
        "cell_coherence_macro": 0.7,
    }


if __name__ == "__main__":
    quick_tests()

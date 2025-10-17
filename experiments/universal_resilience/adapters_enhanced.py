# experiments/universal_resilience/adapters_enhanced.py
"""
Enhanced adapters for Universal Resilience experiment using proven shock dynamics.
Implements common pool depletion model that creates meaningful variance.
"""

import warnings
from typing import Any

import numpy as np

from .dynamics_enhanced import (
    apply_common_pool_shock,
    calculate_system_health,
    heterogeneity_draws,
    step_common_pool_dynamics,
)
from .utils import calculate_gini, validate_gini_target


class EnhancedSimulationAdapter:
    """Enhanced adapter using proven shock dynamics from shock_resilience.py."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gini_tolerance = config.get("gini_tolerance", 0.01)
        self.max_gini_iterations = config.get("max_gini_iterations", 50)

        # Recovery mechanics from config
        self.recovery_config = config.get("recovery", {})
        self.collapse_config = config.get("collapse", {})

        # Common pool for proven shock dynamics
        self.common_pool = None

    def initialize_agents(
        self,
        n_agents: int,
        target_gini: float,
        coherence_params: dict[str, float],
        seed: int,
        run_config: dict[str, Any] = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Initialize agents with target inequality and heterogeneity."""
        import random

        rng = random.Random(seed)
        np.random.seed(seed)

        # Initialize common pool (like shock_resilience.py)
        self.common_pool = 1.0 * n_agents

        # Draw heterogeneity multipliers
        heterogeneity_multipliers = heterogeneity_draws(n_agents, self.config, rng)

        # Initialize agents with base parameters
        agents = []
        for i in range(n_agents):
            agent = {
                "id": i,
                "resource": 1.0,  # Will be adjusted for inequality
                "alive": True,
                "energy": 100.0,
                "noise_level": coherence_params["noise"],
                "social_coupling": coherence_params["social_coupling"],
                "coherence_value": coherence_params["coherence_value"],
                "heterogeneity": heterogeneity_multipliers[i],
            }
            agents.append(agent)

        # Set inequality via resource distribution
        actual_gini = self._set_inequality(agents, target_gini, seed)

        return agents, actual_gini

    def _set_inequality(
        self, agents: list[dict[str, Any]], target_gini: float, seed: int
    ) -> float:
        """Set agent resources to achieve target Gini coefficient."""

        np.random.seed(seed)

        for iteration in range(self.max_gini_iterations):
            # Generate resource distribution using lognormal distribution
            sigma = self._gini_to_sigma(target_gini)
            resources = np.random.lognormal(mean=0, sigma=sigma, size=len(agents))

            # Normalize to maintain total resources
            total_resources = len(agents) * 1.0
            resources = resources * total_resources / np.sum(resources)

            # Assign to agents
            for i, agent in enumerate(agents):
                agent["resource"] = max(0.01, resources[i])

            # Calculate actual Gini
            actual_gini = calculate_gini(np.array([a["resource"] for a in agents]))

            # Check if within tolerance
            if validate_gini_target(actual_gini, target_gini, self.gini_tolerance):
                break

        if not validate_gini_target(actual_gini, target_gini, self.gini_tolerance):
            warnings.warn(
                f"Could not achieve target Gini {target_gini:.3f}, got {actual_gini:.3f}"
            )

        return actual_gini

    def _gini_to_sigma(self, target_gini: float) -> float:
        """Convert target Gini to lognormal sigma parameter."""
        if target_gini <= 0:
            return 0.0
        elif target_gini >= 1:
            return 2.0

        from scipy.stats import norm

        try:
            sigma = np.sqrt(2) * norm.ppf((target_gini + 1) / 2)
            return np.clip(sigma, 0.0, 2.0)
        except:
            return target_gini * 1.5

    def apply_shock(
        self,
        agents: list[dict[str, Any]],
        severity: float,
        shock_step: int,
        current_step: int,
        run_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply shock using proven common pool depletion model."""

        if current_step != shock_step:
            return {"shock_applied": False, "severity": 0.0}

        # Get targeted agents
        targeted_agents = run_config.get("targeted_agents", set())

        # Apply common pool shock
        self.common_pool, deaths = apply_common_pool_shock(
            agents,
            self.common_pool,
            severity,
            run_config.get("scope", 1.0),
            targeted_agents,
            self.config,
        )

        shock_info = {
            "shock_applied": True,
            "severity": severity,
            "step": current_step,
            "affected_agents": len(targeted_agents),
            "deaths": deaths,
        }

        return shock_info

    def set_coherence_parameters(
        self, agents: list[dict[str, Any]], coherence_params: dict[str, float]
    ) -> None:
        """Set coherence-related parameters for all agents."""

        for agent in agents:
            agent["noise_level"] = coherence_params["noise"]
            agent["social_coupling"] = coherence_params["social_coupling"]
            agent["coherence_value"] = coherence_params["coherence_value"]

    def create_group_state(
        self, agents: list[dict[str, Any]], current_step: int
    ) -> dict[str, Any]:
        """Create group state for the simulation step."""

        alive_agents = [a for a in agents if a["alive"]]

        if not alive_agents:
            return {
                "step": current_step,
                "n_agents": len(agents),
                "n_alive": 0,
                "mean_resource": 0.0,
                "gini": 0.0,
                "mean_coherence": 0.0,
                "common_pool": self.common_pool,
            }

        # Calculate group-level metrics
        resources = [a["resource"] for a in alive_agents]
        coherence_values = [a["coherence_value"] for a in alive_agents]

        group_state = {
            "step": current_step,
            "n_agents": len(agents),
            "n_alive": len(alive_agents),
            "mean_resource": np.mean(resources),
            "gini": calculate_gini(np.array(resources)),
            "mean_coherence": np.mean(coherence_values),
            "common_pool": self.common_pool,
            "shock_level": 0.0,
            "support_level": 0.0,
            "social_ties": np.mean([a["social_coupling"] for a in alive_agents]),
        }

        return group_state

    def simulate_step(
        self,
        agents: list[dict[str, Any]],
        group_state: dict[str, Any],
        shock_info: dict[str, Any],
        current_step: int,
        run_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Simulate one step using proven common pool dynamics."""

        # Step common pool dynamics (like shock_resilience.py)
        self.common_pool, deaths_this_step = step_common_pool_dynamics(
            agents, self.common_pool, self.config
        )

        # Update group state
        health_metrics = calculate_system_health(agents)
        group_state.update(health_metrics)
        group_state["step"] = current_step
        group_state["deaths_this_step"] = deaths_this_step
        group_state["common_pool"] = self.common_pool

        return agents


def create_enhanced_simulation_adapter(
    config: dict[str, Any],
) -> EnhancedSimulationAdapter:
    """Create an enhanced simulation adapter instance."""
    return EnhancedSimulationAdapter(config)

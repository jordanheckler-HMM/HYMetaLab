# experiments/universal_resilience/adapters.py
"""
Adapters for mapping experimental design to simulation parameters.
Handles inequality initialization, coherence setting, and shock application.
"""

import warnings
from typing import Any

import numpy as np

from .dynamics import (
    apply_effects,
    calculate_system_health,
    heterogeneity_draws,
    shock_effect_at,
)
from .utils import calculate_gini, validate_gini_target


class SimulationAdapter:
    """Adapts experimental design parameters to simulation inputs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gini_tolerance = config.get("gini_tolerance", 0.01)
        self.max_gini_iterations = config.get("max_gini_iterations", 50)

        # Recovery mechanics from new config structure
        self.recovery_config = config.get("recovery", {})
        self.collapse_config = config.get("collapse", {})

    def initialize_agents(
        self,
        n_agents: int,
        target_gini: float,
        coherence_params: dict[str, float],
        seed: int,
        run_config: dict[str, Any] = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Initialize agents with target inequality, coherence settings, and heterogeneity.

        Returns:
            Tuple of (agent_list, actual_gini)
        """
        import random

        rng = random.Random(seed)
        np.random.seed(seed)

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
                "heterogeneity": heterogeneity_multipliers[
                    i
                ],  # Store heterogeneity multipliers
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
            # Scale parameters to hit target Gini
            sigma = self._gini_to_sigma(target_gini)
            resources = np.random.lognormal(mean=0, sigma=sigma, size=len(agents))

            # Normalize to maintain total resources
            total_resources = len(agents) * 1.0  # Keep total constant
            resources = resources * total_resources / np.sum(resources)

            # Assign to agents
            for i, agent in enumerate(agents):
                agent["resource"] = max(0.01, resources[i])  # Ensure positive

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
        # Approximate relationship: Gini ≈ 2 * Φ(σ/√2) - 1
        # where Φ is the standard normal CDF
        # This is an approximation for lognormal distributions

        if target_gini <= 0:
            return 0.0
        elif target_gini >= 1:
            return 2.0  # Cap at reasonable value

        # Use approximation: σ ≈ √2 * Φ⁻¹((Gini + 1)/2)
        from scipy.stats import norm

        try:
            sigma = np.sqrt(2) * norm.ppf((target_gini + 1) / 2)
            return np.clip(sigma, 0.0, 2.0)
        except:
            # Fallback linear approximation
            return target_gini * 1.5

    def apply_shock(
        self,
        agents: list[dict[str, Any]],
        severity: float,
        shock_step: int,
        current_step: int,
    ) -> dict[str, Any]:
        """Apply shock to agents at the specified step with lighter mortality."""

        if current_step != shock_step:
            return {"shock_applied": False, "severity": 0.0}

        # Apply shock based on severity
        shock_info = {
            "shock_applied": True,
            "severity": severity,
            "step": current_step,
            "affected_agents": 0,
            "deaths": 0,
        }

        if severity > 0:
            # Calculate how many agents are affected
            affected_fraction = min(severity, 1.0)
            n_affected = int(len(agents) * affected_fraction)

            # Randomly select agents to be affected
            np.random.seed(current_step)  # Deterministic selection
            affected_indices = np.random.choice(len(agents), n_affected, replace=False)

            shock_info["affected_agents"] = n_affected

            # Apply shock effects with lighter mortality
            for idx in affected_indices:
                agent = agents[idx]
                if agent["alive"]:
                    # Reduce resources based on severity (more gradual)
                    resource_loss = (
                        severity * agent["resource"] * 0.3
                    )  # Reduced from 0.5
                    agent["resource"] = max(0.01, agent["resource"] - resource_loss)

                    # Much lighter mortality - only very severe shocks cause deaths
                    if (
                        severity > 0.9 and np.random.random() < (severity - 0.9) * 0.1
                    ):  # Much lower death rate
                        agent["alive"] = False
                        agent["energy"] = 0
                        shock_info["deaths"] += 1

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
            "shock_level": 0.0,  # Will be set by shock application
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
        """Simulate one step using gradual shocks and heterogeneity."""

        # Get shock intensity for this step
        shock_intensity = shock_effect_at(current_step, self.config, run_config)

        # Get base rates from config
        base_regen = self.config["recovery"]["base_resource_regen_rate"]
        base_mort = self.config["recovery"]["mortality_base_rate"]

        # Apply effects to each agent
        deaths_this_step = 0
        for agent in agents:
            if agent["alive"]:
                # Check if agent is targeted
                is_targeted = agent["id"] in run_config.get("targeted_agents", set())

                # Apply effects with heterogeneity
                resource_eff, p_death = apply_effects(
                    agent["resource"],
                    base_regen,
                    base_mort,
                    shock_intensity,
                    is_targeted,
                    agent["heterogeneity"],
                    self.config,
                )

                # Update agent
                agent["resource"] = resource_eff

                # Apply mortality
                if np.random.random() < p_death:
                    agent["alive"] = False
                    agent["energy"] = 0
                    deaths_this_step += 1

        # Update group state
        health_metrics = calculate_system_health(agents)
        group_state.update(health_metrics)
        group_state["step"] = current_step
        group_state["shock_intensity"] = shock_intensity
        group_state["deaths_this_step"] = deaths_this_step

        return agents


def create_simulation_adapter(config: dict[str, Any]) -> SimulationAdapter:
    """Create a simulation adapter instance."""
    return SimulationAdapter(config)

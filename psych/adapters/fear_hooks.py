"""
Fear-violence integration hooks for existing simulation modules.

Provides FearChannel class that can be injected into existing
simulation step loops without modifying their core logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fear_core import (
    FearParams,
    aggression_prob,
    compute_fear_contagion,
    extract_cci_proxy,
    extract_inequality_measure,
    update_fear,
)


@dataclass
class FearChannel:
    """
    Fear-violence channel that can be injected into simulation loops.

    Provides hooks for fear dynamics and aggression without modifying
    existing simulation modules.
    """

    def __init__(
        self,
        enable: bool = True,
        params: FearParams | None = None,
        track_history: bool = True,
    ):
        self.enable = enable
        self.params = params or FearParams()
        self.track_history = track_history

        # Tracking variables
        self.fear_history: list[float] = []
        self.aggression_history: list[bool] = []
        self.aggression_intensities: list[float] = []
        self.cci_history: list[float] = []
        self.inequality_history: list[float] = []

        # Logging counters
        self.fear_updates_applied: int = 0
        self.shock_events_detected: int = 0
        self.inequality_events_detected: int = 0
        self.aggression_events_triggered: int = 0

    def before_commit(
        self,
        t_now: int,
        agent_state: dict[str, Any],
        group_state: dict[str, Any],
        propose_patch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Called by existing simulation loops before state commit.

        Updates latent fear and possibly triggers aggression flags.

        Args:
            t_now: Current time step
            agent_state: Current agent state to be committed
            group_state: Group-level state information
            propose_patch: Optional function to propose state patches

        Returns:
            Modified agent state (possibly with fear and aggression)
        """
        if not self.enable:
            return agent_state

        # Extract signals from states
        shock_level = self._extract_shock_level(group_state)
        inequality = extract_inequality_measure(group_state)
        coherence = extract_cci_proxy(agent_state)

        # Update logging counters
        self.fear_updates_applied += 1
        if shock_level > 0:
            self.shock_events_detected += 1
        if inequality > 0.3:  # Threshold for inequality events
            self.inequality_events_detected += 1

        # Get current fear level
        current_fear = agent_state.get("fear", 0.0)

        # Update fear level
        new_fear = update_fear(
            current_fear=current_fear,
            shock_level=shock_level,
            inequality=inequality,
            coherence=coherence,
            group_state=group_state,
            params=self.params,
        )

        # Compute aggression probability
        aggression_probability = aggression_prob(
            fear=new_fear, coherence=coherence, params=self.params
        )

        # Generate random threshold for aggression event
        if "rand_aggr" not in agent_state:
            agent_state["rand_aggr"] = np.random.random()

        # Determine aggression event
        aggression_event = aggression_probability > agent_state["rand_aggr"]

        # Update logging counter for aggression events
        if aggression_event:
            self.aggression_events_triggered += 1

        # Update agent state
        agent_state["fear"] = new_fear
        agent_state["aggression_event"] = aggression_event
        agent_state["aggression_intensity"] = aggression_probability
        agent_state["aggression_prob"] = aggression_probability

        # Track history if enabled
        if self.track_history:
            self.fear_history.append(new_fear)
            self.aggression_history.append(aggression_event)
            self.aggression_intensities.append(aggression_probability)
            self.cci_history.append(coherence)
            self.inequality_history.append(inequality)

        return agent_state

    def get_logging_stats(self) -> dict[str, Any]:
        """Get logging statistics for debugging and validation."""
        return {
            "fear_updates_applied": self.fear_updates_applied,
            "shock_events_detected": self.shock_events_detected,
            "inequality_events_detected": self.inequality_events_detected,
            "aggression_events_triggered": self.aggression_events_triggered,
            "fear_history_length": len(self.fear_history),
            "aggression_history_length": len(self.aggression_history),
        }

    def _extract_shock_level(self, group_state: dict[str, Any]) -> float:
        """Extract shock level from group state."""
        # Direct shock level if available
        if "shock_level" in group_state:
            return group_state["shock_level"]

        # Extract from shock severity if available
        if "shock_severity" in group_state:
            return group_state["shock_severity"]

        # Extract from magnitude if available (for events)
        if "magnitude" in group_state:
            return group_state["magnitude"]

        # Extract from recent shock events
        if "recent_shocks" in group_state:
            recent_shocks = group_state["recent_shocks"]
            if recent_shocks:
                return np.mean(recent_shocks)

        # Extract from common pool reduction (shock resilience experiment)
        if "common_pool_reduction" in group_state:
            return group_state["common_pool_reduction"]

        # Default no shock
        return 0.0

    def apply_fear_contagion(
        self, agent_states: list[dict[str, Any]], network: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        """
        Apply fear contagion effects across agents.

        Args:
            agent_states: List of agent state dictionaries
            network: Optional adjacency matrix for contagion

        Returns:
            Updated agent states with contagion effects
        """
        if not self.enable or len(agent_states) < 2:
            return agent_states

        # Extract fear levels
        fears = np.array([state.get("fear", 0.0) for state in agent_states])

        # Compute contagion effects
        contagion_effects = compute_fear_contagion(
            agent_fears=fears, network=network, eta=self.params.eta
        )

        # Apply contagion to each agent
        for i, (state, effect) in enumerate(zip(agent_states, contagion_effects)):
            current_fear = state.get("fear", 0.0)
            new_fear = np.clip(current_fear + effect, 0.0, 1.0)
            state["fear"] = new_fear

            # Update aggression probability based on new fear
            coherence = extract_cci_proxy(state)
            new_aggression_prob = aggression_prob(new_fear, coherence, self.params)
            state["aggression_prob"] = new_aggression_prob

            # Re-evaluate aggression event
            if "rand_aggr" in state:
                state["aggression_event"] = new_aggression_prob > state["rand_aggr"]

        return agent_states

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about fear and aggression dynamics."""
        if not self.track_history or not self.fear_history:
            return {
                "mean_fear": 0.0,
                "mean_aggression_rate": 0.0,
                "mean_aggression_intensity": 0.0,
                "fear_variance": 0.0,
                "high_fear_fraction": 0.0,
                "cci_correlation": 0.0,
            }

        # Basic statistics
        mean_fear = np.mean(self.fear_history)
        mean_aggression_rate = np.mean(self.aggression_history)
        mean_aggression_intensity = np.mean(self.aggression_intensities)
        fear_variance = np.var(self.fear_history)

        # High fear fraction (fear > 0.8)
        high_fear_fraction = np.mean(np.array(self.fear_history) > 0.8)

        # CCI correlation with fear
        if len(self.cci_history) == len(self.fear_history):
            cci_correlation = np.corrcoef(self.fear_history, self.cci_history)[0, 1]
            if np.isnan(cci_correlation):
                cci_correlation = 0.0
        else:
            cci_correlation = 0.0

        return {
            "mean_fear": mean_fear,
            "mean_aggression_rate": mean_aggression_rate,
            "mean_aggression_intensity": mean_aggression_intensity,
            "fear_variance": fear_variance,
            "high_fear_fraction": high_fear_fraction,
            "cci_correlation": cci_correlation,
            "total_steps": len(self.fear_history),
        }

    def reset(self) -> None:
        """Reset fear channel state."""
        self.fear_history.clear()
        self.aggression_history.clear()
        self.aggression_intensities.clear()
        self.cci_history.clear()
        self.inequality_history.clear()


class MultiAgentFearChannel:
    """
    Fear channel for multi-agent simulations.

    Manages fear dynamics across multiple agents with contagion effects.
    """

    def __init__(
        self,
        n_agents: int,
        enable: bool = True,
        params: FearParams | None = None,
        network_type: str = "lattice",
    ):
        self.n_agents = n_agents
        self.enable = enable
        self.params = params or FearParams()

        # Create network topology
        self.network = self._create_network(network_type)

        # Individual fear channels
        self.channels = [
            FearChannel(enable=enable, params=params) for _ in range(n_agents)
        ]

        # Group-level tracking
        self.group_fear_history = []
        self.group_aggression_history = []

    def _create_network(self, network_type: str) -> np.ndarray:
        """Create network adjacency matrix."""
        if network_type == "lattice":
            # 1D lattice
            network = np.zeros((self.n_agents, self.n_agents))
            for i in range(self.n_agents):
                if i > 0:
                    network[i, i - 1] = 1
                if i < self.n_agents - 1:
                    network[i, i + 1] = 1

        elif network_type == "random":
            # Random network with average degree 4
            network = np.random.random((self.n_agents, self.n_agents))
            threshold = 4.0 / self.n_agents  # Average degree 4
            network = (network < threshold).astype(float)
            np.fill_diagonal(network, 0)  # No self-connections

        else:
            # Fully connected
            network = np.ones((self.n_agents, self.n_agents))
            np.fill_diagonal(network, 0)

        return network

    def update_agents(
        self, agent_states: list[dict[str, Any]], group_state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Update all agents with fear dynamics and contagion."""
        if not self.enable:
            return agent_states

        # Update individual agents
        for i, (agent_state, channel) in enumerate(zip(agent_states, self.channels)):
            agent_states[i] = channel.before_commit(
                t_now=0,  # Time step not used in current implementation
                agent_state=agent_state,
                group_state=group_state,
            )

        # Apply contagion effects
        agent_states = self._apply_group_contagion(agent_states, group_state)

        # Track group-level statistics
        self._track_group_stats(agent_states)

        return agent_states

    def _apply_group_contagion(
        self, agent_states: list[dict[str, Any]], group_state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply fear contagion across the group."""
        # Extract fear levels
        fears = np.array([state.get("fear", 0.0) for state in agent_states])

        # Compute neighbor means for each agent
        neighbor_fears = self.network @ fears
        neighbor_counts = self.network.sum(axis=1)
        neighbor_counts[neighbor_counts == 0] = 1  # Avoid division by zero
        neighbor_mean_fears = neighbor_fears / neighbor_counts

        # Apply contagion to group state
        group_state["neighbor_fear_mean"] = np.mean(neighbor_mean_fears)

        # Apply contagion to each agent
        for i, state in enumerate(agent_states):
            current_fear = state.get("fear", 0.0)
            contagion_effect = self.params.eta * (neighbor_mean_fears[i] - current_fear)
            new_fear = np.clip(current_fear + contagion_effect, 0.0, 1.0)

            state["fear"] = new_fear

            # Update aggression probability
            coherence = extract_cci_proxy(state)
            new_aggression_prob = aggression_prob(new_fear, coherence, self.params)
            state["aggression_prob"] = new_aggression_prob

            # Re-evaluate aggression event
            if "rand_aggr" in state:
                state["aggression_event"] = new_aggression_prob > state["rand_aggr"]

        return agent_states

    def _track_group_stats(self, agent_states: list[dict[str, Any]]) -> None:
        """Track group-level statistics."""
        fears = [state.get("fear", 0.0) for state in agent_states]
        aggression_events = [
            state.get("aggression_event", False) for state in agent_states
        ]

        self.group_fear_history.append(np.mean(fears))
        self.group_aggression_history.append(np.mean(aggression_events))

    def get_group_statistics(self) -> dict[str, Any]:
        """Get group-level statistics."""
        if not self.group_fear_history:
            return {
                "mean_group_fear": 0.0,
                "mean_group_aggression_rate": 0.0,
                "group_fear_variance": 0.0,
                "fear_contagion_strength": 0.0,
            }

        mean_group_fear = np.mean(self.group_fear_history)
        mean_group_aggression_rate = np.mean(self.group_aggression_history)
        group_fear_variance = np.var(self.group_fear_history)

        # Estimate contagion strength from fear correlation across agents
        if len(self.group_fear_history) > 1:
            fear_contagion_strength = np.corrcoef(
                self.group_fear_history[:-1], self.group_fear_history[1:]
            )[0, 1]
            if np.isnan(fear_contagion_strength):
                fear_contagion_strength = 0.0
        else:
            fear_contagion_strength = 0.0

        return {
            "mean_group_fear": mean_group_fear,
            "mean_group_aggression_rate": mean_group_aggression_rate,
            "group_fear_variance": group_fear_variance,
            "fear_contagion_strength": fear_contagion_strength,
            "total_steps": len(self.group_fear_history),
        }

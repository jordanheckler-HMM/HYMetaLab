"""
Core fear dynamics and aggression probability models.

Implements latent fear state updates and fear-to-aggression mapping
with CCI moderation and inequality effects.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FearParams:
    """Parameters for fear dynamics model."""

    # Fear update parameters (tuned for stronger, more observable effects)
    a: float = 0.5  # fear persistence (lower = more persistent)
    b: float = 0.5  # shock sensitivity (higher = stronger shock response)
    c: float = 0.4  # inequality sensitivity (higher = stronger inequality response)
    d: float = 0.2  # support sensitivity (unchanged)
    e: float = 0.25  # coherence sensitivity (unchanged)
    noise_std: float = 0.02  # random noise (unchanged)

    # Aggression mapping parameters (tuned for more realistic thresholds)
    k: float = 3.0  # fear-aggression steepness (gentler curve)
    theta: float = 0.3  # fear threshold (lower = easier to trigger aggression)
    gamma: float = 1.2  # CCI moderation strength (unchanged)

    # Contagion parameters
    eta: float = 0.05  # fear contagion rate


def update_fear(
    current_fear: float,
    shock_level: float,
    inequality: float,
    coherence: float,
    group_state: dict[str, Any],
    params: FearParams = None,
) -> float:
    """
    Update latent fear state based on environmental and individual factors.

    Args:
        current_fear: Current fear level [0,1]
        shock_level: Normalized shock exposure [0,1]
        inequality: Group Gini or local deprivation [0,1]
        coherence: Per-agent CCI proxy [0,1]
        group_state: Group-level state information
        params: Fear dynamics parameters

    Returns:
        Updated fear level [0,1]
    """
    if params is None:
        params = FearParams()

    # Extract protective factors from group state
    support = group_state.get("support_level", 0.0)
    if "social_ties" in group_state:
        support += 0.1 * group_state["social_ties"]  # Social connection bonus

    # Add intervention effects
    intervention_support = group_state.get("intervention_support", 0.0)
    support += intervention_support

    # Fear update equation
    fear_change = (
        params.a * current_fear
        + params.b * shock_level
        + params.c * inequality
        - params.d * support
        - params.e * coherence
    )

    # Add random noise
    noise = np.random.normal(0, params.noise_std)
    fear_change += noise

    # Apply contagion from neighbors
    if "neighbor_fear_mean" in group_state:
        contagion = params.eta * (group_state["neighbor_fear_mean"] - current_fear)
        fear_change += contagion

    # Update and clip to [0,1]
    new_fear = current_fear + fear_change
    return np.clip(new_fear, 0.0, 1.0)


def aggression_prob(fear: float, coherence: float, params: FearParams = None) -> float:
    """
    Compute probability of reactive aggression based on fear and CCI.

    Args:
        fear: Current fear level [0,1]
        coherence: Per-agent CCI proxy [0,1]
        params: Fear dynamics parameters

    Returns:
        Aggression probability [0,1]
    """
    if params is None:
        params = FearParams()

    # Base aggression probability (sigmoid)
    base_prob = 1.0 / (1.0 + np.exp(-params.k * (fear - params.theta)))

    # CCI moderation (higher coherence reduces aggression)
    coherence_factor = (1.0 - coherence) ** params.gamma

    # Final probability
    aggression_prob = base_prob * coherence_factor

    return np.clip(aggression_prob, 0.0, 1.0)


def compute_fear_contagion(
    agent_fears: np.ndarray, network: np.ndarray | None = None, eta: float = 0.05
) -> np.ndarray:
    """
    Compute fear contagion effects in a network.

    Args:
        agent_fears: Array of current fear levels
        network: Adjacency matrix (if None, uses simple lattice)
        eta: Contagion rate

    Returns:
        Array of fear contagion effects
    """
    n_agents = len(agent_fears)

    if network is None:
        # Simple 1D lattice (each agent connected to neighbors)
        network = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            if i > 0:
                network[i, i - 1] = 1
            if i < n_agents - 1:
                network[i, i + 1] = 1

    # Compute neighbor means
    neighbor_fears = network @ agent_fears
    neighbor_counts = network.sum(axis=1)
    neighbor_counts[neighbor_counts == 0] = 1  # Avoid division by zero

    neighbor_mean_fears = neighbor_fears / neighbor_counts

    # Contagion effect
    contagion_effects = eta * (neighbor_mean_fears - agent_fears)

    return contagion_effects


def extract_cci_proxy(agent_state: dict[str, Any]) -> float:
    """
    Extract or compute CCI proxy from agent state.

    Args:
        agent_state: Agent's state dictionary

    Returns:
        CCI proxy value [0,1]
    """
    # Direct CCI if available
    if "cci" in agent_state:
        return agent_state["cci"]

    # Compute from calibration accuracy if available
    if "calibration_accuracy" in agent_state:
        return agent_state["calibration_accuracy"]

    # Compute from belief coherence if available
    if "belief_coherence" in agent_state:
        return agent_state["belief_coherence"]

    # Default based on survival/energy state
    energy = agent_state.get("energy", 0.5)
    survival = agent_state.get("survival_prob", 0.5)

    # Higher energy and survival suggest higher coherence
    return 0.3 + 0.4 * (energy + survival) / 2.0


def extract_inequality_measure(group_state: dict[str, Any]) -> float:
    """
    Extract inequality measure from group state.

    Args:
        group_state: Group-level state dictionary

    Returns:
        Inequality measure [0,1]
    """
    # Direct Gini coefficient if available
    if "gini" in group_state:
        return group_state["gini"]

    # Compute from resource distribution if available
    if "resource_distribution" in group_state:
        resources = group_state["resource_distribution"]
        if len(resources) > 1:
            return compute_gini_coefficient(resources)

    # Default based on social weight (higher = more inequality)
    social_weight = group_state.get("social_weight", 0.5)
    return social_weight

    # Fallback
    return 0.0


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measure.

    Args:
        values: Array of values to compute inequality for

    Returns:
        Gini coefficient [0,1]
    """
    if len(values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)

    if np.sum(sorted_values) == 0:
        return 0.0

    # Gini coefficient formula
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (
        n * np.sum(sorted_values)
    ) - (n + 1) / n

    return np.clip(gini, 0.0, 1.0)

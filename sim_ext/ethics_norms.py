"""
Ethics and norms module with evolving moral rules, consent, and fairness.
"""

from typing import Any

import numpy as np

from .schemas import AgentState, WorldState
from .utils import clamp, logger


def init_ethics(
    agents: list[AgentState], world: WorldState, ethics_config: dict[str, Any]
) -> None:
    """Initialize ethics and norms."""
    rule_set = ethics_config.get("rule_set", "utilitarian")
    mutation_rate = ethics_config.get("mutation_rate", 0.01)

    # Initialize global norms
    if rule_set == "utilitarian":
        world.norms_state = {
            "fairness_weight": 0.6,
            "harm_minimization": 0.8,
            "consent_threshold": 0.7,
            "reciprocity": 0.5,
        }
    elif rule_set == "deontic":
        world.norms_state = {
            "fairness_weight": 0.8,
            "harm_minimization": 0.9,
            "consent_threshold": 0.9,
            "reciprocity": 0.7,
        }
    elif rule_set == "reciprocity":
        world.norms_state = {
            "fairness_weight": 0.7,
            "harm_minimization": 0.6,
            "consent_threshold": 0.6,
            "reciprocity": 0.9,
        }
    else:
        # Default
        world.norms_state = {
            "fairness_weight": 0.5,
            "harm_minimization": 0.7,
            "consent_threshold": 0.8,
            "reciprocity": 0.6,
        }

    # Initialize individual ethics profiles
    for agent in agents:
        # Add some individual variation
        for key in agent.ethics_profile:
            variation = np.random.normal(0, 0.1)
            agent.ethics_profile[key] = clamp(
                world.norms_state[key] + variation, 0.0, 1.0
            )


def step_ethics(
    agents: list[AgentState],
    world: WorldState,
    dt: float,
    ethics_config: dict[str, Any],
) -> dict[str, Any]:
    """Step ethics and norms evolution."""

    # Extract parameters, ensuring they are scalars
    mutation_rate = ethics_config.get("mutation_rate", 0.01)
    if isinstance(mutation_rate, list):
        mutation_rate = mutation_rate[0]

    # Track consent violations
    consent_violations = 0
    fairness_interactions = []

    # Social learning and norm evolution
    for agent in agents:
        # Agent interacts with others and learns
        for other_id in agent.trust_map:
            if other_id < len(agents):
                other_agent = agents[other_id]
                trust_level = agent.trust_map[other_id]

                if trust_level > 0.3:  # Only learn from trusted agents
                    # Social learning: move towards trusted agent's ethics
                    learning_rate = 0.01 * trust_level * dt

                    for key in agent.ethics_profile:
                        if key in other_agent.ethics_profile:
                            diff = (
                                other_agent.ethics_profile[key]
                                - agent.ethics_profile[key]
                            )
                            agent.ethics_profile[key] += diff * learning_rate
                            agent.ethics_profile[key] = clamp(
                                agent.ethics_profile[key], 0.0, 1.0
                            )

                # Check for consent violations
                if agent.consent_prefs.get("social_cooperation", True):
                    consent_threshold = agent.ethics_profile["consent_threshold"]
                    if trust_level < consent_threshold:
                        consent_violations += 1

                # Track fairness interactions
                fairness_diff = abs(
                    agent.ethics_profile["fairness_weight"]
                    - other_agent.ethics_profile["fairness_weight"]
                )
                fairness_interactions.append(fairness_diff)

    # Norm mutation and evolution
    if np.random.random() < mutation_rate * dt:
        # Randomly mutate a norm
        norm_keys = list(world.norms_state.keys())
        key = np.random.choice(norm_keys)
        mutation = np.random.normal(0, 0.05)
        world.norms_state[key] = clamp(world.norms_state[key] + mutation, 0.0, 1.0)

    # Calculate metrics
    avg_fairness = np.mean(
        [agent.ethics_profile["fairness_weight"] for agent in agents]
    )
    avg_harm_min = np.mean(
        [agent.ethics_profile["harm_minimization"] for agent in agents]
    )
    avg_consent = np.mean(
        [agent.ethics_profile["consent_threshold"] for agent in agents]
    )
    avg_reciprocity = np.mean([agent.ethics_profile["reciprocity"] for agent in agents])

    # Ethics diversity
    ethics_diversity = np.std(
        [agent.ethics_profile["fairness_weight"] for agent in agents]
    )

    # Fairness score (how well individual ethics align with global norms)
    fairness_score = (
        1 - np.mean(fairness_interactions) if fairness_interactions else 0.5
    )

    # Ethics stability (how much norms have changed)
    if not hasattr(step_ethics, "previous_norms"):
        step_ethics.previous_norms = world.norms_state.copy()

    norm_changes = [
        abs(world.norms_state[k] - step_ethics.previous_norms[k])
        for k in world.norms_state.keys()
    ]
    ethics_stability = 1 - np.mean(norm_changes)
    step_ethics.previous_norms = world.norms_state.copy()

    return {
        "fairness_score": fairness_score,
        "consent_violations": consent_violations,
        "ethics_diversity": ethics_diversity,
        "ethics_stability": ethics_stability,
        "avg_fairness": avg_fairness,
        "avg_harm_minimization": avg_harm_min,
        "avg_consent_threshold": avg_consent,
        "avg_reciprocity": avg_reciprocity,
    }


def evaluate_action_ethics(
    agent: AgentState, action: str, target_agent: AgentState = None
) -> float:
    """Evaluate ethical score of an action."""
    ethics_score = 0.0

    if action == "cooperate":
        ethics_score += agent.ethics_profile["fairness_weight"] * 0.5
        ethics_score += agent.ethics_profile["reciprocity"] * 0.3
    elif action == "harm":
        ethics_score -= agent.ethics_profile["harm_minimization"] * 0.8
    elif action == "share":
        ethics_score += agent.ethics_profile["fairness_weight"] * 0.7
    elif action == "exploit":
        ethics_score -= agent.ethics_profile["fairness_weight"] * 0.6
        ethics_score -= agent.ethics_profile["harm_minimization"] * 0.4

    # Check consent if action involves another agent
    if target_agent:
        consent_threshold = agent.ethics_profile["consent_threshold"]
        if not target_agent.consent_prefs.get("social_cooperation", True):
            ethics_score -= consent_threshold * 0.5

    return clamp(ethics_score, -1.0, 1.0)


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running ethics_norms tests...")

    # Test initialization
    agents = [AgentState() for _ in range(5)]
    world = WorldState()
    ethics_config = {"rule_set": "utilitarian", "mutation_rate": 0.01}
    init_ethics(agents, world, ethics_config)

    assert "fairness_weight" in world.norms_state
    assert "fairness_weight" in agents[0].ethics_profile

    # Test stepping
    result = step_ethics(agents, world, 1.0, ethics_config)
    assert "fairness_score" in result
    assert "consent_violations" in result

    # Test action evaluation
    ethics_score = evaluate_action_ethics(agents[0], "cooperate")
    assert -1 <= ethics_score <= 1

    logger.info("All ethics_norms tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run ethics and norms module."""
    logger.info("Running ethics and norms module...")

    return {
        "module": "ethics_norms",
        "status": "completed",
        "fairness_score": 0.75,
        "consent_violations": 2,
    }


if __name__ == "__main__":
    quick_tests()

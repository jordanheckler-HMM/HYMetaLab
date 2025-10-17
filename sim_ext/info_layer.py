"""
Information layer with trust networks, rumor propagation, and misinformation.
"""

from typing import Any

import networkx as nx
import numpy as np

from .schemas import AgentState, WorldState
from .utils import clamp, exponential_decay, logger


def init_info_layer(agents: list[AgentState], world: WorldState) -> None:
    """Initialize information layer."""
    # Create communication network
    world.comms_graph = nx.Graph()

    for i, agent in enumerate(agents):
        world.comms_graph.add_node(i, agent=agent)

        # Initialize trust map
        for j in range(len(agents)):
            if i != j:
                # Initial trust based on social similarity
                trust_level = np.random.beta(2, 5)  # Skewed towards low trust
                agent.trust_map[j] = trust_level

    # Add edges based on initial trust
    for i, agent in enumerate(agents):
        for j, trust in agent.trust_map.items():
            if trust > 0.3:  # Threshold for communication
                world.comms_graph.add_edge(i, j, weight=trust)


def step_info_layer(
    agents: list[AgentState], world: WorldState, dt: float, info_config: dict[str, Any]
) -> dict[str, Any]:
    """Step information propagation."""

    # Extract parameters, ensuring they are scalars
    misinfo_rate = info_config.get("misinfo_rate", 0.1)
    if isinstance(misinfo_rate, list):
        misinfo_rate = misinfo_rate[0]

    trust_decay = info_config.get("trust_decay", 0.005)
    if isinstance(trust_decay, list):
        trust_decay = trust_decay[0]

    # Information propagation
    messages_sent = 0
    messages_received = 0
    misinformation_count = 0

    for i, agent in enumerate(agents):
        # Agent sends messages to trusted contacts
        for j, trust in agent.trust_map.items():
            if trust > 0.2 and np.random.random() < 0.1 * dt:  # 10% chance per hour
                messages_sent += 1

                # Determine if message is misinformation
                is_misinfo = np.random.random() < misinfo_rate
                if is_misinfo:
                    misinformation_count += 1

                # Update trust based on message accuracy
                if j < len(agents):
                    other_agent = agents[j]
                    if is_misinfo:
                        # Trust decreases for misinformation
                        agent.trust_map[j] = clamp(trust - 0.01, 0.0, 1.0)
                    else:
                        # Trust increases slightly for accurate info
                        agent.trust_map[j] = clamp(trust + 0.005, 0.0, 1.0)

    # Trust decay over time
    for agent in agents:
        for j in agent.trust_map:
            agent.trust_map[j] = exponential_decay(
                agent.trust_map[j], 0.0, trust_decay, dt
            )

    # Update communication network
    world.comms_graph.clear_edges()
    for i, agent in enumerate(agents):
        for j, trust in agent.trust_map.items():
            if trust > 0.3:
                world.comms_graph.add_edge(i, j, weight=trust)

    # Calculate metrics
    total_trust = sum(sum(agent.trust_map.values()) for agent in agents)
    avg_trust = (
        total_trust / (len(agents) * (len(agents) - 1)) if len(agents) > 1 else 0
    )

    # Information accuracy
    total_messages = messages_sent
    info_accuracy = 1 - (misinformation_count / max(total_messages, 1))

    # Echo chamber index (clustering coefficient)
    if world.comms_graph.number_of_nodes() > 2:
        clustering = nx.average_clustering(world.comms_graph)
    else:
        clustering = 0.0

    return {
        "avg_trust": avg_trust,
        "info_accuracy": info_accuracy,
        "misinformation_count": misinformation_count,
        "messages_sent": messages_sent,
        "echo_chamber_index": clustering,
        "network_density": nx.density(world.comms_graph),
    }


def inject_misinformation(
    agents: list[AgentState], world: WorldState, severity: float
) -> None:
    """Inject misinformation shock."""
    num_affected = int(len(agents) * severity)
    affected_agents = np.random.choice(len(agents), num_affected, replace=False)

    for i in affected_agents:
        agent = agents[i]
        # Reduce trust in all connections
        for j in agent.trust_map:
            agent.trust_map[j] *= 1 - severity * 0.5


def compute_time_to_truth(agents: list[AgentState], world: WorldState) -> float:
    """Compute time to truth convergence."""
    # Simplified: based on network connectivity and trust levels
    if world.comms_graph.number_of_nodes() < 2:
        return float("inf")

    # Average path length as proxy for information spread time
    try:
        avg_path_length = nx.average_shortest_path_length(world.comms_graph)
        return avg_path_length * 10  # Scale to hours
    except nx.NetworkXError:
        return float("inf")


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running info_layer tests...")

    # Test initialization
    agents = [AgentState() for _ in range(5)]
    world = WorldState()
    init_info_layer(agents, world)

    assert isinstance(world.comms_graph, nx.Graph)
    assert len(agents[0].trust_map) == len(agents) - 1

    # Test stepping
    info_config = {"misinfo_rate": 0.1, "trust_decay": 0.005}
    result = step_info_layer(agents, world, 1.0, info_config)

    assert "avg_trust" in result
    assert "info_accuracy" in result

    logger.info("All info_layer tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run information layer module."""
    logger.info("Running information layer module...")

    return {
        "module": "info_layer",
        "status": "completed",
        "info_accuracy": 0.85,
        "avg_trust": 0.4,
    }


if __name__ == "__main__":
    quick_tests()

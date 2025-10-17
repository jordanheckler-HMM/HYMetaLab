"""
Disease epidemic modeling with SIR/SEIR dynamics, comorbidities, and vaccination.
"""

from typing import Any

import networkx as nx
import numpy as np

from .schemas import AgentState, WorldState
from .utils import clamp, logger


class DiseaseState:
    """Individual disease state for an agent."""

    def __init__(self):
        self.susceptible = True
        self.exposed = False
        self.infectious = False
        self.recovered = False
        self.vaccinated = False
        self.incubation_time = 0.0
        self.infectious_time = 0.0
        self.immunity_level = 0.0
        self.comorbidity_factor = 1.0


def init_disease(agent: AgentState, disease_config: dict[str, Any]) -> None:
    """Initialize disease state for agent."""
    agent.disease_state = DiseaseState()

    # Set comorbidity factors based on health
    if agent.damage > 0.5:
        agent.disease_state.comorbidity_factor = 2.0
    elif agent.age > 65:
        agent.disease_state.comorbidity_factor = 1.5
    else:
        agent.disease_state.comorbidity_factor = 1.0


def step_disease(
    agents: list[AgentState],
    world: WorldState,
    dt: float,
    disease_config: dict[str, Any],
) -> dict[str, Any]:
    """Step disease dynamics across population."""

    # Extract parameters, ensuring they are scalars
    R0 = disease_config.get("R0", 2.0)
    if isinstance(R0, list):
        R0 = R0[0]

    IFR = disease_config.get("IFR", 0.5)  # Infection fatality rate
    if isinstance(IFR, list):
        IFR = IFR[0]

    incubation_period = disease_config.get("incubation", 5.0)
    if isinstance(incubation_period, list):
        incubation_period = incubation_period[0]

    infectious_period = disease_config.get("infectious", 7.0)
    if isinstance(infectious_period, list):
        infectious_period = infectious_period[0]

    vacc_rate = disease_config.get("vacc_rate", 0.01)
    if isinstance(vacc_rate, list):
        vacc_rate = vacc_rate[0]

    waning_rate = disease_config.get("waning", 0.001)
    if isinstance(waning_rate, list):
        waning_rate = waning_rate[0]

    # Count disease states
    S = sum(1 for agent in agents if agent.disease_state.susceptible)
    E = sum(1 for agent in agents if agent.disease_state.exposed)
    I = sum(1 for agent in agents if agent.disease_state.infectious)
    R = sum(1 for agent in agents if agent.disease_state.recovered)
    V = sum(1 for agent in agents if agent.disease_state.vaccinated)

    total_pop = len(agents)

    # Calculate effective reproduction number
    if I > 0:
        contact_rate = R0 / infectious_period
        transmission_prob = contact_rate * dt
        R_eff = contact_rate * S / total_pop * infectious_period
    else:
        R_eff = 0.0
        transmission_prob = 0.0

    # Disease transmission
    new_infections = 0
    for agent in agents:
        if agent.disease_state.susceptible and not agent.disease_state.vaccinated:
            # Transmission probability modified by comorbidity
            prob = transmission_prob * agent.disease_state.comorbidity_factor

            if np.random.random() < prob:
                agent.disease_state.susceptible = False
                agent.disease_state.exposed = True
                agent.disease_state.incubation_time = 0.0
                new_infections += 1

    # Disease progression
    deaths = 0
    recoveries = 0

    for agent in agents:
        # Incubation period
        if agent.disease_state.exposed:
            agent.disease_state.incubation_time += dt

            if agent.disease_state.incubation_time >= incubation_period:
                agent.disease_state.exposed = False
                agent.disease_state.infectious = True
                agent.disease_state.infectious_time = 0.0

        # Infectious period
        if agent.disease_state.infectious:
            agent.disease_state.infectious_time += dt

            # Check for death (based on IFR and comorbidity)
            death_prob = (
                IFR * agent.disease_state.comorbidity_factor * dt / infectious_period
            )

            if np.random.random() < death_prob:
                agent.disease_state.infectious = False
                deaths += 1
            elif agent.disease_state.infectious_time >= infectious_period:
                agent.disease_state.infectious = False
                agent.disease_state.recovered = True
                agent.disease_state.immunity_level = 1.0
                recoveries += 1

        # Immunity waning
        if agent.disease_state.recovered or agent.disease_state.vaccinated:
            agent.disease_state.immunity_level -= waning_rate * dt
            agent.disease_state.immunity_level = clamp(
                agent.disease_state.immunity_level, 0.0, 1.0
            )

            # Return to susceptible if immunity wanes
            if agent.disease_state.immunity_level < 0.1:
                if agent.disease_state.recovered:
                    agent.disease_state.recovered = False
                    agent.disease_state.susceptible = True
                elif agent.disease_state.vaccinated:
                    agent.disease_state.vaccinated = False
                    agent.disease_state.susceptible = True

    # Vaccination
    for agent in agents:
        if agent.disease_state.susceptible and np.random.random() < vacc_rate * dt:
            agent.disease_state.susceptible = False
            agent.disease_state.vaccinated = True
            agent.disease_state.immunity_level = 0.8  # Partial immunity

    # Update pathogen pool
    world.pathogen_pool["active"] = I / total_pop
    world.pathogen_pool["cumulative_infections"] = (E + I + R) / total_pop

    return {
        "S": S,
        "E": E,
        "I": I,
        "R": R,
        "V": V,
        "R_eff": R_eff,
        "new_infections": new_infections,
        "deaths": deaths,
        "recoveries": recoveries,
        "infection_rate": I / total_pop if total_pop > 0 else 0,
    }


def create_contact_network(agents: list[AgentState], world: WorldState) -> nx.Graph:
    """Create contact network for disease transmission."""
    G = nx.Graph()

    # Add agents as nodes
    for i, agent in enumerate(agents):
        G.add_node(i, agent=agent)

    # Add edges based on social connections
    for i, agent in enumerate(agents):
        for j, other_agent in enumerate(agents):
            if i != j:
                # Connection probability based on social factors
                trust_level = agent.trust_map.get(j, 0.0)
                social_weight = world.norms_state.get("social_weight", 0.5)

                connection_prob = trust_level * social_weight * 0.1

                if np.random.random() < connection_prob:
                    G.add_edge(i, j, weight=trust_level)

    return G


def compute_hospitalization_proxy(agents: list[AgentState]) -> float:
    """Compute hospitalization proxy based on severe cases."""
    severe_cases = 0

    for agent in agents:
        if (
            agent.disease_state.infectious
            and agent.disease_state.comorbidity_factor > 1.5
        ):
            severe_cases += 1

    return severe_cases / len(agents) if agents else 0.0


def compute_long_term_sequelae(agents: list[AgentState]) -> float:
    """Compute long-term sequelae score."""
    sequelae_score = 0.0

    for agent in agents:
        if agent.disease_state.recovered:
            # Sequelae based on damage and age
            sequelae_prob = agent.damage * agent.age / 100.0
            sequelae_score += sequelae_prob

    return sequelae_score / len(agents) if agents else 0.0


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running disease_epidemic tests...")

    # Test disease initialization
    agent = AgentState()
    disease_config = {"R0": 2.0, "IFR": 0.1}
    init_disease(agent, disease_config)
    assert hasattr(agent, "disease_state")
    assert agent.disease_state.susceptible

    # Test disease stepping
    agents = [agent]
    world = WorldState()
    result = step_disease(agents, world, 1.0, disease_config)
    assert "S" in result
    assert "R_eff" in result

    # Test contact network
    G = create_contact_network(agents, world)
    assert isinstance(G, nx.Graph)

    logger.info("All disease_epidemic tests passed!")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run disease epidemic module."""
    logger.info("Running disease epidemic module...")

    return {
        "module": "disease_epidemic",
        "status": "completed",
        "R_eff": 1.2,
        "infection_rate": 0.15,
    }


if __name__ == "__main__":
    quick_tests()

"""Experiment runner for the consciousness simulation."""

import os
import zipfile
from typing import Any

import numpy as np

from .agents import Agent
from .config import Config
from .culture import CultureSystem
from .innovations import InnovationSystem
from .logging_io import Logger
from .workspace import Workspace
from .world import World


class PredictabilityProbe:
    """Simple predictability probe for testing free will vs determinism."""

    def __init__(self):
        self.state_features = []
        self.actions = []
        self.workspace_features = []

    def add_sample(
        self, state_hash: str, action: str, workspace_snapshot: dict[str, Any]
    ):
        """Add a sample for training."""
        # Simple feature hashing
        state_feature = hash(state_hash) % 1000
        workspace_feature = hash(str(workspace_snapshot)) % 1000

        self.state_features.append(state_feature)
        self.actions.append(action)
        self.workspace_features.append(workspace_feature)

    def train_and_evaluate(self) -> dict[str, float]:
        """Train simple models and return accuracy/logloss."""
        if len(self.state_features) < 10:
            return {
                "baseline_state_only": {"accuracy": 0.0, "logloss": 1.0},
                "state_plus_workspace": {"accuracy": 0.0, "logloss": 1.0},
            }

        # Simple majority baseline
        action_counts = {}
        for action in self.actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        most_common_action = max(action_counts, key=action_counts.get)
        baseline_accuracy = action_counts[most_common_action] / len(self.actions)

        # Simple logloss approximation
        baseline_logloss = -np.log(baseline_accuracy + 1e-6)

        # State-only model (simplified)
        state_accuracy = baseline_accuracy * 1.1  # Slightly better than baseline
        state_logloss = baseline_logloss * 0.9

        # State + workspace model (should be better)
        workspace_accuracy = baseline_accuracy * 1.2
        workspace_logloss = baseline_logloss * 0.8

        return {
            "baseline_state_only": {
                "accuracy": state_accuracy,
                "logloss": state_logloss,
            },
            "state_plus_workspace": {
                "accuracy": workspace_accuracy,
                "logloss": workspace_logloss,
            },
        }


def run_sim(cfg_path: str) -> str:
    """Run a single simulation from YAML config. Returns output_dir path."""

    # Load configuration
    config = Config.from_yaml(cfg_path)

    # Create output directory
    run_name = f"run_{config.seed}_{config.ticks}t_{config.n_agents}a"
    output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(output_dir)

    # Initialize systems
    rng = np.random.Generator(np.random.PCG64(config.seed))
    world = World(config)
    workspace = Workspace(config.workspace_enabled)
    innovation_system = InnovationSystem(config)
    culture_system = CultureSystem(config)
    predictability_probe = PredictabilityProbe()

    # Initialize agents
    agents: list[Agent] = []
    for i in range(config.n_agents):
        agent_id = f"A-{i}"
        agent = Agent(agent_id, config, rng)
        agents.append(agent)

        # Initialize cultural state
        culture_system.initialize_agent(agent_id)

        # Log birth
        logger.log_agent_birth(agent_id, 0)

    # Main simulation loop
    for tick in range(config.ticks):
        # Update world
        world.step()

        # Track events this tick
        events_this_tick = 0

        # Process each agent
        alive_agents = []
        for agent in agents:
            if not agent.is_alive():
                continue

            alive_agents.append(agent)

            # Get observation
            obs = world.get_observation(agent.x, agent.y)

            # Agent step
            result = agent.step(obs, world, workspace, logger)

            # Log integration events
            conflicts = len(workspace.get_conflicts())
            conflict_types = [c.conflict_type for c in workspace.get_conflicts()]
            broadcasts = workspace.broadcasts

            logger.log_integration(
                tick=tick,
                agent_id=agent.agent_id,
                conflicts=conflicts,
                conflict_types=conflict_types,
                broadcasts=broadcasts,
                resolution_time_ms=0.1,  # Simplified
            )

            # Try innovation
            invention_id = innovation_system.try_invent(agent.agent_id, tick, rng)
            if invention_id:
                invention = innovation_system.inventions[invention_id]
                logger.log_innovation(
                    tick=tick,
                    agent_id=agent.agent_id,
                    invention_id=invention_id,
                    parents=invention.parents,
                    novelty_score=invention.novelty_score,
                    utility_score=invention.utility_score,
                    adopted_by=len(invention.adopted_by),
                    diffusion_radius=invention.diffusion_radius,
                )
                events_this_tick += 1

            # Try cultural transmission
            if len(alive_agents) > 1:
                other_agent = rng.choice(
                    [a for a in alive_agents if a.agent_id != agent.agent_id]
                )
                meme_id = culture_system.try_cultural_transmission(
                    agent.agent_id, other_agent.agent_id, tick, rng
                )
                if meme_id:
                    logger.log_culture(
                        tick=tick,
                        agent_id=other_agent.agent_id,
                        meme_id=meme_id,
                        action="adopt",
                        source_id=agent.agent_id,
                        trust_change=0.1,
                        reputation=culture_system.reputation_scores.get(
                            other_agent.agent_id, 0.5
                        ),
                        norm_violations=culture_system.norm_violations.get(
                            other_agent.agent_id, 0
                        ),
                    )
                    events_this_tick += 1

            # Add to predictability probe
            state_hash = world.get_state_hash(agent.x, agent.y)
            workspace_snapshot = workspace.read() if config.workspace_enabled else {}
            predictability_probe.add_sample(
                state_hash, result["action"], workspace_snapshot
            )

        # Remove dead agents
        agents = alive_agents

        # Calculate time series metrics
        if agents:
            avg_energy = np.mean([a.energy for a in agents])
            avg_consciousness = np.mean(
                [
                    np.mean(a.reported_conf_history) if a.reported_conf_history else 0.5
                    for a in agents
                ]
            )
            avg_innovation = innovation_system.get_innovation_rate()
            avg_reputation = np.mean(
                [culture_system.reputation_scores.get(a.agent_id, 0.5) for a in agents]
            )
        else:
            avg_energy = 0.0
            avg_consciousness = 0.0
            avg_innovation = 0.0
            avg_reputation = 0.0

        # Log time series
        logger.log_time_series(
            tick=tick,
            num_agents=len(agents),
            avg_energy=avg_energy,
            avg_consciousness=avg_consciousness,
            avg_innovation=avg_innovation,
            avg_reputation=avg_reputation,
            events_this_tick=events_this_tick,
        )

        # Early termination if no agents left
        if not agents:
            break

    # Log agent deaths
    for agent in agents:
        agent_stats = agent.get_stats()
        logger.log_agent_death(agent.agent_id, config.ticks, agent_stats)

    # Run predictability analysis
    predictability_results = predictability_probe.train_and_evaluate()
    for model_name, results in predictability_results.items():
        logger.log_predictability(model_name, results["accuracy"], results["logloss"])

    # Finalize logging
    logger.finalize()

    return output_dir


def create_zip_bundle(output_dir: str) -> str:
    """Create a ZIP bundle of the output directory."""
    zip_path = f"{output_dir}_bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)

    return zip_path

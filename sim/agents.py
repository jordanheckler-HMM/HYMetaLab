"""Agent implementation for the consciousness simulation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .workspace import Workspace


@dataclass
class ActionCandidate:
    """Represents a possible action with its score."""

    action: str
    score: float


class Agent:
    """An agent with consciousness, metacognition, and decision-making capabilities."""

    def __init__(self, agent_id: str, config, rng: np.random.Generator):
        self.agent_id = agent_id
        self.config = config
        self.rng = rng

        # Position and energy
        self.x = rng.integers(0, config.grid_size)
        self.y = rng.integers(0, config.grid_size)
        self.energy = config.initial_energy

        # Metacognitive state
        self.recent_failures = 0
        self.total_decisions = 0
        self.successful_decisions = 0
        self.reported_conf_history = []

        # Innovation and culture
        self.inventions_made = 0
        self.inventions_adopted = 0
        self.trust_scores = {}  # agent_id -> trust
        self.reputation = 0.5

        # Decision history for predictability
        self.decision_history = []

        # Local RNG seed for reproducibility
        self.local_rng_seed = rng.integers(0, 2**31)
        self.local_rng = np.random.Generator(np.random.PCG64(self.local_rng_seed))

    def perceive(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Process observation and return perceived state."""
        return {
            "position": obs["agent_pos"],
            "food_positions": obs["food_positions"],
            "local_food": obs["local_food"],
            "energy": self.energy,
            "observe_radius": obs["observe_radius"],
        }

    def propose_actions(self, ws: Workspace) -> list[ActionCandidate]:
        """Propose possible actions with scores."""
        candidates = []

        # Basic actions
        actions = ["stay", "move_n", "move_s", "move_e", "move_w", "eat"]

        for action in actions:
            score = self._score_action(action, ws)
            candidates.append(ActionCandidate(action, score))

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _score_action(self, action: str, ws: Workspace) -> float:
        """Score a potential action."""
        base_score = 0.0

        if action == "stay":
            base_score = 0.1  # Low score for staying

        elif action.startswith("move_"):
            # Score based on food availability in that direction
            direction = action.split("_")[1]
            new_x, new_y = self._get_new_position(direction)

            if (
                0 <= new_x < self.config.grid_size
                and 0 <= new_y < self.config.grid_size
            ):
                # Check for food in that direction
                food_score = self._get_food_score(new_x, new_y)
                base_score = food_score * 0.8  # Movement has some cost
            else:
                base_score = -0.5  # Penalty for moving out of bounds

        elif action == "eat":
            # Score based on current food availability
            food_score = self._get_food_score(self.x, self.y)
            base_score = food_score * 1.2  # Eating is generally good

        # Add some randomness
        noise = self.local_rng.normal(0, 0.1)
        return base_score + noise

    def _get_new_position(self, direction: str) -> tuple[int, int]:
        """Get new position after moving in given direction."""
        if direction == "n":
            return self.x, self.y - 1
        elif direction == "s":
            return self.x, self.y + 1
        elif direction == "e":
            return self.x + 1, self.y
        elif direction == "w":
            return self.x - 1, self.y
        else:
            return self.x, self.y

    def _get_food_score(self, x: int, y: int) -> float:
        """Get food score at position (x, y)."""
        # This would normally check the world's food grid
        # For now, return a random score
        return self.local_rng.uniform(0, 1)

    def choose_action(
        self, proposals: list[ActionCandidate]
    ) -> tuple[str, float, float]:
        """Choose action from proposals and return (action, confidence, prediction_prob)."""
        if not proposals:
            return "stay", 0.0, 0.0

        # Choose best action
        chosen = proposals[0]

        # Calculate confidence from score margin
        if len(proposals) > 1:
            best_score = proposals[0].score
            second_score = proposals[1].score
            raw_conf = (best_score - second_score) / max(
                1e-6, abs(best_score) + abs(second_score)
            )
            raw_conf = np.clip(raw_conf, 0, 1)
        else:
            raw_conf = 0.5

        # Apply metacognitive decay
        failure_rate = self.recent_failures / max(1, self.total_decisions)
        conf = raw_conf * (1 - failure_rate * self.config.metacog_decay)

        # Add noise
        conf += self.local_rng.normal(0, self.config.conf_noise_std)
        conf = np.clip(conf, 0, 1)

        # Simple prediction model (for predictability analysis)
        prediction_prob = self._predict_action_probability(chosen.action)

        return chosen.action, conf, prediction_prob

    def _predict_action_probability(self, action: str) -> float:
        """Simple prediction model for action probability."""
        # Count how often this action was chosen recently
        recent_decisions = (
            self.decision_history[-10:]
            if len(self.decision_history) >= 10
            else self.decision_history
        )
        if not recent_decisions:
            return 0.2  # Uniform prior

        action_count = sum(1 for a in recent_decisions if a == action)
        return action_count / len(recent_decisions)

    def confidence(self, best: float, second: float) -> float:
        """Calculate confidence from best and second best scores."""
        if abs(best) + abs(second) < 1e-6:
            return 0.5

        raw_conf = (best - second) / (abs(best) + abs(second))
        raw_conf = np.clip(raw_conf, 0, 1)

        # Apply metacognitive decay
        failure_rate = self.recent_failures / max(1, self.total_decisions)
        conf = raw_conf * (1 - failure_rate * self.config.metacog_decay)

        # Add noise
        conf += self.local_rng.normal(0, self.config.conf_noise_std)
        return np.clip(conf, 0, 1)

    def apply_costs(self, energy: float, ws: Workspace, lookahead_depth: int) -> float:
        """Apply workspace and lookahead costs to energy."""
        # Workspace costs
        ws_cost = self.config.ws_cost * (ws.reads + ws.writes)

        # Lookahead costs
        lookahead_cost = self.config.lookahead_cost * lookahead_depth

        return energy - ws_cost - lookahead_cost

    def execute_action(self, action: str, world) -> float:
        """Execute action and return outcome reward."""
        outcome_reward = 0.0

        if action == "stay":
            outcome_reward = 0.0

        elif action.startswith("move_"):
            direction = action.split("_")[1]
            new_x, new_y = self._get_new_position(direction)

            if (
                0 <= new_x < self.config.grid_size
                and 0 <= new_y < self.config.grid_size
            ):
                self.x, self.y = new_x, new_y
                outcome_reward = -self.config.energy_cost_move
            else:
                outcome_reward = (
                    -self.config.energy_cost_move * 2
                )  # Penalty for hitting boundary

        elif action == "eat":
            energy_gained = world.consume_food(self.x, self.y)
            if energy_gained > 0:
                self.energy += energy_gained
                outcome_reward = energy_gained
            else:
                outcome_reward = -1.0  # Penalty for trying to eat empty cell

        # Update energy
        self.energy += outcome_reward

        # Update decision history
        self.decision_history.append(action)
        self.total_decisions += 1

        if outcome_reward > 0:
            self.successful_decisions += 1
            self.recent_failures = max(0, self.recent_failures - 1)
        else:
            self.recent_failures += 1

        return outcome_reward

    def step(self, obs: dict[str, Any], world, ws: Workspace, logger) -> dict[str, Any]:
        """Execute one step of the agent."""
        # Perceive environment
        perceived = self.perceive(obs)

        # Generate state hash for logging
        state_hash = world.get_state_hash(self.x, self.y)

        # Propose actions
        proposals = self.propose_actions(ws)

        # Choose action
        chosen_action, reported_conf, prediction_prob = self.choose_action(proposals)

        # Apply costs
        lookahead_depth = 1 if self.config.workspace_enabled else 0
        self.energy = self.apply_costs(self.energy, ws, lookahead_depth)

        # Execute action
        outcome_reward = self.execute_action(chosen_action, world)

        # Log decision
        logger.log_decision(
            tick=world.tick,
            agent_id=self.agent_id,
            state_hash=state_hash,
            workspace_reads=ws.reads,
            workspace_writes=ws.writes,
            lookahead_depth=lookahead_depth,
            candidates=[{"action": p.action, "score": p.score} for p in proposals],
            chosen_action=chosen_action,
            reported_conf=reported_conf,
            outcome_reward=outcome_reward,
            prediction_model_p=prediction_prob,
            rng_seed_local=self.local_rng_seed,
        )

        # Update confidence history
        self.reported_conf_history.append(reported_conf)

        return {
            "action": chosen_action,
            "confidence": reported_conf,
            "outcome_reward": outcome_reward,
            "energy": self.energy,
            "position": (self.x, self.y),
        }

    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.energy > 0

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "energy": self.energy,
            "position": (self.x, self.y),
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "recent_failures": self.recent_failures,
            "inventions_made": self.inventions_made,
            "inventions_adopted": self.inventions_adopted,
            "reputation": self.reputation,
            "mean_conf": (
                np.mean(self.reported_conf_history)
                if self.reported_conf_history
                else 0.0
            ),
        }

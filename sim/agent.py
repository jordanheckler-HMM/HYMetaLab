from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    """Lightweight Agent model for simulations.

    This version adds a simple belief state (continuous 0..1) and a couple
    of small update rules to support belief-dynamics experiments (Bayesian
    and delta-rule social influence).
    """

    agent_id: int
    x: int = 0
    y: int = 0
    energy: float = 100.0
    age: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    # belief: probability the agent assigns to a proposition (e.g. "god exists")
    belief: float = 0.5
    # openness controls how strongly social signals move the agent (0..1)
    openness: float = 0.2

    def step(self, energy_cost: float = 1.0):
        """Advance the agent by one tick and apply an energy cost."""
        self.age += 1
        self.energy -= energy_cost
        if self.energy < 0:
            self.energy = 0.0

    def is_alive(self) -> bool:
        return self.energy > 0

    def move(self, dx: int, dy: int, cost: float = 0.5):
        self.x += dx
        self.y += dy
        self.energy -= cost
        if self.energy < 0:
            self.energy = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "age": self.age,
            "belief": self.belief,
        }

    # ------------------ Belief update utilities ------------------
    def _clamp_belief(self):
        if self.belief < 0.0:
            self.belief = 0.0
        if self.belief > 1.0:
            self.belief = 1.0

    def update_belief_bayesian(self, likelihood_ratio: float):
        """Update belief using odds form of Bayes' rule.

        Args:
            likelihood_ratio: P(evidence | proposition true) / P(evidence | proposition false).
        """
        # avoid division by zero
        prior = max(min(self.belief, 1.0 - 1e-12), 1e-12)
        prior_odds = prior / (1.0 - prior)
        posterior_odds = prior_odds * float(likelihood_ratio)
        self.belief = posterior_odds / (1.0 + posterior_odds)
        self._clamp_belief()

    def update_belief_delta(self, evidence_value: float, alpha: float = 0.2):
        """Simple delta-rule (error-driven) belief update.

        evidence_value should be in [0,1] representing observed support for the proposition.
        """
        self.belief = float(self.belief) + float(alpha) * (
            float(evidence_value) - float(self.belief)
        )
        self._clamp_belief()

    def social_influence(
        self, neighbor_beliefs: list[float], influence_weight: float | None = None
    ):
        """Move belief toward the average belief of neighbors.

        influence_weight overrides the agent's openness if provided (0..1).
        """
        if not neighbor_beliefs:
            return
        w = self.openness if influence_weight is None else float(influence_weight)
        mean_neighbor = float(sum(neighbor_beliefs)) / len(neighbor_beliefs)
        self.belief = (1.0 - w) * self.belief + w * mean_neighbor
        self._clamp_belief()

"""Culture system for the consciousness simulation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class MemeAction(Enum):
    """Types of meme actions."""

    ADOPT = "adopt"
    REJECT = "reject"
    TRANSMIT = "transmit"
    MODIFY = "modify"


@dataclass
class Meme:
    """Represents a cultural meme."""

    meme_id: str
    content: str
    creator_id: str
    tick_created: int
    transmission_count: int
    adoption_count: int
    rejection_count: int


class CultureSystem:
    """Manages cultural transmission and meme evolution."""

    def __init__(self, config):
        self.config = config
        self.memes: dict[str, Meme] = {}
        self.next_meme_id = 1
        self.agent_memes: dict[str, list[str]] = {}  # agent_id -> meme_ids
        self.trust_network: dict[str, dict[str, float]] = (
            {}
        )  # agent_id -> {other_agent_id: trust}
        self.reputation_scores: dict[str, float] = {}  # agent_id -> reputation
        self.norm_violations: dict[str, int] = {}  # agent_id -> violation_count

    def initialize_agent(self, agent_id: str):
        """Initialize cultural state for a new agent."""
        self.agent_memes[agent_id] = []
        self.trust_network[agent_id] = {}
        self.reputation_scores[agent_id] = 0.5  # Start with neutral reputation
        self.norm_violations[agent_id] = 0

    def try_cultural_transmission(
        self, sender_id: str, receiver_id: str, tick: int, rng: np.random.Generator
    ) -> str | None:
        """Try to transmit a meme from sender to receiver."""
        if sender_id not in self.agent_memes or receiver_id not in self.agent_memes:
            return None

        sender_memes = self.agent_memes[sender_id]
        if not sender_memes:
            return None

        # Choose meme to transmit
        meme_id = rng.choice(sender_memes)
        meme = self.memes[meme_id]

        # Calculate transmission probability based on trust and reputation
        trust = self.get_trust(sender_id, receiver_id)
        reputation = self.reputation_scores.get(sender_id, 0.5)
        transmission_prob = trust * reputation * self.config.culture_transmission_prob

        if rng.random() < transmission_prob:
            # Receiver decides whether to adopt
            adoption_prob = self._calculate_adoption_probability(
                meme, receiver_id, trust
            )

            if rng.random() < adoption_prob:
                # Adopt meme
                self.agent_memes[receiver_id].append(meme_id)
                meme.adoption_count += 1
                meme.transmission_count += 1

                # Update trust
                self._update_trust(receiver_id, sender_id, 0.1)

                return meme_id
            else:
                # Reject meme
                meme.rejection_count += 1
                meme.transmission_count += 1

                # Slight trust decrease
                self._update_trust(receiver_id, sender_id, -0.05)

        return None

    def create_meme(self, agent_id: str, content: str, tick: int) -> str:
        """Create a new meme."""
        meme_id = f"meme-{self.next_meme_id}"
        self.next_meme_id += 1

        meme = Meme(
            meme_id=meme_id,
            content=content,
            creator_id=agent_id,
            tick_created=tick,
            transmission_count=0,
            adoption_count=1,
            rejection_count=0,
        )

        self.memes[meme_id] = meme

        # Add to creator's memes
        if agent_id not in self.agent_memes:
            self.agent_memes[agent_id] = []
        self.agent_memes[agent_id].append(meme_id)

        return meme_id

    def _calculate_adoption_probability(
        self, meme: Meme, receiver_id: str, trust: float
    ) -> float:
        """Calculate probability of adopting a meme."""
        base_prob = 0.3

        # Trust factor
        trust_factor = trust

        # Reputation factor
        creator_reputation = self.reputation_scores.get(meme.creator_id, 0.5)
        reputation_factor = creator_reputation

        # Meme success factor (adoption vs rejection ratio)
        total_interactions = meme.adoption_count + meme.rejection_count
        if total_interactions > 0:
            success_ratio = meme.adoption_count / total_interactions
        else:
            success_ratio = 0.5

        # Calculate final probability
        adoption_prob = base_prob * trust_factor * reputation_factor * success_ratio
        return min(0.9, max(0.1, adoption_prob))

    def get_trust(self, agent_id: str, other_id: str) -> float:
        """Get trust level between two agents."""
        if agent_id not in self.trust_network:
            return 0.5  # Default neutral trust

        return self.trust_network[agent_id].get(other_id, 0.5)

    def _update_trust(self, agent_id: str, other_id: str, change: float):
        """Update trust level between two agents."""
        if agent_id not in self.trust_network:
            self.trust_network[agent_id] = {}

        current_trust = self.trust_network[agent_id].get(other_id, 0.5)
        new_trust = current_trust + change * self.config.trust_update_rate
        self.trust_network[agent_id][other_id] = np.clip(new_trust, 0.0, 1.0)

    def update_reputation(self, agent_id: str, change: float):
        """Update agent's reputation."""
        if agent_id not in self.reputation_scores:
            self.reputation_scores[agent_id] = 0.5

        current_reputation = self.reputation_scores[agent_id]
        new_reputation = current_reputation + change * self.config.trust_update_rate
        self.reputation_scores[agent_id] = np.clip(new_reputation, 0.0, 1.0)

    def record_norm_violation(self, agent_id: str):
        """Record a norm violation by an agent."""
        if agent_id not in self.norm_violations:
            self.norm_violations[agent_id] = 0

        self.norm_violations[agent_id] += 1

        # Decrease reputation
        self.update_reputation(agent_id, -0.1)

    def get_cultural_stats(self) -> dict[str, Any]:
        """Get overall cultural statistics."""
        if not self.memes:
            return {
                "total_memes": 0,
                "mean_transmissions": 0.0,
                "mean_adoptions": 0.0,
                "cultural_diversity": 0.0,
            }

        total_memes = len(self.memes)
        mean_transmissions = np.mean(
            [m.transmission_count for m in self.memes.values()]
        )
        mean_adoptions = np.mean([m.adoption_count for m in self.memes.values()])

        # Calculate cultural diversity (unique memes per agent)
        total_agents = len(self.agent_memes)
        if total_agents > 0:
            total_unique_memes = len(
                set(meme_id for memes in self.agent_memes.values() for meme_id in memes)
            )
            cultural_diversity = total_unique_memes / total_agents
        else:
            cultural_diversity = 0.0

        return {
            "total_memes": total_memes,
            "mean_transmissions": mean_transmissions,
            "mean_adoptions": mean_adoptions,
            "cultural_diversity": cultural_diversity,
        }

    def get_agent_cultural_stats(self, agent_id: str) -> dict[str, Any]:
        """Get cultural statistics for a specific agent."""
        memes_created = len(
            [m for m in self.memes.values() if m.creator_id == agent_id]
        )
        memes_adopted = len(self.agent_memes.get(agent_id, []))
        reputation = self.reputation_scores.get(agent_id, 0.5)
        norm_violations = self.norm_violations.get(agent_id, 0)

        return {
            "memes_created": memes_created,
            "memes_adopted": memes_adopted,
            "reputation": reputation,
            "norm_violations": norm_violations,
        }

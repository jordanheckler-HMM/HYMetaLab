"""Innovation system for the consciousness simulation."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Invention:
    """Represents an invention in the simulation."""

    invention_id: str
    creator_id: str
    tick_created: int
    parents: list[str]
    novelty_score: float
    utility_score: float
    adopted_by: list[str]
    diffusion_radius: int


class InnovationSystem:
    """Manages innovation and invention diffusion."""

    def __init__(self, config):
        self.config = config
        self.inventions: dict[str, Invention] = {}
        self.next_invention_id = 1
        self.adoption_network: dict[str, list[str]] = (
            {}
        )  # agent_id -> adopted inventions

    def try_invent(
        self, agent_id: str, tick: int, rng: np.random.Generator
    ) -> str | None:
        """Try to create an invention for an agent."""
        if rng.random() > self.config.innovation_prob:
            return None

        # Create new invention
        invention_id = f"inv-{self.next_invention_id}"
        self.next_invention_id += 1

        # Determine parents (previous inventions this agent adopted)
        parents = self.adoption_network.get(agent_id, [])

        # Calculate novelty and utility
        novelty_score = rng.uniform(0.3, 0.9)
        utility_score = rng.uniform(0.2, 0.8)

        # Create invention
        invention = Invention(
            invention_id=invention_id,
            creator_id=agent_id,
            tick_created=tick,
            parents=parents,
            novelty_score=novelty_score,
            utility_score=utility_score,
            adopted_by=[agent_id],
            diffusion_radius=1,
        )

        self.inventions[invention_id] = invention

        # Add to agent's adopted inventions
        if agent_id not in self.adoption_network:
            self.adoption_network[agent_id] = []
        self.adoption_network[agent_id].append(invention_id)

        return invention_id

    def try_adopt(
        self, agent_id: str, neighbor_id: str, rng: np.random.Generator
    ) -> str | None:
        """Try to adopt an invention from a neighbor."""
        if neighbor_id not in self.adoption_network:
            return None

        # Get inventions this neighbor has that the agent doesn't
        neighbor_inventions = set(self.adoption_network[neighbor_id])
        agent_inventions = set(self.adoption_network.get(agent_id, []))
        available_inventions = neighbor_inventions - agent_inventions

        if not available_inventions:
            return None

        # Choose invention to potentially adopt
        invention_id = rng.choice(list(available_inventions))
        invention = self.inventions[invention_id]

        # Adoption probability based on utility and trust
        adoption_prob = invention.utility_score * 0.5  # Base adoption rate
        if rng.random() < adoption_prob:
            # Adopt invention
            if agent_id not in self.adoption_network:
                self.adoption_network[agent_id] = []
            self.adoption_network[agent_id].append(invention_id)
            invention.adopted_by.append(agent_id)

            # Update diffusion radius
            invention.diffusion_radius = max(
                invention.diffusion_radius,
                self._calculate_diffusion_radius(invention_id),
            )

            return invention_id

        return None

    def _calculate_diffusion_radius(self, invention_id: str) -> int:
        """Calculate diffusion radius for an invention."""
        invention = self.inventions[invention_id]

        # Simple calculation: number of unique agents who adopted
        return len(set(invention.adopted_by))

    def get_invention_stats(self, invention_id: str) -> dict[str, Any]:
        """Get statistics for an invention."""
        if invention_id not in self.inventions:
            return {}

        invention = self.inventions[invention_id]
        return {
            "invention_id": invention_id,
            "creator_id": invention.creator_id,
            "tick_created": invention.tick_created,
            "parents": invention.parents,
            "novelty_score": invention.novelty_score,
            "utility_score": invention.utility_score,
            "adopted_by": len(invention.adopted_by),
            "diffusion_radius": invention.diffusion_radius,
        }

    def get_agent_inventions(self, agent_id: str) -> list[str]:
        """Get inventions made by an agent."""
        return [
            inv_id
            for inv_id, inv in self.inventions.items()
            if inv.creator_id == agent_id
        ]

    def get_agent_adoptions(self, agent_id: str) -> list[str]:
        """Get inventions adopted by an agent."""
        return self.adoption_network.get(agent_id, [])

    def get_innovation_rate(self) -> float:
        """Get current innovation rate (inventions per tick per agent)."""
        if not self.inventions:
            return 0.0

        total_inventions = len(self.inventions)
        # This would need to be calculated based on total ticks and agents
        # For now, return a simple metric
        return total_inventions / 1000.0  # Rough estimate

    def get_diffusion_stats(self) -> dict[str, float]:
        """Get diffusion statistics."""
        if not self.inventions:
            return {"mean_diffusion": 0.0, "max_diffusion": 0.0}

        diffusion_radii = [inv.diffusion_radius for inv in self.inventions.values()]
        return {
            "mean_diffusion": np.mean(diffusion_radii),
            "max_diffusion": np.max(diffusion_radii),
            "total_inventions": len(self.inventions),
        }

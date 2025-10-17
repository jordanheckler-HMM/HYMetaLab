"""World environment for the consciousness simulation."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FoodCell:
    """Represents a food cell in the world."""

    x: int
    y: int
    energy: float = 10.0
    depleted: bool = False


class World:
    """The world environment containing food and agents."""

    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size
        self.food_regen_prob = config.food_regen_prob
        self.volatility_period = config.volatility_period
        self.volatility_strength = config.volatility_strength
        self.observe_radius = config.observe_radius

        # Initialize food grid
        self.food_grid = np.zeros((self.grid_size, self.grid_size))
        self._populate_food()

        # Volatility tracking
        self.tick = 0
        self.last_volatility_tick = 0

    def _populate_food(self) -> None:
        """Initialize food cells randomly across the grid."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if np.random.random() < self.food_regen_prob:
                    self.food_grid[x, y] = np.random.uniform(5.0, 15.0)

    def get_observation(self, agent_x: int, agent_y: int) -> dict[str, Any]:
        """Get local observation for an agent."""
        # Create observation window
        x_min = max(0, agent_x - self.observe_radius)
        x_max = min(self.grid_size, agent_x + self.observe_radius + 1)
        y_min = max(0, agent_y - self.observe_radius)
        y_max = min(self.grid_size, agent_y + self.observe_radius + 1)

        local_food = self.food_grid[x_min:x_max, y_min:y_max].copy()

        return {
            "agent_pos": (agent_x, agent_y),
            "local_food": local_food,
            "food_positions": self._get_food_positions(x_min, x_max, y_min, y_max),
            "grid_bounds": (x_min, x_max, y_min, y_max),
            "observe_radius": self.observe_radius,
        }

    def _get_food_positions(
        self, x_min: int, x_max: int, y_min: int, y_max: int
    ) -> list[tuple[int, int, float]]:
        """Get positions and values of food in the observation window."""
        positions = []
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if self.food_grid[x, y] > 0:
                    positions.append((x, y, self.food_grid[x, y]))
        return positions

    def consume_food(self, x: int, y: int) -> float:
        """Consume food at position (x, y) and return energy gained."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            energy = self.food_grid[x, y]
            self.food_grid[x, y] = 0.0
            return energy
        return 0.0

    def step(self) -> None:
        """Advance world by one tick."""
        self.tick += 1

        # Regenerate food
        self._regenerate_food()

        # Apply volatility if it's time
        if self.tick - self.last_volatility_tick >= self.volatility_period:
            self._apply_volatility()
            self.last_volatility_tick = self.tick

    def _regenerate_food(self) -> None:
        """Regenerate food cells randomly."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (
                    self.food_grid[x, y] == 0
                    and np.random.random() < self.food_regen_prob
                ):
                    self.food_grid[x, y] = np.random.uniform(5.0, 15.0)

    def _apply_volatility(self) -> None:
        """Apply volatility shocks to the world."""
        # Randomly modify food values
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.food_grid[x, y] > 0:
                    # Apply random shock
                    shock = np.random.normal(0, self.volatility_strength)
                    self.food_grid[x, y] = max(0, self.food_grid[x, y] + shock)

        # Randomly add/remove food cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if np.random.random() < self.volatility_strength * 0.1:
                    if self.food_grid[x, y] == 0:
                        self.food_grid[x, y] = np.random.uniform(5.0, 15.0)
                    else:
                        self.food_grid[x, y] = 0.0

    def get_food_density(self) -> float:
        """Get current food density in the world."""
        return np.sum(self.food_grid > 0) / (self.grid_size * self.grid_size)

    def get_total_food(self) -> float:
        """Get total food energy in the world."""
        return np.sum(self.food_grid)

    def get_state_hash(self, agent_x: int, agent_y: int) -> str:
        """Generate a hash of the agent's local state for predictability analysis."""
        obs = self.get_observation(agent_x, agent_y)
        # Create a compact representation for hashing
        state_tuple = (
            agent_x,
            agent_y,
            tuple(sorted(obs["food_positions"])),
            obs["observe_radius"],
        )
        return f"sha1:{hash(state_tuple)}"

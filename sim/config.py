"""Configuration management for the consciousness simulation."""

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Config:
    """Configuration container for simulation parameters."""

    # Simulation parameters
    ticks: int = 2000
    n_agents: int = 80
    seed: int = 1234

    # Workspace costs
    ws_cost: float = 0.01
    lookahead_cost: float = 0.02

    # Metacognition
    conf_noise_std: float = 0.08
    metacog_decay: float = 0.5

    # World parameters
    volatility_period: int = 50
    volatility_strength: float = 0.2
    observe_radius: int = 3

    # Workspace settings
    workspace_enabled: bool = True

    # Grid parameters
    grid_size: int = 20
    food_regen_prob: float = 0.1

    # Agent parameters
    initial_energy: float = 50.0
    energy_cost_move: float = 1.0
    energy_gain_food: float = 10.0

    # Innovation parameters
    innovation_prob: float = 0.01
    innovation_utility_base: float = 0.1

    # Culture parameters
    culture_transmission_prob: float = 0.05
    trust_update_rate: float = 0.1

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Create config with defaults, then update with file data
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "ticks": self.ticks,
            "n_agents": self.n_agents,
            "seed": self.seed,
            "ws_cost": self.ws_cost,
            "lookahead_cost": self.lookahead_cost,
            "conf_noise_std": self.conf_noise_std,
            "metacog_decay": self.metacog_decay,
            "volatility_period": self.volatility_period,
            "volatility_strength": self.volatility_strength,
            "observe_radius": self.observe_radius,
            "workspace_enabled": self.workspace_enabled,
            "grid_size": self.grid_size,
            "food_regen_prob": self.food_regen_prob,
            "initial_energy": self.initial_energy,
            "energy_cost_move": self.energy_cost_move,
            "energy_gain_food": self.energy_gain_food,
            "innovation_prob": self.innovation_prob,
            "innovation_utility_base": self.innovation_utility_base,
            "culture_transmission_prob": self.culture_transmission_prob,
            "trust_update_rate": self.trust_update_rate,
        }

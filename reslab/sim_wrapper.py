#!/usr/bin/env python3
"""
Wrapper function for the consciousness simulation to work with reslab adapter.
This creates a temporary config file and calls the simulation.
"""

import os
import tempfile

import yaml

from sim.experiments import run_sim


def run_consciousness_sim(
    n_agents=80,
    ticks=2000,
    seed=1234,
    coordination_strength=0.5,
    goal_inequality=0.25,
    noise=0.05,
    memory_cap=None,
    energy_drift=False,
    shock_level=0.4,
    **kwargs,
):
    """
    Wrapper function that maps reslab parameters to consciousness simulation parameters.

    Args:
        n_agents: Number of agents
        ticks: Number of simulation ticks
        seed: Random seed
        coordination_strength: Maps to workspace_enabled (coordination via global workspace)
        goal_inequality: Maps to volatility_strength (environmental inequality)
        noise: Maps to conf_noise_std (confidence noise)
        memory_cap: Not directly used in this sim
        energy_drift: Maps to energy_cost_move (energy degradation)
        shock_level: Maps to volatility_strength (environmental shocks)
        **kwargs: Additional parameters

    Returns:
        str: Output directory path
    """

    # Create temporary config file
    config_data = {
        "ticks": ticks,
        "n_agents": n_agents,
        "seed": seed,
        # Map coordination to workspace usage with more granular control
        "workspace_enabled": coordination_strength > 0.05,
        "ws_cost": coordination_strength * 0.02,  # Scale with coordination strength
        # Map goal inequality and shock level to volatility with more sensitivity
        "volatility_strength": goal_inequality + shock_level,  # Additive instead of max
        "volatility_period": max(
            10, int(50 / (1 + shock_level))
        ),  # More frequent shocks for higher levels
        # Map noise to confidence noise with more range
        "conf_noise_std": noise * 2.0,  # Scale noise for more effect
        # Map energy drift to energy costs with more variation
        "energy_cost_move": 1.0
        + (energy_drift * 0.8),  # More gradual energy degradation
        "initial_energy": 50.0,
        "energy_gain_food": 10.0,
        # Other parameters
        "observe_radius": 3,
        "grid_size": 20,
        "food_regen_prob": 0.1,
        "metacog_decay": 0.5,
        "lookahead_cost": 0.02,
        "innovation_prob": 0.01,
        "innovation_utility_base": 0.1,
        "culture_transmission_prob": 0.05,
        "trust_update_rate": 0.1,
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Run simulation
        output_dir = run_sim(config_path)
        return output_dir
    finally:
        # Clean up temporary file
        os.unlink(config_path)

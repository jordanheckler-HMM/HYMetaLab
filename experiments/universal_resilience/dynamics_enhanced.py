"""
Enhanced dynamics utilities for Universal Resilience experiment.
Implements proven shock mechanisms that create meaningful variance.
"""

import math
import random
from typing import Any

import numpy as np


def exp_decay_factor(dt: int, half_life: float) -> float:
    """Calculate exponential decay factor."""
    if dt <= 0:
        return 1.0
    if half_life <= 0:
        return 0.0
    return 0.5 ** (dt / float(half_life))


def heterogeneity_draws(
    n: int, cfg: dict[str, Any], rng: random.Random
) -> list[dict[str, float]]:
    """Return per-agent dicts: resilience_mult, regen_mult, mort_base_mult."""
    if not cfg["heterogeneity"]["enable"]:
        return [
            {"resilience_mult": 1.0, "regen_mult": 1.0, "mort_base_mult": 1.0}
            for _ in range(n)
        ]

    sig = cfg["heterogeneity"]["resilience_sigma"]
    regen_j = cfg["heterogeneity"]["regen_jitter"]
    mort_j = cfg["heterogeneity"]["mort_base_jitter"]

    out = []
    for _ in range(n):
        # lognormal resilience multiplier ~ exp(N(0, sig^2))
        res = math.exp(rng.gauss(0.0, sig))
        regen = 1.0 + (rng.random() * 2 - 1) * regen_j
        mort = 1.0 + (rng.random() * 2 - 1) * mort_j
        out.append(
            {
                "resilience_mult": res,
                "regen_mult": max(0.2, regen),
                "mort_base_mult": max(0.2, mort),
            }
        )
    return out


def shock_effect_at(t: int, cfg: dict[str, Any], cell: dict[str, Any]) -> float:
    """Return [0..1] shock intensity at time t based on severity, duration, taper."""
    s0 = cell["severity"]
    start = cell["shock_start"]
    end = cell["shock_end"]

    if t < start:
        return 0.0

    if t <= end:
        # inside shock window: intensity ramps in/out across duration
        dur = max(1, end - start)
        pos = (t - start + 1) / dur
        envelope = min(pos, 1.0 - (t - start) / dur) * 2.0  # triangular ramp
        return max(0.0, min(1.0, s0 * envelope))

    # post-shock taper
    dt = t - end
    return s0 * exp_decay_factor(dt, cfg["shock"]["taper_half_life"])


def apply_effects_enhanced(
    resource: float,
    base_regen: float,
    base_mort: float,
    shock_intensity: float,
    is_targeted: bool,
    het: dict[str, float],
    cfg: dict[str, Any],
    common_pool: float,
    n_alive: int,
) -> tuple[float, float, float]:
    """Enhanced effects using common pool depletion model (proven to work)."""

    # Common pool dynamics (like shock_resilience.py)
    if n_alive > 0:
        per_agent_share = common_pool / n_alive
        take = min(0.5, per_agent_share)  # Agents can take up to 0.5 from pool
        resource += take

    # Consumption (baseline cost) - increased for more variance
    resource -= 0.4

    # Shock damage if targeted (more aggressive)
    if is_targeted and shock_intensity > 0:
        # Direct resource loss proportional to shock intensity
        resource_loss = shock_intensity * 0.4  # Increased from 0.3
        resource = max(0.01, resource - resource_loss)

    # Mortality based on resource level (more aggressive)
    if resource <= 0:
        p_death = 1.0  # Immediate death if no resources
    else:
        # Logistic mortality based on resource deficit
        deficit = max(0.0, 1.0 - resource)
        k = cfg["recovery"]["mortality_k"]
        c = cfg["recovery"]["mortality_center"]
        logistic = 1.0 / (1.0 + math.exp(-k * (deficit - c)))
        p_death = min(1.0, base_mort + logistic * 0.1)  # Increased mortality

    # Apply heterogeneity
    resource *= het["resilience_mult"]
    p_death *= het["mort_base_mult"]

    return resource, p_death, take


def apply_effects(
    resource: float,
    base_regen: float,
    base_mort: float,
    shock_intensity: float,
    is_targeted: bool,
    het: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[float, float]:
    """Legacy wrapper for backward compatibility."""
    # Use enhanced effects with dummy common pool
    resource_eff, p_death, _ = apply_effects_enhanced(
        resource,
        base_regen,
        base_mort,
        shock_intensity,
        is_targeted,
        het,
        cfg,
        1000.0,
        1,  # Dummy values for backward compatibility
    )
    return resource_eff, p_death


def calculate_system_health(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates system-wide health metrics."""
    alive_agents = [a for a in agents if a["alive"]]
    n_alive = len(alive_agents)
    n_total = len(agents)
    alive_fraction = n_alive / n_total if n_total > 0 else 0.0

    return {"n_alive": n_alive, "alive_fraction": alive_fraction}


def apply_common_pool_shock(
    agents: list[dict[str, Any]],
    common_pool: float,
    shock_intensity: float,
    scope: float,
    targeted_agents: set[int],
    cfg: dict[str, Any],
) -> tuple[float, int]:
    """Apply shock using common pool depletion model (proven approach)."""

    # Deplete common pool based on shock intensity
    pool_loss = shock_intensity * 0.3  # Reduce pool by up to 30%
    common_pool *= 1.0 - pool_loss

    # Apply immediate mortality to targeted agents
    deaths = 0
    for agent in agents:
        if agent["alive"] and agent["id"] in targeted_agents:
            # Probability of death based on shock intensity
            death_prob = shock_intensity * 0.2  # Up to 20% immediate death
            if np.random.random() < death_prob:
                agent["alive"] = False
                agent["energy"] = 0
                deaths += 1

    return common_pool, deaths


def step_common_pool_dynamics(
    agents: list[dict[str, Any]], common_pool: float, cfg: dict[str, Any]
) -> tuple[float, int]:
    """Step the common pool dynamics (like shock_resilience.py)."""

    alive_agents = [a for a in agents if a["alive"]]
    n_alive = len(alive_agents)

    if n_alive == 0:
        return common_pool, 0

    # Each alive agent draws from pool
    per_agent_share = common_pool / n_alive
    deaths_this_step = 0

    for agent in alive_agents:
        # Take resources from pool
        take = min(0.5, per_agent_share)
        agent["resource"] += take

        # Consumption - increased for more variance
        agent["resource"] -= 0.4

        # Check for death
        if agent["resource"] <= 0:
            agent["alive"] = False
            agent["energy"] = 0
            deaths_this_step += 1

    # Replenish pool (regrowth)
    regrowth_rate = cfg["recovery"]["base_resource_regen_rate"] * 10  # Scale up
    common_pool += regrowth_rate * len(agents)

    return common_pool, deaths_this_step

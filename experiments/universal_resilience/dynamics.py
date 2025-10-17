"""
Dynamics utilities for Universal Resilience experiment.
Handles gradual shocks, heterogeneity, and recovery mechanics.
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


def apply_effects(
    resource: float,
    base_regen: float,
    base_mort: float,
    shock_intensity: float,
    is_targeted: bool,
    het: dict[str, float],
    cfg: dict[str, Any],
) -> tuple[float, float]:
    """Compute effective resource & mortality updates (one agent, one step)."""
    # regen with heterogeneity
    regen = base_regen * het["regen_mult"]
    resource = resource + regen * (cfg["recovery"]["capacity"] - resource)

    # shock damage if targeted
    eff = 1.0
    if is_targeted and shock_intensity > 0:
        # combine severity, duration, scope via configured weights
        dmg = cfg["shock"]["severity_to_damage"] * shock_intensity
        eff -= max(0.0, min(0.9, dmg))  # cap damage per step

    resource_eff = max(0.0, min(1.5, resource * eff * het["resilience_mult"]))

    # mortality (baseline with heterogeneity; logistic on deficit)
    base = base_mort * het["mort_base_mult"]
    k = cfg["recovery"]["mortality_k"]
    c = cfg["recovery"]["mortality_center"]
    deficit = max(0.0, 1.0 - resource_eff)
    logistic = 1.0 / (1.0 + math.exp(-k * (deficit - c)))
    p_death = min(1.0, base + logistic * base)

    return resource_eff, p_death


def apply_mortality_step(
    agents: list[dict[str, Any]],
    cfg: dict[str, Any],
    shock_multiplier: float,
    extra_mortality: float,
) -> int:
    """Apply mortality to agents based on their effective resources and shock effects."""
    deaths_this_step = 0
    for agent in agents:
        if agent["alive"]:
            effective_resource = agent["resource"] * shock_multiplier
            prob_death = mortality_prob(effective_resource, cfg) + extra_mortality
            if np.random.random() < prob_death:
                agent["alive"] = False
                agent["energy"] = 0
                deaths_this_step += 1
    return deaths_this_step


def apply_recovery_step(agents: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    """Apply resource regeneration to alive agents."""
    for agent in agents:
        if agent["alive"]:
            agent["resource"] = step_resource_regen(agent["resource"], cfg)


def step_resource_regen(resource: float, cfg: dict[str, Any]) -> float:
    """Applies resource regeneration."""
    cap = cfg["recovery"]["capacity"]
    rate = cfg["recovery"]["base_resource_regen_rate"]
    return resource + rate * (cap - resource)


def mortality_prob(resource_effective: float, cfg: dict[str, Any]) -> float:
    """Logistic death probability as function of resource deficit."""
    base = cfg["recovery"]["mortality_base_rate"]
    deficit = max(0.0, 1.0 - resource_effective)
    k = cfg["recovery"]["mortality_k"]
    c = cfg["recovery"]["mortality_center"]
    # logistic in [0,1]; add base
    logistic = 1.0 / (1.0 + math.exp(-k * (deficit - c)))
    return min(1.0, base + logistic * base)  # scale by base for gentleness


def calculate_system_health(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates system-wide health metrics."""
    alive_agents = [a for a in agents if a["alive"]]
    n_alive = len(alive_agents)
    n_total = len(agents)
    alive_fraction = n_alive / n_total if n_total > 0 else 0.0

    return {"n_alive": n_alive, "alive_fraction": alive_fraction}


def apply_shock_effects(
    state: dict[str, Any], cfg: dict[str, Any], t: int, shock_step: int
) -> tuple[float, float]:
    """Return (effective_resource_mult, extra_mortality) for this step."""
    if t < shock_step:
        return 1.0, 0.0
    dt = t - shock_step
    decay = exp_decay_factor(dt, cfg["shock"]["taper_half_life"])
    eff_mult = 1.0 - cfg["shock"]["severity_to_damage"] * decay
    extra_mort = cfg["recovery"]["mortality_base_rate"] * exp_decay_factor(
        dt, cfg["shock"]["taper_half_life"]
    )
    return eff_mult, extra_mort

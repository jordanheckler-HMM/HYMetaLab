# data_sources/earth_calibration.py
"""
Earth → Sim parameter calibration (non-intrusive, optional).
Maps real-world indicators to the sim's CivParams fields.

Usage patterns:
- import get_calibrated_params; pass into your runner when --realworld provided
- default behavior (no call) leaves all prior experiments unchanged
"""

from dataclasses import dataclass


# Minimal CivParams shadow to avoid tight coupling; runners convert to their native CivParams
@dataclass
class CalibratedParams:
    population: int
    goal_diversity: int
    social_weight: float
    inequality: float
    init_tech: float
    init_cci: float
    innovation_rate: float


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def estimate_cci_from_signals(
    internet_penetration: float,  # 0..1
    ai_capability_index: float,  # 0..1 (your proxy)
    polarization_index: float,  # 0..1 (higher = more polarized)
) -> float:
    # Info integration up, coherence down with polarization; lightly weight AI as integration amplifier
    base = 0.35 + 0.45 * internet_penetration + 0.20 * ai_capability_index
    penalty = 0.30 * polarization_index
    return _clip(base - penalty, 0.1, 0.95)


def estimate_goal_diversity(sdg_count_active: int, geopolitics_intensity: float) -> int:
    # 3-4 optimal; push toward 5+ as geopolitics heats up or too many simultaneous agendas
    raw = 3.5 + 1.0 * geopolitics_intensity + 0.2 * max(0, sdg_count_active - 4)
    return int(_clip(round(raw), 2, 6))


def estimate_social_weight(trust_index: float, institutional_capacity: float) -> float:
    # social_weight ~ coordination strength
    return _clip(
        0.2 + 0.6 * (0.6 * trust_index + 0.4 * institutional_capacity), 0.2, 0.9
    )


def estimate_init_tech(energy_per_capita_norm: float, r_and_d_share: float) -> float:
    # 0..1 scale; slight >1 allowed inside runners if desired
    return _clip(0.6 * energy_per_capita_norm + 0.4 * r_and_d_share, 0.3, 1.0)


def estimate_innovation_rate(r_and_d_share: float, startup_rate_norm: float) -> float:
    return _clip(
        0.01 + 0.03 * (0.6 * r_and_d_share + 0.4 * startup_rate_norm), 0.005, 0.05
    )


def get_calibrated_params(indicators: dict) -> CalibratedParams:
    """
    indicators dict keys (normalized 0..1 unless noted):
      - internet_penetration, ai_capability_index, polarization_index
      - trust_index, institutional_capacity
      - energy_per_capita_norm, r_and_d_share
      - startup_rate_norm
      - wealth_gini (0..1), income_gini (0..1)
      - world_pop (absolute, e.g., 8_100_000_000)
      - sdg_count_active (int)
      - geopolitics_intensity (0..1)
    """
    pop = indicators.get("world_pop", 8_000_000_000)
    # Map absolute to sim-complexity bands (100, 300, 500 representative cohorts)
    population = 500 if pop > 3e9 else (300 if pop > 3e8 else 100)

    inequality = _clip(
        max(
            indicators.get("wealth_gini", 0.6) * 0.6
            + indicators.get("income_gini", 0.4) * 0.4,
            0.12,
        ),
        0.12,
        0.6,
    )
    goal_diversity = estimate_goal_diversity(
        indicators.get("sdg_count_active", 5),
        indicators.get("geopolitics_intensity", 0.5),
    )
    social_weight = estimate_social_weight(
        indicators.get("trust_index", 0.45),
        indicators.get("institutional_capacity", 0.55),
    )
    init_tech = estimate_init_tech(
        indicators.get("energy_per_capita_norm", 0.75),
        indicators.get("r_and_d_share", 0.25),
    )
    init_cci = estimate_cci_from_signals(
        indicators.get("internet_penetration", 0.67),
        indicators.get("ai_capability_index", 0.55),
        indicators.get("polarization_index", 0.55),
    )
    innovation_rate = estimate_innovation_rate(
        indicators.get("r_and_d_share", 0.25), indicators.get("startup_rate_norm", 0.5)
    )

    return CalibratedParams(
        population=population,
        goal_diversity=goal_diversity,
        social_weight=social_weight,
        inequality=inequality,
        init_tech=init_tech,
        init_cci=init_cci,
        innovation_rate=innovation_rate,
    )

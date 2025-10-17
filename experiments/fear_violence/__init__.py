"""Fear-violence experiment suite."""

from .exp_suite import (
    run_all_fear_experiments,
    run_cci_moderation,
    run_contagion_hotspots,
    run_inequality_collapse,
    run_intervention_effects,
    run_shock_fear_aggression,
)

__all__ = [
    "run_shock_fear_aggression",
    "run_cci_moderation",
    "run_inequality_collapse",
    "run_intervention_effects",
    "run_contagion_hotspots",
    "run_all_fear_experiments",
]

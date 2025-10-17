"""
Psychology simulation framework for fear-violence studies.

This package provides adapters and hooks for existing simulation modules
to study fear-driven aggression while preserving established behavioral laws.
"""

__version__ = "0.1.0"
__author__ = "Research Copilot"

from .adapters.fear_core import aggression_prob, update_fear
from .adapters.fear_hooks import FearChannel
from .adapters.fear_metrics import compute_fear_metrics_bundle
from .adapters.interventions import FearIntervention

__all__ = [
    "update_fear",
    "aggression_prob",
    "FearChannel",
    "compute_fear_metrics_bundle",
    "FearIntervention",
]

"""
Spacetime simulation framework for retrocausality studies.

This package provides adapters and hooks for existing simulation modules
to study time travel effects while preserving established physical laws.
"""

__version__ = "0.1.0"
__author__ = "Research Copilot"

from .core import Event, RetroLink, SpacetimeGraph
from .retro.hooks import RetroChannel
from .retro.metrics import compute_retro_metrics_bundle

__all__ = [
    "Event",
    "RetroLink",
    "SpacetimeGraph",
    "RetroChannel",
    "compute_retro_metrics_bundle",
]

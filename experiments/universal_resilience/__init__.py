# experiments/universal_resilience/__init__.py
"""
Universal Resilience Experiment Module

Tests the Universal Resilience hypothesis:
Resilience increases with constructive stress and coherence, and decreases with inequality.

This module jointly varies:
- Shock severity (constructiveness)
- Inequality (Gini coefficient)
- Coherence (CCI proxy)

And analyzes outcomes to validate the UR_score predictor.
"""

__version__ = "1.0.0"
__author__ = "Consciousness Proxy Simulation Team"

from .run import main

__all__ = ["main"]

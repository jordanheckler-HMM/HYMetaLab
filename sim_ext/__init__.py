"""
Extended Simulation Framework
Medically grounded physiology, disease, ethics, information propagation,
multi-scale coupling, phenomenology, self-modeling, thermodynamics, and Bayesian inference.
"""

__version__ = "1.0.0"
__all__ = ["run_extended", "AgentState", "WorldState", "ExperimentConfig", "Metrics"]

from .extended_sweep import run_extended
from .schemas import AgentState, ExperimentConfig, Metrics, WorldState

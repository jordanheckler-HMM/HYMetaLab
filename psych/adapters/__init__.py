"""Fear-violence adaptation modules."""

from .fear_core import aggression_prob, update_fear
from .fear_hooks import FearChannel
from .fear_metrics import compute_fear_metrics_bundle
from .interventions import FearIntervention

__all__ = [
    "update_fear",
    "aggression_prob",
    "FearChannel",
    "compute_fear_metrics_bundle",
    "FearIntervention",
]

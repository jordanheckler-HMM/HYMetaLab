"""Civilization Legacy Theme - Study artifact generation, repurposing, and misinterpretation.

This theme explores how civilizations create artifacts based on their goals and values,
how these artifacts survive shocks and get repurposed, and how future observers
misinterpret their original function.
"""

from .legacy_metrics import (
    artifact_portfolio_entropy,
    dominant_function_alignment_score,
    intended_vs_observed_confusion,
    misinterpret_curve,
    persistence_by_type,
    repurpose_rate,
    repurpose_sequences,
)
from .legacy_models import (
    Artifact,
    ArtifactType,
    CivState,
    LegacyTrace,
    evolve_legacy,
    generate_artifacts,
    observer_inference,
)
from .legacy_report import generate_report
from .legacy_sweep import run_sweep

__all__ = [
    "ArtifactType",
    "CivState",
    "Artifact",
    "LegacyTrace",
    "generate_artifacts",
    "evolve_legacy",
    "observer_inference",
    "artifact_portfolio_entropy",
    "intended_vs_observed_confusion",
    "misinterpret_curve",
    "repurpose_rate",
    "repurpose_sequences",
    "persistence_by_type",
    "dominant_function_alignment_score",
    "run_sweep",
    "generate_report",
]

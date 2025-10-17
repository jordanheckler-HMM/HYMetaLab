#!/usr/bin/env python3
"""
Epistemic Boundary Utilities
Provides standardized hedging and epistemic boundary declarations for HYMetaLab outputs.

This module ensures all research outputs include appropriate epistemic qualifiers
to maintain Guardian alignment and intellectual humility.
"""

# Standard epistemic boundary declaration for HYMetaLab outputs
BOUNDARY = (
    "Epistemic boundary: Results are contingent on dataset scope, fixed seeds, "
    "and current model versions; claims should be read as provisional and subject to replication."
)


def hedge(s: str) -> str:
    """
    Add epistemic hedging to a claim or statement.

    Args:
        s: The statement to hedge

    Returns:
        The hedged statement with epistemic qualifiers

    Example:
        >>> hedge("X causes Y")
        "X causes Y This suggests, but does not prove, the effect; alternative explanations may exist."
    """
    return f"{s} This suggests, but does not prove, the effect; alternative explanations may exist."


def boundary_header() -> str:
    """
    Get the standard epistemic boundary header for reports.

    Returns:
        Formatted boundary declaration
    """
    return f"---\n{BOUNDARY}\n---\n"


def wrap_claim(claim: str, confidence: str = "moderate") -> str:
    """
    Wrap a claim with appropriate epistemic qualifiers based on confidence level.

    Args:
        claim: The claim to wrap
        confidence: Confidence level ('low', 'moderate', 'high')

    Returns:
        Claim with confidence-appropriate hedging
    """
    qualifiers = {
        "low": "may suggest that",
        "moderate": "suggests that",
        "high": "provides evidence that",
    }

    qualifier = qualifiers.get(confidence.lower(), qualifiers["moderate"])
    return f"The data {qualifier} {claim}. However, alternative explanations should be considered."


def add_boundary_to_report(report: str) -> str:
    """
    Add epistemic boundary to a report.

    Args:
        report: The report content

    Returns:
        Report with boundary header prepended
    """
    return boundary_header() + "\n" + report


# Guardian-aligned epistemic markers
EPISTEMIC_MARKERS = {
    "appears": "Observational claim, subject to alternative interpretation",
    "suggests": "Indicative pattern, not causal proof",
    "may": "Possibility, requires further validation",
    "likely": "Probabilistic claim, not certainty",
    "tends to": "General pattern, exceptions may exist",
    "associated with": "Correlation, not causation",
    "consistent with": "Compatibility, not confirmation",
}


def get_marker_explanation(marker: str) -> str:
    """
    Get explanation for an epistemic marker.

    Args:
        marker: The epistemic marker to explain

    Returns:
        Explanation of what the marker implies epistemically
    """
    return EPISTEMIC_MARKERS.get(
        marker.lower(), "Epistemic qualifier used to maintain intellectual humility"
    )

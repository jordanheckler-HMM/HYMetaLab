"""Metrics for analyzing civilization legacy patterns.

Provides functions to analyze artifact portfolios, repurposing patterns,
misinterpretation rates, and persistence across different artifact types.
"""

from collections import Counter

import numpy as np
import pandas as pd

from .legacy_models import Artifact, LegacyTrace


def artifact_portfolio_entropy(artifacts: list[Artifact]) -> float:
    """
    Calculate Shannon entropy of artifact portfolio.

    Higher entropy indicates more diverse artifact types.

    Args:
        artifacts: List of artifacts

    Returns:
        Shannon entropy of artifact type distribution
    """
    if not artifacts:
        return 0.0

    # Count artifact types
    type_counts = Counter(artifact.atype for artifact in artifacts)
    total = len(artifacts)

    # Calculate entropy
    entropy = 0.0
    for count in type_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def intended_vs_observed_confusion(legacies: list[LegacyTrace]) -> pd.DataFrame:
    """
    Create confusion matrix between intended and observed artifact types.

    Args:
        legacies: List of legacy traces

    Returns:
        DataFrame with confusion matrix
    """
    if not legacies:
        return pd.DataFrame()

    # Create confusion matrix
    confusion_data = []
    for trace in legacies:
        confusion_data.append(
            {
                "intended": trace.artifact.atype.value,
                "observed": trace.observer_inference.value,
                "misinterpret_prob": trace.misinterpret_prob,
            }
        )

    df = pd.DataFrame(confusion_data)

    # Create confusion matrix
    confusion_matrix = pd.crosstab(df["intended"], df["observed"], margins=True)

    return confusion_matrix


def misinterpret_curve(legacies: list[LegacyTrace]) -> pd.DataFrame:
    """
    Analyze misinterpretation probability vs. time and collapse severity.

    Args:
        legacies: List of legacy traces

    Returns:
        DataFrame with misinterpretation analysis
    """
    if not legacies:
        return pd.DataFrame()

    data = []
    for trace in legacies:
        collapse_severity = len(trace.repurpose_history)
        data.append(
            {
                "survival_time": trace.survival_time,
                "collapse_severity": collapse_severity,
                "misinterpret_prob": trace.misinterpret_prob,
                "artifact_type": trace.artifact.atype.value,
                "repurposed": trace.repurposed,
            }
        )

    df = pd.DataFrame(data)

    # Group by time bins and collapse severity
    df["time_bin"] = pd.cut(
        df["survival_time"],
        bins=5,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    df["severity_bin"] = pd.cut(
        df["collapse_severity"], bins=3, labels=["Low", "Medium", "High"]
    )

    # Calculate mean misinterpretation by bins
    curve_data = (
        df.groupby(["time_bin", "severity_bin"])["misinterpret_prob"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return curve_data


def repurpose_rate(legacies: list[LegacyTrace]) -> float:
    """
    Calculate overall repurposing rate.

    Args:
        legacies: List of legacy traces

    Returns:
        Fraction of artifacts that were repurposed
    """
    if not legacies:
        return 0.0

    repurposed_count = sum(1 for trace in legacies if trace.repurposed)
    return repurposed_count / len(legacies)


def repurpose_sequences(legacies: list[LegacyTrace]) -> dict[str, int]:
    """
    Count repurposing sequences.

    Args:
        legacies: List of legacy traces

    Returns:
        Dictionary mapping repurpose sequences to counts
    """
    sequences = {}

    for trace in legacies:
        if trace.repurpose_history:
            # Create sequence string
            sequence = " -> ".join(trace.repurpose_history)
            sequences[sequence] = sequences.get(sequence, 0) + 1

    return sequences


def persistence_by_type(legacies: list[LegacyTrace]) -> pd.DataFrame:
    """
    Calculate persistence statistics by artifact type (Kaplan-Meier-like).

    Args:
        legacies: List of legacy traces

    Returns:
        DataFrame with persistence statistics by type
    """
    if not legacies:
        return pd.DataFrame()

    data = []
    for trace in legacies:
        data.append(
            {
                "artifact_type": trace.artifact.atype.value,
                "survival_time": trace.survival_time,
                "durability": trace.artifact.durability,
                "maintenance_need": trace.artifact.maintenance_need,
                "repurposed": trace.repurposed,
            }
        )

    df = pd.DataFrame(data)

    # Calculate statistics by type
    stats = (
        df.groupby("artifact_type")
        .agg(
            {
                "survival_time": ["mean", "std", "median", "min", "max"],
                "durability": "mean",
                "maintenance_need": "mean",
                "repurposed": "sum",
            }
        )
        .round(3)
    )

    # Flatten column names
    stats.columns = ["_".join(col).strip() for col in stats.columns]
    stats = stats.reset_index()

    return stats


def dominant_function_alignment_score(
    artifacts: list[Artifact], civ_goals_vector: dict[str, float]
) -> float:
    """
    Calculate alignment between artifact functions and civilization goals.

    Args:
        artifacts: List of artifacts
        civ_goals_vector: Civilization goal weights

    Returns:
        Alignment score (0-1)
    """
    if not artifacts or not civ_goals_vector:
        return 0.0

    # Aggregate artifact function vectors
    total_functions = {}
    for artifact in artifacts:
        for func, weight in artifact.intended_function_vector.items():
            total_functions[func] = total_functions.get(func, 0) + weight

    # Normalize artifact functions
    total_weight = sum(total_functions.values())
    if total_weight == 0:
        return 0.0

    artifact_functions = {k: v / total_weight for k, v in total_functions.items()}

    # Calculate alignment (cosine similarity)
    common_functions = set(artifact_functions.keys()) & set(civ_goals_vector.keys())
    if not common_functions:
        return 0.0

    dot_product = sum(
        artifact_functions[f] * civ_goals_vector[f] for f in common_functions
    )

    norm_artifact = np.sqrt(sum(v**2 for v in artifact_functions.values()))
    norm_goals = np.sqrt(sum(v**2 for v in civ_goals_vector.values()))

    if norm_artifact == 0 or norm_goals == 0:
        return 0.0

    alignment = dot_product / (norm_artifact * norm_goals)
    return max(0.0, min(1.0, alignment))

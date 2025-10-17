"""Core data models for civilization legacy simulation.

Defines artifact types, civilization states, artifacts, and legacy traces
for studying how civilizations create and maintain cultural artifacts.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ArtifactType(Enum):
    """Types of artifacts that civilizations can create."""

    COORDINATION_MONUMENT = "coordination_monument"  # Ritual/coordination hubs
    RESOURCE_STORE = "resource_store"  # Granaries, caches
    SIGNALING_TOWER = "signaling_tower"  # Beacons, astronomical markers
    POWER_INFRA = "power_infra"  # Energy/logistics infrastructure
    BURIAL_TOMB = "burial_tomb"  # Mortuary/ancestral functions
    KNOWLEDGE_ARCHIVE = "knowledge_archive"  # Scriptoria, libraries


@dataclass
class CivState:
    """State of a civilization at a point in time."""

    cci: float  # Collective Consciousness Index
    gini: float  # Gini coefficient (inequality)
    population: int  # Population size
    goal_diversity: int  # Number of distinct goals
    social_weight: float  # Social influence weight
    shock_severity: float  # Current shock severity
    time: int  # Time step


@dataclass
class Artifact:
    """An artifact created by a civilization."""

    atype: ArtifactType  # Type of artifact
    build_time: int  # When it was built
    intended_function_vector: dict[str, float]  # Intended function weights
    materials: dict[str, float]  # Material composition
    durability: float  # Durability score (0-1)
    visibility: float  # Visibility score (0-1)
    maintenance_need: float  # Maintenance requirement (0-1)


@dataclass
class LegacyTrace:
    """Trace of an artifact through time, including repurposing."""

    artifact: Artifact  # Original artifact
    repurposed: bool  # Whether it was repurposed
    repurpose_history: list[str]  # History of repurposing
    survival_time: int  # How long it survived
    observer_inference: ArtifactType  # What observers think it is
    misinterpret_prob: float  # Probability of misinterpretation


def generate_artifacts(civ: CivState, rng: np.random.Generator) -> list[Artifact]:
    """
    Generate artifacts based on civilization state.

    Heuristics:
    - High CCI, balanced Gini -> more COORDINATION_MONUMENT + KNOWLEDGE_ARCHIVE
    - High inequality (Gini>0.3) -> monumental displays + resource hoarding
    - Low goal diversity -> monoculture artifacts; high diversity -> mixed portfolio
    - If social_weight ~0.5 and CCI>0.7 -> SIGNALING_TOWER prevalence
    - POWER_INFRA emerges only if goal mix includes logistics/energy

    Args:
        civ: Current civilization state
        rng: Random number generator

    Returns:
        List of generated artifacts
    """
    artifacts = []

    # Base artifact count scales with population and CCI
    base_count = int(civ.population * 0.01 * (0.5 + civ.cci))
    artifact_count = max(1, rng.poisson(base_count))

    for _ in range(artifact_count):
        # Determine artifact type based on civilization characteristics
        atype = _select_artifact_type(civ, rng)

        # Generate function vector based on civilization goals
        function_vector = _generate_function_vector(civ, atype, rng)

        # Generate materials based on CCI and resource concentration
        materials = _generate_materials(civ, rng)

        # Durability scales with CCI and material quality
        durability = min(1.0, civ.cci * 0.7 + np.mean(list(materials.values())) * 0.3)

        # Visibility depends on artifact type and inequality
        visibility = _calculate_visibility(atype, civ.gini)

        # Maintenance need inversely related to durability
        maintenance_need = max(0.0, 1.0 - durability + rng.normal(0, 0.1))

        artifact = Artifact(
            atype=atype,
            build_time=civ.time,
            intended_function_vector=function_vector,
            materials=materials,
            durability=durability,
            visibility=visibility,
            maintenance_need=max(0.0, min(1.0, maintenance_need)),
        )
        artifacts.append(artifact)

    return artifacts


def _select_artifact_type(civ: CivState, rng: np.random.Generator) -> ArtifactType:
    """Select artifact type based on civilization characteristics."""
    # High CCI, balanced Gini -> coordination and knowledge
    if civ.cci > 0.7 and civ.gini < 0.3:
        weights = {
            ArtifactType.COORDINATION_MONUMENT: 0.3,
            ArtifactType.KNOWLEDGE_ARCHIVE: 0.3,
            ArtifactType.SIGNALING_TOWER: 0.2,
            ArtifactType.RESOURCE_STORE: 0.1,
            ArtifactType.BURIAL_TOMB: 0.05,
            ArtifactType.POWER_INFRA: 0.05,
        }
    # High inequality -> monumental displays and resource hoarding
    elif civ.gini > 0.3:
        weights = {
            ArtifactType.COORDINATION_MONUMENT: 0.4,
            ArtifactType.RESOURCE_STORE: 0.3,
            ArtifactType.BURIAL_TOMB: 0.15,
            ArtifactType.SIGNALING_TOWER: 0.1,
            ArtifactType.KNOWLEDGE_ARCHIVE: 0.03,
            ArtifactType.POWER_INFRA: 0.02,
        }
    # Low goal diversity -> monoculture
    elif civ.goal_diversity <= 2:
        weights = {
            ArtifactType.COORDINATION_MONUMENT: 0.6,
            ArtifactType.RESOURCE_STORE: 0.2,
            ArtifactType.BURIAL_TOMB: 0.1,
            ArtifactType.SIGNALING_TOWER: 0.05,
            ArtifactType.KNOWLEDGE_ARCHIVE: 0.03,
            ArtifactType.POWER_INFRA: 0.02,
        }
    # High diversity -> mixed portfolio
    else:
        weights = {
            ArtifactType.COORDINATION_MONUMENT: 0.2,
            ArtifactType.RESOURCE_STORE: 0.2,
            ArtifactType.SIGNALING_TOWER: 0.2,
            ArtifactType.KNOWLEDGE_ARCHIVE: 0.15,
            ArtifactType.BURIAL_TOMB: 0.15,
            ArtifactType.POWER_INFRA: 0.1,
        }

    # Adjust for social weight and CCI
    if 0.4 <= civ.social_weight <= 0.6 and civ.cci > 0.7:
        weights[ArtifactType.SIGNALING_TOWER] *= 1.5

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Sample based on weights
    types = list(normalized_weights.keys())
    probs = list(normalized_weights.values())
    return rng.choice(types, p=probs)


def _generate_function_vector(
    civ: CivState, atype: ArtifactType, rng: np.random.Generator
) -> dict[str, float]:
    """Generate function vector for an artifact."""
    functions = ["coordination", "storage", "signaling", "power", "burial", "knowledge"]

    # Base weights depend on artifact type
    base_weights = {
        ArtifactType.COORDINATION_MONUMENT: [0.8, 0.1, 0.05, 0.02, 0.01, 0.02],
        ArtifactType.RESOURCE_STORE: [0.1, 0.8, 0.05, 0.02, 0.01, 0.02],
        ArtifactType.SIGNALING_TOWER: [0.1, 0.05, 0.8, 0.02, 0.01, 0.02],
        ArtifactType.POWER_INFRA: [0.1, 0.1, 0.1, 0.7, 0.0, 0.0],
        ArtifactType.BURIAL_TOMB: [0.1, 0.1, 0.05, 0.02, 0.8, 0.03],
        ArtifactType.KNOWLEDGE_ARCHIVE: [0.1, 0.1, 0.05, 0.02, 0.01, 0.8],
    }

    weights = base_weights[atype].copy()

    # Add noise based on goal diversity
    noise_scale = 0.1 * (6 - civ.goal_diversity) / 5  # More noise with less diversity
    for i in range(len(weights)):
        weights[i] += rng.normal(0, noise_scale)
        weights[i] = max(0.0, weights[i])

    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    return dict(zip(functions, weights))


def _generate_materials(civ: CivState, rng: np.random.Generator) -> dict[str, float]:
    """Generate material composition for an artifact."""
    materials = ["stone", "wood", "metal", "ceramic", "organic"]

    # Higher CCI -> better materials
    if civ.cci > 0.7:
        base_weights = [0.3, 0.2, 0.3, 0.15, 0.05]
    elif civ.cci > 0.5:
        base_weights = [0.4, 0.3, 0.2, 0.08, 0.02]
    else:
        base_weights = [0.5, 0.4, 0.05, 0.04, 0.01]

    # Add noise
    weights = [w + rng.normal(0, 0.05) for w in base_weights]
    weights = [max(0.0, w) for w in weights]

    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    return dict(zip(materials, weights))


def _calculate_visibility(atype: ArtifactType, gini: float) -> float:
    """Calculate visibility score for an artifact type."""
    base_visibility = {
        ArtifactType.COORDINATION_MONUMENT: 0.9,
        ArtifactType.SIGNALING_TOWER: 0.95,
        ArtifactType.BURIAL_TOMB: 0.7,
        ArtifactType.RESOURCE_STORE: 0.6,
        ArtifactType.POWER_INFRA: 0.8,
        ArtifactType.KNOWLEDGE_ARCHIVE: 0.5,
    }

    visibility = base_visibility[atype]

    # High inequality increases visibility of monuments
    if atype == ArtifactType.COORDINATION_MONUMENT and gini > 0.3:
        visibility = min(1.0, visibility * 1.2)

    return visibility


def evolve_legacy(
    artifacts: list[Artifact],
    shocks: list[float],
    cci_traj: list[float],
    rng: np.random.Generator,
) -> list[LegacyTrace]:
    """
    Evolve artifacts through time with shocks and repurposing.

    Args:
        artifacts: Initial artifacts
        shocks: Shock severity at each time step
        cci_traj: CCI trajectory over time
        rng: Random number generator

    Returns:
        List of legacy traces
    """
    traces = []

    for artifact in artifacts:
        trace = LegacyTrace(
            artifact=artifact,
            repurposed=False,
            repurpose_history=[],
            survival_time=0,
            observer_inference=artifact.atype,
            misinterpret_prob=0.0,
        )

        current_artifact = artifact
        current_type = artifact.atype

        for t, (shock, cci) in enumerate(zip(shocks, cci_traj)):
            # Check for abandonment/failure
            failure_prob = shock * (1.0 - artifact.durability) * (1.0 - cci)
            if rng.random() < failure_prob:
                break

            # Check for repurposing
            if shock > 0.3:  # Significant shock
                repurpose_prob = shock * 0.5 * (1.0 - artifact.maintenance_need)
                if rng.random() < repurpose_prob:
                    # Repurpose to a different type
                    new_type = _select_repurpose_type(current_type, shock, rng)
                    trace.repurpose_history.append(
                        f"{current_type.value} -> {new_type.value}"
                    )
                    current_type = new_type
                    trace.repurposed = True

            trace.survival_time += 1

        # Set observer inference and misinterpretation probability
        trace.observer_inference, trace.misinterpret_prob = observer_inference(
            trace,
            observer_noise=0.2,
            cultural_distance=0.5,
            time_gap=trace.survival_time,
            rng=rng,
        )

        traces.append(trace)

    return traces


def _select_repurpose_type(
    current_type: ArtifactType, shock_severity: float, rng: np.random.Generator
) -> ArtifactType:
    """Select new type when repurposing an artifact."""
    # Common repurposing patterns
    repurpose_map = {
        ArtifactType.COORDINATION_MONUMENT: [
            ArtifactType.BURIAL_TOMB,
            ArtifactType.RESOURCE_STORE,
            ArtifactType.SIGNALING_TOWER,
        ],
        ArtifactType.RESOURCE_STORE: [
            ArtifactType.COORDINATION_MONUMENT,
            ArtifactType.BURIAL_TOMB,
        ],
        ArtifactType.SIGNALING_TOWER: [
            ArtifactType.COORDINATION_MONUMENT,
            ArtifactType.RESOURCE_STORE,
        ],
        ArtifactType.POWER_INFRA: [
            ArtifactType.RESOURCE_STORE,
            ArtifactType.COORDINATION_MONUMENT,
        ],
        ArtifactType.BURIAL_TOMB: [
            ArtifactType.COORDINATION_MONUMENT,
            ArtifactType.RESOURCE_STORE,
        ],
        ArtifactType.KNOWLEDGE_ARCHIVE: [
            ArtifactType.RESOURCE_STORE,
            ArtifactType.COORDINATION_MONUMENT,
        ],
    }

    candidates = repurpose_map.get(current_type, list(ArtifactType))
    return rng.choice(candidates)


def observer_inference(
    trace: LegacyTrace,
    observer_noise: float,
    cultural_distance: float,
    time_gap: int,
    rng: np.random.Generator,
) -> tuple[ArtifactType, float]:
    """
    Simulate observer inference about artifact function.

    Args:
        trace: Legacy trace to analyze
        observer_noise: Base noise level
        cultural_distance: Cultural distance from original civilization
        time_gap: Time gap since creation
        rng: Random number generator

    Returns:
        Tuple of (inferred_type, misinterpret_prob)
    """
    # Base misinterpretation probability
    base_prob = observer_noise

    # Increase with time gap
    time_factor = min(1.0, time_gap / 100.0)
    base_prob += time_factor * 0.3

    # Increase with cultural distance
    base_prob += cultural_distance * 0.2

    # Increase with collapse severity (proxied by number of repurposings)
    collapse_factor = len(trace.repurpose_history) * 0.1
    base_prob += collapse_factor

    # Decrease if knowledge archives survived
    if trace.artifact.atype == ArtifactType.KNOWLEDGE_ARCHIVE:
        base_prob *= 0.5

    # Cap at 0.95
    misinterpret_prob = min(0.95, base_prob)

    # If misinterpretation occurs, choose a different type
    if rng.random() < misinterpret_prob:
        # Common misinterpretations
        misinterpret_map = {
            ArtifactType.COORDINATION_MONUMENT: [
                ArtifactType.BURIAL_TOMB,
                ArtifactType.SIGNALING_TOWER,
            ],
            ArtifactType.RESOURCE_STORE: [
                ArtifactType.BURIAL_TOMB,
                ArtifactType.COORDINATION_MONUMENT,
            ],
            ArtifactType.SIGNALING_TOWER: [
                ArtifactType.COORDINATION_MONUMENT,
                ArtifactType.BURIAL_TOMB,
            ],
            ArtifactType.POWER_INFRA: [
                ArtifactType.RESOURCE_STORE,
                ArtifactType.COORDINATION_MONUMENT,
            ],
            ArtifactType.BURIAL_TOMB: [
                ArtifactType.COORDINATION_MONUMENT,
                ArtifactType.RESOURCE_STORE,
            ],
            ArtifactType.KNOWLEDGE_ARCHIVE: [
                ArtifactType.RESOURCE_STORE,
                ArtifactType.COORDINATION_MONUMENT,
            ],
        }

        candidates = misinterpret_map.get(trace.artifact.atype, list(ArtifactType))
        inferred_type = rng.choice(candidates)
    else:
        inferred_type = trace.artifact.atype

    return inferred_type, misinterpret_prob

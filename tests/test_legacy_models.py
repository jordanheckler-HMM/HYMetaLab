"""Tests for civilization legacy models.

Tests the core data models, artifact generation, and legacy evolution
functionality.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from themes.theme10_civilization_legacy.legacy_models import (
    Artifact,
    ArtifactType,
    CivState,
    LegacyTrace,
    evolve_legacy,
    generate_artifacts,
    observer_inference,
)


class TestArtifactGeneration:
    """Test artifact generation functionality."""

    def test_generate_artifacts_basic(self):
        """Test basic artifact generation."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        assert len(artifacts) > 0, "Should generate at least one artifact"
        assert all(
            isinstance(a, Artifact) for a in artifacts
        ), "All items should be Artifact instances"
        assert all(
            isinstance(a.atype, ArtifactType) for a in artifacts
        ), "All artifacts should have valid types"

    def test_generate_artifacts_high_cci(self):
        """Test artifact generation for high CCI civilizations."""
        civ = CivState(
            cci=0.9,
            gini=0.2,
            population=1000,
            goal_diversity=5,
            social_weight=0.6,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # High CCI should produce more coordination monuments and knowledge archives
        coordination_count = sum(
            1 for a in artifacts if a.atype == ArtifactType.COORDINATION_MONUMENT
        )
        knowledge_count = sum(
            1 for a in artifacts if a.atype == ArtifactType.KNOWLEDGE_ARCHIVE
        )

        assert coordination_count > 0, "High CCI should produce coordination monuments"
        assert knowledge_count > 0, "High CCI should produce knowledge archives"

    def test_generate_artifacts_high_inequality(self):
        """Test artifact generation for high inequality civilizations."""
        civ = CivState(
            cci=0.5,
            gini=0.4,
            population=1000,
            goal_diversity=3,
            social_weight=0.3,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # High inequality should produce more resource stores and burial tombs
        resource_count = sum(
            1 for a in artifacts if a.atype == ArtifactType.RESOURCE_STORE
        )
        burial_count = sum(1 for a in artifacts if a.atype == ArtifactType.BURIAL_TOMB)

        assert resource_count > 0, "High inequality should produce resource stores"
        assert burial_count > 0, "High inequality should produce burial tombs"

    def test_generate_artifacts_low_diversity(self):
        """Test artifact generation for low goal diversity."""
        civ = CivState(
            cci=0.6,
            gini=0.3,
            population=1000,
            goal_diversity=1,
            social_weight=0.4,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Low diversity should produce more coordination monuments (monoculture)
        coordination_count = sum(
            1 for a in artifacts if a.atype == ArtifactType.COORDINATION_MONUMENT
        )
        total_count = len(artifacts)

        assert (
            coordination_count / total_count > 0.4
        ), "Low diversity should produce mostly coordination monuments"

    def test_artifact_properties(self):
        """Test artifact property ranges."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        for artifact in artifacts:
            assert 0.0 <= artifact.durability <= 1.0, "Durability should be in [0, 1]"
            assert 0.0 <= artifact.visibility <= 1.0, "Visibility should be in [0, 1]"
            assert (
                0.0 <= artifact.maintenance_need <= 1.0
            ), "Maintenance need should be in [0, 1]"
            assert (
                len(artifact.intended_function_vector) > 0
            ), "Function vector should not be empty"
            assert len(artifact.materials) > 0, "Materials should not be empty"


class TestLegacyEvolution:
    """Test legacy evolution functionality."""

    def test_evolve_legacy_basic(self):
        """Test basic legacy evolution."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Create shock schedule
        shocks = [0.0] * 100 + [0.5] + [0.0] * 99
        cci_traj = [0.7] * 200

        legacies = evolve_legacy(artifacts, shocks, cci_traj, rng)

        assert len(legacies) == len(artifacts), "Should have one legacy per artifact"
        assert all(
            isinstance(l, LegacyTrace) for l in legacies
        ), "All items should be LegacyTrace instances"
        assert all(
            l.survival_time >= 0 for l in legacies
        ), "Survival time should be non-negative"

    def test_evolve_legacy_with_shocks(self):
        """Test legacy evolution with significant shocks."""
        civ = CivState(
            cci=0.5,
            gini=0.3,
            population=1000,
            goal_diversity=3,
            social_weight=0.4,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Create high-severity shock schedule
        shocks = [0.0] * 50 + [0.8] + [0.0] * 49
        cci_traj = [0.5] * 100

        legacies = evolve_legacy(artifacts, shocks, cci_traj, rng)

        # Some artifacts should be repurposed due to high shock
        repurposed_count = sum(1 for l in legacies if l.repurposed)
        assert repurposed_count > 0, "High shocks should cause some repurposing"

    def test_evolve_legacy_durability_effect(self):
        """Test that durability affects survival."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Create moderate shock schedule
        shocks = [0.0] * 100 + [0.3] + [0.0] * 99
        cci_traj = [0.7] * 200

        legacies = evolve_legacy(artifacts, shocks, cci_traj, rng)

        # Durable artifacts should survive longer
        durable_artifacts = [l for l in legacies if l.artifact.durability > 0.7]
        fragile_artifacts = [l for l in legacies if l.artifact.durability < 0.3]

        if durable_artifacts and fragile_artifacts:
            avg_durable_survival = np.mean([l.survival_time for l in durable_artifacts])
            avg_fragile_survival = np.mean([l.survival_time for l in fragile_artifacts])

            assert (
                avg_durable_survival >= avg_fragile_survival
            ), "Durable artifacts should survive longer"


class TestObserverInference:
    """Test observer inference functionality."""

    def test_observer_inference_basic(self):
        """Test basic observer inference."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Create a simple legacy trace
        artifact = artifacts[0]
        trace = LegacyTrace(
            artifact=artifact,
            repurposed=False,
            repurpose_history=[],
            survival_time=100,
            observer_inference=artifact.atype,
            misinterpret_prob=0.0,
        )

        inferred_type, misinterpret_prob = observer_inference(
            trace, observer_noise=0.1, cultural_distance=0.2, time_gap=50, rng=rng
        )

        assert isinstance(
            inferred_type, ArtifactType
        ), "Inferred type should be ArtifactType"
        assert (
            0.0 <= misinterpret_prob <= 1.0
        ), "Misinterpret probability should be in [0, 1]"

    def test_observer_inference_time_effect(self):
        """Test that time gap affects misinterpretation."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        artifact = artifacts[0]
        trace = LegacyTrace(
            artifact=artifact,
            repurposed=False,
            repurpose_history=[],
            survival_time=100,
            observer_inference=artifact.atype,
            misinterpret_prob=0.0,
        )

        # Test with short time gap
        _, prob_short = observer_inference(
            trace, observer_noise=0.1, cultural_distance=0.2, time_gap=10, rng=rng
        )

        # Test with long time gap
        _, prob_long = observer_inference(
            trace, observer_noise=0.1, cultural_distance=0.2, time_gap=200, rng=rng
        )

        assert (
            prob_long >= prob_short
        ), "Longer time gaps should increase misinterpretation probability"

    def test_observer_inference_cultural_distance_effect(self):
        """Test that cultural distance affects misinterpretation."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        artifact = artifacts[0]
        trace = LegacyTrace(
            artifact=artifact,
            repurposed=False,
            repurpose_history=[],
            survival_time=100,
            observer_inference=artifact.atype,
            misinterpret_prob=0.0,
        )

        # Test with low cultural distance
        _, prob_low = observer_inference(
            trace, observer_noise=0.1, cultural_distance=0.1, time_gap=50, rng=rng
        )

        # Test with high cultural distance
        _, prob_high = observer_inference(
            trace, observer_noise=0.1, cultural_distance=0.8, time_gap=50, rng=rng
        )

        assert (
            prob_high >= prob_low
        ), "Higher cultural distance should increase misinterpretation probability"


class TestDeterministicBehavior:
    """Test deterministic behavior with fixed seeds."""

    def test_deterministic_artifact_generation(self):
        """Test that artifact generation is deterministic with fixed seed."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        # Generate artifacts with same seed twice
        rng1 = np.random.Generator(np.random.PCG64(42))
        artifacts1 = generate_artifacts(civ, rng1)

        rng2 = np.random.Generator(np.random.PCG64(42))
        artifacts2 = generate_artifacts(civ, rng2)

        assert len(artifacts1) == len(
            artifacts2
        ), "Same seed should produce same number of artifacts"

        # Check artifact types match
        types1 = [a.atype for a in artifacts1]
        types2 = [a.atype for a in artifacts2]
        assert types1 == types2, "Same seed should produce same artifact types"

    def test_deterministic_legacy_evolution(self):
        """Test that legacy evolution is deterministic with fixed seed."""
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        rng1 = np.random.Generator(np.random.PCG64(42))
        artifacts1 = generate_artifacts(civ, rng1)

        rng2 = np.random.Generator(np.random.PCG64(42))
        artifacts2 = generate_artifacts(civ, rng2)

        # Create same shock schedule
        shocks = [0.0] * 100 + [0.3] + [0.0] * 99
        cci_traj = [0.7] * 200

        # Evolve with same seed
        rng1 = np.random.Generator(np.random.PCG64(42))
        legacies1 = evolve_legacy(artifacts1, shocks, cci_traj, rng1)

        rng2 = np.random.Generator(np.random.PCG64(42))
        legacies2 = evolve_legacy(artifacts2, shocks, cci_traj, rng2)

        assert len(legacies1) == len(
            legacies2
        ), "Same seed should produce same number of legacies"

        # Check survival times match
        survival_times1 = [l.survival_time for l in legacies1]
        survival_times2 = [l.survival_time for l in legacies2]
        assert (
            survival_times1 == survival_times2
        ), "Same seed should produce same survival times"

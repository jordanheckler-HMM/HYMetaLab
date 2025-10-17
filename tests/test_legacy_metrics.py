"""Tests for civilization legacy metrics.

Tests the metrics calculation functions for analyzing artifact portfolios,
repurposing patterns, and misinterpretation rates.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from themes.theme10_civilization_legacy.legacy_metrics import (
    artifact_portfolio_entropy,
    dominant_function_alignment_score,
    intended_vs_observed_confusion,
    misinterpret_curve,
    persistence_by_type,
    repurpose_rate,
    repurpose_sequences,
)
from themes.theme10_civilization_legacy.legacy_models import (
    Artifact,
    ArtifactType,
    CivState,
    LegacyTrace,
    evolve_legacy,
    generate_artifacts,
)


class TestPortfolioEntropy:
    """Test portfolio entropy calculation."""

    def test_portfolio_entropy_empty(self):
        """Test portfolio entropy with empty artifact list."""
        entropy = artifact_portfolio_entropy([])
        assert entropy == 0.0, "Empty portfolio should have zero entropy"

    def test_portfolio_entropy_single_type(self):
        """Test portfolio entropy with single artifact type."""
        artifacts = [
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 1.0},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            ),
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 1.0},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            ),
        ]

        entropy = artifact_portfolio_entropy(artifacts)
        assert entropy == 0.0, "Single type portfolio should have zero entropy"

    def test_portfolio_entropy_diverse(self):
        """Test portfolio entropy with diverse artifact types."""
        artifacts = [
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 1.0},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            ),
            Artifact(
                atype=ArtifactType.RESOURCE_STORE,
                build_time=0,
                intended_function_vector={"storage": 1.0},
                materials={"wood": 1.0},
                durability=0.6,
                visibility=0.5,
                maintenance_need=0.4,
            ),
            Artifact(
                atype=ArtifactType.KNOWLEDGE_ARCHIVE,
                build_time=0,
                intended_function_vector={"knowledge": 1.0},
                materials={"ceramic": 1.0},
                durability=0.7,
                visibility=0.3,
                maintenance_need=0.3,
            ),
        ]

        entropy = artifact_portfolio_entropy(artifacts)
        assert entropy > 0.0, "Diverse portfolio should have positive entropy"
        assert entropy <= np.log2(3), "Entropy should not exceed log2(number of types)"


class TestConfusionMatrix:
    """Test confusion matrix calculation."""

    def test_confusion_matrix_empty(self):
        """Test confusion matrix with empty legacy list."""
        confusion_df = intended_vs_observed_confusion([])
        assert (
            confusion_df.empty
        ), "Empty legacy list should produce empty confusion matrix"

    def test_confusion_matrix_basic(self):
        """Test basic confusion matrix calculation."""
        legacies = [
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.COORDINATION_MONUMENT,
                    build_time=0,
                    intended_function_vector={"coordination": 1.0},
                    materials={"stone": 1.0},
                    durability=0.8,
                    visibility=0.9,
                    maintenance_need=0.2,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=100,
                observer_inference=ArtifactType.COORDINATION_MONUMENT,
                misinterpret_prob=0.1,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.RESOURCE_STORE,
                    build_time=0,
                    intended_function_vector={"storage": 1.0},
                    materials={"wood": 1.0},
                    durability=0.6,
                    visibility=0.5,
                    maintenance_need=0.4,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=80,
                observer_inference=ArtifactType.BURIAL_TOMB,
                misinterpret_prob=0.8,
            ),
        ]

        confusion_df = intended_vs_observed_confusion(legacies)
        assert not confusion_df.empty, "Should produce non-empty confusion matrix"
        assert (
            "coordination_monument" in confusion_df.index
        ), "Should include coordination monument"
        assert "resource_store" in confusion_df.index, "Should include resource store"


class TestMisinterpretCurve:
    """Test misinterpretation curve calculation."""

    def test_misinterpret_curve_empty(self):
        """Test misinterpretation curve with empty legacy list."""
        curve_df = misinterpret_curve([])
        assert curve_df.empty, "Empty legacy list should produce empty curve"

    def test_misinterpret_curve_basic(self):
        """Test basic misinterpretation curve calculation."""
        legacies = [
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.COORDINATION_MONUMENT,
                    build_time=0,
                    intended_function_vector={"coordination": 1.0},
                    materials={"stone": 1.0},
                    durability=0.8,
                    visibility=0.9,
                    maintenance_need=0.2,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=100,
                observer_inference=ArtifactType.COORDINATION_MONUMENT,
                misinterpret_prob=0.1,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.RESOURCE_STORE,
                    build_time=0,
                    intended_function_vector={"storage": 1.0},
                    materials={"wood": 1.0},
                    durability=0.6,
                    visibility=0.5,
                    maintenance_need=0.4,
                ),
                repurposed=True,
                repurpose_history=["resource_store -> burial_tomb"],
                survival_time=80,
                observer_inference=ArtifactType.BURIAL_TOMB,
                misinterpret_prob=0.8,
            ),
        ]

        curve_df = misinterpret_curve(legacies)
        assert not curve_df.empty, "Should produce non-empty curve"
        assert "time_bin" in curve_df.columns, "Should include time_bin column"
        assert "severity_bin" in curve_df.columns, "Should include severity_bin column"
        # Note: misinterpret_prob is aggregated, so it's in the result columns


class TestRepurposeMetrics:
    """Test repurposing metrics."""

    def test_repurpose_rate_empty(self):
        """Test repurpose rate with empty legacy list."""
        rate = repurpose_rate([])
        assert rate == 0.0, "Empty legacy list should have zero repurpose rate"

    def test_repurpose_rate_mixed(self):
        """Test repurpose rate with mixed repurposing."""
        legacies = [
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.COORDINATION_MONUMENT,
                    build_time=0,
                    intended_function_vector={"coordination": 1.0},
                    materials={"stone": 1.0},
                    durability=0.8,
                    visibility=0.9,
                    maintenance_need=0.2,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=100,
                observer_inference=ArtifactType.COORDINATION_MONUMENT,
                misinterpret_prob=0.1,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.RESOURCE_STORE,
                    build_time=0,
                    intended_function_vector={"storage": 1.0},
                    materials={"wood": 1.0},
                    durability=0.6,
                    visibility=0.5,
                    maintenance_need=0.4,
                ),
                repurposed=True,
                repurpose_history=["resource_store -> burial_tomb"],
                survival_time=80,
                observer_inference=ArtifactType.BURIAL_TOMB,
                misinterpret_prob=0.8,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.KNOWLEDGE_ARCHIVE,
                    build_time=0,
                    intended_function_vector={"knowledge": 1.0},
                    materials={"ceramic": 1.0},
                    durability=0.7,
                    visibility=0.3,
                    maintenance_need=0.3,
                ),
                repurposed=True,
                repurpose_history=["knowledge_archive -> resource_store"],
                survival_time=60,
                observer_inference=ArtifactType.RESOURCE_STORE,
                misinterpret_prob=0.6,
            ),
        ]

        rate = repurpose_rate(legacies)
        assert rate == 2 / 3, "Should calculate correct repurpose rate"

        sequences = repurpose_sequences(legacies)
        assert len(sequences) == 2, "Should identify two unique sequences"
        assert (
            "resource_store -> burial_tomb" in sequences
        ), "Should include resource store sequence"
        assert (
            "knowledge_archive -> resource_store" in sequences
        ), "Should include knowledge archive sequence"


class TestPersistenceMetrics:
    """Test persistence metrics."""

    def test_persistence_by_type_empty(self):
        """Test persistence by type with empty legacy list."""
        persistence_df = persistence_by_type([])
        assert (
            persistence_df.empty
        ), "Empty legacy list should produce empty persistence data"

    def test_persistence_by_type_basic(self):
        """Test basic persistence by type calculation."""
        legacies = [
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.COORDINATION_MONUMENT,
                    build_time=0,
                    intended_function_vector={"coordination": 1.0},
                    materials={"stone": 1.0},
                    durability=0.8,
                    visibility=0.9,
                    maintenance_need=0.2,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=100,
                observer_inference=ArtifactType.COORDINATION_MONUMENT,
                misinterpret_prob=0.1,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.COORDINATION_MONUMENT,
                    build_time=0,
                    intended_function_vector={"coordination": 1.0},
                    materials={"stone": 1.0},
                    durability=0.8,
                    visibility=0.9,
                    maintenance_need=0.2,
                ),
                repurposed=False,
                repurpose_history=[],
                survival_time=120,
                observer_inference=ArtifactType.COORDINATION_MONUMENT,
                misinterpret_prob=0.1,
            ),
            LegacyTrace(
                artifact=Artifact(
                    atype=ArtifactType.RESOURCE_STORE,
                    build_time=0,
                    intended_function_vector={"storage": 1.0},
                    materials={"wood": 1.0},
                    durability=0.6,
                    visibility=0.5,
                    maintenance_need=0.4,
                ),
                repurposed=True,
                repurpose_history=["resource_store -> burial_tomb"],
                survival_time=80,
                observer_inference=ArtifactType.BURIAL_TOMB,
                misinterpret_prob=0.8,
            ),
        ]

        persistence_df = persistence_by_type(legacies)
        assert not persistence_df.empty, "Should produce non-empty persistence data"
        assert (
            "artifact_type" in persistence_df.columns
        ), "Should include artifact_type column"
        assert (
            "survival_time_mean" in persistence_df.columns
        ), "Should include survival_time_mean column"
        assert (
            "durability_mean" in persistence_df.columns
        ), "Should include durability_mean column"

        # Check that coordination monuments have higher average survival time
        coord_data = persistence_df[
            persistence_df["artifact_type"] == "coordination_monument"
        ]
        resource_data = persistence_df[
            persistence_df["artifact_type"] == "resource_store"
        ]

        if not coord_data.empty and not resource_data.empty:
            coord_survival = coord_data["survival_time_mean"].iloc[0]
            resource_survival = resource_data["survival_time_mean"].iloc[0]
            assert (
                coord_survival > resource_survival
            ), "Coordination monuments should survive longer"


class TestAlignmentScore:
    """Test function-goal alignment score."""

    def test_alignment_score_empty(self):
        """Test alignment score with empty inputs."""
        score = dominant_function_alignment_score([], {})
        assert score == 0.0, "Empty inputs should produce zero alignment score"

    def test_alignment_score_perfect(self):
        """Test alignment score with robust alignment."""
        artifacts = [
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 1.0, "storage": 0.0},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            )
        ]

        civ_goals = {"coordination": 1.0, "storage": 0.0}

        score = dominant_function_alignment_score(artifacts, civ_goals)
        assert score == 1.0, "robust alignment should produce score of 1.0"

    def test_alignment_score_no_overlap(self):
        """Test alignment score with no function overlap."""
        artifacts = [
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 1.0},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            )
        ]

        civ_goals = {"storage": 1.0}

        score = dominant_function_alignment_score(artifacts, civ_goals)
        assert score == 0.0, "No overlap should produce zero alignment score"

    def test_alignment_score_partial(self):
        """Test alignment score with partial alignment."""
        artifacts = [
            Artifact(
                atype=ArtifactType.COORDINATION_MONUMENT,
                build_time=0,
                intended_function_vector={"coordination": 0.7, "storage": 0.3},
                materials={"stone": 1.0},
                durability=0.8,
                visibility=0.9,
                maintenance_need=0.2,
            ),
            Artifact(
                atype=ArtifactType.RESOURCE_STORE,
                build_time=0,
                intended_function_vector={"coordination": 0.3, "storage": 0.7},
                materials={"wood": 1.0},
                durability=0.6,
                visibility=0.5,
                maintenance_need=0.4,
            ),
        ]

        civ_goals = {"coordination": 0.6, "storage": 0.4}

        score = dominant_function_alignment_score(artifacts, civ_goals)
        assert (
            0.0 < score < 1.0
        ), "Partial alignment should produce score between 0 and 1"


class TestIntegration:
    """Test integration between models and metrics."""

    def test_full_pipeline(self):
        """Test full pipeline from artifact generation to metrics calculation."""
        # Generate civilization state
        civ = CivState(
            cci=0.7,
            gini=0.25,
            population=1000,
            goal_diversity=4,
            social_weight=0.5,
            shock_severity=0.0,
            time=0,
        )

        # Generate artifacts
        rng = np.random.Generator(np.random.PCG64(42))
        artifacts = generate_artifacts(civ, rng)

        # Evolve legacy
        shocks = [0.0] * 100 + [0.3] + [0.0] * 99
        cci_traj = [0.7] * 200
        legacies = evolve_legacy(artifacts, shocks, cci_traj, rng)

        # Calculate metrics
        entropy = artifact_portfolio_entropy(artifacts)
        confusion_df = intended_vs_observed_confusion(legacies)
        curve_df = misinterpret_curve(legacies)
        rate = repurpose_rate(legacies)
        sequences = repurpose_sequences(legacies)
        persistence_df = persistence_by_type(legacies)

        # Generate civilization goals
        civ_goals = {
            "coordination": 0.4,
            "storage": 0.3,
            "signaling": 0.2,
            "power": 0.05,
            "burial": 0.03,
            "knowledge": 0.02,
        }
        alignment = dominant_function_alignment_score(artifacts, civ_goals)

        # Verify all metrics are calculated
        assert entropy >= 0.0, "Entropy should be non-negative"
        assert not confusion_df.empty, "Confusion matrix should not be empty"
        assert not curve_df.empty, "Misinterpret curve should not be empty"
        assert 0.0 <= rate <= 1.0, "Repurpose rate should be in [0, 1]"
        assert isinstance(sequences, dict), "Sequences should be a dictionary"
        assert not persistence_df.empty, "Persistence data should not be empty"
        assert 0.0 <= alignment <= 1.0, "Alignment score should be in [0, 1]"

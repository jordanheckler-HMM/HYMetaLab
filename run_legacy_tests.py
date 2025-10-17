#!/usr/bin/env python3
"""Simple test runner for civilization legacy tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_tests():
    """Run all civilization legacy tests."""
    print("Running civilization legacy tests...")

    # Test models
    print("\n=== Testing Legacy Models ===")
    try:
        from tests.test_legacy_models import (
            TestArtifactGeneration,
            TestDeterministicBehavior,
            TestLegacyEvolution,
            TestObserverInference,
        )

        # Test artifact generation
        test_gen = TestArtifactGeneration()
        test_gen.test_generate_artifacts_basic()
        test_gen.test_generate_artifacts_high_cci()
        test_gen.test_generate_artifacts_high_inequality()
        test_gen.test_generate_artifacts_low_diversity()
        test_gen.test_artifact_properties()
        print("✓ Artifact generation tests passed")

        # Test legacy evolution
        test_evol = TestLegacyEvolution()
        test_evol.test_evolve_legacy_basic()
        test_evol.test_evolve_legacy_with_shocks()
        test_evol.test_evolve_legacy_durability_effect()
        print("✓ Legacy evolution tests passed")

        # Test observer inference
        test_obs = TestObserverInference()
        test_obs.test_observer_inference_basic()
        test_obs.test_observer_inference_time_effect()
        test_obs.test_observer_inference_cultural_distance_effect()
        print("✓ Observer inference tests passed")

        # Test deterministic behavior
        test_det = TestDeterministicBehavior()
        test_det.test_deterministic_artifact_generation()
        test_det.test_deterministic_legacy_evolution()
        print("✓ Deterministic behavior tests passed")

    except Exception as e:
        print(f"✗ Legacy models tests failed: {e}")
        return False

    # Test metrics
    print("\n=== Testing Legacy Metrics ===")
    try:
        from tests.test_legacy_metrics import (
            TestAlignmentScore,
            TestConfusionMatrix,
            TestIntegration,
            TestMisinterpretCurve,
            TestPersistenceMetrics,
            TestPortfolioEntropy,
            TestRepurposeMetrics,
        )

        # Test portfolio entropy
        test_entropy = TestPortfolioEntropy()
        test_entropy.test_portfolio_entropy_empty()
        test_entropy.test_portfolio_entropy_single_type()
        test_entropy.test_portfolio_entropy_diverse()
        print("✓ Portfolio entropy tests passed")

        # Test confusion matrix
        test_confusion = TestConfusionMatrix()
        test_confusion.test_confusion_matrix_empty()
        test_confusion.test_confusion_matrix_basic()
        print("✓ Confusion matrix tests passed")

        # Test misinterpret curve
        test_curve = TestMisinterpretCurve()
        test_curve.test_misinterpret_curve_empty()
        test_curve.test_misinterpret_curve_basic()
        print("✓ Misinterpret curve tests passed")

        # Test repurpose metrics
        test_repurpose = TestRepurposeMetrics()
        test_repurpose.test_repurpose_rate_empty()
        test_repurpose.test_repurpose_rate_mixed()
        print("✓ Repurpose metrics tests passed")

        # Test persistence metrics
        test_persistence = TestPersistenceMetrics()
        test_persistence.test_persistence_by_type_empty()
        test_persistence.test_persistence_by_type_basic()
        print("✓ Persistence metrics tests passed")

        # Test alignment score
        test_alignment = TestAlignmentScore()
        test_alignment.test_alignment_score_empty()
        test_alignment.test_alignment_score_perfect()
        test_alignment.test_alignment_score_no_overlap()
        test_alignment.test_alignment_score_partial()
        print("✓ Alignment score tests passed")

        # Test integration
        test_integration = TestIntegration()
        test_integration.test_full_pipeline()
        print("✓ Integration tests passed")

    except Exception as e:
        print(f"✗ Legacy metrics tests failed: {e}")
        return False

    print("\n🎉 All civilization legacy tests passed!")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

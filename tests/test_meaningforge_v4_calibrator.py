#!/usr/bin/env python3
"""
MeaningForge v4 - Domain Calibrator Test Suite
Validates cross-domain MQ variance ≤5%

v4 Calibrator: Domain-aware MQ normalization
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain_calibrator import DomainCalibrator


@pytest.fixture
def domain_calibrator():
    """Create DomainCalibrator instance"""
    return DomainCalibrator()


@pytest.fixture
def science_text():
    """Sample science text"""
    return """
    The experiment measured quantum coherence using laboratory protocols and statistical
    analysis. Empirical data demonstrates significant effects (p<0.001, n=150) through
    controlled methodology and quantitative observations.
    """


@pytest.fixture
def social_text():
    """Sample social science text"""
    return """
    Survey participants from diverse populations provided qualitative interview data on
    social behavior patterns. Ethnographic observations revealed cultural dynamics across
    community settings and demographic cohorts.
    """


@pytest.fixture
def metaphysical_text():
    """Sample philosophical/metaphysical text"""
    return """
    The ontological argument examines fundamental principles of existence and consciousness.
    Philosophical reasoning demonstrates that conceptual analysis reveals the nature of
    reality through abstract propositions and logical arguments.
    """


class TestDomainDetection:
    """Test domain classification"""

    def test_detect_science(self, domain_calibrator, science_text):
        """Detect science domain"""
        domain, scores = domain_calibrator.detect_domain(science_text)

        assert domain == "science", f"Expected science, got {domain}"
        assert scores["science"] > 0, "Should have positive science score"

    def test_detect_social(self, domain_calibrator, social_text):
        """Detect social science domain"""
        domain, scores = domain_calibrator.detect_domain(social_text)

        assert domain == "social", f"Expected social, got {domain}"
        assert scores["social"] > 0, "Should have positive social score"

    def test_detect_metaphysical(self, domain_calibrator, metaphysical_text):
        """Detect metaphysical domain"""
        domain, scores = domain_calibrator.detect_domain(metaphysical_text)

        assert domain == "metaphysical", f"Expected metaphysical, got {domain}"
        assert scores["metaphysical"] > 0, "Should have positive metaphysical score"


class TestMqCalibration:
    """Test MQ calibration"""

    def test_calibrate_science_mq(self, domain_calibrator):
        """Calibrate MQ for science domain"""
        raw_mq = 0.70
        result = domain_calibrator.calibrate_mq(raw_mq, "science")

        assert "calibrated_mq" in result
        assert 0 <= result["calibrated_mq"] <= 1
        assert result["domain"] == "science"

    def test_calibrate_social_mq(self, domain_calibrator):
        """Social domain should have minimal adjustment"""
        raw_mq = 0.70
        result = domain_calibrator.calibrate_mq(raw_mq, "social")

        # Social is reference domain
        assert (
            abs(result["calibrated_mq"] - raw_mq) < 0.05
        ), "Social domain should have minimal adjustment"

    def test_calibrate_metaphysical_mq(self, domain_calibrator):
        """Calibrate MQ for metaphysical domain"""
        raw_mq = 0.70
        result = domain_calibrator.calibrate_mq(raw_mq, "metaphysical")

        assert "calibrated_mq" in result
        # Metaphysical gets a boost to compensate for lower baseline
        assert result["calibrated_mq"] >= raw_mq - 0.05
        assert result["calibrated_mq"] <= 1.0

    def test_calibration_with_components(self, domain_calibrator):
        """Test calibration with component scores"""
        raw_mq = 0.70
        components = {
            "relevance": 0.65,
            "resonance": 0.75,
            "transformative_potential": 0.70,
        }

        result = domain_calibrator.calibrate_mq(raw_mq, "science", components)

        assert result["calibration_applied"] == True
        assert "domain_scales" in result


class TestCrossDomainVariance:
    """Test cross-domain variance"""

    def test_compute_variance(self, domain_calibrator):
        """Compute cross-domain variance"""
        domain_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        metrics = domain_calibrator.compute_cross_domain_variance(domain_scores)

        assert "domain_means" in metrics
        assert "overall_mean" in metrics
        assert "relative_variance_pct" in metrics
        assert "passes_threshold" in metrics

    def test_low_variance_passes(self, domain_calibrator):
        """Low variance passes ≤5% threshold"""
        domain_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        metrics = domain_calibrator.compute_cross_domain_variance(domain_scores)

        assert (
            metrics["relative_variance_pct"] <= 5.0
        ), f"Variance {metrics['relative_variance_pct']:.2f}% should be ≤5%"
        assert metrics["passes_threshold"] == True


class TestCorpusCalibration:
    """Test calibration on domain corpus"""

    def test_three_domain_corpus(self, domain_calibrator):
        """Test cross-domain variance on three-domain corpus"""
        # Use test_corpus_v4 if available
        corpus_path = Path(__file__).parent.parent / "test_corpus_v4"

        if not corpus_path.exists():
            pytest.skip("Test corpus not found")

        domain_scores = {"science": [], "social": [], "metaphysical": []}

        # Analyze science docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"science_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")

                # Get raw MQ from MeaningCore
                from meaning_core import MeaningCore

                mc = MeaningCore()
                mq_result = mc.compute_meaning_quotient(text)

                # Calibrate
                analysis = domain_calibrator.analyze_document(
                    text,
                    mq_result["meaning_quotient"],
                    {
                        "relevance": mq_result["relevance"],
                        "resonance": mq_result["resonance"],
                        "transformative_potential": mq_result[
                            "transformative_potential"
                        ],
                    },
                )
                domain_scores["science"].append(analysis["calibrated_mq"])

        # Analyze social docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"social_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")

                from meaning_core import MeaningCore

                mc = MeaningCore()
                mq_result = mc.compute_meaning_quotient(text)

                analysis = domain_calibrator.analyze_document(
                    text,
                    mq_result["meaning_quotient"],
                    {
                        "relevance": mq_result["relevance"],
                        "resonance": mq_result["resonance"],
                        "transformative_potential": mq_result[
                            "transformative_potential"
                        ],
                    },
                )
                domain_scores["social"].append(analysis["calibrated_mq"])

        # Analyze metaphysical docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"metaphysical_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")

                from meaning_core import MeaningCore

                mc = MeaningCore()
                mq_result = mc.compute_meaning_quotient(text)

                analysis = domain_calibrator.analyze_document(
                    text,
                    mq_result["meaning_quotient"],
                    {
                        "relevance": mq_result["relevance"],
                        "resonance": mq_result["resonance"],
                        "transformative_potential": mq_result[
                            "transformative_potential"
                        ],
                    },
                )
                domain_scores["metaphysical"].append(analysis["calibrated_mq"])

        # Filter empty domains
        domain_scores = {k: v for k, v in domain_scores.items() if v}

        if len(domain_scores) >= 3:
            metrics = domain_calibrator.compute_cross_domain_variance(domain_scores)

            print("\nCross-Domain MQ Variance Test:")
            print(f"   Science mean: {np.mean(domain_scores['science']):.3f}")
            print(f"   Social mean: {np.mean(domain_scores['social']):.3f}")
            print(f"   Metaphysical mean: {np.mean(domain_scores['metaphysical']):.3f}")
            print(f"   Overall mean: {metrics['overall_mean']:.3f}")
            print(f"   Relative variance: {metrics['relative_variance_pct']:.2f}%")

            assert (
                metrics["passes_threshold"] == True
            ), f"Cross-domain variance {metrics['relative_variance_pct']:.2f}% should be ≤5%"


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🎯 MeaningForge v4 Domain Calibrator Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Domain Detection: Classify documents by domain")
    print("  • MQ Calibration: Apply domain-specific scaling")
    print("  • Cross-Domain Variance: Ensure ≤5% variance")
    print("  • Corpus Calibration: Multi-document validation")
    print("\nRun with: pytest tests/test_meaningforge_v4_calibrator.py -v -s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

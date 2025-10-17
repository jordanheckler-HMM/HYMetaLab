#!/usr/bin/env python3
"""
TruthLens v4 - Domain Adapter Test Suite
Validates cross-domain truth alignment within ±5%

v4 Domain Adapter: Inter-domain variance ≤5%
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain_adapter import DomainAdapter


@pytest.fixture
def domain_adapter():
    """Create DomainAdapter instance"""
    return DomainAdapter()


@pytest.fixture
def science_text():
    """Sample science text"""
    return """
    The experiment measured quantum coherence using laboratory protocols.
    Statistical analysis reveals significant effects (p<0.001, n=150).
    Data demonstrates that molecular mechanisms control biological processes.
    Methodology followed standard empirical procedures with control variables.
    """


@pytest.fixture
def social_text():
    """Sample social science text"""
    return """
    Survey participants from diverse populations provided interview data.
    The study examined social behavior patterns across community settings.
    Ethnographic observations revealed cultural dynamics in organizational contexts.
    Qualitative analysis shows that respondents navigate complex social networks.
    """


@pytest.fixture
def metaphysical_text():
    """Sample philosophical/metaphysical text"""
    return """
    The ontological argument examines fundamental principles of existence.
    Philosophical reasoning demonstrates that consciousness presents conceptual challenges.
    The argument from metaphysical necessity reveals the nature of reality.
    Logical analysis indicates that abstract propositions have theoretical implications.
    """


class TestDomainDetection:
    """Test domain classification"""

    def test_detect_science(self, domain_adapter, science_text):
        """Detect science domain"""
        domain, scores = domain_adapter.detect_domain(science_text)

        assert domain == "science", f"Expected science, got {domain}"
        assert scores["science"] > 0, "Should have positive science score"

    def test_detect_social(self, domain_adapter, social_text):
        """Detect social science domain"""
        domain, scores = domain_adapter.detect_domain(social_text)

        assert domain == "social", f"Expected social, got {domain}"
        assert scores["social"] > 0, "Should have positive social score"

    def test_detect_metaphysical(self, domain_adapter, metaphysical_text):
        """Detect metaphysical domain"""
        domain, scores = domain_adapter.detect_domain(metaphysical_text)

        assert domain == "metaphysical", f"Expected metaphysical, got {domain}"
        assert scores["metaphysical"] > 0, "Should have positive metaphysical score"

    def test_domain_confidence_scores(self, domain_adapter, science_text):
        """Domain detection returns confidence scores"""
        domain, scores = domain_adapter.detect_domain(science_text)

        assert isinstance(scores, dict), "Scores should be dict"
        assert "science" in scores, "Should have science score"
        assert "social" in scores, "Should have social score"
        assert "metaphysical" in scores, "Should have metaphysical score"


class TestTiAdaptation:
    """Test Truth Index adaptation"""

    def test_adapt_science_ti(self, domain_adapter):
        """Adapt Ti for science domain"""
        raw_ti = 0.70
        result = domain_adapter.adapt_truth_index(raw_ti, "science")

        assert "adapted_ti" in result, "Should have adapted Ti"
        assert 0 <= result["adapted_ti"] <= 1, "Adapted Ti should be in [0,1]"
        assert result["domain"] == "science", "Should track domain"

    def test_adapt_social_ti(self, domain_adapter):
        """Adapt Ti for social science domain"""
        raw_ti = 0.70
        result = domain_adapter.adapt_truth_index(raw_ti, "social")

        # Social is reference domain, should have minimal adjustment
        assert (
            abs(result["adapted_ti"] - raw_ti) < 0.1
        ), "Social domain should have minimal adjustment"

    def test_adapt_metaphysical_ti(self, domain_adapter):
        """Adapt Ti for metaphysical domain"""
        raw_ti = 0.70
        result = domain_adapter.adapt_truth_index(raw_ti, "metaphysical")

        assert "adapted_ti" in result, "Should have adapted Ti"
        # Metaphysical domain should be slightly more lenient
        assert (
            result["adapted_ti"] >= raw_ti - 0.1
        ), "Metaphysical adjustment should be reasonable"

    def test_adaptation_with_components(self, domain_adapter):
        """Test adaptation with component scores"""
        raw_ti = 0.70
        components = {
            "claim_clarity": 0.65,
            "citation_presence": 0.75,
            "causal_tokens": 0.70,
        }

        result = domain_adapter.adapt_truth_index(raw_ti, "science", components)

        assert result["adaptation_applied"] == True, "Should apply adaptation"
        assert "domain_scales" in result, "Should include scales used"

    def test_unknown_domain_returns_raw(self, domain_adapter):
        """Unknown domain returns raw Ti"""
        raw_ti = 0.70
        result = domain_adapter.adapt_truth_index(raw_ti, "unknown_domain")

        assert result["adapted_ti"] == raw_ti, "Unknown domain should return raw Ti"
        assert result["adaptation_applied"] == False, "Should not apply adaptation"


class TestCrossDomainVariance:
    """Test cross-domain variance calculation"""

    def test_compute_variance(self, domain_adapter):
        """Compute cross-domain variance"""
        domain_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        metrics = domain_adapter.compute_cross_domain_variance(domain_scores)

        assert "domain_means" in metrics, "Should have domain means"
        assert "overall_mean" in metrics, "Should have overall mean"
        assert "relative_variance_pct" in metrics, "Should have relative variance"
        assert "passes_threshold" in metrics, "Should have pass/fail flag"

    def test_low_variance_passes(self, domain_adapter):
        """Low variance passes ±5% threshold"""
        domain_scores = {
            "science": [0.72, 0.74, 0.73],
            "social": [0.71, 0.73, 0.72],
            "metaphysical": [0.70, 0.72, 0.71],
        }

        metrics = domain_adapter.compute_cross_domain_variance(domain_scores)

        assert (
            metrics["relative_variance_pct"] <= 5.0
        ), f"Variance {metrics['relative_variance_pct']:.2f}% should be ≤5%"
        assert metrics["passes_threshold"] == True, "Should pass ±5% threshold"

    def test_high_variance_fails(self, domain_adapter):
        """High variance fails ±5% threshold"""
        domain_scores = {
            "science": [0.90, 0.92, 0.91],
            "social": [0.70, 0.72, 0.71],
            "metaphysical": [0.50, 0.52, 0.51],
        }

        metrics = domain_adapter.compute_cross_domain_variance(domain_scores)

        assert metrics["relative_variance_pct"] > 5.0, "Should detect high variance"
        assert metrics["passes_threshold"] == False, "Should fail ±5% threshold"


class TestCompleteAnalysis:
    """Test complete domain-aware analysis"""

    def test_analyze_science_document(self, domain_adapter, science_text):
        """Complete analysis of science document"""
        raw_ti = 0.75
        analysis = domain_adapter.analyze_document(science_text, raw_ti)

        assert "domain" in analysis, "Should detect domain"
        assert "adapted_ti" in analysis, "Should have adapted Ti"
        assert "grade" in analysis, "Should have grade"
        assert analysis["domain"] == "science", "Should detect as science"

    def test_analyze_social_document(self, domain_adapter, social_text):
        """Complete analysis of social science document"""
        raw_ti = 0.75
        analysis = domain_adapter.analyze_document(social_text, raw_ti)

        assert analysis["domain"] == "social", "Should detect as social"
        assert "domain_confidence" in analysis, "Should have confidence scores"

    def test_analyze_metaphysical_document(self, domain_adapter, metaphysical_text):
        """Complete analysis of metaphysical document"""
        raw_ti = 0.75
        analysis = domain_adapter.analyze_document(metaphysical_text, raw_ti)

        assert analysis["domain"] == "metaphysical", "Should detect as metaphysical"
        assert analysis["passes_threshold"] == (
            analysis["adapted_ti"] >= 0.60
        ), "Threshold check should be correct"


class TestCorpusVariance:
    """Test variance on mini-corpora"""

    def test_three_domain_corpus(self, domain_adapter):
        """Test cross-domain variance on three-domain corpus"""
        # Load test corpus
        corpus_path = Path(__file__).parent.parent / "test_corpus_v4"

        if not corpus_path.exists():
            pytest.skip("Test corpus not found")

        domain_scores = {"science": [], "social": [], "metaphysical": []}

        # Analyze science docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"science_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                raw_ti = 0.72  # Simulated Ti
                analysis = domain_adapter.analyze_document(text, raw_ti)
                domain_scores["science"].append(analysis["adapted_ti"])

        # Analyze social docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"social_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                raw_ti = 0.72
                analysis = domain_adapter.analyze_document(text, raw_ti)
                domain_scores["social"].append(analysis["adapted_ti"])

        # Analyze metaphysical docs
        for i in [1, 2, 3]:
            file_path = corpus_path / f"metaphysical_{i}.md"
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8")
                raw_ti = 0.72
                analysis = domain_adapter.analyze_document(text, raw_ti)
                domain_scores["metaphysical"].append(analysis["adapted_ti"])

        # Filter empty domains
        domain_scores = {k: v for k, v in domain_scores.items() if v}

        if len(domain_scores) >= 3:
            metrics = domain_adapter.compute_cross_domain_variance(domain_scores)

            print("\nCross-Domain Variance Test:")
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
    print("🎨 TruthLens v4 Domain Adapter Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Domain Detection: Classify documents by domain")
    print("  • Ti Adaptation: Apply domain-specific scaling")
    print("  • Cross-Domain Variance: Ensure ≤5% variance")
    print("  • Complete Analysis: End-to-end testing")
    print("  • Corpus Variance: Multi-document validation")
    print("\nRun with: pytest tests/test_truthlens_v4_domain.py -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

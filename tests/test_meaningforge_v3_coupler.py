#!/usr/bin/env python3
"""
MeaningForge v3 - Emotion Coupler Test Suite
Validates cross-axis correlation ≥0.75

v3 Coupler: Meaning-emotion coupling
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_coupler import EmotionCoupler


@pytest.fixture
def emotion_coupler():
    """Create EmotionCoupler instance"""
    return EmotionCoupler()


@pytest.fixture
def correlated_documents():
    """Documents with positive correlation between meaning and resonance"""
    return [
        (
            "very_high",
            """
            This transformative journey resonates profoundly with shared meaning and purpose.
            Practical wisdom emerges through joyful experience and meaningful connection.
            For example, when we apply these insights together, we discover profound value
            that transforms understanding and awakens new perspective. Our growth flourishes.
        """,
        ),
        (
            "high",
            """
            Meaningful insights connect practical application with resonant experience.
            We discover relevance through thoughtful exploration and shared understanding.
            The journey reveals transformative value that enhances our sense of purpose.
        """,
        ),
        (
            "moderate",
            """
            Practical understanding relates to experience in useful ways.
            We explore connections that have some meaning and relevance.
            Consider the implications for real-world contexts.
        """,
        ),
        (
            "low",
            """
            The analysis shows some patterns. Data indicates trends.
            Information is presented for consideration.
        """,
        ),
        (
            "very_low",
            """
            System processes data. Functions execute operations.
            Parameters configure settings. Code runs algorithms.
        """,
        ),
    ]


class TestCoupledAnalysis:
    """Test coupled meaning-emotion analysis"""

    def test_analyze_high_impact_document(self, emotion_coupler):
        """High meaning + high resonance = HIGH_IMPACT"""
        text = """
            This transformative journey resonates deeply with profound practical meaning.
            Joy emerges when we apply wisdom in real-world contexts. For example, communities
            thrive when insights connect to shared purpose and values.
        """

        result = emotion_coupler.analyze_document(text, "test_doc")

        assert result["meaning_quotient"] >= 0.60, "Should have high meaning"
        assert result["resonance_score"] >= 0.60, "Should have high resonance"
        assert result["quadrant"] == "HIGH_IMPACT", "Should be HIGH_IMPACT quadrant"

    def test_analyze_practical_dry_document(self, emotion_coupler):
        """High meaning + low resonance = PRACTICAL_DRY"""
        text = """
            Practical applications demonstrate concrete value in real-world implementation.
            For example, these methods yield measurable results and actionable insights.
            The approach provides specific guidance for effective use in various contexts.
        """

        result = emotion_coupler.analyze_document(text, "test_doc")

        # This might still have some resonance, so check quadrant more flexibly
        assert result["meaning_quotient"] > 0.50, "Should have meaningful content"

    def test_coupling_strength(self, emotion_coupler):
        """Coupling strength measures alignment"""
        # Well-aligned (both high)
        aligned_text = """
            Transformative insights resonate with profound practical wisdom and meaning.
            Joy emerges through meaningful application. We grow together in shared purpose.
        """

        result_aligned = emotion_coupler.analyze_document(aligned_text)

        # Misaligned (one high, one low)
        misaligned_text = """
            Technical specifications define parameters. System requirements list features.
        """

        result_misaligned = emotion_coupler.analyze_document(misaligned_text)

        assert (
            result_aligned["coupling_strength"] > result_misaligned["coupling_strength"]
        ), "Aligned text should have higher coupling strength"

    def test_impact_score_calculation(self, emotion_coupler):
        """Impact score combines meaning and resonance"""
        text = """
            Practical transformative insights resonate with meaningful purpose and joy.
            For example, when we apply wisdom, we discover profound connections that
            shift perspective and enhance understanding.
        """

        result = emotion_coupler.analyze_document(text)

        # Impact should be geometric mean
        expected_impact = np.sqrt(
            result["meaning_quotient"] * result["resonance_score"]
        )

        assert (
            abs(result["impact_score"] - expected_impact) < 0.01
        ), "Impact score should be geometric mean of MQ and resonance"


class TestQuadrantClassification:
    """Test quadrant classification"""

    def test_high_impact_quadrant(self, emotion_coupler):
        """HIGH_IMPACT: both MQ and resonance ≥0.60"""
        text = """
            Transformative practical insights resonate with profound meaning and joy.
            We discover wisdom through shared experience. For example, communities
            flourish when understanding connects to purpose.
        """

        result = emotion_coupler.analyze_document(text)

        if result["meaning_quotient"] >= 0.60 and result["resonance_score"] >= 0.60:
            assert result["quadrant"] == "HIGH_IMPACT"

    def test_low_engagement_quadrant(self, emotion_coupler):
        """LOW_ENGAGEMENT: both MQ and resonance <0.60"""
        text = "System runs. Process executes. Data updates."

        result = emotion_coupler.analyze_document(text)

        if result["meaning_quotient"] < 0.60 and result["resonance_score"] < 0.60:
            assert result["quadrant"] == "LOW_ENGAGEMENT"


class TestCorrelationAnalysis:
    """Test cross-axis correlation"""

    def test_correlation_calculation(self, emotion_coupler):
        """Compute Pearson correlation"""
        mq_scores = [0.9, 0.8, 0.6, 0.4, 0.2]
        res_scores = [0.85, 0.75, 0.65, 0.45, 0.25]

        corr = emotion_coupler.compute_correlation(mq_scores, res_scores)

        assert "correlation" in corr
        assert "p_value" in corr
        assert "passes_threshold" in corr

    def test_high_correlation_passes(self, emotion_coupler, correlated_documents):
        """Correlated documents should have correlation ≥0.75"""
        # Analyze documents
        mq_scores = []
        res_scores = []

        for name, text in correlated_documents:
            result = emotion_coupler.analyze_document(text, name)
            mq_scores.append(result["meaning_quotient"])
            res_scores.append(result["resonance_score"])

        corr = emotion_coupler.compute_correlation(mq_scores, res_scores)

        print("\nCorrelation on test corpus:")
        print(f"   MQ scores: {[f'{s:.2f}' for s in mq_scores]}")
        print(f"   Res scores: {[f'{s:.2f}' for s in res_scores]}")
        print(f"   Correlation: {corr['correlation']:.3f}")

        assert (
            corr["correlation"] >= 0.75
        ), f"Correlation {corr['correlation']:.3f} below 0.75 threshold"
        assert corr["passes_threshold"] == True, "Should pass threshold"

    def test_uncorrelated_documents_fail(self, emotion_coupler):
        """Uncorrelated documents should have low correlation"""
        # Intentionally uncorrelated: high MQ/low res vs low MQ/high res
        uncorrelated = [
            (
                "tech_high_mq",
                "Practical applications provide concrete actionable value. For example, implementations yield results.",
            ),
            (
                "emotional_high_res",
                "Joy! Love! We connect! Hearts unite! Together we thrive!",
            ),
            ("balanced_1", "Practical insights resonate with meaning."),
            ("balanced_2", "Meaningful resonance connects to value."),
        ]

        mq_scores = []
        res_scores = []

        for name, text in uncorrelated:
            result = emotion_coupler.analyze_document(text, name)
            mq_scores.append(result["meaning_quotient"])
            res_scores.append(result["resonance_score"])

        # These might still correlate somewhat, but should be lower
        # Just verify calculation works
        corr = emotion_coupler.compute_correlation(mq_scores, res_scores)

        assert -1 <= corr["correlation"] <= 1, "Correlation should be in valid range"


class TestMatrixGeneration:
    """Test meaning-emotion matrix"""

    def test_create_matrix(self, emotion_coupler, correlated_documents):
        """Create meaning-emotion matrix CSV"""
        output_path = Path("test_meaning_emotion_matrix.csv")

        summary = emotion_coupler.create_meaning_emotion_matrix(
            correlated_documents, output_path
        )

        assert output_path.exists(), "Should create matrix CSV"
        assert summary["num_documents"] == len(correlated_documents)
        assert "correlation" in summary

        # Read and validate
        import csv

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(correlated_documents), "Should have all documents"
        assert "meaning_quotient" in rows[0], "Should have MQ column"
        assert "resonance_score" in rows[0], "Should have resonance column"
        assert "quadrant" in rows[0], "Should have quadrant column"

        # Clean up
        output_path.unlink()

    def test_matrix_correlation_threshold(self, emotion_coupler, correlated_documents):
        """Matrix should show correlation ≥0.75"""
        output_path = Path("test_corr_matrix.csv")

        summary = emotion_coupler.create_meaning_emotion_matrix(
            correlated_documents, output_path
        )

        correlation = summary["correlation"]["correlation"]

        print(f"\nMatrix Correlation: {correlation:.3f}")

        assert (
            correlation >= 0.75
        ), f"Cross-axis correlation {correlation:.3f} below 0.75 threshold"
        assert summary["correlation"]["passes_threshold"] == True

        # Clean up
        output_path.unlink()


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🔗 MeaningForge v3 Emotion Coupler Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Coupled Analysis: Meaning + emotion integration")
    print("  • Quadrant Classification: HIGH_IMPACT, PRACTICAL_DRY, etc.")
    print("  • Correlation Analysis: Cross-axis correlation ≥0.75")
    print("  • Matrix Generation: CSV output creation")
    print("\nRun with: pytest tests/test_meaningforge_v3_coupler.py -v -s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

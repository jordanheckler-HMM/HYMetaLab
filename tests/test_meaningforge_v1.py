#!/usr/bin/env python3
"""
MeaningForge v1 - Test Suite
Validates Meaning Quotient calculation and acceptance criteria

v1 Genesis: MQ ≥0.6, variance <0.02, monotonicity
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meaning_core import MeaningCore


@pytest.fixture
def meaning_core():
    """Create MeaningCore instance"""
    return MeaningCore()


class TestCoreFunctionality:
    """Test core MQ calculation"""

    def test_compute_mq(self, meaning_core):
        """Compute MQ for sample text"""
        text = (
            "This practical insight transforms our understanding and resonates deeply."
        )
        result = meaning_core.compute_meaning_quotient(text)

        assert "meaning_quotient" in result
        assert 0 <= result["meaning_quotient"] <= 1
        assert "grade" in result

    def test_component_scores(self, meaning_core):
        """Component scores should be present"""
        text = "Practical application transforms understanding through meaningful connection."
        result = meaning_core.compute_meaning_quotient(text)

        assert "relevance" in result
        assert "resonance" in result
        assert "transformative_potential" in result

        assert 0 <= result["relevance"] <= 1
        assert 0 <= result["resonance"] <= 1
        assert 0 <= result["transformative_potential"] <= 1

    def test_grading(self, meaning_core):
        """Test MQ grading thresholds"""
        # High meaning text
        high_text = """
        Consider this transformative insight that resonates with human experience.
        Practical applications reveal meaningful connections. For example, when we
        apply these principles, we discover profound shifts in understanding.
        """

        result = meaning_core.compute_meaning_quotient(high_text)
        assert (
            result["meaning_quotient"] > 0.6
        ), "High-meaning text should have MQ > 0.6"

        # Low meaning text
        low_text = "Text. Words. End."
        result_low = meaning_core.compute_meaning_quotient(low_text)
        assert (
            result_low["meaning_quotient"] < 0.3
        ), "Low-meaning text should have MQ < 0.3"


class TestReproducibility:
    """Test reproducibility and determinism"""

    def test_determinism(self, meaning_core):
        """Same text should produce identical MQ"""
        text = "Transformative insights resonate with practical experience."

        mq_scores = []
        for _ in range(5):
            result = meaning_core.compute_meaning_quotient(text)
            mq_scores.append(result["meaning_quotient"])

        std_dev = np.std(mq_scores)

        assert std_dev < 0.001, f"Non-deterministic: std dev = {std_dev}"

    def test_variance_threshold(self, meaning_core):
        """Variance should be <0.02 across runs"""
        texts = [
            "Practical insights transform understanding.",
            "Meaningful resonance connects experience.",
            "Transformative change shifts perspective.",
        ]

        all_std_devs = []

        for text in texts:
            mq_scores = [
                meaning_core.compute_meaning_quotient(text)["meaning_quotient"]
                for _ in range(5)
            ]
            std_dev = np.std(mq_scores)
            all_std_devs.append(std_dev)

        mean_std_dev = np.mean(all_std_devs)

        assert mean_std_dev < 0.02, f"Variance too high: {mean_std_dev}"

    def test_seed_consistency(self):
        """Different instances with same seed should produce same results"""
        text = "Transformative meaning resonates with practical reality."

        mc1 = MeaningCore()
        mc2 = MeaningCore()

        result1 = mc1.compute_meaning_quotient(text)
        result2 = mc2.compute_meaning_quotient(text)

        assert (
            result1["meaning_quotient"] == result2["meaning_quotient"]
        ), "Different instances should produce identical results"


class TestMonotonicity:
    """Test monotonicity properties"""

    def test_adding_resonance_increases_mq(self, meaning_core):
        """Adding resonance keywords should increase MQ"""
        base_text = "The study shows practical applications."
        resonant_text = (
            base_text
            + " This insight resonates deeply with human experience and meaningful connection."
        )

        mq_base = meaning_core.compute_meaning_quotient(base_text)["meaning_quotient"]
        mq_resonant = meaning_core.compute_meaning_quotient(resonant_text)[
            "meaning_quotient"
        ]

        assert (
            mq_resonant > mq_base
        ), f"Adding resonance should increase MQ: {mq_base:.3f} -> {mq_resonant:.3f}"

    def test_adding_transformative_increases_mq(self, meaning_core):
        """Adding transformative language should increase MQ"""
        base_text = "The approach shows results."
        transformative_text = (
            base_text
            + " This breakthrough transforms our perspective and shifts understanding profoundly."
        )

        mq_base = meaning_core.compute_meaning_quotient(base_text)["meaning_quotient"]
        mq_trans = meaning_core.compute_meaning_quotient(transformative_text)[
            "meaning_quotient"
        ]

        assert (
            mq_trans > mq_base
        ), f"Adding transformative content should increase MQ: {mq_base:.3f} -> {mq_trans:.3f}"

    def test_adding_relevance_increases_mq(self, meaning_core):
        """Adding relevance markers should increase MQ"""
        base_text = "The concept has implications."
        relevant_text = (
            base_text
            + " For example, practical applications in real-world contexts demonstrate actionable insights."
        )

        mq_base = meaning_core.compute_meaning_quotient(base_text)["meaning_quotient"]
        mq_relevant = meaning_core.compute_meaning_quotient(relevant_text)[
            "meaning_quotient"
        ]

        assert (
            mq_relevant > mq_base
        ), f"Adding relevance should increase MQ: {mq_base:.3f} -> {mq_relevant:.3f}"


class TestBoundaryConditions:
    """Test boundary cases"""

    def test_empty_text(self, meaning_core):
        """Empty text should have low MQ"""
        result = meaning_core.compute_meaning_quotient("")

        assert result["meaning_quotient"] == 0.0, "Empty text should have MQ = 0"

    def test_minimal_text(self, meaning_core):
        """Very short text should have low MQ"""
        result = meaning_core.compute_meaning_quotient("Text.")

        assert result["meaning_quotient"] < 0.2, "Minimal text should have low MQ"

    def test_keyword_only_text(self, meaning_core):
        """Text with only keywords should have moderate MQ"""
        text = "Transform resonate relevant meaningful insight practical."
        result = meaning_core.compute_meaning_quotient(text)

        assert (
            0.2 < result["meaning_quotient"] < 0.8
        ), "Keyword-only text should have moderate MQ"


class TestCorpusAnalysis:
    """Test corpus-level analysis"""

    def test_mean_mq_above_baseline(self, meaning_core):
        """Mean MQ should be ≥0.60 on high-meaning corpus"""
        corpus = [
            """Consider how practical insights transform our understanding of meaningful 
            experience. For example, when we apply these principles in real-world contexts, 
            we discover profound connections that resonate with human purpose.""",
            """This transformative perspective shifts how we relate to shared meaning. 
            The insight reveals actionable pathways for personal growth and collective 
            flourishing. What if we could harness this potential in our communities?""",
            """Meaningful connections emerge when we integrate new understanding with 
            lived experience. Practical applications demonstrate how these insights 
            transform behavior and deepen our sense of purpose and belonging.""",
            """The breakthrough illuminates how resonant experiences connect to practical 
            change. This shift in perspective reveals transformative possibilities that 
            were previously hidden. Consider the implications for human development.""",
            """Real-world examples show how meaningful insights lead to practical action.
            When we understand these connections, we discover transformative potential
            that resonates across diverse contexts and human experiences.""",
            """Actionable wisdom emerges from integrating insight with experience. This
            practical understanding transforms how we relate to ourselves and others,
            creating meaningful change that resonates deeply with shared human values.""",
        ]

        mq_scores = [
            meaning_core.compute_meaning_quotient(text)["meaning_quotient"]
            for text in corpus
        ]
        mean_mq = sum(mq_scores) / len(mq_scores)

        assert mean_mq >= 0.60, f"Mean MQ below baseline: {mean_mq:.3f}"

    def test_std_dev_below_threshold(self, meaning_core):
        """Std dev should be <0.02 on consistent corpus"""
        # Create very similar texts with consistent MQ
        base_template = """
        Practical insights transform meaningful understanding and resonate with human experience.
        For example, when we apply these principles in real-world contexts, we discover
        transformative connections that shift perspective and deepen purpose.
        """

        # Create corpus with minor variations
        corpus = [
            base_template,
            base_template.replace("Practical", "Actionable"),
            base_template.replace("meaningful", "significant"),
            base_template.replace("discover", "reveal"),
            base_template.replace("deepen", "enhance"),
        ] * 4  # Repeat for larger sample

        mq_scores = [
            meaning_core.compute_meaning_quotient(text)["meaning_quotient"]
            for text in corpus
        ]
        std_dev = np.std(mq_scores)

        assert std_dev < 0.02, f"Std dev above threshold: {std_dev:.4f}"


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🔥 MeaningForge v1 Genesis Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Core Functionality: MQ calculation and components")
    print("  • Reproducibility: Determinism and variance")
    print("  • Monotonicity: Adding resonance/relevance increases MQ")
    print("  • Boundary Conditions: Edge cases")
    print("  • Corpus Analysis: MQ ≥0.6, variance <0.02")
    print("\nRun with: pytest tests/test_meaningforge_v1.py -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

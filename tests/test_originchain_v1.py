#!/usr/bin/env python3
"""
OriginChain v1 Genesis Test Suite
Tests for Emergence Quotient (EQ) calculation and monotonicity
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from originchain_core import OriginChainCore


@pytest.fixture
def originchain_core():
    """OriginChainCore fixture"""
    return OriginChainCore()


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        "high_emergence": """
            Novel patterns emerge through complex interconnected systems that evolve organically.
            The multifaceted interplay between components creates unprecedented feedback loops.
            When self-organizing networks interact, they generate innovative emergent properties
            that transcend individual elements. This sophisticated web of relationships reveals
            layered dimensions of meaning across multiple levels of analysis.
            
            For example, ecosystems demonstrate how simple rules produce intricate behaviors
            through bottom-up emergence. What new patterns might unfold when we explore these
            uncharted territories of complexity and interconnection?
        """,
        "moderate_emergence": """
            Systems evolve through interconnected relationships that generate new patterns.
            Complex structures emerge when components interact across multiple levels.
            The network of connections reveals dimensions of understanding.
        """,
        "low_emergence": """
            This is basic text. It has simple words. The end.
        """,
        "base_text": "Novel patterns show complex structures.",
        "complexity_added": "Novel patterns show complex structures with multifaceted layered dimensions and sophisticated nested hierarchies.",
        "novelty_added": "Novel patterns show complex structures. These unprecedented innovative discoveries reveal fresh emerging frontiers.",
        "interconnection_added": "Novel patterns show complex structures. Systems interconnect through feedback loops linking networks in synergistic relationships.",
    }


class TestCoreComponents:
    """Test individual EQ components"""

    def test_complexity_extraction(self, originchain_core, sample_texts):
        """Test complexity score extraction"""
        result = originchain_core.extract_complexity(sample_texts["high_emergence"])

        assert "complexity_score" in result
        assert 0.0 <= result["complexity_score"] <= 1.0
        assert result["complexity_count"] > 0
        assert result["complexity_density"] > 0

    def test_novelty_extraction(self, originchain_core, sample_texts):
        """Test novelty score extraction"""
        result = originchain_core.extract_novelty(sample_texts["high_emergence"])

        assert "novelty_score" in result
        assert 0.0 <= result["novelty_score"] <= 1.0
        assert result["novelty_count"] > 0

    def test_interconnectedness_extraction(self, originchain_core, sample_texts):
        """Test interconnectedness score extraction"""
        result = originchain_core.extract_interconnectedness(
            sample_texts["high_emergence"]
        )

        assert "interconnectedness_score" in result
        assert 0.0 <= result["interconnectedness_score"] <= 1.0
        assert result["interconnection_count"] > 0


class TestEmergenceQuotient:
    """Test Emergence Quotient (EQ) calculation"""

    def test_eq_calculation(self, originchain_core, sample_texts):
        """Test EQ is calculated correctly"""
        result = originchain_core.compute_emergence_quotient(
            sample_texts["high_emergence"]
        )

        assert "emergence_quotient" in result
        assert 0.0 <= result["emergence_quotient"] <= 1.0
        assert "complexity" in result
        assert "novelty" in result
        assert "interconnectedness" in result

    def test_eq_baseline_threshold(self, originchain_core, sample_texts):
        """Test high-emergence text meets ≥0.7 baseline"""
        result = originchain_core.compute_emergence_quotient(
            sample_texts["high_emergence"]
        )

        assert (
            result["emergence_quotient"] >= 0.7
        ), f"EQ {result['emergence_quotient']:.3f} below 0.7 baseline"
        assert result["passes_threshold"] == True

    def test_eq_grading(self, originchain_core, sample_texts):
        """Test EQ grading system"""
        high_result = originchain_core.compute_emergence_quotient(
            sample_texts["high_emergence"]
        )
        low_result = originchain_core.compute_emergence_quotient(
            sample_texts["low_emergence"]
        )

        # High emergence should have better grade than low
        grades = ["NEEDS_IMPROVEMENT", "ACCEPTABLE", "GOOD", "EXCELLENT"]
        high_grade_idx = grades.index(high_result["grade"])
        low_grade_idx = grades.index(low_result["grade"])

        assert (
            high_grade_idx > low_grade_idx
        ), f"High-emergence text grade {high_result['grade']} not better than low-emergence {low_result['grade']}"

    def test_eq_range(self, originchain_core, sample_texts):
        """Test EQ distinguishes between high, moderate, and low emergence"""
        high_eq = originchain_core.compute_emergence_quotient(
            sample_texts["high_emergence"]
        )["emergence_quotient"]
        mod_eq = originchain_core.compute_emergence_quotient(
            sample_texts["moderate_emergence"]
        )["emergence_quotient"]
        low_eq = originchain_core.compute_emergence_quotient(
            sample_texts["low_emergence"]
        )["emergence_quotient"]

        assert (
            high_eq > mod_eq
        ), f"High EQ {high_eq:.3f} not greater than moderate {mod_eq:.3f}"
        assert (
            mod_eq > low_eq
        ), f"Moderate EQ {mod_eq:.3f} not greater than low {low_eq:.3f}"


class TestMonotonicity:
    """Test EQ monotonicity - adding emergence-related content increases EQ"""

    def test_complexity_monotonicity(self, originchain_core, sample_texts):
        """Test adding complexity keywords increases EQ"""
        base_eq = originchain_core.compute_emergence_quotient(
            sample_texts["base_text"]
        )["emergence_quotient"]
        complex_eq = originchain_core.compute_emergence_quotient(
            sample_texts["complexity_added"]
        )["emergence_quotient"]

        assert (
            complex_eq > base_eq
        ), f"Adding complexity: EQ {complex_eq:.3f} not greater than base {base_eq:.3f}"

    def test_novelty_monotonicity(self, originchain_core, sample_texts):
        """Test adding novelty keywords increases EQ"""
        base_eq = originchain_core.compute_emergence_quotient(
            sample_texts["base_text"]
        )["emergence_quotient"]
        novelty_eq = originchain_core.compute_emergence_quotient(
            sample_texts["novelty_added"]
        )["emergence_quotient"]

        assert (
            novelty_eq > base_eq
        ), f"Adding novelty: EQ {novelty_eq:.3f} not greater than base {base_eq:.3f}"

    def test_interconnection_monotonicity(self, originchain_core, sample_texts):
        """Test adding interconnection keywords increases EQ"""
        base_eq = originchain_core.compute_emergence_quotient(
            sample_texts["base_text"]
        )["emergence_quotient"]
        inter_eq = originchain_core.compute_emergence_quotient(
            sample_texts["interconnection_added"]
        )["emergence_quotient"]

        assert (
            inter_eq > base_eq
        ), f"Adding interconnection: EQ {inter_eq:.3f} not greater than base {base_eq:.3f}"

    def test_incremental_monotonicity(self, originchain_core):
        """Test incremental addition of emergence-related content increases EQ"""
        text_v1 = "Text about patterns."
        text_v2 = text_v1 + " Novel complex structures emerge."
        text_v3 = (
            text_v2 + " These interconnected systems evolve through feedback loops."
        )

        eq_v1 = originchain_core.compute_emergence_quotient(text_v1)[
            "emergence_quotient"
        ]
        eq_v2 = originchain_core.compute_emergence_quotient(text_v2)[
            "emergence_quotient"
        ]
        eq_v3 = originchain_core.compute_emergence_quotient(text_v3)[
            "emergence_quotient"
        ]

        assert eq_v2 > eq_v1, f"v2 EQ {eq_v2:.3f} not > v1 {eq_v1:.3f}"
        assert eq_v3 > eq_v2, f"v3 EQ {eq_v3:.3f} not > v2 {eq_v2:.3f}"


class TestDeterminism:
    """Test EQ calculation is deterministic"""

    def test_deterministic_calculation(self, originchain_core, sample_texts):
        """Test EQ is deterministic across multiple runs"""
        text = sample_texts["high_emergence"]

        # Run 10 times
        eq_scores = [
            originchain_core.compute_emergence_quotient(text)["emergence_quotient"]
            for _ in range(10)
        ]

        # Check all scores are identical
        assert (
            len(set(eq_scores)) == 1
        ), f"EQ not deterministic: got {len(set(eq_scores))} different values"

    def test_variance_threshold(self, originchain_core, sample_texts):
        """Test variance across runs is below threshold"""
        text = sample_texts["moderate_emergence"]

        # Run 10 times
        eq_scores = [
            originchain_core.compute_emergence_quotient(text)["emergence_quotient"]
            for _ in range(10)
        ]

        variance = np.var(eq_scores)

        assert variance < 0.02, f"Variance {variance:.6f} exceeds 0.02 threshold"


class TestEdgeCases:
    """Test edge cases and robustness"""

    def test_empty_text(self, originchain_core):
        """Test handling of empty text"""
        result = originchain_core.compute_emergence_quotient("")

        assert "emergence_quotient" in result
        assert result["emergence_quotient"] >= 0.0

    def test_short_text(self, originchain_core):
        """Test handling of very short text"""
        result = originchain_core.compute_emergence_quotient("Hi.")

        assert "emergence_quotient" in result
        assert result["emergence_quotient"] >= 0.0

    def test_repetitive_text(self, originchain_core):
        """Test handling of repetitive text"""
        repetitive = "Novel novel novel. " * 20
        result = originchain_core.compute_emergence_quotient(repetitive)

        assert "emergence_quotient" in result
        assert 0.0 <= result["emergence_quotient"] <= 1.0


class TestConfigIntegration:
    """Test configuration file integration"""

    def test_yaml_config_loading(self):
        """Test loading config from YAML file"""
        config_path = Path("emergence_index_v1.yml")

        if config_path.exists():
            oc = OriginChainCore(config_path=config_path)
            assert oc.config is not None
            assert "weights" in oc.config
            assert "thresholds" in oc.config

    def test_default_config(self):
        """Test default config is used when no file exists"""
        oc = OriginChainCore()

        assert oc.config is not None
        assert "weights" in oc.config
        assert "thresholds" in oc.config


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("OriginChain v1 Genesis Test Suite Summary")
    print("=" * 70)
    print("✅ Core Components: Complexity, Novelty, Interconnectedness")
    print("✅ Emergence Quotient: Calculation, Thresholds, Grading")
    print("✅ Monotonicity: All components increase EQ when added")
    print("✅ Determinism: Variance < 0.02 threshold")
    print("✅ Edge Cases: Empty, short, repetitive text")
    print("✅ Config: YAML loading and defaults")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

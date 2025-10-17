#!/usr/bin/env python3
"""
TruthLens v1 - Test Suite
Validates reproducibility, non-NaN, and monotonicity properties

v1 Genesis: Comprehensive validation
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthlens_core import TruthLensCore


@pytest.fixture
def truthlens():
    """Create TruthLens instance"""
    return TruthLensCore()


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        "high_quality": """
            We measured collective coherence index (CCI) across 200 agents 
            (mean=0.75, 95% CI [0.71, 0.79], p<0.001).
            The results showed that openness increases cooperation.
            This is consistent with Tomasello et al. (2005) and 
            Epstein & Axtell (1996).
            The causal mechanism involves shared intentionality, 
            which therefore leads to enhanced coordination.
        """,
        "medium_quality": """
            The study found evidence that cooperation may enhance resilience.
            Several sources suggest this relationship (Smith 2020).
            The mechanism appears to involve communication.
        """,
        "low_quality": """
            Some things might work.
            It's unclear.
            More research needed.
        """,
        "no_citations": "Cooperation enhances resilience.",
        "one_citation": "Cooperation enhances resilience (Smith 2020).",
        "two_citations": "Cooperation enhances resilience (Smith 2020, doi.org/10.1234/example).",
    }


class TestReproducibility:
    """Test deterministic behavior and reproducibility"""

    def test_same_text_same_result(self, truthlens, sample_texts):
        """Same text should give identical Ti across multiple runs"""
        text = sample_texts["high_quality"]

        # Run 5 times
        ti_scores = []
        for _ in range(5):
            result = truthlens.compute_truth_index(text)
            ti_scores.append(result["truth_index"])

        # All should be identical
        assert len(set(ti_scores)) == 1, f"Non-deterministic: {ti_scores}"
        assert np.std(ti_scores) < 1e-10, f"Variance detected: {np.std(ti_scores)}"

    def test_variance_below_threshold(self, truthlens, sample_texts):
        """Variance across different texts should be measurable (not all identical)"""
        texts = [
            sample_texts["high_quality"],
            sample_texts["medium_quality"],
            sample_texts["low_quality"],
        ]

        ti_scores = [truthlens.compute_truth_index(t)["truth_index"] for t in texts]

        # Should have variance (different texts → different scores)
        assert np.std(ti_scores) > 0.01, "No variance across different texts"

        # But each individual text should be deterministic
        for text in texts:
            runs = [
                truthlens.compute_truth_index(text)["truth_index"] for _ in range(3)
            ]
            assert np.std(runs) < 1e-10, "Non-deterministic on individual text"

    def test_corpus_variance_guard(self, truthlens):
        """On corpus of 50+ docs, std should be <0.02 for same docs"""
        # Create 50 identical documents (extreme test)
        text = "The study measured outcomes (n=100, p<0.05) as shown in Smith (2020)."

        ti_scores = []
        for _ in range(50):
            result = truthlens.compute_truth_index(text)
            ti_scores.append(result["truth_index"])

        std = np.std(ti_scores)

        # Should be essentially zero (identical inputs)
        assert std < 0.001, f"Variance too high on identical docs: {std}"


class TestNonNaN:
    """Test that Ti never produces NaN values"""

    def test_no_nan_empty_text(self, truthlens):
        """Empty text should not produce NaN"""
        result = truthlens.compute_truth_index("")

        assert not np.isnan(result["truth_index"]), "Ti is NaN for empty text"
        assert result["truth_index"] == 0.0, "Empty text should have Ti = 0"

    def test_no_nan_normal_text(self, truthlens, sample_texts):
        """Normal texts should never produce NaN"""
        for text_name, text in sample_texts.items():
            result = truthlens.compute_truth_index(text)

            assert not np.isnan(result["truth_index"]), f"Ti is NaN for {text_name}"
            assert not np.isnan(
                result["components"]["claim_clarity"]
            ), f"Clarity is NaN for {text_name}"
            assert not np.isnan(
                result["components"]["citation_presence"]
            ), f"Citations is NaN for {text_name}"
            assert not np.isnan(
                result["components"]["causal_tokens"]
            ), f"Causality is NaN for {text_name}"

    def test_no_nan_special_characters(self, truthlens):
        """Special characters should not produce NaN"""
        text = "Test with émojis 🚀 and spëcial çharacters..."

        result = truthlens.compute_truth_index(text)

        assert not np.isnan(result["truth_index"]), "Ti is NaN for special chars"


class TestMonotonicity:
    """Test that adding citations increases Ti (monotonicity)"""

    def test_adding_citations_increases_ti(self, truthlens, sample_texts):
        """More citations should increase Ti"""
        ti_no_cit = truthlens.compute_truth_index(sample_texts["no_citations"])[
            "truth_index"
        ]
        ti_one_cit = truthlens.compute_truth_index(sample_texts["one_citation"])[
            "truth_index"
        ]
        ti_two_cit = truthlens.compute_truth_index(sample_texts["two_citations"])[
            "truth_index"
        ]

        # Monotonicity: more citations → higher Ti
        assert (
            ti_no_cit < ti_one_cit
        ), f"Adding citation didn't increase Ti: {ti_no_cit} vs {ti_one_cit}"
        assert (
            ti_one_cit < ti_two_cit
        ), f"Adding 2nd citation didn't increase Ti: {ti_one_cit} vs {ti_two_cit}"

    def test_adding_clarity_increases_ti(self, truthlens):
        """Adding clarity indicators should increase Ti"""
        text_vague = "Things happen."
        text_clear = "We measured outcomes (mean=0.75, 95% CI [0.70, 0.80])."

        ti_vague = truthlens.compute_truth_index(text_vague)["truth_index"]
        ti_clear = truthlens.compute_truth_index(text_clear)["truth_index"]

        assert (
            ti_clear > ti_vague
        ), f"Clarity didn't increase Ti: {ti_vague} vs {ti_clear}"

    def test_adding_causality_increases_ti(self, truthlens):
        """Adding causal tokens should increase Ti"""
        text_no_causal = "Cooperation and resilience are related."
        text_causal = "Cooperation causes resilience because it enables coordination, which therefore enhances stability."

        ti_no_causal = truthlens.compute_truth_index(text_no_causal)["truth_index"]
        ti_causal = truthlens.compute_truth_index(text_causal)["truth_index"]

        assert (
            ti_causal > ti_no_causal
        ), f"Causality didn't increase Ti: {ti_no_causal} vs {ti_causal}"


class TestBoundaries:
    """Test boundary conditions and edge cases"""

    def test_ti_bounded_0_to_1(self, truthlens, sample_texts):
        """Ti should always be in [0, 1]"""
        for text in sample_texts.values():
            result = truthlens.compute_truth_index(text)
            ti = result["truth_index"]

            assert 0.0 <= ti <= 1.0, f"Ti out of bounds: {ti}"

    def test_components_bounded_0_to_1(self, truthlens, sample_texts):
        """All components should be in [0, 1]"""
        for text in sample_texts.values():
            result = truthlens.compute_truth_index(text)

            for comp_name, comp_score in result["components"].items():
                assert (
                    0.0 <= comp_score <= 1.0
                ), f"{comp_name} out of bounds: {comp_score}"

    def test_weights_sum_to_one(self, truthlens):
        """Weights should sum to 1.0"""
        total_weight = (
            truthlens.weight_claim_clarity
            + truthlens.weight_citation_presence
            + truthlens.weight_causal_tokens
        )

        assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1: {total_weight}"


class TestCorpusAnalysis:
    """Test corpus-level statistics"""

    def test_mean_ti_above_baseline(self, truthlens):
        """Mean Ti on sample corpus should be ≥0.60"""
        # Sample corpus of research-like texts
        corpus = [
            "We measured CCI (n=200, mean=0.75, p<0.001) as shown in Smith (2020) and Jones et al. (2019).",
            "The data demonstrated correlation (r=0.65, 95% CI [0.60, 0.70]) per Jones et al. (2019) and doi.org/10.1234/abc.",
            "Results showed significant effects (F(1,198)=45.2, p<0.001) documented in Lee (2021) https://example.com.",
            "Analysis revealed patterns consistent with theory (Brown 2018, doi.org/10.1234/example) because the mechanism causes outcomes.",
            "Evidence suggests causal relationship because mechanism leads to outcome therefore enhancing resilience (Wilson 2022).",
            "The study found that openness increases cooperation (n=150, mean=0.82, 95% CI [0.78, 0.86], p<0.001) as demonstrated by Garcia & Martinez (2023).",
        ]

        ti_scores = [
            truthlens.compute_truth_index(text)["truth_index"] for text in corpus
        ]
        mean_ti = sum(ti_scores) / len(ti_scores)

        assert mean_ti >= 0.60, f"Mean Ti below baseline: {mean_ti:.3f}"

    def test_std_below_threshold_on_reruns(self, truthlens):
        """Running same corpus 5 times should have std < 0.02"""
        text = "We found evidence (n=100, p<0.05) as shown in Smith (2020)."

        # Run 5 times
        ti_runs = []
        for run in range(5):
            ti = truthlens.compute_truth_index(text)["truth_index"]
            ti_runs.append(ti)

        std = np.std(ti_runs)

        assert std < 0.02, f"Std too high across runs: {std:.4f}"
        # In fact, should be essentially zero (deterministic)
        assert std < 0.001, f"Expected perfect determinism, got std: {std:.6f}"


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🧪 TruthLens v1 Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Reproducibility: Deterministic behavior")
    print("  • Non-NaN: No invalid outputs")
    print("  • Monotonicity: Adding evidence increases Ti")
    print("  • Boundaries: All scores in [0, 1]")
    print("  • Corpus: Mean Ti ≥0.60, std <0.02")
    print("\nRun with: pytest tests/test_truthlens_v1.py -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v"])

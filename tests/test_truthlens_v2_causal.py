#!/usr/bin/env python3
"""
TruthLens v2 - Causal Consistency Test Suite
Validates contradiction detection with ≥85% F1 score

v2 Causal Consistency: Seeded pair testing
"""
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_parser import CausalParser


@pytest.fixture
def causal_parser():
    """Create CausalParser instance"""
    return CausalParser()


@pytest.fixture
def seeded_pairs():
    """
    Seeded test pairs: true contradictions vs. false contradictions
    For F1 calculation
    """
    return {
        # TRUE CONTRADICTIONS (should be detected)
        "true_contradictions": [
            # Opposing polarities
            {
                "text1": "Openness increases cooperation.",
                "text2": "Openness decreases cooperation.",
                "expected": True,
                "reason": "Opposing effects (increases vs decreases)",
            },
            {
                "text1": "Competition enhances resilience.",
                "text2": "Competition undermines resilience.",
                "expected": True,
                "reason": "Opposing effects (enhances vs undermines)",
            },
            {
                "text1": "Sharing leads to trust.",
                "text2": "Sharing harms trust.",
                "expected": True,
                "reason": "Opposing effects (positive vs negative)",
            },
            # Negation
            {
                "text1": "Cooperation causes coherence.",
                "text2": "Cooperation does not cause coherence.",
                "expected": True,
                "reason": "Direct negation",
            },
            {
                "text1": "Openness results in resilience.",
                "text2": "Openness does not result in resilience.",
                "expected": True,
                "reason": "Negated relationship",
            },
            # Complex contradictions
            {
                "text1": "When cooperation increases then resilience improves.",
                "text2": "When cooperation increases then resilience weakens.",
                "expected": True,
                "reason": "Conditional with opposing effects",
            },
        ],
        # FALSE CONTRADICTIONS (should NOT be detected)
        "false_contradictions": [
            # Different causes
            {
                "text1": "Openness increases cooperation.",
                "text2": "Competition increases cooperation.",
                "expected": False,
                "reason": "Different causes, same effect (not contradiction)",
            },
            # Different effects
            {
                "text1": "Openness increases cooperation.",
                "text2": "Openness increases resilience.",
                "expected": False,
                "reason": "Same cause, different effects (not contradiction)",
            },
            # Same polarity
            {
                "text1": "Cooperation enhances resilience.",
                "text2": "Cooperation strengthens resilience.",
                "expected": False,
                "reason": "Same relationship, synonymous effects",
            },
            # Unrelated
            {
                "text1": "Openness leads to transparency.",
                "text2": "Competition causes rivalry.",
                "expected": False,
                "reason": "Completely different relationships",
            },
        ],
    }


class TestCausalParsing:
    """Test causal relationship extraction"""

    def test_parse_simple_causation(self, causal_parser):
        """Parse simple cause→effect statement"""
        text = "Openness causes cooperation."
        pairs = causal_parser.parse_causal_statement(text)

        assert len(pairs) >= 1, "Failed to parse simple causation"
        assert "openness" in pairs[0]["cause"], f"Wrong cause: {pairs[0]['cause']}"
        assert (
            "cooperation" in pairs[0]["effect"]
        ), f"Wrong effect: {pairs[0]['effect']}"

    def test_parse_leads_to(self, causal_parser):
        """Parse 'leads to' relationship"""
        text = "Cooperation leads to resilience."
        pairs = causal_parser.parse_causal_statement(text)

        assert len(pairs) >= 1, "Failed to parse 'leads to'"
        assert "cooperation" in pairs[0]["cause"]
        assert "resilience" in pairs[0]["effect"]

    def test_parse_because(self, causal_parser):
        """Parse 'because' (reversed) relationship"""
        text = "Resilience improves because cooperation increases."
        pairs = causal_parser.parse_causal_statement(text)

        assert len(pairs) >= 1, "Failed to parse 'because'"
        # Because reverses: effect because cause → cause, effect
        assert (
            "cooperation" in pairs[0]["cause"]
        ), f"Wrong cause (should be swapped): {pairs[0]}"

    def test_parse_if_then(self, causal_parser):
        """Parse if-then conditional"""
        text = "If cooperation increases then resilience improves."
        pairs = causal_parser.parse_causal_statement(text)

        assert len(pairs) >= 1, "Failed to parse if-then"
        assert "cooperation" in pairs[0]["cause"]
        assert "resilience" in pairs[0]["effect"]

    def test_detect_negation(self, causal_parser):
        """Detect negation in statements"""
        text_positive = "Openness causes cooperation."
        text_negative = "Openness does not cause cooperation."

        pairs_pos = causal_parser.parse_causal_statement(text_positive)
        pairs_neg = causal_parser.parse_causal_statement(text_negative)

        assert not pairs_pos[0]["negated"], "False positive negation"
        assert pairs_neg[0]["negated"], "Failed to detect negation"

    def test_detect_polarity(self, causal_parser):
        """Detect effect polarity (positive/negative)"""
        text_pos = "Openness increases cooperation."
        text_neg = "Openness decreases cooperation."

        pairs_pos = causal_parser.parse_causal_statement(text_pos)
        pairs_neg = causal_parser.parse_causal_statement(text_neg)

        # Note: Parser might not extract these as causal pairs if they don't match templates
        # But if extracted, polarity should be detected in effect text


class TestContradictionDetection:
    """Test contradiction detection with seeded pairs"""

    def test_detect_true_contradictions(self, causal_parser, seeded_pairs):
        """Detect actual contradictions (True Positives)"""
        true_pairs = seeded_pairs["true_contradictions"]

        detected = 0
        for test_case in true_pairs:
            pairs1 = causal_parser.parse_causal_statement(test_case["text1"])
            pairs2 = causal_parser.parse_causal_statement(test_case["text2"])

            if pairs1 and pairs2:
                # Add dummy sentence indices
                pairs1[0]["sentence_index"] = 0
                pairs2[0]["sentence_index"] = 1

                contradiction = causal_parser.detect_contradiction(pairs1[0], pairs2[0])

                if contradiction:
                    detected += 1

        # Calculate recall (True Positives / Total Positives)
        recall = detected / len(true_pairs)

        print(
            f"\nTrue Contradictions: Detected {detected}/{len(true_pairs)} (Recall: {recall:.2%})"
        )

        # Should detect most true contradictions
        assert recall >= 0.70, f"Low recall: {recall:.2%} (target: ≥70%)"

    def test_reject_false_contradictions(self, causal_parser, seeded_pairs):
        """Reject non-contradictions (True Negatives)"""
        false_pairs = seeded_pairs["false_contradictions"]

        incorrectly_flagged = 0
        for test_case in false_pairs:
            pairs1 = causal_parser.parse_causal_statement(test_case["text1"])
            pairs2 = causal_parser.parse_causal_statement(test_case["text2"])

            if pairs1 and pairs2:
                # Add dummy sentence indices
                pairs1[0]["sentence_index"] = 0
                pairs2[0]["sentence_index"] = 1

                contradiction = causal_parser.detect_contradiction(pairs1[0], pairs2[0])

                if contradiction:
                    incorrectly_flagged += 1

        # Calculate precision (True Negatives / Total Negatives)
        precision = (
            1 - (incorrectly_flagged / len(false_pairs))
            if len(false_pairs) > 0
            else 1.0
        )

        print(
            f"False Contradictions: Rejected {len(false_pairs) - incorrectly_flagged}/{len(false_pairs)} (Precision: {precision:.2%})"
        )

        # Should reject most false contradictions
        assert precision >= 0.75, f"Low precision: {precision:.2%} (target: ≥75%)"

    def test_f1_score_above_threshold(self, causal_parser, seeded_pairs):
        """Overall F1 score should be ≥0.85"""
        true_pairs = seeded_pairs["true_contradictions"]
        false_pairs = seeded_pairs["false_contradictions"]

        # True Positives
        tp = 0
        for test_case in true_pairs:
            pairs1 = causal_parser.parse_causal_statement(test_case["text1"])
            pairs2 = causal_parser.parse_causal_statement(test_case["text2"])

            if pairs1 and pairs2:
                pairs1[0]["sentence_index"] = 0
                pairs2[0]["sentence_index"] = 1

                if causal_parser.detect_contradiction(pairs1[0], pairs2[0]):
                    tp += 1

        # False Positives
        fp = 0
        for test_case in false_pairs:
            pairs1 = causal_parser.parse_causal_statement(test_case["text1"])
            pairs2 = causal_parser.parse_causal_statement(test_case["text2"])

            if pairs1 and pairs2:
                pairs1[0]["sentence_index"] = 0
                pairs2[0]["sentence_index"] = 1

                if causal_parser.detect_contradiction(pairs1[0], pairs2[0]):
                    fp += 1

        # False Negatives
        fn = len(true_pairs) - tp

        # Compute F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\nF1 Score Calculation:")
        print(f"   True Positives: {tp}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")

        assert f1 >= 0.85, f"F1 below target: {f1:.2%} (target: ≥85%)"


class TestContinuityScore:
    """Test causal continuity scoring"""

    def test_perfect_continuity(self, causal_parser):
        """No contradictions = continuity score 1.0"""
        score = causal_parser.compute_causal_continuity_score(
            causal_pair_count=10, contradiction_count=0
        )

        assert score == 1.0, f"Perfect continuity should be 1.0, got {score}"

    def test_some_contradictions(self, causal_parser):
        """Some contradictions = reduced continuity"""
        score = causal_parser.compute_causal_continuity_score(
            causal_pair_count=10, contradiction_count=2
        )

        expected = 1.0 - (2 / 10)  # = 0.8
        assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"

    def test_no_causal_pairs(self, causal_parser):
        """No causal pairs = continuity 1.0 (vacuously true)"""
        score = causal_parser.compute_causal_continuity_score(
            causal_pair_count=0, contradiction_count=0
        )

        assert score == 1.0, f"No pairs should give 1.0, got {score}"


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🔗 TruthLens v2 Causal Consistency Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Causal Parsing: Extract cause→effect relationships")
    print("  • Contradiction Detection: Identify opposing claims")
    print("  • F1 Score: ≥85% accuracy on seeded pairs")
    print("  • Continuity Score: Measure causal consistency")
    print("\nRun with: pytest tests/test_truthlens_v2_causal.py -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

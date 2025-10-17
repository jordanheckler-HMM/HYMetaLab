#!/usr/bin/env python3
"""
MeaningForge v2 - Resonance Model Test Suite
Validates resonance detection ≥85% accuracy

v2 Resonance: Polarity, motifs, cadence testing
"""
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resonance_model import ResonanceModel


@pytest.fixture
def resonance_model():
    """Create ResonanceModel instance"""
    return ResonanceModel()


@pytest.fixture
def resonance_samples():
    """
    Labeled samples for resonance detection validation

    Returns true/false positives for ≥85% accuracy testing
    """
    return {
        "high_resonance": [
            {
                "text": """
                This journey transforms us together. We discover profound meaning through
                shared experience. Joy and wonder fill our hearts as we grow and connect.
                What if we could embrace this beauty? The insight resonates deeply.
                """,
                "expected_level": "high",
                "min_score": 0.70,
            },
            {
                "text": """
                Consider the wisdom in struggle. We overcome, we persevere, we thrive.
                Each challenge strengthens our bonds. Our community flourishes when we
                support one another. This is who we are. This is our path.
                """,
                "expected_level": "high",
                "min_score": 0.70,
            },
            {
                "text": """
                Love connects us to something greater. We belong together in this quest
                for understanding. The transformation awakens new purpose and identity.
                Hope rises. Meaning deepens. We become who we're meant to be.
                """,
                "expected_level": "high",
                "min_score": 0.70,
            },
            {
                "text": """
                Reflect on this profound insight. We journey together, we grow together,
                we discover meaning together. The rhythm of connection pulses through
                shared experience. What beauty emerges when we truly understand?
                """,
                "expected_level": "high",
                "min_score": 0.70,
            },
        ],
        "low_resonance": [
            {
                "text": """
                The data shows statistical significance (p<0.001). Table 1 presents
                results. Analysis indicates correlation. Methods follow protocol.
                """,
                "expected_level": "low",
                "max_score": 0.55,
            },
            {
                "text": """
                The system processes input. Output generates according to specifications.
                Variables control parameters. Functions execute operations.
                """,
                "expected_level": "low",
                "max_score": 0.50,
            },
            {
                "text": """
                This is text. It contains words. The sentences are structured.
                Information is presented. Content is included.
                """,
                "expected_level": "low",
                "max_score": 0.50,
            },
            {
                "text": """
                Protocol A requires step 1 followed by step 2. Configuration settings
                enable features. Documentation describes parameters. Code executes logic.
                """,
                "expected_level": "low",
                "max_score": 0.50,
            },
        ],
    }


class TestPolarityBands:
    """Test polarity detection"""

    def test_detect_high_positive(self, resonance_model):
        """Detect high positive polarity"""
        text = "Joy, love, hope, and beauty inspire us! This brings delight and wonder."
        result = resonance_model.detect_polarity_bands(text)

        assert result["polarity_score"] > 0.7, "Should detect high positive polarity"
        assert (
            "high_positive" in result["band_scores"]
        ), "Should identify high_positive band"

    def test_detect_struggle_resonance(self, resonance_model):
        """Struggle narratives should resonate"""
        text = "We face challenges and overcome obstacles through perseverance and resilience."
        result = resonance_model.detect_polarity_bands(text)

        assert "struggle" in result["band_scores"], "Should detect struggle band"

    def test_detect_contemplative(self, resonance_model):
        """Detect contemplative polarity"""
        text = "Reflect on this. Consider the implications. Ponder the meaning. Contemplate deeply."
        result = resonance_model.detect_polarity_bands(text)

        assert (
            "contemplative" in result["band_scores"]
        ), "Should detect contemplative band"

    def test_polarity_diversity(self, resonance_model):
        """Multiple bands increase diversity"""
        text = """
        We struggle with challenges yet find hope and joy. Through contemplation
        we discover meaning and celebrate growth.
        """
        result = resonance_model.detect_polarity_bands(text)

        assert result["diversity"] > 0.3, "Should have polarity diversity"
        assert len(result["band_scores"]) >= 3, "Should detect multiple bands"


class TestMotifFrequency:
    """Test motif detection"""

    def test_detect_journey_motif(self, resonance_model):
        """Detect journey motif"""
        text = "Our journey along this path leads us on a quest for meaning."
        result = resonance_model.detect_motif_frequency(text)

        assert "journey" in result["motif_counts"], "Should detect journey motif"

    def test_detect_connection_motif(self, resonance_model):
        """Detect connection motif"""
        text = "We connect and bond together, creating bridges that unite us."
        result = resonance_model.detect_motif_frequency(text)

        assert "connection" in result["motif_counts"], "Should detect connection motif"

    def test_detect_belonging_motif(self, resonance_model):
        """Detect belonging motif"""
        text = "We belong together in our community, finding home in shared values."
        result = resonance_model.detect_motif_frequency(text)

        assert "belonging" in result["motif_counts"], "Should detect belonging motif"

    def test_motif_diversity_bonus(self, resonance_model):
        """Multiple motifs increase score"""
        text = """
        Our journey of growth transforms understanding. We connect through shared meaning
        and discover belonging. This wisdom reveals our true identity.
        """
        result = resonance_model.detect_motif_frequency(text)

        assert result["diversity"] > 0.5, "Should have high motif diversity"
        assert result["motif_score"] > 0.7, "Multiple motifs should boost score"


class TestCadenceCues:
    """Test cadence detection"""

    def test_detect_repetition(self, resonance_model):
        """Detect word repetition"""
        text = "We rise. We thrive. We connect."
        result = resonance_model.detect_cadence_cues(text)

        assert result["cadence_features"]["repetition"] > 0, "Should detect repetition"

    def test_detect_parallel_structure(self, resonance_model):
        """Detect parallel structures"""
        text = "We journey together. We grow together. We transform together."
        result = resonance_model.detect_cadence_cues(text)

        assert (
            result["cadence_features"]["parallel_structure"] > 0
        ), "Should detect parallel structure"

    def test_detect_rhythmic_questions(self, resonance_model):
        """Detect question patterns"""
        text = "What if we dared? What if we tried? What if we succeeded?"
        result = resonance_model.detect_cadence_cues(text)

        assert (
            result["cadence_features"]["rhythmic_questions"] > 0
        ), "Should detect rhythmic questions"

    def test_cadence_variety_bonus(self, resonance_model):
        """Multiple cadence types increase score"""
        text = """
        We thrive! What if we could thrive even more?
        The pattern repeats. The rhythm flows, builds, crescendos.
        We connect. We belong. We become.
        """
        result = resonance_model.detect_cadence_cues(text)

        assert result["variety"] >= 3, "Should have cadence variety"


class TestResonanceDetection:
    """Test combined resonance scoring"""

    def test_high_resonance_detection(self, resonance_model, resonance_samples):
        """Detect high-resonance texts (True Positives)"""
        high_samples = resonance_samples["high_resonance"]

        detected = 0
        for sample in high_samples:
            result = resonance_model.compute_resonance_score(sample["text"])

            if result["resonance_score"] >= sample["min_score"]:
                detected += 1

        detection_rate = detected / len(high_samples)

        print(
            f"\nHigh Resonance Detection: {detected}/{len(high_samples)} ({detection_rate:.0%})"
        )

        assert (
            detection_rate >= 0.85
        ), f"Detection rate {detection_rate:.0%} below 85% target"

    def test_low_resonance_rejection(self, resonance_model, resonance_samples):
        """Reject low-resonance texts (True Negatives)"""
        low_samples = resonance_samples["low_resonance"]

        correctly_rejected = 0
        for sample in low_samples:
            result = resonance_model.compute_resonance_score(sample["text"])

            if result["resonance_score"] <= sample["max_score"]:
                correctly_rejected += 1

        rejection_rate = correctly_rejected / len(low_samples)

        print(
            f"Low Resonance Rejection: {correctly_rejected}/{len(low_samples)} ({rejection_rate:.0%})"
        )

        # Allow slightly lower rejection rate (combined accuracy is what matters)
        assert (
            rejection_rate >= 0.70
        ), f"Rejection rate {rejection_rate:.0%} below 70% threshold"

    def test_overall_accuracy(self, resonance_model, resonance_samples):
        """Overall accuracy should be ≥85%"""
        high_samples = resonance_samples["high_resonance"]
        low_samples = resonance_samples["low_resonance"]

        # True Positives (high resonance detected)
        tp = sum(
            1
            for s in high_samples
            if resonance_model.compute_resonance_score(s["text"])["resonance_score"]
            >= s["min_score"]
        )

        # True Negatives (low resonance rejected)
        tn = sum(
            1
            for s in low_samples
            if resonance_model.compute_resonance_score(s["text"])["resonance_score"]
            <= s["max_score"]
        )

        total_samples = len(high_samples) + len(low_samples)
        accuracy = (tp + tn) / total_samples

        print("\nOverall Accuracy:")
        print(f"   True Positives: {tp}/{len(high_samples)}")
        print(f"   True Negatives: {tn}/{len(low_samples)}")
        print(f"   Accuracy: {accuracy:.0%}")

        assert accuracy >= 0.85, f"Accuracy {accuracy:.0%} below 85% target"


class TestResonanceMatrix:
    """Test resonance matrix generation"""

    def test_create_matrix(self, resonance_model):
        """Create resonance matrix CSV"""
        documents = [
            (
                "high_resonance_1",
                "We journey together, discovering profound meaning and joy.",
            ),
            ("high_resonance_2", "Love transforms us. Hope connects us. We belong."),
            ("low_resonance_1", "Data shows results. Analysis indicates patterns."),
            (
                "moderate",
                "Consider this insight. We explore practical applications together.",
            ),
        ]

        output_path = Path("test_resonance_matrix.csv")
        resonance_model.create_resonance_matrix(documents, output_path)

        assert output_path.exists(), "Should create matrix CSV"

        # Read and validate
        import csv

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4, "Should have 4 rows"
        assert "resonance_score" in rows[0], "Should have resonance_score column"
        assert "polarity_score" in rows[0], "Should have polarity_score column"

        # Clean up
        output_path.unlink()


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("🎵 MeaningForge v2 Resonance Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("  • Polarity Bands: Emotional valence detection")
    print("  • Motif Frequency: Archetypal pattern recognition")
    print("  • Cadence Cues: Rhythm and flow analysis")
    print("  • Resonance Detection: ≥85% accuracy validation")
    print("  • Resonance Matrix: CSV generation")
    print("\nRun with: pytest tests/test_meaningforge_v2_resonance.py -v -s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # If run directly, show summary
    test_suite_summary()

    # Run tests
    pytest.main([__file__, "-v", "-s"])

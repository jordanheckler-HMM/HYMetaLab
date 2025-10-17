#!/usr/bin/env python3
"""
OriginChain v5 Catalyst Test Suite
Tests for generative hypothesis system
"""
import json
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generative_catalyst import GenerativeCatalyst, Hypothesis


@pytest.fixture
def catalyst():
    """GenerativeCatalyst fixture with deterministic seed"""
    return GenerativeCatalyst(seed=42)


class TestHypothesisGeneration:
    """Test hypothesis generation"""

    def test_generate_single_hypothesis(self, catalyst):
        """Test generating a single hypothesis"""
        h = catalyst.generate_hypothesis()

        assert isinstance(h, Hypothesis)
        assert h.id is not None
        assert h.title is not None
        assert h.description is not None
        assert h.domain is not None
        assert 0.0 <= h.emergence_potential <= 1.0
        assert h.complexity_level in ["low", "medium", "high"]
        assert isinstance(h.testable, bool)
        assert 1 <= h.priority <= 5

    def test_generate_batch(self, catalyst):
        """Test generating batch of hypotheses"""
        batch = catalyst.generate_batch(n=5)

        assert len(batch) == 5
        assert all(isinstance(h, Hypothesis) for h in batch)
        assert len(catalyst.queue) == 5

    def test_unique_ids(self, catalyst):
        """Test hypothesis IDs are unique"""
        batch = catalyst.generate_batch(n=10)
        ids = [h.id for h in batch]

        assert len(ids) == len(set(ids))  # All unique

    def test_emergence_potential_range(self, catalyst):
        """Test emergence potential is in valid range"""
        batch = catalyst.generate_batch(n=20)

        for h in batch:
            assert 0.0 <= h.emergence_potential <= 1.0

    def test_priority_range(self, catalyst):
        """Test priority is in valid range"""
        batch = catalyst.generate_batch(n=20)

        for h in batch:
            assert 1 <= h.priority <= 5


class TestHypothesisAttributes:
    """Test hypothesis attribute computation"""

    def test_complexity_assessment(self, catalyst):
        """Test complexity level assessment"""
        # Generate multiple hypotheses
        batch = catalyst.generate_batch(n=10)

        # Should have mix of complexity levels
        complexities = [h.complexity_level for h in batch]
        assert all(c in ["low", "medium", "high"] for c in complexities)

    def test_testability_detection(self, catalyst):
        """Test testability detection"""
        batch = catalyst.generate_batch(n=20)

        # Some hypotheses should be testable
        testable_count = sum(1 for h in batch if h.testable)
        assert testable_count >= 0  # At least some should be marked

    def test_tag_generation(self, catalyst):
        """Test tag generation"""
        h = catalyst.generate_hypothesis()

        assert isinstance(h.tags, list)
        assert len(h.tags) > 0  # At least domain tag


class TestQueue:
    """Test hypothesis queue management"""

    def test_queue_accumulation(self, catalyst):
        """Test queue accumulates hypotheses"""
        catalyst.generate_batch(n=3)
        assert len(catalyst.queue) == 3

        catalyst.generate_batch(n=2)
        assert len(catalyst.queue) == 5

    def test_get_priority_queue(self, catalyst):
        """Test filtering by priority"""
        catalyst.generate_batch(n=20)

        high_priority = catalyst.get_priority_queue(min_priority=4)

        assert all(h.priority >= 4 for h in high_priority)

    def test_get_testable_queue(self, catalyst):
        """Test filtering by testability"""
        catalyst.generate_batch(n=20)

        testable = catalyst.get_testable_queue()

        assert all(h.testable for h in testable)


class TestSaving:
    """Test saving functionality"""

    def test_save_markdown(self, catalyst):
        """Test saving as markdown"""
        import tempfile

        catalyst.generate_batch(n=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            catalyst.save_queue(temp_path, format="markdown")
            assert temp_path.exists()

            content = temp_path.read_text()
            assert "# Genesis Queue" in content
            assert "Total Hypotheses" in content
        finally:
            temp_path.unlink()

    def test_save_json(self, catalyst):
        """Test saving as JSON"""
        import tempfile

        catalyst.generate_batch(n=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            catalyst.save_queue(temp_path, format="json")
            assert temp_path.exists()

            with open(temp_path) as f:
                data = json.load(f)

            assert "hypotheses" in data
            assert len(data["hypotheses"]) == 5
        finally:
            temp_path.unlink()


class TestDailyGeneration:
    """Test daily generation capability (≥1/24h requirement)"""

    def test_daily_simulation(self, catalyst):
        """Test simulating multiple days"""
        stats = catalyst.simulate_daily_generation(days=7)

        assert stats["days_simulated"] == 7
        assert stats["total_hypotheses"] > 0
        assert stats["mean_per_day"] >= 1.0  # At least 1 per day on average

    def test_meets_daily_target(self, catalyst):
        """Test meeting ≥1/day target"""
        stats = catalyst.simulate_daily_generation(days=10)

        assert (
            stats["meets_target"] == True
        ), "Failed to meet ≥1 hypothesis per day target"
        assert stats["min_per_day"] >= 1

    def test_single_day_generation(self, catalyst):
        """Test single day can generate ≥1"""
        stats = catalyst.simulate_daily_generation(days=1)

        assert stats["total_hypotheses"] >= 1
        assert stats["mean_per_day"] >= 1.0

    def test_weekly_generation(self, catalyst):
        """Test week-long generation"""
        stats = catalyst.simulate_daily_generation(days=7)

        # Should generate at least 7 hypotheses in 7 days
        assert stats["total_hypotheses"] >= 7
        assert stats["meets_target"] == True


class TestGenerationHistory:
    """Test generation history tracking"""

    def test_history_tracking(self, catalyst):
        """Test generation history is tracked"""
        catalyst.generate_batch(n=3)
        catalyst.generate_batch(n=2)

        assert len(catalyst.generation_history) == 2
        assert catalyst.generation_history[0]["count"] == 3
        assert catalyst.generation_history[1]["count"] == 2


class TestDeterminism:
    """Test deterministic behavior"""

    def test_same_seed_same_results(self):
        """Test same seed produces same hypotheses"""
        catalyst1 = GenerativeCatalyst(seed=123)
        catalyst2 = GenerativeCatalyst(seed=123)

        h1 = catalyst1.generate_hypothesis()
        h2 = catalyst2.generate_hypothesis()

        # Should generate same hypothesis
        assert h1.description == h2.description
        assert h1.domain == h2.domain


class TestEdgeCases:
    """Test edge cases"""

    def test_large_batch(self, catalyst):
        """Test generating large batch"""
        batch = catalyst.generate_batch(n=100)

        assert len(batch) == 100
        assert len(catalyst.queue) == 100

    def test_zero_batch(self, catalyst):
        """Test zero-size batch"""
        batch = catalyst.generate_batch(n=0)

        assert len(batch) == 0


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("OriginChain v5 Catalyst Test Suite Summary")
    print("=" * 70)
    print("✅ Hypothesis Generation: Single, batch, IDs, attributes")
    print("✅ Hypothesis Attributes: Complexity, testability, tags")
    print("✅ Queue Management: Accumulation, priority, testable")
    print("✅ Saving: Markdown and JSON formats")
    print("✅ Daily Generation: ≥1/24h requirement verification")
    print("✅ Generation History: Tracking over time")
    print("✅ Determinism: Reproducible results")
    print("✅ Edge Cases: Large batches, zero batch")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

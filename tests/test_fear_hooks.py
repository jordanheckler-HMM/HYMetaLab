#!/usr/bin/env python3
"""
Unit tests for fear-violence hooks integration.

Tests that fear updates are actually called within shock and inequality
simulation steps and that logging counters work correctly.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psych.adapters.fear_core import FearParams
from psych.adapters.fear_hooks import FearChannel


class TestFearHooks(unittest.TestCase):
    """Test cases for fear-violence hooks integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = FearParams()
        self.fear_channel = FearChannel(
            enable=True, params=self.params, track_history=True
        )

    def test_fear_updates_applied_in_shock_scenario(self):
        """Test that fear updates are applied during shock events."""
        # Create agent state
        agent_state = {"id": 1, "fear": 0.1, "cci": 0.7, "rand_aggr": 0.5}

        # Create group state with shock
        group_state = {"shock_level": 0.6, "gini": 0.25, "support_level": 0.8}

        # Apply fear update
        updated_state = self.fear_channel.before_commit(
            t_now=10, agent_state=agent_state, group_state=group_state
        )

        # Check that fear was updated
        self.assertIn("fear", updated_state)
        self.assertGreater(updated_state["fear"], 0.0)

        # Check that logging counters were updated
        stats = self.fear_channel.get_logging_stats()
        self.assertGreater(stats["fear_updates_applied"], 0)
        self.assertGreater(stats["shock_events_detected"], 0)

        # Check that aggression probability was calculated
        self.assertIn("aggression_prob", updated_state)
        self.assertGreaterEqual(updated_state["aggression_prob"], 0.0)
        self.assertLessEqual(updated_state["aggression_prob"], 1.0)

    def test_fear_updates_applied_in_inequality_scenario(self):
        """Test that fear updates are applied during inequality events."""
        # Create agent state
        agent_state = {"id": 2, "fear": 0.2, "cci": 0.6, "rand_aggr": 0.3}

        # Create group state with high inequality
        group_state = {
            "shock_level": 0.0,
            "gini": 0.45,
            "support_level": 0.6,
        }  # High inequality

        # Apply fear update
        updated_state = self.fear_channel.before_commit(
            t_now=15, agent_state=agent_state, group_state=group_state
        )

        # Check that fear was updated
        self.assertIn("fear", updated_state)
        self.assertGreater(updated_state["fear"], 0.0)

        # Check that logging counters were updated
        stats = self.fear_channel.get_logging_stats()
        self.assertGreater(stats["fear_updates_applied"], 0)
        self.assertGreater(stats["inequality_events_detected"], 0)

    def test_aggression_events_triggered(self):
        """Test that aggression events can be triggered."""
        # Create agent state with low random threshold (likely to trigger aggression)
        agent_state = {
            "id": 3,
            "fear": 0.8,  # High fear
            "cci": 0.3,  # Low CCI (more likely to be aggressive)
            "rand_aggr": 0.1,  # Low threshold
        }

        # Create group state with shock and inequality
        group_state = {"shock_level": 0.7, "gini": 0.4, "support_level": 0.5}

        # Apply fear update
        updated_state = self.fear_channel.before_commit(
            t_now=20, agent_state=agent_state, group_state=group_state
        )

        # Check that aggression event was potentially triggered
        self.assertIn("aggression_event", updated_state)
        self.assertIn("aggression_intensity", updated_state)

        # Check logging stats
        stats = self.fear_channel.get_logging_stats()
        self.assertGreaterEqual(stats["aggression_events_triggered"], 0)

    def test_no_fear_updates_when_disabled(self):
        """Test that fear updates are not applied when channel is disabled."""
        # Create disabled fear channel
        disabled_channel = FearChannel(enable=False, params=self.params)

        # Create agent state
        agent_state = {"id": 4, "fear": 0.1, "cci": 0.7, "rand_aggr": 0.5}

        # Create group state with shock
        group_state = {"shock_level": 0.6, "gini": 0.25}

        # Apply fear update
        updated_state = disabled_channel.before_commit(
            t_now=25, agent_state=agent_state, group_state=group_state
        )

        # Check that state was not modified
        self.assertEqual(updated_state["fear"], 0.1)
        self.assertNotIn("aggression_event", updated_state)

        # Check that no logging occurred
        stats = disabled_channel.get_logging_stats()
        self.assertEqual(stats["fear_updates_applied"], 0)
        self.assertEqual(stats["shock_events_detected"], 0)

    def test_multiple_fear_updates_accumulate_counters(self):
        """Test that multiple fear updates accumulate logging counters correctly."""
        # Create agent state
        agent_state = {"id": 5, "fear": 0.0, "cci": 0.6, "rand_aggr": 0.5}

        # Create group state with shock
        group_state = {"shock_level": 0.5, "gini": 0.35, "support_level": 0.7}

        # Apply multiple fear updates
        for t in range(5):
            agent_state = self.fear_channel.before_commit(
                t_now=t, agent_state=agent_state, group_state=group_state
            )

        # Check that counters accumulated correctly
        stats = self.fear_channel.get_logging_stats()
        self.assertEqual(stats["fear_updates_applied"], 5)
        self.assertEqual(
            stats["shock_events_detected"], 5
        )  # Shock detected in each update
        self.assertGreaterEqual(stats["inequality_events_detected"], 0)

    def test_different_shock_sources_detected(self):
        """Test that different shock sources are properly detected."""
        # Test shock_level
        agent_state = {"id": 6, "fear": 0.0, "cci": 0.6, "rand_aggr": 0.5}
        group_state = {"shock_level": 0.4, "gini": 0.25}

        updated_state = self.fear_channel.before_commit(
            t_now=30, agent_state=agent_state, group_state=group_state
        )

        stats = self.fear_channel.get_logging_stats()
        self.assertGreater(stats["shock_events_detected"], 0)

        # Test shock_severity
        agent_state = {"id": 7, "fear": 0.0, "cci": 0.6, "rand_aggr": 0.5}
        group_state = {"shock_severity": 0.3, "gini": 0.25}

        updated_state = self.fear_channel.before_commit(
            t_now=31, agent_state=agent_state, group_state=group_state
        )

        stats = self.fear_channel.get_logging_stats()
        self.assertGreater(stats["shock_events_detected"], 1)  # Should have increased

        # Test magnitude
        agent_state = {"id": 8, "fear": 0.0, "cci": 0.6, "rand_aggr": 0.5}
        group_state = {"magnitude": 0.2, "gini": 0.25}

        updated_state = self.fear_channel.before_commit(
            t_now=32, agent_state=agent_state, group_state=group_state
        )

        stats = self.fear_channel.get_logging_stats()
        self.assertGreater(
            stats["shock_events_detected"], 2
        )  # Should have increased again

    def test_history_tracking_when_enabled(self):
        """Test that history tracking works when enabled."""
        # Create agent state
        agent_state = {"id": 9, "fear": 0.1, "cci": 0.7, "rand_aggr": 0.5}

        # Create group state
        group_state = {"shock_level": 0.3, "gini": 0.3, "support_level": 0.8}

        # Apply multiple updates
        for t in range(3):
            agent_state = self.fear_channel.before_commit(
                t_now=t, agent_state=agent_state, group_state=group_state
            )

        # Check that history was tracked
        stats = self.fear_channel.get_logging_stats()
        self.assertEqual(stats["fear_history_length"], 3)
        self.assertEqual(stats["aggression_history_length"], 3)

        # Check that history contains expected values
        self.assertEqual(len(self.fear_channel.fear_history), 3)
        self.assertEqual(len(self.fear_channel.aggression_history), 3)
        self.assertEqual(len(self.fear_channel.cci_history), 3)
        self.assertEqual(len(self.fear_channel.inequality_history), 3)


if __name__ == "__main__":
    unittest.main()

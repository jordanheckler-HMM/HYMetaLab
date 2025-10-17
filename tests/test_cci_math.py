#!/usr/bin/env python3
"""
Unit tests for stabilized CCI math with epsilon guards and semantics.

Tests monotonicity, boundedness, and edge cases with near-zero noise.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[0].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cci_math import (
    DEFAULT_CCI_CONFIG,
    CCIConfig,
    classify_consciousness_level,
    compute_cci_from_data,
    compute_cci_stabilized,
    validate_cci_monotonicity,
)


class TestCCIMath(unittest.TestCase):
    """Test cases for CCI math stabilization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CCIConfig()

    def test_epsilon_guard_prevents_division_by_zero(self):
        """Test that epsilon guard prevents division by zero."""
        # Test with zero noise
        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.8,
            coherence_score=0.7,
            emergence_index=0.9,
            noise_level=0.0,
            config=self.config,
        )

        # Should not be infinite or NaN
        self.assertFalse(np.isinf(cci_value))
        self.assertFalse(np.isnan(cci_value))
        self.assertTrue(cci_value > 0)

        # Should have applied epsilon guard
        self.assertTrue(metadata["epsilon_guard_applied"])
        self.assertEqual(metadata["effective_noise"], self.config.EPSILON)

    def test_epsilon_guard_with_very_small_noise(self):
        """Test epsilon guard with noise smaller than epsilon."""
        very_small_noise = self.config.EPSILON / 10

        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.9,
            coherence_score=0.8,
            emergence_index=0.7,
            noise_level=very_small_noise,
            config=self.config,
        )

        # Should apply epsilon guard
        self.assertTrue(metadata["epsilon_guard_applied"])
        self.assertEqual(metadata["effective_noise"], self.config.EPSILON)

        # Should have "highest observed accuracy" semantics
        self.assertEqual(
            metadata["calibration_semantics"],
            "highest_observed_accuracy_at_minimal_noise",
        )

    def test_standard_calibration_semantics(self):
        """Test standard calibration semantics with normal noise."""
        normal_noise = 0.1

        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.7,
            coherence_score=0.6,
            emergence_index=0.8,
            noise_level=normal_noise,
            config=self.config,
        )

        # Should not apply epsilon guard
        self.assertFalse(metadata["epsilon_guard_applied"])
        self.assertEqual(metadata["effective_noise"], normal_noise)

        # Should have standard calibration semantics
        self.assertEqual(metadata["calibration_semantics"], "standard_calibration")

    def test_cci_boundedness(self):
        """Test that CCI values are bounded."""
        # Test with maximum possible inputs
        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=1.0,
            coherence_score=1.0,
            emergence_index=1.0,
            noise_level=self.config.EPSILON,  # Minimum effective noise
            config=self.config,
        )

        # Should be capped at MAX_CCI
        self.assertLessEqual(cci_value, self.config.MAX_CCI)

    def test_input_clamping(self):
        """Test that inputs are properly clamped to valid ranges."""
        # Test with out-of-range inputs
        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=1.5,  # > 1.0
            coherence_score=-0.1,  # < 0.0
            emergence_index=2.0,  # > 1.0
            noise_level=-0.5,  # < 0.0
            config=self.config,
        )

        # Should not be infinite or NaN
        self.assertFalse(np.isinf(cci_value))
        self.assertFalse(np.isnan(cci_value))
        self.assertTrue(cci_value >= 0)

    def test_cci_from_data_with_beta_prior(self):
        """Test CCI computation from data with beta prior smoothing."""
        # Create test data
        n_points = 100
        reported_confidence = np.random.uniform(0.3, 0.9, n_points)
        correct = np.random.binomial(1, 0.7, n_points)  # 70% accuracy

        cci_value, metadata = compute_cci_from_data(
            reported_confidence=reported_confidence, correct=correct, config=self.config
        )

        # Should have valid CCI value
        self.assertFalse(np.isinf(cci_value))
        self.assertFalse(np.isnan(cci_value))
        self.assertTrue(cci_value >= 0)

        # Should have beta prior smoothing
        self.assertEqual(metadata["beta_prior_alpha"], self.config.BETA_PRIOR_SUCCESS)
        self.assertEqual(metadata["beta_prior_beta"], self.config.BETA_PRIOR_FAIL)

        # Should have data statistics
        self.assertEqual(metadata["n_data_points"], n_points)
        self.assertIn("n_success", metadata)
        self.assertIn("n_failure", metadata)

    def test_cci_from_data_with_empty_input(self):
        """Test CCI computation with empty/invalid data."""
        # Test with empty arrays
        cci_value, metadata = compute_cci_from_data(
            reported_confidence=np.array([]), correct=np.array([]), config=self.config
        )

        self.assertEqual(cci_value, 0.0)
        self.assertIn("error", metadata)

        # Test with all NaN values
        cci_value, metadata = compute_cci_from_data(
            reported_confidence=np.array([np.nan, np.nan]),
            correct=np.array([np.nan, np.nan]),
            config=self.config,
        )

        self.assertEqual(cci_value, 0.0)
        self.assertIn("error", metadata)

    def test_consciousness_level_classification(self):
        """Test consciousness level classification."""
        test_cases = [
            (0.1, "unconscious"),
            (0.4, "pre_conscious"),
            (0.7, "conscious"),
            (0.9, "highly_conscious"),
            (1.5, "transcendent"),
        ]

        for cci_value, expected_level in test_cases:
            classification = classify_consciousness_level(cci_value)
            self.assertEqual(classification["level"], expected_level)
            self.assertIn("interpretation", classification)
            self.assertIn("range", classification)

    def test_cci_monotonicity_validation(self):
        """Test CCI monotonicity validation."""
        # Create monotonic data (CCI decreases with noise)
        noise_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cci_values = np.array([8.0, 4.0, 2.67, 2.0, 1.6])

        validation = validate_cci_monotonicity(noise_levels, cci_values)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["n_violations"], 0)

        # Create non-monotonic data (CCI increases with noise)
        cci_values_violation = np.array([4.0, 6.0, 5.0, 3.0, 2.0])

        validation = validate_cci_monotonicity(noise_levels, cci_values_violation)
        self.assertFalse(validation["valid"])
        self.assertGreater(validation["n_violations"], 0)
        self.assertIn("violations", validation)

    def test_edge_case_near_zero_noise(self):
        """Test edge case with noise very close to zero."""
        # Test with noise just above robust threshold
        noise_above_threshold = self.config.PERFECT_NOISE_THRESHOLD * 2

        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.95,
            coherence_score=0.9,
            emergence_index=0.85,
            noise_level=noise_above_threshold,
            config=self.config,
        )

        # Should not apply epsilon guard
        self.assertFalse(metadata["epsilon_guard_applied"])

        # Should have standard calibration semantics
        self.assertEqual(metadata["calibration_semantics"], "standard_calibration")

    def test_edge_case_noise_below_perfect_threshold(self):
        """Test edge case with noise below robust threshold."""
        noise_below_threshold = self.config.PERFECT_NOISE_THRESHOLD / 2

        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.99,
            coherence_score=0.95,
            emergence_index=0.9,
            noise_level=noise_below_threshold,
            config=self.config,
        )

        # Should have highest observed accuracy semantics
        self.assertEqual(
            metadata["calibration_semantics"],
            "highest_observed_accuracy_at_minimal_noise",
        )

    def test_configuration_persistence(self):
        """Test that configuration is properly stored in metadata."""
        custom_config = CCIConfig(EPSILON=1e-4, MAX_CCI=5.0)

        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.8,
            coherence_score=0.7,
            emergence_index=0.6,
            noise_level=0.1,
            config=custom_config,
        )

        # Should store configuration in metadata
        self.assertIn("config_used", metadata)
        self.assertEqual(metadata["config_used"]["epsilon"], custom_config.EPSILON)
        self.assertEqual(metadata["config_used"]["max_cci"], custom_config.MAX_CCI)

    def test_default_config_usage(self):
        """Test that default configuration works correctly."""
        cci_value, metadata = compute_cci_stabilized(
            calibration_accuracy=0.7,
            coherence_score=0.6,
            emergence_index=0.8,
            noise_level=0.2,
            # No config provided, should use default
        )

        # Should work without errors
        self.assertFalse(np.isinf(cci_value))
        self.assertFalse(np.isnan(cci_value))

        # Should use default configuration values
        self.assertEqual(metadata["config_used"]["epsilon"], DEFAULT_CCI_CONFIG.EPSILON)
        self.assertEqual(metadata["config_used"]["max_cci"], DEFAULT_CCI_CONFIG.MAX_CCI)


if __name__ == "__main__":
    unittest.main()

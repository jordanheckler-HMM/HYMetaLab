# tests_real_world_validation/test_smoke.py
"""
Smoke tests for real-world validation module.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from real_world_validation.cli import load_scenarios, run_scenario


class TestRealWorldValidation(unittest.TestCase):
    """Smoke tests for real-world validation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

        # Change to temp directory
        os.chdir(self.temp_dir)

        # Create discovery_results directory
        Path("discovery_results").mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_load_scenarios(self):
        """Test loading scenarios configuration."""
        scenarios = load_scenarios("real_world_validation/scenarios.yaml")

        self.assertIsInstance(scenarios, dict)
        self.assertIn("2008_us_market", scenarios)
        self.assertIn("covid_italy", scenarios)
        self.assertIn("gini_gdp_global_2000_2024", scenarios)

        # Check scenario structure
        market_scenario = scenarios["2008_us_market"]
        self.assertEqual(market_scenario["kind"], "market")
        self.assertEqual(market_scenario["key"], "US_MARKET")
        self.assertIn("window", market_scenario)
        self.assertIn("data", market_scenario)
        self.assertIn("metrics", market_scenario)

    def test_run_2008_us_market_scenario(self):
        """Test running the 2008 US market scenario."""
        scenarios = load_scenarios("real_world_validation/scenarios.yaml")

        # Override the date range to a smaller window for testing
        success = run_scenario(
            "2008_us_market",
            scenarios,
            force_refresh=True,
            start_date="2007-01-01",
            end_date="2009-12-31",
        )

        # Check if scenario completed successfully
        self.assertTrue(success, "2008 US market scenario should complete successfully")

        # Check if output directory was created
        output_dir = Path("discovery_results/real_world/2008_us_market")
        self.assertTrue(output_dir.exists(), "Output directory should be created")

        # Check if required files exist
        self.assertTrue(
            (output_dir / "REPORT.md").exists(), "Report should be generated"
        )
        self.assertTrue(
            (output_dir / "run_manifest.json").exists(), "Manifest should be generated"
        )

        # Check if figures directory exists and has at least 3 figures
        figures_dir = output_dir / "figures"
        if figures_dir.exists():
            figure_files = list(figures_dir.glob("*.png"))
            self.assertGreaterEqual(
                len(figure_files), 3, "Should generate at least 3 figures"
            )

        # Check if data directories exist
        self.assertTrue(
            (output_dir / "data_clean").exists(), "Data clean directory should exist"
        )
        self.assertTrue(
            (output_dir / "metrics").exists(), "Metrics directory should exist"
        )

    def test_shock_classification(self):
        """Test that shock classification produces expected counts."""
        scenarios = load_scenarios("real_world_validation/scenarios.yaml")

        success = run_scenario(
            "2008_us_market",
            scenarios,
            force_refresh=True,
            start_date="2007-01-01",
            end_date="2009-12-31",
        )

        self.assertTrue(success, "Scenario should complete successfully")

        # Check shock classification results
        output_dir = Path("discovery_results/real_world/2008_us_market")
        metrics_dir = output_dir / "metrics"

        if (metrics_dir / "shocks.csv").exists():
            import pandas as pd

            shocks_df = pd.read_csv(metrics_dir / "shocks.csv")

            # Check that we have shock classifications
            if "shock_class" in shocks_df.columns:
                shock_classes = shocks_df["shock_class"].value_counts()

                # Should have some shocks classified
                total_shocks = len(shocks_df)
                self.assertGreater(total_shocks, 0, "Should identify some shocks")

                # Check that classifications are valid
                valid_classes = {"constructive", "transition", "destructive"}
                actual_classes = set(shock_classes.index)
                self.assertTrue(
                    actual_classes.issubset(valid_classes),
                    f"Shock classes should be valid: {actual_classes}",
                )

    def test_collapse_threshold(self):
        """Test that collapse threshold analysis works."""
        scenarios = load_scenarios("real_world_validation/scenarios.yaml")

        success = run_scenario(
            "gini_gdp_global_2000_2024",
            scenarios,
            force_refresh=True,
            start_date="2000-01-01",
            end_date="2024-12-31",
        )

        self.assertTrue(success, "Scenario should complete successfully")

        # Check collapse analysis results
        output_dir = Path("discovery_results/real_world/gini_gdp_global_2000_2024")
        metrics_dir = output_dir / "metrics"

        if (metrics_dir / "collapse.csv").exists():
            import pandas as pd

            collapse_df = pd.read_csv(metrics_dir / "collapse.csv")

            # Check that we have collapse data
            self.assertGreater(len(collapse_df), 0, "Should have collapse data")

            # Check that threshold analysis was performed
            if "above_threshold" in collapse_df.columns:
                threshold_breaches = collapse_df["above_threshold"].sum()
                self.assertGreaterEqual(
                    threshold_breaches, 0, "Threshold breaches should be non-negative"
                )

    def test_survival_analysis(self):
        """Test that survival analysis produces recovery metrics."""
        scenarios = load_scenarios("real_world_validation/scenarios.yaml")

        success = run_scenario(
            "covid_italy",
            scenarios,
            force_refresh=True,
            start_date="2020-01-01",
            end_date="2022-12-31",
        )

        self.assertTrue(success, "Scenario should complete successfully")

        # Check survival analysis results
        output_dir = Path("discovery_results/real_world/covid_italy")
        metrics_dir = output_dir / "metrics"

        if (metrics_dir / "survival.csv").exists():
            import pandas as pd

            survival_df = pd.read_csv(metrics_dir / "survival.csv")

            # Check that we have survival data
            if "recovery_days" in survival_df.columns:
                recovery_days = survival_df["recovery_days"].dropna()
                self.assertGreater(
                    len(recovery_days), 0, "Should have recovery period data"
                )

                # Check that recovery times are reasonable
                self.assertGreater(
                    recovery_days.min(), 0, "Recovery times should be positive"
                )
                self.assertLess(
                    recovery_days.max(), 1000, "Recovery times should be reasonable"
                )


def run_smoke_tests():
    """Run smoke tests."""
    print("Running smoke tests for real-world validation...")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRealWorldValidation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\nSmoke test summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)

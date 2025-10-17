# real_world_validation/cli.py
"""
Command-line interface for real-world validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .etl import run_etl_pipeline
from .fetchers import fetch_scenario_data
from .figures import generate_figures
from .mapping import map_data_to_simulation
from .metrics_bridge import bridge_to_simulation
from .reporters import generate_reports
from .utils import now_iso, safe_mkdirs


def load_scenarios(
    scenarios_file: str = "real_world_validation/scenarios.yaml",
) -> dict[str, Any]:
    """Load scenarios configuration."""
    try:
        with open(scenarios_file) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Scenarios file not found: {scenarios_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing scenarios file: {e}")
        sys.exit(1)


def run_scenario(
    scenario_id: str,
    scenarios: dict[str, Any],
    force_refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> bool:
    """Run a single scenario."""

    if scenario_id not in scenarios:
        print(f"Error: Scenario '{scenario_id}' not found in scenarios file")
        print(f"Available scenarios: {list(scenarios.keys())}")
        return False

    scenario_config = scenarios[scenario_id]

    # Override date window if provided
    if start_date or end_date:
        if "window" not in scenario_config:
            scenario_config["window"] = {}
        if start_date:
            scenario_config["window"]["start"] = start_date
        if end_date:
            scenario_config["window"]["end"] = end_date

    print(f"\n{'='*60}")
    print(f"RUNNING SCENARIO: {scenario_id}")
    print(f"{'='*60}")
    print(f"Type: {scenario_config.get('kind', 'unknown')}")
    print(f"Key: {scenario_config.get('key', 'unknown')}")
    print(
        f"Window: {scenario_config.get('window', {}).get('start', 'N/A')} to {scenario_config.get('window', {}).get('end', 'N/A')}"
    )

    # Create output directory
    timestamp = now_iso().replace(":", "-").split(".")[0]
    output_dir = Path(f"discovery_results/real_world/{scenario_id}")
    safe_mkdirs(output_dir)

    try:
        # Step 1: Fetch data
        print("\n🔄 Step 1: Fetching data...")
        fetch_results = fetch_scenario_data(scenario_config, force_refresh)

        if not fetch_results["data"]:
            print("❌ No data fetched successfully")
            if fetch_results["errors"]:
                print("Errors:")
                for error in fetch_results["errors"]:
                    print(f"  - {error}")
            return False

        # Step 2: ETL pipeline
        print("\n🔄 Step 2: ETL pipeline...")
        harmonized_df = run_etl_pipeline(scenario_config, fetch_results["data"])

        if harmonized_df.empty:
            print("❌ ETL pipeline failed")
            return False

        # Save cleaned data
        data_clean_dir = output_dir / "data_clean"
        safe_mkdirs(data_clean_dir)
        harmonized_df.to_csv(data_clean_dir / "harmonized_data.csv", index=False)

        # Step 3: Map to simulation constructs
        print("\n🔄 Step 3: Mapping to simulation constructs...")
        mapped_data = map_data_to_simulation(scenario_config, harmonized_df)

        if not mapped_data:
            print("❌ Data mapping failed")
            return False

        # Save mapped data
        metrics_dir = output_dir / "metrics"
        safe_mkdirs(metrics_dir)
        for data_type, df in mapped_data.items():
            df.to_csv(metrics_dir / f"{data_type}.csv", index=False)

        # Step 4: Bridge to simulation modules
        print("\n🔄 Step 4: Bridging to simulation modules...")
        bridge_results = bridge_to_simulation(scenario_config, mapped_data)

        # Step 5: Generate figures
        print("\n🔄 Step 5: Generating figures...")
        figure_paths = generate_figures(
            scenario_config,
            bridge_results["processed_data"],
            harmonized_df,
            str(output_dir),
        )

        # Step 6: Generate reports
        print("\n🔄 Step 6: Generating reports...")
        report_results = generate_reports(
            scenario_id,
            scenario_config,
            bridge_results["processed_data"],
            bridge_results["summary_metrics"],
            figure_paths,
            fetch_results["metadata"],
            str(output_dir),
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"SCENARIO COMPLETED: {scenario_id}")
        print(f"{'='*60}")
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Data points: {len(harmonized_df)}")
        print(f"✓ Figures: {len(figure_paths)}")
        print(f"✓ Report: {report_results['report']}")
        print(f"✓ Manifest: {report_results['manifest']}")

        # Print quick metrics
        metrics = bridge_results["summary_metrics"].get("metrics", {})
        if "shocks" in metrics:
            shock_metrics = metrics["shocks"]
            print(
                f"✓ Shocks: {shock_metrics.get('total_shocks', 0)} total, {shock_metrics.get('destructive_shocks', 0)} destructive"
            )

        if "survival" in metrics:
            survival_metrics = metrics["survival"]
            print(
                f"✓ Recovery periods: {survival_metrics.get('total_recovery_periods', 0)}, median {survival_metrics.get('median_recovery_days', 0):.1f} days"
            )

        if "collapse" in metrics:
            collapse_metrics = metrics["collapse"]
            print(
                f"✓ Threshold breaches: {collapse_metrics.get('threshold_breaches', 0)}"
            )

        return True

    except Exception as e:
        print(f"❌ Scenario failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_scenarios(
    scenarios: dict[str, Any],
    force_refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, bool]:
    """Run all scenarios."""

    results = {}

    print(f"\n{'='*60}")
    print("RUNNING ALL SCENARIOS")
    print(f"{'='*60}")
    print(f"Total scenarios: {len(scenarios)}")

    for scenario_id in scenarios.keys():
        print(f"\n--- Running {scenario_id} ---")
        success = run_scenario(
            scenario_id, scenarios, force_refresh, start_date, end_date
        )
        results[scenario_id] = success

        if success:
            print(f"✅ {scenario_id} completed successfully")
        else:
            print(f"❌ {scenario_id} failed")

    # Print summary
    print(f"\n{'='*60}")
    print("ALL SCENARIOS COMPLETED")
    print(f"{'='*60}")

    successful = sum(results.values())
    total = len(results)

    print(f"Successful: {successful}/{total}")

    if successful < total:
        print("Failed scenarios:")
        for scenario_id, success in results.items():
            if not success:
                print(f"  - {scenario_id}")

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-world validation CLI for consciousness proxy simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m real_world_validation.cli run --scenario 2008_us_market
  python -m real_world_validation.cli run --scenario covid_italy --fresh
  python -m real_world_validation.cli run --all --start 2020-01-01 --end 2022-12-31
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run validation scenarios")
    run_parser.add_argument("--scenario", help="Scenario ID to run")
    run_parser.add_argument("--all", action="store_true", help="Run all scenarios")
    run_parser.add_argument(
        "--fresh", action="store_true", help="Ignore cache and refetch data"
    )
    run_parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    run_parser.add_argument(
        "--scenarios-file",
        default="real_world_validation/scenarios.yaml",
        help="Path to scenarios configuration file",
    )

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    # Load scenarios
    scenarios = load_scenarios(args.scenarios_file)

    if args.all:
        # Run all scenarios
        results = run_all_scenarios(scenarios, args.fresh, args.start, args.end)

        # Exit with error code if any scenario failed
        if not all(results.values()):
            sys.exit(1)

    elif args.scenario:
        # Run single scenario
        success = run_scenario(
            args.scenario, scenarios, args.fresh, args.start, args.end
        )

        if not success:
            sys.exit(1)

    else:
        print("Error: Must specify either --scenario or --all")
        run_parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

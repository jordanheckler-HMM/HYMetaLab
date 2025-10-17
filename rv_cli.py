"""
Command-line interface for real-world validation system.
"""

from datetime import datetime
from pathlib import Path

import typer

from real_world_validation.aggregate import AggregateAnalyzer
from real_world_validation.clean import DataCleaner
from real_world_validation.features import FeatureEngineer
from real_world_validation.fetch_runner import FetchRunner
from real_world_validation.ingest import DataIngester
from real_world_validation.models import ModelFitter
from real_world_validation.plots import PlotGenerator
from real_world_validation.report import ReportGenerator

app = typer.Typer(help="Real World Validation CLI for fear-violence hypothesis testing")


@app.command()
def fetch(
    group: str | None = typer.Option(
        None, "--group", "-g", help="Fetch data for a city group"
    ),
    city: str | None = typer.Option(
        None, "--city", "-c", help="Fetch data for a single city"
    ),
    registry_path: str = typer.Option(
        "real_world_validation/registry.yaml",
        "--registry",
        help="Path to registry YAML",
    ),
    force: bool = typer.Option(
        False, "--force", help="Force re-fetch even if data exists"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Fetch data from external sources for cities or groups."""

    print("Real World Validation - Data Fetching")
    print("=" * 50)

    # Initialize fetch runner
    try:
        fetch_runner = FetchRunner(registry_path)
    except Exception as e:
        print(f"❌ Failed to initialize fetch runner: {str(e)}")
        return

    if group:
        # Fetch for entire group
        print(f"Fetching data for group: {group}")

        try:
            results = fetch_runner.fetch_group(group)

            successful_cities = [city for city, data in results.items() if data]
            failed_cities = [city for city, data in results.items() if not data]

            print("\nFetch Results:")
            print(f"  ✅ Successful: {len(successful_cities)} cities")
            print(f"  ❌ Failed: {len(failed_cities)} cities")

            if successful_cities:
                print("\nSuccessful cities:")
                for city in successful_cities:
                    print(f"  - {city}")

            if failed_cities:
                print("\nFailed cities:")
                for city in failed_cities:
                    print(f"  - {city}")

            print("\n✓ Data saved to data/ directory")

        except Exception as e:
            print(f"❌ Group fetch failed: {str(e)}")
            if verbose:
                import traceback

                traceback.print_exc()

    elif city:
        # Fetch for single city
        print(f"Fetching data for city: {city}")

        try:
            results = fetch_runner.fetch_city(city)

            if results:
                print(f"✓ Successfully fetched data for {city}")
                print(f"  Sources: {list(results.keys())}")
                print(f"✓ Data saved to data/{city}/")
            else:
                print(f"❌ No data fetched for {city}")

        except Exception as e:
            print(f"❌ City fetch failed: {str(e)}")
            if verbose:
                import traceback

                traceback.print_exc()

    else:
        print("❌ Must specify either --group or --city")
        return


@app.command()
def autopilot(
    group: str = typer.Option(
        ..., "--group", "-g", help="Run end-to-end validation for a city group"
    ),
    start: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    freq: str = typer.Option("week", "--freq", help="Aggregation frequency (day/week)"),
    collapse_q: float = typer.Option(
        0.9, "--collapse-q", help="Collapse quantile threshold"
    ),
    collapse_min_weeks: int = typer.Option(
        3, "--collapse-min-weeks", help="Minimum weeks for collapse"
    ),
    fear_terms: str | None = typer.Option(
        None, "--fear-terms", help="Comma-separated fear terms"
    ),
    output_dir: str | None = typer.Option(None, "--output", help="Output directory"),
    force_fetch: bool = typer.Option(
        False, "--force-fetch", help="Force re-fetch data"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run end-to-end validation: fetch → clean → feature → model → plot → report."""

    print("Real World Validation - Autopilot Mode")
    print("=" * 50)
    print(f"Group: {group}")

    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"discovery_results/validation/{group}_{timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Step 1: Fetch data
    print(f"\n🔄 Step 1: Fetching data for {group}...")
    try:
        fetch_runner = FetchRunner()
        fetch_results = fetch_runner.fetch_group(group)

        successful_cities = [city for city, data in fetch_results.items() if data]
        print(f"✓ Fetched data for {len(successful_cities)} cities")

        if not successful_cities:
            print("❌ No data fetched successfully")
            return

    except Exception as e:
        print(f"❌ Data fetch failed: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        return

    # Step 2: Run validation analysis
    print("\n🔄 Step 2: Running validation analysis...")
    try:
        # Initialize components
        ingester = DataIngester()
        cleaner = DataCleaner(
            {
                "aggregation_freq": (
                    "W" if freq == "week" else "D"
                ),  # Convert to pandas frequency
                "collapse_quantile": collapse_q,
                "collapse_min_weeks": collapse_min_weeks,
            }
        )
        feature_engineer = FeatureEngineer(
            {"collapse_quantile": collapse_q, "collapse_min_weeks": collapse_min_weeks}
        )
        model_fitter = ModelFitter()
        plot_generator = PlotGenerator()
        report_generator = ReportGenerator()
        aggregate_analyzer = AggregateAnalyzer()

        # Process each city
        city_results = {}
        processed_cities = []

        for city_key in successful_cities:
            try:
                print(f"  Processing {city_key}...")

                # Load city data
                city_data = ingester.load_city_data(city_key)
                if not city_data:
                    print(f"    ❌ No data loaded for {city_key}")
                    continue

                # Get city config
                city_config = ingester.get_city_config(city_key)
                if not city_config:
                    print(f"    ❌ No config found for {city_key}")
                    continue

                city_info = city_config["city"]

                # Clean data
                cleaned_data = cleaner.clean_city_data(
                    city_data, city_info["timezone"], start, end
                )

                if not cleaned_data:
                    print(f"    ❌ No cleaned data for {city_key}")
                    continue

                # Create features
                features_df = feature_engineer.create_features(cleaned_data)

                if features_df.empty:
                    print(f"    ❌ No features created for {city_key}")
                    continue

                # Fit models
                model_results = model_fitter.fit_all_models(features_df, city_key)

                # Generate plots
                plot_files = plot_generator.generate_city_plots(
                    features_df, model_results, city_key, output_dir
                )

                # Generate report
                report_file = report_generator.generate_city_report(
                    city_key, features_df, model_results, plot_files, output_dir
                )

                city_results[city_key] = {
                    "features_df": features_df,
                    "model_results": model_results,
                    "plot_files": plot_files,
                    "report_file": report_file,
                }

                processed_cities.append(city_key)
                print(f"    ✓ Completed {city_key}")

            except Exception as e:
                print(f"    ❌ Failed {city_key}: {str(e)}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                continue

        print(f"✓ Processed {len(processed_cities)} cities")

        # Step 3: Generate aggregate analysis
        if processed_cities:
            print("\n🔄 Step 3: Generating aggregate analysis...")

            # Extract model results for aggregation
            aggregate_results = {
                city: results["model_results"] for city, results in city_results.items()
            }

            aggregate_analysis = aggregate_analyzer.analyze_group_results(
                group, aggregate_results, output_dir
            )

            print("✓ Aggregate analysis complete")

        print("\n🎉 Autopilot complete!")
        print(f"✓ Results saved to {output_dir}")
        print(f"✓ Processed {len(processed_cities)} cities successfully")

    except Exception as e:
        print(f"❌ Validation analysis failed: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()


@app.command()
def run(
    group: str | None = typer.Option(
        None, "--group", "-g", help="Run validation for a city group"
    ),
    city: str | None = typer.Option(
        None, "--city", "-c", help="Run validation for a single city"
    ),
    start: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    freq: str = typer.Option("week", "--freq", help="Aggregation frequency (day/week)"),
    collapse_q: float = typer.Option(
        0.9, "--collapse-q", help="Collapse quantile threshold"
    ),
    collapse_min_weeks: int = typer.Option(
        3, "--collapse-min-weeks", help="Minimum weeks for collapse"
    ),
    fear_terms: str | None = typer.Option(
        None, "--fear-terms", help="Comma-separated fear terms"
    ),
    align_timescales: bool = typer.Option(
        True, "--align-timescales", help="Enable time scale alignment (default: True)"
    ),
    output_dir: str | None = typer.Option(None, "--output", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run validation analysis for cities or groups."""

    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"discovery_results/validation/validation_{timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Real World Validation - Output: {output_dir}")

    # Initialize components
    ingester = DataIngester()
    cleaner = DataCleaner(
        {
            "aggregation_freq": (
                "W" if freq == "week" else "D"
            ),  # Convert to pandas frequency
            "collapse_quantile": collapse_q,
            "collapse_min_weeks": collapse_min_weeks,
        }
    )
    feature_engineer = FeatureEngineer(
        {
            "collapse_quantile": collapse_q,
            "collapse_min_weeks": collapse_min_weeks,
            "align_timescales": align_timescales,
            "gini_resample_method": "ffill",  # Default to forward-fill
        }
    )
    model_fitter = ModelFitter(
        {"align_timescales": align_timescales, "check_gini_variance": True}
    )
    plot_generator = PlotGenerator()
    report_generator = ReportGenerator()
    aggregate_analyzer = AggregateAnalyzer()

    if group:
        # Run for entire group
        print(f"Running validation for group: {group}")

        try:
            # Load group data
            group_data = ingester.load_group_data(group)

            if not group_data:
                print(f"❌ No data loaded for group {group}")
                return

            print(f"✓ Loaded data for {len(group_data)} cities")

            # Process each city
            city_results = {}
            successful_cities = []

            for city_key, city_data in group_data.items():
                try:
                    print(f"\nProcessing {city_key}...")

                    # Get city config
                    city_config = ingester.get_city_config(city_key)
                    if not city_config:
                        print(f"  ❌ No config found for {city_key}")
                        continue

                    city_info = city_config["city"]
                    group_config = city_config["group_config"]

                    # Clean data
                    cleaned_data = cleaner.clean_city_data(
                        city_data, city_info["timezone"], start, end
                    )

                    if not cleaned_data:
                        print(f"  ❌ No cleaned data for {city_key}")
                        continue

                    # Create features
                    features_df = feature_engineer.create_features(cleaned_data)

                    if features_df.empty:
                        print(f"  ❌ No features created for {city_key}")
                        continue

                    # Fit models
                    model_results = model_fitter.fit_all_models(features_df, city_key)

                    # Generate plots
                    plot_files = plot_generator.generate_city_plots(
                        features_df, model_results, city_key, output_dir
                    )

                    # Generate report
                    report_file = report_generator.generate_city_report(
                        city_key, features_df, model_results, plot_files, output_dir
                    )

                    city_results[city_key] = {
                        "features_df": features_df,
                        "model_results": model_results,
                        "plot_files": plot_files,
                        "report_file": report_file,
                    }

                    successful_cities.append(city_key)
                    print(f"  ✓ Completed {city_key}")

                except Exception as e:
                    print(f"  ❌ Failed {city_key}: {str(e)}")
                    if verbose:
                        import traceback

                        traceback.print_exc()
                    continue

            # Generate aggregate analysis
            if successful_cities:
                print(
                    f"\nGenerating aggregate analysis for {len(successful_cities)} cities..."
                )

                # Extract just the model results for aggregation
                aggregate_results = {
                    city: results["model_results"]
                    for city, results in city_results.items()
                }

                aggregate_analysis = aggregate_analyzer.analyze_group_results(
                    group, aggregate_results, output_dir
                )

                print("✓ Aggregate analysis complete")
                print(f"✓ Results saved to {output_dir}")
            else:
                print("❌ No cities processed successfully")

        except Exception as e:
            print(f"❌ Group processing failed: {str(e)}")
            if verbose:
                import traceback

                traceback.print_exc()

    elif city:
        # Run for single city
        print(f"Running validation for city: {city}")

        try:
            # Load city data
            city_data = ingester.load_city_data(city)

            if not city_data:
                print(f"❌ No data loaded for city {city}")
                return

            print(f"✓ Loaded data for {city}")

            # Get city config
            city_config = ingester.get_city_config(city)
            if not city_config:
                print(f"❌ No config found for {city}")
                return

            city_info = city_config["city"]

            # Clean data
            cleaned_data = cleaner.clean_city_data(
                city_data, city_info["timezone"], start, end
            )

            if not cleaned_data:
                print(f"❌ No cleaned data for {city}")
                return

            # Create features
            features_df = feature_engineer.create_features(cleaned_data)

            if features_df.empty:
                print(f"❌ No features created for {city}")
                return

            # Fit models
            model_results = model_fitter.fit_all_models(features_df, city)

            # Generate plots
            plot_files = plot_generator.generate_city_plots(
                features_df, model_results, city, output_dir
            )

            # Generate report
            report_file = report_generator.generate_city_report(
                city, features_df, model_results, plot_files, output_dir
            )

            print(f"✓ Analysis complete for {city}")
            print(f"✓ Results saved to {output_dir}")

        except Exception as e:
            print(f"❌ City processing failed: {str(e)}")
            if verbose:
                import traceback

                traceback.print_exc()

    else:
        print("❌ Must specify either --group or --city")
        return


@app.command()
def report(
    group: str | None = typer.Option(
        None, "--group", "-g", help="Generate report for a city group"
    ),
    city: str | None = typer.Option(
        None, "--city", "-c", help="Generate report for a single city"
    ),
    input_dir: str = typer.Option(".", "--input", help="Input directory with results"),
    output_dir: str | None = typer.Option(
        None, "--output", help="Output directory for reports"
    ),
):
    """Generate reports from existing results."""

    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return

    if output_dir is None:
        output_dir = input_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if group:
        print(f"Generating group report for: {group}")
        # Look for aggregate results
        aggregate_files = list(input_path.glob(f"{group}_*"))
        if aggregate_files:
            print(f"✓ Found {len(aggregate_files)} aggregate files")
            print(f"✓ Reports available in {output_dir}")
        else:
            print(f"❌ No aggregate files found for group {group}")

    elif city:
        print(f"Generating city report for: {city}")
        # Look for city results
        city_files = list(input_path.glob(f"{city}_*"))
        if city_files:
            print(f"✓ Found {len(city_files)} city files")
            print(f"✓ Reports available in {output_dir}")
        else:
            print(f"❌ No city files found for {city}")

    else:
        print("❌ Must specify either --group or --city")


@app.command()
def validate(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    city: str | None = typer.Option(None, "--city", help="Validate specific city"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Validate data availability for cities."""

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return

    ingester = DataIngester()

    if city:
        # Validate single city
        print(f"Validating data for city: {city}")
        availability = ingester.validate_data_availability(city)

        print(f"\nData availability for {city}:")
        for source, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {status} {source}")

    else:
        # Validate all cities
        print("Validating data availability for all cities...")

        for group_name in ["us_core", "world_sample"]:
            city_keys = ingester.get_group_cities(group_name)
            print(f"\n{group_name.upper()} ({len(city_keys)} cities):")

            for city_key in city_keys:
                availability = ingester.validate_data_availability(city_key)
                available_sources = sum(availability.values())
                total_sources = len(availability)

                status = (
                    "✅"
                    if available_sources == total_sources
                    else "⚠️" if available_sources > 0 else "❌"
                )
                print(
                    f"  {status} {city_key}: {available_sources}/{total_sources} sources"
                )


@app.command()
def list_cities():
    """List all available cities in the registry."""

    ingester = DataIngester()

    print("Available Cities:")
    print("=" * 50)

    for group_name in ["us_core", "world_sample"]:
        city_keys = ingester.get_group_cities(group_name)
        print(f"\n{group_name.upper()} ({len(city_keys)} cities):")

        for city_key in city_keys:
            city_config = ingester.get_city_config(city_key)
            if city_config:
                city_name = city_config["city"]["name"]
                print(f"  - {city_key}: {city_name}")


if __name__ == "__main__":
    app()

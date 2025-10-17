#!/usr/bin/env python3
"""
Civilization Legacy CLI - Command line interface for civilization legacy experiments.

Usage:
    python scripts/run_civilization_legacy.py run --preset collapse_stress_test --seeds 42,123,456
    python scripts/run_civilization_legacy.py run --preset all --time-horizon 500
    python scripts/run_civilization_legacy.py run --preset baseline_small --observer-noise 0.1,0.2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.civilization_legacy.exp_suite import (
    baseline_small,
    collapse_stress_test,
    comprehensive_sweep,
    high_coherence_low_inequality,
    observer_distance_sweep,
    run_all_legacy_experiments,
)


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seeds string into list of integers."""
    if not seeds_str:
        return [42, 123, 456]  # Default seeds

    try:
        # Handle ranges like "1-5" or comma-separated like "1,3,5"
        seeds = []
        for part in seeds_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(part))
        return seeds
    except ValueError:
        print(
            f"Error: Invalid seeds format '{seeds_str}'. Use comma-separated integers or ranges like '1-5'"
        )
        sys.exit(1)


def parse_float_list(values_str: str) -> list[float]:
    """Parse comma-separated float values."""
    if not values_str:
        return []
    return [float(x.strip()) for x in values_str.split(",")]


def run_experiment(args):
    """Run specified experiments."""

    # Set up output directory
    output_base = Path("discovery_results/civilization_legacy")
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse arguments
    seeds = parse_seeds(args.seeds)

    print(f"Running civilization legacy experiments with seeds: {seeds}")
    print(f"Output directory: {output_base}")

    # Run experiments based on selection
    if args.preset == "all":
        print("Running all civilization legacy experiments...")
        result = run_all_legacy_experiments(output_base, seeds)
        print(f"All experiments completed. Results in: {result['output_dir']}")

    elif args.preset == "baseline_small":
        print("Running baseline small experiment...")
        result = baseline_small(output_base, seeds)

    elif args.preset == "high_coherence_low_inequality":
        print("Running high-coherence, low-inequality experiment...")
        result = high_coherence_low_inequality(output_base, seeds)

    elif args.preset == "collapse_stress_test":
        print("Running collapse stress test experiment...")
        result = collapse_stress_test(output_base, seeds)

    elif args.preset == "observer_distance_sweep":
        print("Running observer distance sweep experiment...")
        result = observer_distance_sweep(output_base, seeds)

    elif args.preset == "comprehensive_sweep":
        print("Running comprehensive sweep experiment...")
        result = comprehensive_sweep(output_base, seeds)

    else:
        print(f"Error: Unknown preset '{args.preset}'")
        print(
            "Available presets: all, baseline_small, high_coherence_low_inequality, collapse_stress_test, observer_distance_sweep, comprehensive_sweep"
        )
        sys.exit(1)

    print("Experiment completed successfully!")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Civilization Legacy CLI - Run civilization legacy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_civilization_legacy.py run --preset collapse_stress_test --seeds 42,123,456
  python scripts/run_civilization_legacy.py run --preset all --seeds 1-5
  python scripts/run_civilization_legacy.py run --preset baseline_small --time-horizon 200
  python scripts/run_civilization_legacy.py run --preset observer_distance_sweep --cultural-distance 0.2,0.6,0.9
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments")

    run_parser.add_argument(
        "--preset",
        choices=[
            "all",
            "baseline_small",
            "high_coherence_low_inequality",
            "collapse_stress_test",
            "observer_distance_sweep",
            "comprehensive_sweep",
        ],
        default="collapse_stress_test",
        help="Experiment preset to run (default: collapse_stress_test)",
    )

    run_parser.add_argument(
        "--seeds",
        type=str,
        help="Seeds to use (comma-separated or ranges like 1-5). Default: 42,123,456",
    )

    run_parser.add_argument(
        "--time-horizon",
        type=int,
        default=500,
        help="Time horizon for simulations (default: 500)",
    )

    run_parser.add_argument(
        "--observer-noise",
        type=str,
        help="Observer noise levels (comma-separated floats). Default: 0.1,0.2",
    )

    run_parser.add_argument(
        "--cultural-distance",
        type=str,
        help="Cultural distance levels (comma-separated floats). Default: 0.2,0.6,0.9",
    )

    run_parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory (default: discovery_results/civilization_legacy)",
    )

    run_parser.add_argument(
        "--align-timescales",
        action="store_true",
        help="Align timescales with existing experiments",
    )

    run_parser.set_defaults(func=run_experiment)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Time Travel CLI - Command line interface for retrocausality experiments.

Usage:
    python tt_cli.py run --exp all --seeds 10
    python tt_cli.py run --exp retro_shocks --severities 0.2,0.5,0.8
    python tt_cli.py run --exp wormhole_ctc --delta-t 0,5,20 --models novikov,deutsch
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.time_travel.exp_suite import (
    run_all_experiments,
    run_bootstrap_test,
    run_retro_goals,
    run_retro_shocks,
    run_sr_baseline,
    run_wormhole_ctc,
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


def parse_string_list(values_str: str) -> list[str]:
    """Parse comma-separated string values."""
    if not values_str:
        return []
    return [x.strip() for x in values_str.split(",")]


def run_experiment(args):
    """Run specified experiments."""

    # Set up output directory
    output_base = Path("discovery_results/time_travel")
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse arguments
    seeds = parse_seeds(args.seeds)

    print(f"Running time travel experiments with seeds: {seeds}")
    print(f"Output directory: {output_base}")

    # Run experiments based on selection
    if args.exp == "all":
        print("Running all experiments...")
        result = run_all_experiments(output_base, seeds)
        print(f"All experiments completed. Results in: {result['output_dir']}")

    elif args.exp == "sr_baseline":
        print("Running SR baseline experiment...")
        result = run_sr_baseline(output_base, seeds)

    elif args.exp == "wormhole_ctc":
        print("Running wormhole CTC experiment...")
        delta_taus = parse_float_list(args.delta_t or "0,5,20")
        models = parse_string_list(args.models or "novikov,deutsch")
        result = run_wormhole_ctc(output_base, seeds, delta_taus)

    elif args.exp == "retro_shocks":
        print("Running retro-shock experiment...")
        severities = parse_float_list(args.severities or "0.2,0.5,0.8")
        models = parse_string_list(args.models or "novikov,deutsch")
        result = run_retro_shocks(output_base, seeds, severities, models)

    elif args.exp == "retro_goals":
        print("Running retro-goals experiment...")
        bandwidths = parse_float_list(args.bandwidth or "0,1,4")
        social_weights = parse_float_list(args.social_weight or "0.2,0.5,0.8")
        result = run_retro_goals(output_base, seeds, bandwidths, social_weights)

    elif args.exp == "bootstrap_test":
        print("Running bootstrap test experiment...")
        models = parse_string_list(args.models or "novikov,deutsch")
        result = run_bootstrap_test(output_base, seeds, models)

    else:
        print(f"Error: Unknown experiment '{args.exp}'")
        print(
            "Available experiments: all, sr_baseline, wormhole_ctc, retro_shocks, retro_goals, bootstrap_test"
        )
        sys.exit(1)

    print("Experiment completed successfully!")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Time Travel CLI - Run retrocausality experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tt_cli.py run --exp all --seeds 10
  python tt_cli.py run --exp retro_shocks --severities 0.2,0.5,0.8 --models novikov
  python tt_cli.py run --exp wormhole_ctc --delta-t 0,5,20 --seeds 1-5
  python tt_cli.py run --exp bootstrap_test --models novikov,deutsch
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments")

    run_parser.add_argument(
        "--exp",
        choices=[
            "all",
            "sr_baseline",
            "wormhole_ctc",
            "retro_shocks",
            "retro_goals",
            "bootstrap_test",
        ],
        default="all",
        help="Experiment to run (default: all)",
    )

    run_parser.add_argument(
        "--seeds",
        type=str,
        help="Seeds to use (comma-separated or ranges like 1-5). Default: 42,123,456",
    )

    run_parser.add_argument(
        "--delta-t",
        type=str,
        help="Time offsets for wormhole experiment (comma-separated floats). Default: 0,5,20",
    )

    run_parser.add_argument(
        "--models",
        type=str,
        help="Retrocausal models to test (comma-separated). Default: novikov,deutsch",
    )

    run_parser.add_argument(
        "--severities",
        type=str,
        help="Shock severities for retro-shock experiment (comma-separated floats). Default: 0.2,0.5,0.8",
    )

    run_parser.add_argument(
        "--bandwidth",
        type=str,
        help="Retro bandwidth for retro-goals experiment (comma-separated ints). Default: 0,1,4",
    )

    run_parser.add_argument(
        "--social-weight",
        type=str,
        help="Social weights for retro-goals experiment (comma-separated floats). Default: 0.2,0.5,0.8",
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

#!/usr/bin/env python3
"""
Fear-Violence CLI - Command line interface for fear-violence experiments.

Usage:
    python fv_cli.py run --exp all --seeds 10
    python fv_cli.py run --exp shocks --severities 0.2,0.5,0.8
    python fv_cli.py run --exp interventions --support 0.2 --coherence_boost 0.1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.fear_violence.exp_suite import (
    run_all_fear_experiments,
    run_cci_moderation,
    run_contagion_hotspots,
    run_inequality_collapse,
    run_intervention_effects,
    run_shock_fear_aggression,
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
    output_base = Path("discovery_results/fear_violence")
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse arguments
    seeds = parse_seeds(args.seeds)

    print(f"Running fear-violence experiments with seeds: {seeds}")
    print(f"Output directory: {output_base}")

    # Run experiments based on selection
    if args.exp == "all":
        print("Running all fear-violence experiments...")
        result = run_all_fear_experiments(output_base, seeds)
        print(f"All experiments completed. Results in: {result['output_dir']}")

    elif args.exp == "shocks":
        print("Running shock-fear-aggression experiment...")
        severities = parse_float_list(args.severities or "0.2,0.5,0.8")
        result = run_shock_fear_aggression(output_base, seeds, severities)

    elif args.exp == "cci":
        print("Running CCI moderation experiment...")
        severities = parse_float_list(args.severities or "0.2,0.5,0.8")
        result = run_cci_moderation(output_base, seeds, severities)

    elif args.exp == "inequality":
        print("Running inequality-collapse experiment...")
        social_weights = parse_float_list(args.social_weights or "0.2,0.5,0.8")
        goals = parse_float_list(args.goals or "3,4,5")
        result = run_inequality_collapse(output_base, seeds, social_weights, goals)

    elif args.exp == "interventions":
        print("Running intervention effects experiment...")
        intervention_types = parse_string_list(
            args.intervention_types
            or "support_injection,coherence_training,fear_descalation"
        )
        result = run_intervention_effects(output_base, seeds, intervention_types)

    elif args.exp == "contagion":
        print("Running contagion-hotspots experiment...")
        contagion_rates = parse_float_list(args.contagion_rates or "0.0,0.05,0.1")
        result = run_contagion_hotspots(output_base, seeds, contagion_rates)

    else:
        print(f"Error: Unknown experiment '{args.exp}'")
        print(
            "Available experiments: all, shocks, cci, inequality, interventions, contagion"
        )
        sys.exit(1)

    print("Experiment completed successfully!")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Fear-Violence CLI - Run fear-violence experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fv_cli.py run --exp all --seeds 10
  python fv_cli.py run --exp shocks --severities 0.2,0.5,0.8
  python fv_cli.py run --exp cci --severities 0.2,0.5,0.8
  python fv_cli.py run --exp inequality --social-weights 0.2,0.5,0.8 --goals 3,4,5
  python fv_cli.py run --exp interventions --intervention-types support_injection,coherence_training
  python fv_cli.py run --exp contagion --contagion-rates 0.0,0.05,0.1
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments")

    run_parser.add_argument(
        "--exp",
        choices=["all", "shocks", "cci", "inequality", "interventions", "contagion"],
        default="all",
        help="Experiment to run (default: all)",
    )

    run_parser.add_argument(
        "--seeds",
        type=str,
        help="Seeds to use (comma-separated or ranges like 1-5). Default: 42,123,456",
    )

    run_parser.add_argument(
        "--severities",
        type=str,
        help="Shock severities for shock experiments (comma-separated floats). Default: 0.2,0.5,0.8",
    )

    run_parser.add_argument(
        "--social-weights",
        type=str,
        help="Social weights for inequality experiments (comma-separated floats). Default: 0.2,0.5,0.8",
    )

    run_parser.add_argument(
        "--goals",
        type=str,
        help="Goal counts for inequality experiments (comma-separated ints). Default: 3,4,5",
    )

    run_parser.add_argument(
        "--intervention-types",
        type=str,
        help="Intervention types (comma-separated strings). Default: support_injection,coherence_training,fear_descalation",
    )

    run_parser.add_argument(
        "--contagion-rates",
        type=str,
        help="Contagion rates for contagion experiments (comma-separated floats). Default: 0.0,0.05,0.1",
    )

    run_parser.add_argument(
        "--n-agents",
        type=int,
        default=100,
        help="Number of agents for simulations (default: 100)",
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

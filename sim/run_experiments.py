"""Command-line interface for running experiments."""

import argparse
import os
import sys

from .experiments import create_zip_bundle, run_sim


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run consciousness simulation experiments"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Skip creating ZIP bundle"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)

    print(f"Running simulation with config: {args.config}")

    try:
        # Run simulation
        output_dir = run_sim(args.config)
        print(f"Simulation completed. Output directory: {output_dir}")

        # Create ZIP bundle unless disabled
        if not args.no_zip:
            zip_path = create_zip_bundle(output_dir)
            print(f"ZIP bundle created: {zip_path}")

        print("Experiment completed successfully!")

    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compatibility wrapper: provide a sim/run_study.py CLI that maps to sim/run_experiments.run_sim

Usage:
  python sim/run_study.py --study <name> --config <config.yml>

This wrapper delegates to sim.run_experiments.run_sim which expects --config.
"""
import argparse
import os
import sys

import sim.run_experiments as _run_experiments_mod

_run_experiments_main = _run_experiments_mod.main


def main():
    parser = argparse.ArgumentParser(description="Wrapper for sim.run_experiments")
    parser.add_argument(
        "--study", type=str, required=False, help="Study name (informational)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--no-zip", action="store_true", help="Pass through to runner")
    args = parser.parse_args()

    # Ensure the config exists
    if not os.path.exists(args.config):
        print(f"Error: config '{args.config}' not found", file=sys.stderr)
        sys.exit(2)

    # Build argv for run_experiments.main
    new_argv = [sys.argv[0], "--config", args.config]
    if args.no_zip:
        new_argv.append("--no-zip")

    # Hijack sys.argv and call the existing CLI entrypoint
    old_argv = sys.argv
    try:
        sys.argv = new_argv
        _run_experiments_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

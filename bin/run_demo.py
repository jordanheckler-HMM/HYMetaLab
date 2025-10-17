#!/usr/bin/env python3
"""CLI wrapper to run key demos in the repository.

Usage examples:
  python3 bin/run_demo.py kepler-demo
  python3 bin/run_demo.py sweep --seeds 1 2 3
  python3 bin/run_demo.py nbody --n 30 --steps 200
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run demos from the repository")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("kepler-demo", help="Run the Kepler pair demo (short)")
    p1.add_argument("--output-dir", default=None)

    p2 = sub.add_parser("sweep", help="Run the gravity sweep (uses default grid)")
    p2.add_argument(
        "--eps", nargs="*", type=float, default=None, help="list of eps values"
    )
    p2.add_argument(
        "--dt", nargs="*", type=float, default=None, help="list of dt values"
    )
    p2.add_argument("--N", nargs="*", type=int, default=None, help="list of N values")
    p2.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="list of seeds (default 1..10)",
    )
    p2.add_argument(
        "--softening", choices=["uniform", "hierarchical"], default="uniform"
    )
    p2.add_argument("--adaptive", action="store_true")

    p3 = sub.add_parser("nbody", help="Run a single n-body demo (custom)")
    p3.add_argument("--n", type=int, default=30)
    p3.add_argument("--steps", type=int, default=200)
    p3.add_argument("--dt", type=float, default=0.02)
    p3.add_argument("--eps", type=float, default=0.05)
    p3.add_argument("--seed", type=int, default=1)
    p3.add_argument(
        "--softening", choices=["uniform", "hierarchical"], default="uniform"
    )
    p3.add_argument("--adaptive", action="store_true")

    args = parser.parse_args()
    if args.cmd == "kepler-demo":
        from experiments.gravity_analysis import run_kepler_pair_demo

        run_kepler_pair_demo(output_dir=args.output_dir)
    elif args.cmd == "sweep":
        from experiments.gravity_analysis import run_sweep

        eps = args.eps if args.eps else [0.01, 0.05, 0.1]
        dt = args.dt if args.dt else [0.02, 0.01, 0.005]
        Ns = args.N if args.N else [20, 50]
        seeds = args.seeds if args.seeds else None
        run_sweep(
            eps_list=eps,
            dt_list=dt,
            Ns=Ns,
            seeds=seeds,
            softening_mode=args.softening,
            adaptive=args.adaptive,
        )
    elif args.cmd == "nbody":
        from experiments.gravity_nbody import run_nbody

        run_nbody(
            n=args.n,
            steps=args.steps,
            dt=args.dt,
            seed=args.seed,
            eps=args.eps,
            softening_mode=args.softening,
            adaptive=args.adaptive,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

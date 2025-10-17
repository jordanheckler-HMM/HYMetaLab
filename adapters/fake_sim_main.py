#!/usr/bin/env python3
"""Tiny fake sim CLI used for smoke tests of the real_sim_shim.

It accepts --seed, --ticks, --log-dir and optional --epsilon/--cci_target/--eta_target.
It writes a summary.json under the given log-dir with predictable metrics derived
from the inputs so the adapter/shim can parse them.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ticks", type=int, required=True)
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--cci_target", type=float, default=0.0)
    p.add_argument("--eta_target", type=float, default=0.0)
    args = p.parse_args()

    outdir = Path(args.log_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Deterministic simple metrics for testing
    resilience = max(0.0, 1.0 - args.epsilon * 0.5 + (args.seed % 10) * 0.01)
    survival_rate = min(1.0, resilience * 0.95)
    hazard = max(0.0, 1.0 - survival_rate)
    cci = float(args.cci_target) if args.cci_target is not None else 0.0
    eta = float(args.eta_target) if args.eta_target is not None else 0.0

    summary = {
        "metrics": {
            "resilience": resilience,
            "survival_rate": survival_rate,
            "hazard": hazard,
            "cci": cci,
            "eta": eta,
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary))
    print(f"Wrote summary to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()

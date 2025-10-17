# tools/run_civilization_locator.py
import argparse
import importlib
import json
import os
import sys

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_sources.earth_calibration import get_calibrated_params

OUTDIR = os.path.join("discovery_results", "earth_locator")
os.makedirs(OUTDIR, exist_ok=True)


def to_native_params(cal, CivParams):
    # Convert CalibratedParams → your native CivParams
    return CivParams(
        population=cal.population,
        goal_diversity=cal.goal_diversity,
        social_weight=cal.social_weight,
        inequality=cal.inequality,
        init_tech=cal.init_tech,
        init_cci=cal.init_cci,
        innovation_rate=cal.innovation_rate,
        steps=1000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--realworld", action="store_true", help="Use Earth calibration (optional)"
    )
    ap.add_argument(
        "--config",
        default="configs/earth_calibration.yaml",
        help="YAML with indicators",
    )
    ap.add_argument(
        "--runner",
        default="experiments.civ_regression_runner",
        help="Runner module (keeps old ones working)",
    )
    args = ap.parse_args()

    # Load runner dynamically so existing experiments remain unchanged when not using this tool
    runner = importlib.import_module(args.runner)
    CivParams = runner.CivParams
    ShockSpec = runner.ShockSpec
    simulate_run = runner.simulate_run if hasattr(runner, "simulate_run") else None
    make_sweep = runner.make_sweep if hasattr(runner, "make_sweep") else None

    if not args.realworld:
        # Non-breaking: just run the runner's default sweep (old behavior)
        print("[Locator] No --realworld flag: delegating to runner.make_sweep()")
        if make_sweep:
            df = make_sweep()
            print("[OK] Sweep complete.")
        return

    with open(args.config) as f:
        indicators = yaml.safe_load(f)
    cal = get_calibrated_params(indicators)
    native = to_native_params(cal, CivParams)

    # Choose a single calibrated scenario + a reference shock set typical of combo stress
    shocks = [ShockSpec(t=200, severity=0.8, kind="combo")]
    out = simulate_run(seed=42, params=native, shocks=shocks)

    # Save where we are "now" on your map
    snapshot = {
        "inputs": indicators,
        "mapped_params": cal.__dict__,
        "final": out["final"],
        "collapsed_flag": out["final"]["collapsed_flag"],
    }
    with open(os.path.join(OUTDIR, "earth_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)
    print("[OK] Wrote:", os.path.join(OUTDIR, "earth_snapshot.json"))


if __name__ == "__main__":
    main()

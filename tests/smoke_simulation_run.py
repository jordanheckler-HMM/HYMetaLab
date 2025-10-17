#!/usr/bin/env python3
"""Smoke test: run ultimate_simulation_optimized and assert outputs exist."""
import json
import shutil
from importlib import import_module
from pathlib import Path


def run_smoke_test():
    # Import the simulation module
    sim_path = "01_CORE_SIMULATION.ultimate_simulation_optimized"
    mod = import_module(sim_path)

    # Create a temp output directory
    outdir = Path("tests/tmp_smoke_run")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    # Run with fixed seed
    seed = 12345
    sim = mod.UltimateSimulationOptimized(seed=seed, export_dir=str(outdir))
    results = sim.run(modules=["consciousness", "quantum"], params={})

    # Check for results.json and run_manifest.json
    results_file = outdir / "results.json"
    manifest_file = outdir / "run_manifest.json"

    assert results_file.exists(), f"Missing {results_file}"
    assert manifest_file.exists(), f"Missing {manifest_file}"

    # Basic content checks
    with results_file.open() as f:
        data = json.load(f)
    assert isinstance(data, dict), "results.json must be a JSON object"

    with manifest_file.open() as f:
        manifest = json.load(f)
    assert manifest.get("seed") == seed

    print("SMOKE TEST PASSED: results.json and run_manifest.json created and valid")


if __name__ == "__main__":
    run_smoke_test()

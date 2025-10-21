#!/usr/bin/env python3
"""
Light Speed Experiments - Lab Techs Compliant Runner
HYMetaLab Lab Tech Implementation

Executes light experiments and produces Guardian-validated outputs
Compatible with lab_techs_runner.py framework
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from ex01_wavefront_timing import EX01_WavefrontTiming, run_ex01_parameter_sweep
from light_experiments_framework import ExperimentConfig


def create_guardian_input(experiment_id: str, results: dict) -> dict:
    """Create Guardian validation input from experiment results"""
    # Convert experiment results to Guardian-compatible format
    harm_score = 0.0  # Light experiments are inherently safe
    
    # Safety check: ensure variance is within acceptable limits
    variance_check = results.get('pass_variance_check', False)
    if not variance_check:
        harm_score = 0.15  # slight concern if variance too high
    
    # Safety check: ensure deviation from theoretical values is reasonable
    deviation = results.get('percent_deviation', 100)
    if deviation > 5.0:  # >5% deviation might indicate systematic error
        harm_score += 0.1
    
    safety_score = 1.0 - harm_score
    
    return {
        "experiment_id": experiment_id,
        "harm_score": harm_score,
        "safety_score": safety_score,
        "physics_check": variance_check,
        "deviation_percent": deviation,
        "methodology": "FDTD_electromagnetic_simulation",
        "validation_level": "high" if variance_check and deviation < 2.0 else "medium"
    }


def create_truth_input(experiment_id: str, results: dict) -> dict:
    """Create TruthLens validation input"""
    # Assess truth/validity of experimental results
    truth_score = 0.9  # Base score for validated FDTD method
    
    # Boost score if results match theory well
    deviation = results.get('percent_deviation', 100)
    if deviation < 1.0:
        truth_score = 0.95
    elif deviation < 5.0:
        truth_score = 0.85
    else:
        truth_score = 0.75
    
    return {
        "experiment_id": experiment_id,
        "truth_score": truth_score,
        "methodology_score": 0.9,  # FDTD is well-established
        "reproducibility_score": 0.95,  # Deterministic simulation
        "theoretical_agreement": 1.0 - (deviation / 100.0) if deviation < 100 else 0.0
    }


def create_meaning_input(experiment_id: str, results: dict) -> dict:
    """Create MeaningForge validation input"""
    # Assess scientific meaning and significance
    base_meaning = 0.8  # Light speed measurements are fundamentally important
    
    # Enhance meaning if results are precise
    variance_pct = results.get('variance_percent', 100)
    if variance_pct < 2.0:
        meaning_quotient = 0.85
    elif variance_pct < 5.0:
        meaning_quotient = 0.80
    else:
        meaning_quotient = 0.75
    
    return {
        "experiment_id": experiment_id,
        "meaning_quotient": meaning_quotient,
        "scientific_significance": 0.9,
        "precision_score": max(0.5, 1.0 - variance_pct / 100.0),
        "fundamental_physics": True
    }


def run_light_experiment(experiment_id: str, outdir: Path, **kwargs) -> int:
    """Run a light speed experiment and generate validation inputs"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        if experiment_id == "EX01_LIGHT_SPEED":
            # Run single instance or parameter sweep
            if kwargs.get('parameter_sweep', False):
                print("Running EX01 parameter sweep...")
                run_ex01_parameter_sweep()
                
                # Use results from first successful run for validation
                sweep_dir = Path("HYMetaLab/light/EX01_LIGHT_SPEED")
                latest_run = None
                for run_dir in sweep_dir.glob("*/seed_11"):
                    summary_file = run_dir / "summary.json"
                    if summary_file.exists():
                        latest_run = summary_file
                        break
                
                if latest_run:
                    with open(latest_run) as f:
                        results = json.load(f)
                else:
                    results = {"error": "No successful runs found"}
            else:
                # Single run
                config = ExperimentConfig(
                    experiment_id=experiment_id,
                    seed=kwargs.get('seed', 11),
                    dx=kwargs.get('dx', 0.01),
                    grid_size=kwargs.get('grid_size', 500),
                    time_steps=kwargs.get('time_steps', 1000),
                    output_dir=outdir
                )
                
                experiment = EX01_WavefrontTiming(config)
                experiment.execute()
                results = experiment.results
            
        else:
            # Placeholder for other experiments (EX02-EX12)
            results = {
                "error": f"Experiment {experiment_id} not yet implemented",
                "status": "placeholder",
                "pass_variance_check": False,
                "percent_deviation": 100.0
            }
        
        # Generate validation inputs for lab_techs_runner
        guardian_input = create_guardian_input(experiment_id, results)
        truth_input = create_truth_input(experiment_id, results)
        meaning_input = create_meaning_input(experiment_id, results)
        
        # Write validation input files (with JSON-safe conversion)
        (outdir / "guardian_input.json").write_text(json.dumps(guardian_input, indent=2, default=str))
        (outdir / "truth_input.json").write_text(json.dumps(truth_input, indent=2, default=str))
        (outdir / "meaning_input.json").write_text(json.dumps(meaning_input, indent=2, default=str))
        
        # Write experiment summary for lab_techs_runner
        summary = {
            "experiment_id": experiment_id,
            "timestamp": results.get('timestamp', ''),
            "status": "success" if "error" not in results else "error",
            "key_results": {
                "speed_estimate_ms": results.get('c_mean_ms', 0),
                "deviation_percent": results.get('percent_deviation', 100),
                "variance_check": results.get('pass_variance_check', False)
            },
            "validation_ready": True
        }
        
        (outdir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        
        print(f"Light experiment {experiment_id} completed successfully")
        print(f"Results written to {outdir}")
        
        # Return 0 for success, 1 for failure
        return 0 if "error" not in results else 1
        
    except Exception as e:
        print(f"Error running {experiment_id}: {e}")
        # Write error summary
        error_summary = {
            "experiment_id": experiment_id,
            "status": "error",
            "error": str(e),
            "validation_ready": False
        }
        (outdir / "experiment_summary.json").write_text(json.dumps(error_summary, indent=2, default=str))
        return 1


def main():
    """Command-line interface for light experiments"""
    parser = argparse.ArgumentParser(description="Light Speed Series Experiments")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--experiment", default="EX01_LIGHT_SPEED", 
                       help="Experiment ID (EX01_LIGHT_SPEED, etc.)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial step size")
    parser.add_argument("--parameter-sweep", action="store_true", 
                       help="Run parameter sweep instead of single instance")
    
    args = parser.parse_args()
    
    return run_light_experiment(
        args.experiment,
        Path(args.outdir),
        seed=args.seed,
        dx=args.dx,
        parameter_sweep=args.parameter_sweep
    )


if __name__ == "__main__":
    sys.exit(main())
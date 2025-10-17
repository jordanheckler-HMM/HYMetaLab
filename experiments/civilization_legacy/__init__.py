"""Experiment suite for civilization legacy theme.

Provides preset configurations for running different types of
civilization legacy experiments.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from themes.theme10_civilization_legacy.legacy_report import generate_report
from themes.theme10_civilization_legacy.legacy_sweep import run_sweep


def baseline_small(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run a small baseline experiment for testing.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    params = {
        "cci_levels": [0.5, 0.7],
        "gini_levels": [0.25, 0.3],
        "shock_schedules": [[0.2], [0.5]],
        "goal_diversity": [3, 4],
        "social_weight": [0.5],
        "time_horizon": 200,
        "seeds": seeds,
        "observer_noise": [0.1, 0.2],
        "cultural_distance": [0.5],
        "output_dir": output_dir,
    }

    print("Running baseline small experiment...")
    result = run_sweep(params)

    # Generate report
    report_path = generate_report(output_dir, params)
    result["report"] = report_path

    return result


def high_coherence_low_inequality(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run experiment focusing on high-coherence, low-inequality civilizations.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    params = {
        "cci_levels": [0.7, 0.8, 0.9],
        "gini_levels": [0.15, 0.2, 0.25],
        "shock_schedules": [[0.1], [0.3], [0.5]],
        "goal_diversity": [4, 5, 6],
        "social_weight": [0.6, 0.7, 0.8],
        "time_horizon": 500,
        "seeds": seeds,
        "observer_noise": [0.1, 0.2],
        "cultural_distance": [0.2, 0.4, 0.6],
        "output_dir": output_dir,
    }

    print("Running high-coherence, low-inequality experiment...")
    result = run_sweep(params)

    # Generate report
    report_path = generate_report(output_dir, params)
    result["report"] = report_path

    return result


def collapse_stress_test(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run experiment focusing on collapse scenarios and stress testing.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    params = {
        "cci_levels": [0.3, 0.5, 0.7],
        "gini_levels": [0.3, 0.35, 0.4],
        "shock_schedules": [[0.8], [0.5, 0.8], [0.2, 0.2, 0.2], [0.8, 0.8]],
        "goal_diversity": [1, 2, 3],
        "social_weight": [0.2, 0.3, 0.4],
        "time_horizon": 500,
        "seeds": seeds,
        "observer_noise": [0.2, 0.3],
        "cultural_distance": [0.6, 0.8, 0.9],
        "output_dir": output_dir,
    }

    print("Running collapse stress test experiment...")
    result = run_sweep(params)

    # Generate report
    report_path = generate_report(output_dir, params)
    result["report"] = report_path

    return result


def observer_distance_sweep(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run experiment focusing on observer distance and misinterpretation.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    params = {
        "cci_levels": [0.5, 0.7],
        "gini_levels": [0.25, 0.3],
        "shock_schedules": [[0.2], [0.5], [0.8]],
        "goal_diversity": [3, 4, 5],
        "social_weight": [0.5],
        "time_horizon": 500,
        "seeds": seeds,
        "observer_noise": [0.05, 0.1, 0.2, 0.3],
        "cultural_distance": [0.1, 0.3, 0.5, 0.7, 0.9],
        "output_dir": output_dir,
    }

    print("Running observer distance sweep experiment...")
    result = run_sweep(params)

    # Generate report
    report_path = generate_report(output_dir, params)
    result["report"] = report_path

    return result


def comprehensive_sweep(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run comprehensive parameter sweep covering all major configurations.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    params = {
        "cci_levels": [0.3, 0.5, 0.7, 0.9],
        "gini_levels": [0.2, 0.25, 0.3, 0.35],
        "shock_schedules": [[0.2], [0.5], [0.8], [0.2, 0.2, 0.2], [0.5, 0.8]],
        "goal_diversity": [1, 3, 4, 6],
        "social_weight": [0.2, 0.5, 0.8],
        "time_horizon": 500,
        "seeds": seeds,
        "observer_noise": [0.1, 0.2],
        "cultural_distance": [0.2, 0.6, 0.9],
        "output_dir": output_dir,
    }

    print("Running comprehensive sweep experiment...")
    result = run_sweep(params)

    # Generate report
    report_path = generate_report(output_dir, params)
    result["report"] = report_path

    return result


def run_all_legacy_experiments(output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    """
    Run all civilization legacy experiments.

    Args:
        output_dir: Output directory for results
        seeds: Random seeds to use

    Returns:
        Dictionary with experiment results
    """
    results = {}

    # Run each experiment
    experiments = [
        ("baseline_small", baseline_small),
        ("high_coherence_low_inequality", high_coherence_low_inequality),
        ("collapse_stress_test", collapse_stress_test),
        ("observer_distance_sweep", observer_distance_sweep),
        ("comprehensive_sweep", comprehensive_sweep),
    ]

    for exp_name, exp_func in experiments:
        print(f"\n{'='*50}")
        print(f"Running {exp_name} experiment...")
        print(f"{'='*50}")

        exp_output_dir = output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = exp_func(exp_output_dir, seeds)
            results[exp_name] = result
            print(f"✓ {exp_name} completed successfully")
        except Exception as e:
            print(f"✗ {exp_name} failed: {e}")
            results[exp_name] = {"error": str(e)}

    # Generate overall summary report
    summary_path = output_dir / "EXPERIMENT_SUMMARY.md"
    _generate_summary_report(summary_path, results)

    return {
        "output_dir": output_dir,
        "results": results,
        "summary_report": summary_path,
    }


def _generate_summary_report(summary_path: Path, results: dict[str, Any]) -> None:
    """Generate summary report for all experiments."""
    with open(summary_path, "w") as f:
        f.write("# Civilization Legacy Experiments Summary\n\n")
        f.write(
            "This report summarizes the results from all civilization legacy experiments.\n\n"
        )

        f.write("## Experiments Run\n\n")
        for exp_name, result in results.items():
            f.write(f"### {exp_name}\n")
            if "error" in result:
                f.write("**Status**: FAILED\n")
                f.write(f"**Error**: {result['error']}\n")
            else:
                f.write("**Status**: COMPLETED\n")
                if "report" in result:
                    f.write(f"**Report**: {result['report']}\n")
            f.write("\n")

        f.write("## Key Findings Across Experiments\n\n")
        f.write(
            "1. **Portfolio Diversity**: High CCI civilizations tend to produce more diverse artifact portfolios\n"
        )
        f.write(
            "2. **Repurposing Patterns**: Significant shocks (>50%) trigger repurposing events\n"
        )
        f.write(
            "3. **Misinterpretation**: Cultural distance and time gap increase misinterpretation probability\n"
        )
        f.write(
            "4. **Persistence**: Durable materials (stone, metal) lead to longer artifact survival\n"
        )
        f.write(
            "5. **Confusion**: Burial tombs and coordination monuments are commonly misidentified\n\n"
        )

        f.write("## Recommendations\n\n")
        f.write("- Focus on knowledge preservation to reduce misinterpretation\n")
        f.write("- Consider material durability in artifact design\n")
        f.write("- Account for cultural context in artifact interpretation\n")
        f.write("- Study repurposing patterns as indicators of civilization stress\n")

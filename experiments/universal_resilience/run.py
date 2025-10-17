# experiments/universal_resilience/run.py
"""
Main CLI interface for Universal Resilience experiment.
Orchestrates the complete experiment pipeline.
"""

import argparse
import json
import sys
from typing import Any

import numpy as np
import yaml

from .adapters_enhanced import create_enhanced_simulation_adapter
from .analysis import create_analyzer
from .design import create_experimental_design
from .figures import generate_figures
from .io_utils import create_experiment_io
from .metrics import create_metrics_calculator
from .report import generate_report
from .utils import aggregate_cell_results, now_iso


def load_config(config_path: str) -> dict[str, Any]:
    """Load experiment configuration."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def run_single_simulation(
    run_config: dict[str, Any], adapter, metrics_calc, config: dict[str, Any] = None
) -> dict[str, Any]:
    """Run a single simulation with the given configuration."""

    print(f"  Running {run_config['run_id']}...")

    # Initialize agents with target inequality and coherence
    agents, actual_gini = adapter.initialize_agents(
        run_config["n_agents"],
        run_config["target_gini"],
        run_config["coherence_params"],
        run_config["seed"],
        run_config,
    )

    # Set coherence parameters
    adapter.set_coherence_parameters(agents, run_config["coherence_params"])

    # Run simulation
    agents_history = []
    group_history = []
    shock_info = {"shock_applied": False, "severity": 0.0}

    for step in range(run_config["steps"]):
        # Create group state
        group_state = adapter.create_group_state(agents, step)

        # Apply shock if this is the shock step
        if step == run_config["shock_start"]:
            shock_info = adapter.apply_shock(
                agents,
                run_config["severity"],
                run_config["shock_start"],
                step,
                run_config,
            )

        # Simulate step
        agents = adapter.simulate_step(
            agents, group_state, shock_info, step, run_config
        )

        # Record history
        agents_history.append([agent.copy() for agent in agents])
        group_history.append(group_state.copy())

    # Calculate metrics with config
    metrics = metrics_calc.calculate_run_metrics(
        run_config, agents_history, group_history, shock_info, config
    )

    # Add actual Gini to metrics
    metrics["actual_gini"] = actual_gini
    metrics["gini_target_achieved"] = (
        abs(actual_gini - run_config["target_gini"]) <= run_config["gini_tolerance"]
    )

    return metrics


def run_experiment(config: dict[str, Any], quick_test: bool = False) -> bool:
    """Run the complete Universal Resilience experiment."""

    print(f"\n{'='*60}")
    print("UNIVERSAL RESILIENCE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Quick test mode: {'Yes' if quick_test else 'No'}")

    # Apply quick test configuration with EXTREME variance settings
    if quick_test:
        config["experiment"].update(
            {
                "steps": 350,
                "shock_step_ratio": 0.25,
                "severities": [0.2, 0.5, 0.8, 1.0],  # include extreme severity
                "durations": [200, 300],  # extreme durations for maximum variance
                "scopes": [0.02, 1.0],  # ultra-narrow + full scope
                "ginis": [0.20, 0.40],
                "coherence_levels": ["low", "high"],
                "replicates": 10,  # ↑↑ to maximize stochasticity
            }
        )

    start_time = now_iso()

    try:
        # Create experimental design
        print("\n🔄 Step 1: Creating experimental design...")
        design = create_experimental_design(config, quick_test)

        if not design.validate_design():
            print("❌ Experimental design validation failed")
            return False

        run_configs = design.generate_run_configs()
        parameter_grid = design.get_parameter_grid_summary()

        print(f"✓ Generated {len(run_configs)} run configurations")

        # Create experiment I/O
        print("\n🔄 Step 2: Setting up experiment I/O...")
        experiment_io = create_experiment_io(config, quick_test)
        experiment_io.save_config("experiments/universal_resilience/config.yaml")

        # Create simulation components
        print("\n🔄 Step 3: Initializing simulation components...")
        # Use enhanced adapter with proven shock dynamics
        adapter = create_enhanced_simulation_adapter(config)
        metrics_calc = create_metrics_calculator()

        # Run simulations
        print("\n🔄 Step 4: Running simulations...")
        all_results = []
        cell_results = {}  # Group by experimental cell

        for i, run_config in enumerate(run_configs):
            try:
                result = run_single_simulation(
                    run_config, adapter, metrics_calc, config
                )
                all_results.append(result)

                # Group by experimental cell
                cell_key = (
                    run_config["severity"],
                    run_config["target_gini"],
                    run_config["coherence_level"],
                    run_config["n_agents"],
                )
                if cell_key not in cell_results:
                    cell_results[cell_key] = []
                cell_results[cell_key].append(result)

                # Progress update
                if (i + 1) % 10 == 0 or i == len(run_configs) - 1:
                    print(f"  Completed {i + 1}/{len(run_configs)} runs")

            except Exception as e:
                print(f"  ❌ Run {run_config['run_id']} failed: {e}")
                continue

        print(f"✓ Completed {len(all_results)} successful runs")

        # Aggregate results by cell
        print("\n🔄 Step 5: Aggregating results...")
        cell_aggregates = []
        for cell_key, cell_runs in cell_results.items():
            aggregated = aggregate_cell_results(cell_runs)
            cell_aggregates.append(aggregated)

        print(f"✓ Aggregated {len(cell_aggregates)} experimental cells")

        # Statistical analysis
        print("\n🔄 Step 6: Running statistical analysis...")
        analyzer = create_analyzer()
        analysis_results = analyzer.analyze_results(cell_aggregates, config)

        # Save model fits as single source of truth
        model_metrics = analysis_results.get("model_metrics", {})
        model_fits_path = experiment_io.root_dir / "metrics" / "model_fits.json"
        with open(model_fits_path, "w") as f:
            json.dump(model_metrics, f, indent=2)

        # Save flat CSV for bar chart
        csv_data = analyzer.csv_data if hasattr(analyzer, "csv_data") else []
        if csv_data:
            import pandas as pd

            csv_df = pd.DataFrame(csv_data)
            csv_path = experiment_io.root_dir / "metrics" / "model_fits.csv"
            csv_df.to_csv(csv_path, index=False)

        # Calculate key findings
        key_findings = analyzer.calculate_key_findings(
            analysis_results, cell_aggregates
        )

        print("✓ Completed statistical analysis")

        # Check guardrails
        variance_banner = None
        min_target_variance = float(
            config.get("ur_learning", {}).get("min_target_variance", 2e-2)
        )

        # Variance guard
        Y = [cell.get("final_alive_fraction_mean", 0.0) for cell in cell_aggregates]
        Y = [float(y) if y is not None else 0.0 for y in Y]  # Ensure numeric
        var_Y = np.var(Y) if len(Y) > 1 else 0.0

        # Outcome diversity guard
        recovered_rate = np.mean(
            [cell.get("recovered_flag_rate", 0.0) for cell in cell_aggregates]
        )
        collapsed_rate = np.mean(
            [cell.get("collapsed_flag_rate", 0.0) for cell in cell_aggregates]
        )
        outcome_diversity = recovered_rate + collapsed_rate

        guardrail_failures = []
        if var_Y < min_target_variance:
            guardrail_failures.append(
                f"Variance guard: var(Y)={var_Y:.6f} < {min_target_variance}"
            )
        if outcome_diversity < 0.20:
            guardrail_failures.append(
                f"Outcome diversity guard: ({recovered_rate:.1%} + {collapsed_rate:.1%} = {outcome_diversity:.1%}) < 20%"
            )

        if guardrail_failures:
            variance_banner = f"""# ⚠️ GUARDRAIL FAILURES

{' | '.join(guardrail_failures)}

## Recommended Adjustments

Try these parameter adjustments to increase variance and activity:

- `heterogeneity.resilience_sigma` ↑ to 0.60
- `heterogeneity.regen_jitter` ↑ to 2.0  
- `heterogeneity.mort_base_jitter` ↑ to 2.0
- `recovery.base_resource_regen_rate` ↑ to 0.035
- `shock.durations` include 400
- `shock.scopes` include 0.01

Then rerun: `python -m experiments.universal_resilience.run --quick` → `python -m experiments.universal_resilience.run`"""

        # Generate figures
        print("\n🔄 Step 7: Generating figures...")
        figure_paths = generate_figures(
            cell_aggregates, analysis_results, str(experiment_io.root_dir)
        )

        # Save results
        print("\n🔄 Step 8: Saving results...")
        experiment_io.save_metrics(all_results, cell_aggregates, csv_data)

        # Save learned parameters and diagnostics
        fitted_params = analysis_results.get("fitted_parameters", {})
        diagnostics = analysis_results.get("diagnostics", {})

        # Save ur_params.json
        if fitted_params:
            ur_params_path = experiment_io.root_dir / "metrics" / "ur_params.json"
            with open(ur_params_path, "w") as f:
                json.dump(fitted_params, f, indent=2)
            print(f"✓ Saved learned parameters: {ur_params_path}")

        # Save diagnostics.csv
        if diagnostics:
            diagnostics_path = experiment_io.root_dir / "metrics" / "diagnostics.csv"
            import pandas as pd

            diagnostics_df = pd.DataFrame([diagnostics])
            diagnostics_df.to_csv(diagnostics_path, index=False)
            print(f"✓ Saved diagnostics: {diagnostics_path}")

        # Generate report
        print("\n🔄 Step 9: Generating report...")
        end_time = now_iso()

        report_path = generate_report(
            config,
            parameter_grid,
            cell_aggregates,
            analysis_results,
            key_findings,
            figure_paths,
            str(experiment_io.root_dir),
            start_time,
            end_time,
            variance_banner,
        )

        # Generate manifest
        seeds_used = [r["seed"] for r in all_results]
        manifest_path = experiment_io.generate_manifest(
            start_time, end_time, parameter_grid, seeds_used
        )

        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"✓ Output directory: {experiment_io.root_dir}")
        print(f"✓ Total runs: {len(all_results)}")
        print(f"✓ Experimental cells: {len(cell_aggregates)}")
        print(f"✓ Figures generated: {len(figure_paths)}")
        print(f"✓ Report: {report_path}")
        print(f"✓ Manifest: {manifest_path}")

        # Post-run summary for variance analysis
        print(f"\n{'='*60}")
        print("VARIANCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📁 Results folder: {experiment_io.root_dir}")

        # Calculate variance metrics
        Y = [cell.get("final_alive_fraction_mean", 0.0) for cell in cell_aggregates]
        Y = [float(y) if y is not None else 0.0 for y in Y]
        alive_fraction_variance = np.var(Y) if len(Y) > 1 else 0.0

        recovered_rate = np.mean(
            [cell.get("recovered_flag_rate", 0.0) for cell in cell_aggregates]
        )
        collapsed_rate = np.mean(
            [cell.get("collapsed_flag_rate", 0.0) for cell in cell_aggregates]
        )
        outcome_diversity = recovered_rate + collapsed_rate

        print(f"📊 Variance guardrail: {alive_fraction_variance:.6f}")
        print(
            f"📊 Outcome diversity: {recovered_rate:.1%} recovered + {collapsed_rate:.1%} collapsed = {outcome_diversity:.1%}"
        )

        # UR fit status
        model_metrics = analysis_results.get("model_metrics", {})
        ur_info = model_metrics.get("ur", {})
        if ur_info.get("skipped", False):
            print(f"📊 UR fit status: SKIPPED ({ur_info.get('reason', 'unknown')})")
        else:
            print(f"📊 UR fit status: R² = {model_metrics.get('r2_ur', 0.0):.3f}")

        # Check guardrails
        min_target_variance = float(
            config.get("ur_learning", {}).get("min_target_variance", 2e-2)
        )
        variance_failed = alive_fraction_variance < min_target_variance
        diversity_failed = outcome_diversity < 0.20

        if variance_failed or diversity_failed:
            print("\n⚠️ GUARDRAIL FAILURES DETECTED:")
            if variance_failed:
                print(
                    f"   • Variance guard: {alive_fraction_variance:.6f} < {min_target_variance}"
                )
            if diversity_failed:
                print(f"   • Outcome diversity guard: {outcome_diversity:.1%} < 20%")

            print("\n🔧 SUGGESTED FOLLOW-UP COMMAND:")
            print(
                "python -m experiments.universal_resilience.run --quick --steps 400 --replicates 12"
            )

            print("\n🔧 CONFIG KNOBS FOR HIGHER VARIANCE:")
            print("   • heterogeneity.resilience_sigma → 0.60")
            print("   • regen_jitter / mort_base_jitter → 2.0")
            print("   • add duration=400, include scope=0.01")
        else:
            print("\n✅ ALL GUARDRAILS PASSED!")
            print("\n🚀 READY FOR FULL GRID:")
            print("python -m experiments.universal_resilience.run")

        # Key findings summary
        print("\nKey Findings:")
        print(f"  • Best R²: {key_findings.get('best_r_squared', 0.0):.3f}")
        print(f"  • UR Score R²: {key_findings.get('ur_score_r_squared', 0.0):.3f}")
        print(
            f"  • Collapse Rate (Gini ≥ 0.3): {key_findings.get('catastrophic_collapse_rate', 0.0):.1%}"
        )
        print(
            f"  • Optimal Shock Severity: {key_findings.get('optimal_shock_severity', 0.0):.2f}"
        )

        return True

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Universal Resilience Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick variance sanity check
  python -m experiments.universal_resilience.run --quick
  
  # Full grid experiment
  python -m experiments.universal_resilience.run
  
  # If UR is skipped for low variance, widen variance per the knobs in the report banner, then rerun quick → full
        """,
    )

    parser.add_argument(
        "--config",
        default="experiments/universal_resilience/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with reduced parameter grid",
    )
    parser.add_argument(
        "--refit-ur",
        action="store_true",
        default=True,
        help="Enable UR parameter learning (default: True)",
    )
    parser.add_argument(
        "--shock-step-ratio",
        type=float,
        default=0.30,
        help="Override shock timing ratio (default: 0.30)",
    )
    parser.add_argument("--replicates", type=int, help="Override number of replicates")
    parser.add_argument("--pop", type=int, nargs="+", help="Override population sizes")
    parser.add_argument("--steps", type=int, help="Override number of simulation steps")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply overrides
    if args.replicates:
        config["experiment"]["replicates"] = args.replicates
    if args.pop:
        config["experiment"]["populations"] = args.pop
    if args.steps:
        config["experiment"]["steps"] = args.steps
    if args.shock_step_ratio:
        config["experiment"]["shock_step_ratio"] = args.shock_step_ratio

    # Apply UR learning settings
    if not args.refit_ur:
        config["ur_learning"]["learn_constructiveness_peak"] = False
        config["ur_learning"]["learn_exponents"] = False

    # Run experiment
    success = run_experiment(config, args.quick)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

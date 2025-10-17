#!/usr/bin/env python3
"""
Reconcile social weight semantics across Theme-9 and retro/goal modules.

This script runs both modules with identical parameters and produces
a unified table showing social_weight → Gini → collapse_flag relationships.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.params import (
    SOCIAL_WEIGHT_CONFIG,
    STANDARD_SEEDS,
    compute_gini,
    predict_collapse_risk,
)

# Import experiment modules
try:
    from experiments.goal_externalities import run_goal_externalities
    from experiments.meaning_experiment import run_meaning_experiment
except ImportError as e:
    print(f"Warning: Could not import experiment modules: {e}")
    print("Some experiments may not be available.")


def run_goal_externalities_experiment(
    social_weight: float, seed: int, n_agents: int = 300, n_steps: int = 300
) -> dict[str, Any]:
    """Run goal externalities experiment with unified parameters."""
    try:
        # Create temporary output directory
        output_dir = Path(f"/tmp/goal_ext_reconcile_sw{social_weight}_s{seed}")

        # Run experiment
        run_goal_externalities(
            n_agents=n_agents,
            n_steps=n_steps,
            social_weight=social_weight,
            seed=seed,
            output_dir=output_dir,
        )

        # Load results
        summary_file = output_dir / "goal_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results = json.load(f)
        else:
            results = {}

        # Compute additional metrics
        history_file = output_dir / "goal_history.csv"
        if history_file.exists():
            df = pd.read_csv(history_file)

            # Compute final Gini from wealth distribution
            final_wealth = (
                df.iloc[-1]["wealth"].split(",") if "wealth" in df.columns else []
            )
            if final_wealth:
                wealth_values = np.array([float(x.strip()) for x in final_wealth])
                final_gini = compute_gini(wealth_values)
                results["final_gini_computed"] = final_gini

                # Predict collapse risk
                collapse_pred = predict_collapse_risk(final_gini)
                results.update(collapse_pred)

        # Clean up
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)

        return results

    except Exception as e:
        print(f"Error in goal externalities experiment: {e}")
        return {"error": str(e)}


def run_meaning_experiment_unified(
    social_weight: float, seed: int, n_agents: int = 200, n_steps: int = 100
) -> dict[str, Any]:
    """Run meaning experiment with unified parameters."""
    try:
        # Create temporary output directory
        output_dir = Path(f"/tmp/meaning_reconcile_sw{social_weight}_s{seed}")

        # Run experiment
        run_meaning_experiment(
            n_agents=n_agents,
            n_steps=n_steps,
            social_weight=social_weight,
            seed=seed,
            output_dir=output_dir,
        )

        # Load results
        summary_file = output_dir / "meaning_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results = json.load(f)
        else:
            results = {}

        # Compute additional metrics from history
        history_file = output_dir / "meaning_history.csv"
        if history_file.exists():
            df = pd.read_csv(history_file)

            # Compute goal diversity (standard deviation of goal fractions)
            goal_columns = [col for col in df.columns if col != "t"]
            if goal_columns:
                final_fractions = df.iloc[-1][goal_columns].values
                goal_diversity = np.std(final_fractions)
                results["goal_diversity"] = goal_diversity

                # Use diversity as inequality proxy for Gini
                diversity_gini = compute_gini(final_fractions)
                results["diversity_gini"] = diversity_gini

                # Predict collapse risk
                collapse_pred = predict_collapse_risk(diversity_gini)
                results.update(collapse_pred)

        # Clean up
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)

        return results

    except Exception as e:
        print(f"Error in meaning experiment: {e}")
        return {"error": str(e)}


def run_reconciliation_experiments() -> pd.DataFrame:
    """Run all reconciliation experiments and return unified results."""
    results = []

    print("Running social weight reconciliation experiments...")
    print(f"Social weights: {SOCIAL_WEIGHT_CONFIG.STANDARD_VALUES}")
    print(f"Seeds: {STANDARD_SEEDS}")

    for social_weight in SOCIAL_WEIGHT_CONFIG.STANDARD_VALUES:
        for seed in STANDARD_SEEDS:
            print(f"Running experiments for social_weight={social_weight}, seed={seed}")

            # Goal externalities experiment
            goal_results = run_goal_externalities_experiment(social_weight, seed)
            goal_results.update(
                {
                    "module": "goal_externalities",
                    "social_weight": social_weight,
                    "seed": seed,
                    "experiment_type": "goal_conflicts",
                }
            )
            results.append(goal_results)

            # Meaning experiment
            meaning_results = run_meaning_experiment_unified(social_weight, seed)
            meaning_results.update(
                {
                    "module": "meaning_experiment",
                    "social_weight": social_weight,
                    "seed": seed,
                    "experiment_type": "goal_adoption",
                }
            )
            results.append(meaning_results)

    return pd.DataFrame(results)


def generate_reconciliation_report(results_df: pd.DataFrame) -> str:
    """Generate reconciliation report with key findings."""
    report = []
    report.append("# Social Weight Reconciliation Report")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")

    # Group by module and social weight
    module_stats = (
        results_df.groupby(["module", "social_weight"])
        .agg(
            {
                "final_gini": ["mean", "std", "min", "max"],
                "final_gini_computed": ["mean", "std", "min", "max"],
                "diversity_gini": ["mean", "std", "min", "max"],
                "above_threshold": "sum",
            }
        )
        .round(3)
    )

    report.append("### Gini Coefficient by Module and Social Weight")
    report.append("")
    report.append(
        "| Module | Social Weight | Mean Gini | Std Gini | Max Gini | Threshold Breaches |"
    )
    report.append(
        "|--------|---------------|-----------|----------|----------|-------------------|"
    )

    for (module, sw), stats in module_stats.iterrows():
        mean_gini = (
            stats[("final_gini", "mean")]
            if not pd.isna(stats[("final_gini", "mean")])
            else stats[("diversity_gini", "mean")]
        )
        std_gini = (
            stats[("final_gini", "std")]
            if not pd.isna(stats[("final_gini", "std")])
            else stats[("diversity_gini", "std")]
        )
        max_gini = (
            stats[("final_gini", "max")]
            if not pd.isna(stats[("final_gini", "max")])
            else stats[("diversity_gini", "max")]
        )
        breaches = stats[("above_threshold", "sum")]

        report.append(
            f"| {module} | {sw} | {mean_gini:.3f} | {std_gini:.3f} | {max_gini:.3f} | {breaches} |"
        )

    report.append("")

    # Key findings
    report.append("## Key Findings")
    report.append("")

    # Check for contradictions
    goal_gini = results_df[results_df["module"] == "goal_externalities"][
        "final_gini_computed"
    ].dropna()
    meaning_gini = results_df[results_df["module"] == "meaning_experiment"][
        "diversity_gini"
    ].dropna()

    if not goal_gini.empty and not meaning_gini.empty:
        goal_mean = goal_gini.mean()
        meaning_mean = meaning_gini.mean()

        report.append("1. **Gini Coefficient Differences:**")
        report.append(f"   - Goal externalities mean Gini: {goal_mean:.3f}")
        report.append(f"   - Meaning experiment mean Gini: {meaning_mean:.3f}")
        report.append(f"   - Difference: {abs(goal_mean - meaning_mean):.3f}")

        if abs(goal_mean - meaning_mean) > 0.1:
            report.append(
                "   - **Note:** Significant difference detected - modules measure different aspects of inequality"
            )
        else:
            report.append("   - **Note:** Consistent Gini computation across modules")

    # Social weight effects
    sw_effects = (
        results_df.groupby(["module", "social_weight"])["final_gini_computed"]
        .mean()
        .unstack()
    )
    if not sw_effects.empty:
        report.append("")
        report.append("2. **Social Weight Effects:**")
        for module in sw_effects.index:
            if module in sw_effects.index:
                effects = sw_effects.loc[module].dropna()
                if len(effects) > 1:
                    sw_correlation = effects.corr(
                        pd.Series(SOCIAL_WEIGHT_CONFIG.STANDARD_VALUES[: len(effects)])
                    )
                    report.append(
                        f"   - {module}: Social weight correlation with Gini = {sw_correlation:.3f}"
                    )

    # Collapse predictions
    collapse_rate = results_df["above_threshold"].sum() / len(results_df)
    report.append("")
    report.append("3. **Collapse Predictions:**")
    report.append(f"   - Overall collapse rate: {collapse_rate:.1%}")
    report.append(f"   - Threshold used: {GINI_CONFIG.COLLAPSE_THRESHOLD}")

    if collapse_rate > 0.5:
        report.append(
            "   - **High collapse risk detected** - social weight may destabilize systems"
        )
    elif collapse_rate > 0.2:
        report.append(
            "   - **Moderate collapse risk** - some social weights lead to instability"
        )
    else:
        report.append(
            "   - **Low collapse risk** - social weight generally stabilizes systems"
        )

    return "\n".join(report)


def main():
    """Main reconciliation script."""
    print("Starting social weight reconciliation...")

    # Run experiments
    results_df = run_reconciliation_experiments()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("discovery_results") / f"v2_reconciliation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "reconciled_social_weight.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Generate and save report
    report = generate_reconciliation_report(results_df)
    report_path = output_dir / "reconciled_social_weight.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Print summary
    print("\nReconciliation Summary:")
    print(f"Total experiments: {len(results_df)}")
    print(f"Modules tested: {results_df['module'].nunique()}")
    print(f"Social weights: {sorted(results_df['social_weight'].unique())}")
    print(f"Collapse rate: {results_df['above_threshold'].sum() / len(results_df):.1%}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import numpy as np

    main()

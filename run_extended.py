#!/usr/bin/env python3
"""
Top-level script for running extended simulation framework.
"""

import sys
import time

from sim_ext.extended_sweep import run_extended


def main():
    """Run extended simulation with example configuration."""

    # Example configuration for extended simulation
    config = {
        # Basic Parameters
        "n_agents": [50, 100],
        "timesteps": 2000,
        "dt": 1.0,
        "noise": [0.0, 0.1],
        # Shocks
        "shocks": [
            {"severity": 0.2, "timing": 500, "type": "external"},
            {"severity": 0.8, "timing": 1000, "type": "combo"},
        ],
        # Disease Parameters
        "disease": {
            "R0": [1.2, 2.0, 3.0],
            "IFR": [0.2, 0.6],
            "vacc_rate": [0.0, 0.01],
            "waning": [0.0, 0.002],
        },
        # Information Layer
        "info": {"misinfo_rate": [0.0, 0.1, 0.3], "trust_decay": [0.001, 0.01]},
        # Ethics
        "ethics": {
            "rule_set": ["utilitarian", "deontic", "reciprocity"],
            "mutation_rate": [0.0, 0.02],
        },
        # Multi-scale
        "multiscale": {"coupling_coeff": [0.2, 0.6, 0.9], "cell_agents": 16},
        # Energy/Thermodynamics
        "energy": {"softening": [0.01, 0.05, 0.1], "heat_loss": [0.01, 0.05]},
        # Additional Parameters
        "valence_weighting": [0.3, 0.6],
        "enable_uq": True,
        "enable_bayes": False,
        "seeds": [111, 222, 333],
        # Performance Options
        "use_parallel": True,  # Enable parallel processing
        "n_workers": 4,  # Number of parallel workers
        "limit_history": True,  # Optimize memory usage
    }

    print("Starting Extended Simulation Framework...")
    print(f"Configuration: {len(config)} parameter groups")

    start_time = time.time()

    try:
        # Run the extended simulation
        results = run_extended(config)

        end_time = time.time()
        runtime = end_time - start_time

        print("\n✅ Extended simulation completed successfully!")
        print(f"📊 Results saved to: {results['output_dir']}")
        print(f"🔢 Simulations run: {results['n_simulations']}")
        print(f"⏱️  Runtime: {runtime:.1f} seconds")

        # Print summary
        summary = results["summary"]
        print("\n📈 Summary:")
        print(
            f"   Energy Valid: {summary['energy_valid_simulations']}/{summary['n_simulations']}"
        )
        print(f"   Avg Survival: {summary['avg_survival_rate']:.3f}")
        print(f"   Avg CCI: {summary['avg_cci']:.3f}")
        print(f"   Avg Valence: {summary['avg_valence']:.3f}")
        print(f"   Avg Energy Drift: {summary['avg_energy_drift']:.3f}%")

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

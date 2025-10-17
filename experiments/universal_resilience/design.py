# experiments/universal_resilience/design.py
"""
Experimental design for Universal Resilience experiment.
Creates factorial sweep of shock severity, inequality (Gini), and coherence.
"""

import itertools
from typing import Any

import numpy as np

from .utils import generate_deterministic_seeds


class ExperimentalDesign:
    """Designs the factorial experiment for Universal Resilience testing."""

    def __init__(self, config: dict[str, Any], quick_test: bool = False):
        self.config = config
        self.quick_test = quick_test

        # Get experiment parameters
        if quick_test:
            # Use experiment params but override with quick test values
            self.params = config["experiment"].copy()
        else:
            self.params = config["experiment"]

        self.coherence_map = config["coherence_map"]
        self.gini_tolerance = config.get("gini_tolerance", 0.01)
        self.max_gini_iterations = config.get("max_gini_iterations", 50)

    def generate_run_configs(self) -> list[dict[str, Any]]:
        """Generate all run configurations for the factorial design."""

        # Extract parameter ranges
        severities = self.params["severities"]
        durations = self.params["durations"]
        scopes = self.params["scopes"]
        ginis = self.params["ginis"]
        coherence_levels = self.params["coherence_levels"]
        populations = self.params["populations"]
        replicates = self.params["replicates"]
        steps = self.params["steps"]
        shock_step_ratio = self.params["shock_step_ratio"]
        seeds_base = self.params["seeds_base"]

        # Calculate shock start from ratio
        shock_start = int(steps * shock_step_ratio)

        # Generate all combinations (now includes durations and scopes)
        param_combinations = list(
            itertools.product(
                severities, durations, scopes, ginis, coherence_levels, populations
            )
        )

        # Generate seeds
        total_runs = len(param_combinations) * replicates
        seeds = generate_deterministic_seeds(seeds_base, total_runs)

        # Create run configurations
        run_configs = []
        seed_idx = 0

        for (
            severity,
            duration,
            scope,
            target_gini,
            coherence_level,
            n_agents,
        ) in param_combinations:
            for replicate in range(replicates):
                # Calculate shock timing
                shock_end = shock_start + duration

                # Pre-select targeted agent indices (deterministic by seed)
                target_count = int(scope * n_agents)
                np.random.seed(
                    seeds[seed_idx]
                )  # Use same seed for deterministic sampling
                targeted_agents = set(
                    np.random.choice(n_agents, target_count, replace=False)
                )

                run_config = {
                    "run_id": f"s{severity:.2f}_d{duration}_sc{scope:.1f}_g{target_gini:.2f}_c{coherence_level}_p{n_agents}_r{replicate}",
                    "severity": severity,
                    "duration": duration,
                    "scope": scope,
                    "target_gini": target_gini,
                    "coherence_level": coherence_level,
                    "n_agents": n_agents,
                    "replicate": replicate,
                    "steps": steps,
                    "shock_start": shock_start,
                    "shock_end": shock_end,
                    "targeted_agents": targeted_agents,
                    "seed": seeds[seed_idx],
                    "coherence_params": self.coherence_map[coherence_level].copy(),
                    "gini_tolerance": self.gini_tolerance,
                    "max_gini_iterations": self.max_gini_iterations,
                }

                # Calculate derived metrics
                run_config["constructiveness"] = self._calculate_constructiveness(
                    severity
                )
                run_config["coherence_value"] = float(
                    self.coherence_map[coherence_level]["coherence_value"]
                )
                run_config["ur_score"] = self._calculate_ur_score(
                    run_config["constructiveness"],
                    run_config["coherence_value"],
                    target_gini,
                )

                run_configs.append(run_config)
                seed_idx += 1

        print(f"Generated {len(run_configs)} run configurations")
        print(f"Parameter combinations: {len(param_combinations)}")
        print(f"Replicates per combination: {replicates}")

        return run_configs

    def _calculate_constructiveness(self, severity: float) -> float:
        """Calculate constructiveness from shock severity."""
        constructiveness = 1 - abs(severity - 0.5) / 0.5
        return np.clip(constructiveness, 0.0, 1.0)

    def _calculate_ur_score(
        self, constructiveness: float, coherence_value: float, gini: float
    ) -> float:
        """Calculate Universal Resilience score."""
        return (constructiveness * coherence_value) / (gini + 1e-6)

    def get_parameter_grid_summary(self) -> dict[str, Any]:
        """Get summary of the parameter grid."""
        return {
            "severities": self.params["severities"],
            "durations": self.params["durations"],
            "scopes": self.params["scopes"],
            "ginis": self.params["ginis"],
            "coherence_levels": self.params["coherence_levels"],
            "populations": self.params["populations"],
            "replicates": self.params["replicates"],
            "steps": self.params["steps"],
            "shock_step_ratio": self.params["shock_step_ratio"],
            "shock_start": int(self.params["steps"] * self.params["shock_step_ratio"]),
            "total_combinations": len(
                list(
                    itertools.product(
                        self.params["severities"],
                        self.params["durations"],
                        self.params["scopes"],
                        self.params["ginis"],
                        self.params["coherence_levels"],
                        self.params["populations"],
                    )
                )
            ),
            "total_runs": len(self.generate_run_configs()),
            "quick_test": self.quick_test,
        }

    def get_coherence_mapping(self) -> dict[str, dict[str, float]]:
        """Get the coherence level to parameter mapping."""
        return self.coherence_map.copy()

    def validate_design(self) -> bool:
        """Validate the experimental design."""
        try:
            # Check parameter ranges
            severities = self.params["severities"]
            durations = self.params["durations"]
            scopes = self.params["scopes"]
            ginis = self.params["ginis"]
            coherence_levels = self.params["coherence_levels"]

            # Validate severity range
            if not all(0 <= s <= 1 for s in severities):
                print("Error: All severities must be in [0, 1]")
                return False

            # Validate duration range
            if not all(d > 0 for d in durations):
                print("Error: All durations must be positive")
                return False

            # Validate scope range
            if not all(0 < s <= 1 for s in scopes):
                print("Error: All scopes must be in (0, 1]")
                return False

            # Validate Gini range
            if not all(0 <= g <= 1 for g in ginis):
                print("Error: All Gini values must be in [0, 1]")
                return False

            # Validate coherence levels
            valid_coherence = set(self.coherence_map.keys())
            if not all(c in valid_coherence for c in coherence_levels):
                print(f"Error: Invalid coherence levels. Valid: {valid_coherence}")
                return False

            # Check coherence parameter completeness
            for level in coherence_levels:
                params = self.coherence_map[level]
                required_keys = ["noise", "social_coupling", "coherence_value"]
                if not all(key in params for key in required_keys):
                    print(
                        f"Error: Coherence level '{level}' missing required parameters"
                    )
                    return False

            print("✓ Experimental design validation passed")
            return True

        except Exception as e:
            print(f"Error validating design: {e}")
            return False


def create_experimental_design(
    config: dict[str, Any], quick_test: bool = False
) -> ExperimentalDesign:
    """Create an experimental design instance."""
    return ExperimentalDesign(config, quick_test)

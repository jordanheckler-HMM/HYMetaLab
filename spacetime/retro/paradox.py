"""
Paradox battery tests for retrocausality simulations.

Implements canonical paradox tests to verify self-consistency
of retrocausal solutions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ParadoxType(Enum):
    """Types of temporal paradoxes."""

    GRANDPARENT = "grandparent"
    BOOTSTRAP = "bootstrap"
    INFORMATION = "information"
    PREDICTION = "prediction"


class FailureMode(Enum):
    """Types of paradox resolution failures."""

    INCONSISTENT = "inconsistent"
    ENERGY_VIOLATION = "energy_violation"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    CAUSAL_LOOP = "causal_loop"
    NO_SOLUTION = "no_solution"


@dataclass
class ParadoxTest:
    """Represents a single paradox test case."""

    paradox_type: ParadoxType
    description: str
    initial_condition: dict[str, Any]
    retro_action: dict[str, Any]
    expected_outcome: dict[str, Any]
    energy_cost: float = 0.0


@dataclass
class ParadoxResult:
    """Result of a paradox test."""

    test: ParadoxTest
    consistency_score: float  # 0..1
    failure_mode: FailureMode = None
    resolved: bool = False
    iterations_to_resolve: int = 0
    energy_violation: bool = False
    details: dict[str, Any] = None


class ParadoxBattery:
    """
    Battery of paradox tests for retrocausality validation.

    Tests various canonical temporal paradoxes to ensure
    retrocausal solutions are self-consistent.
    """

    def __init__(self):
        self.tests = self._create_test_battery()

    def _create_test_battery(self) -> list[ParadoxTest]:
        """Create comprehensive battery of paradox tests."""

        tests = []

        # Grandparent Paradox
        tests.append(
            ParadoxTest(
                paradox_type=ParadoxType.GRANDPARENT,
                description="Agent tries to prevent their own existence",
                initial_condition={
                    "agent_exists": True,
                    "agent_age": 25,
                    "parent_age": 50,
                    "grandparent_age": 75,
                },
                retro_action={
                    "target_time": 0,
                    "action": "prevent_grandparent_meeting",
                    "success_probability": 0.8,
                },
                expected_outcome={"agent_exists": False, "paradox_resolved": True},
                energy_cost=10.0,
            )
        )

        # Bootstrap Paradox - Information
        tests.append(
            ParadoxTest(
                paradox_type=ParadoxType.BOOTSTRAP,
                description="Information appears from nowhere",
                initial_condition={
                    "knowledge_base": {},
                    "information_source": None,
                    "proof_string": None,
                },
                retro_action={
                    "target_time": 10,
                    "action": "send_proof",
                    "proof_content": "Fermat's Last Theorem: a^n + b^n ≠ c^n for n>2",
                },
                expected_outcome={
                    "knowledge_base": {"fermat_proof": True},
                    "information_source": "future_self",
                    "proof_string": "Fermat's Last Theorem: a^n + b^n ≠ c^n for n>2",
                },
                energy_cost=5.0,
            )
        )

        # Bootstrap Paradox - Invention
        tests.append(
            ParadoxTest(
                paradox_type=ParadoxType.BOOTSTRAP,
                description="Invention appears from future",
                initial_condition={
                    "inventions": [],
                    "inventor": None,
                    "time_machine_design": None,
                },
                retro_action={
                    "target_time": 15,
                    "action": "send_design",
                    "design_specs": {
                        "type": "time_machine",
                        "power_requirements": "1.21 Gigawatts",
                        "materials": ["plutonium", "flux_capacitor"],
                    },
                },
                expected_outcome={
                    "inventions": ["time_machine"],
                    "inventor": "future_self",
                    "time_machine_design": {
                        "type": "time_machine",
                        "power_requirements": "1.21 Gigawatts",
                        "materials": ["plutonium", "flux_capacitor"],
                    },
                },
                energy_cost=20.0,
            )
        )

        # Information Paradox - Prediction
        tests.append(
            ParadoxTest(
                paradox_type=ParadoxType.INFORMATION,
                description="Self-fulfilling prophecy",
                initial_condition={
                    "market_state": "normal",
                    "stock_price": 100.0,
                    "prediction": None,
                    "confidence": 0.0,
                },
                retro_action={
                    "target_time": 5,
                    "action": "send_prediction",
                    "prediction": "Stock will crash at time 10",
                    "confidence": 0.95,
                },
                expected_outcome={
                    "market_state": "crashed",
                    "stock_price": 50.0,
                    "prediction": "Stock will crash at time 10",
                    "confidence": 0.95,
                },
                energy_cost=3.0,
            )
        )

        # Information Paradox - Knowledge
        tests.append(
            ParadoxTest(
                paradox_type=ParadoxType.INFORMATION,
                description="Knowledge without learning",
                initial_condition={
                    "agent_knowledge": [],
                    "learning_source": None,
                    "quantum_mechanics": False,
                },
                retro_action={
                    "target_time": 8,
                    "action": "send_knowledge",
                    "knowledge": {
                        "subject": "quantum_mechanics",
                        "concept": "wave_function_collapse",
                        "equation": "ψ(x,t) = Σ c_n φ_n(x) e^(-iE_n t/ℏ)",
                    },
                },
                expected_outcome={
                    "agent_knowledge": ["quantum_mechanics"],
                    "learning_source": "future_self",
                    "quantum_mechanics": True,
                },
                energy_cost=7.0,
            )
        )

        return tests

    def run_test(
        self, test: ParadoxTest, solver_result: dict[str, Any], energy_guard: Any = None
    ) -> ParadoxResult:
        """
        Run a single paradox test.

        Args:
            test: Paradox test to run
            solver_result: Result from Novikov/Deutsch solver
            energy_guard: Energy conservation guard

        Returns:
            ParadoxResult with test outcome
        """

        # Check energy conservation
        energy_violation = False
        if energy_guard is not None:
            energy_drift = energy_guard.get_energy_drift()
            if abs(energy_drift) > 1e-6:
                energy_violation = True

        # Check if solver found a solution
        if not solver_result.get("converged", False):
            return ParadoxResult(
                test=test,
                consistency_score=0.0,
                failure_mode=FailureMode.NO_SOLUTION,
                resolved=False,
                energy_violation=energy_violation,
            )

        # Evaluate consistency
        consistency_score = self._evaluate_consistency(test, solver_result)

        # Determine if paradox was resolved
        resolved = consistency_score > 0.8 and not energy_violation

        # Identify failure mode if not resolved
        failure_mode = None
        if not resolved:
            if energy_violation:
                failure_mode = FailureMode.ENERGY_VIOLATION
            elif consistency_score < 0.5:
                failure_mode = FailureMode.LOGICAL_CONTRADICTION
            elif consistency_score < 0.8:
                failure_mode = FailureMode.INCONSISTENT
            else:
                failure_mode = FailureMode.CAUSAL_LOOP

        return ParadoxResult(
            test=test,
            consistency_score=consistency_score,
            failure_mode=failure_mode,
            resolved=resolved,
            iterations_to_resolve=solver_result.get("iterations", 0),
            energy_violation=energy_violation,
            details=solver_result,
        )

    def _evaluate_consistency(
        self, test: ParadoxTest, solver_result: dict[str, Any]
    ) -> float:
        """Evaluate how consistent the solver result is with the test."""

        final_state = solver_result.get("state_past_consistent", {})
        expected = test.expected_outcome

        if not final_state or not expected:
            return 0.0

        # Check each expected outcome
        consistency_scores = []

        for key, expected_value in expected.items():
            actual_value = final_state.get(key)

            if isinstance(expected_value, bool):
                # Boolean consistency
                if actual_value == expected_value:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)

            elif isinstance(expected_value, (int, float)):
                # Numerical consistency
                if isinstance(actual_value, (int, float)):
                    if abs(expected_value) > 1e-10:
                        consistency = 1.0 - abs(actual_value - expected_value) / abs(
                            expected_value
                        )
                    else:
                        consistency = 1.0 if abs(actual_value) < 1e-10 else 0.0
                    consistency_scores.append(max(0.0, consistency))
                else:
                    consistency_scores.append(0.0)

            elif isinstance(expected_value, str):
                # String consistency
                if actual_value == expected_value:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)

            elif isinstance(expected_value, dict):
                # Dictionary consistency
                if isinstance(actual_value, dict):
                    dict_consistency = self._evaluate_dict_consistency(
                        actual_value, expected_value
                    )
                    consistency_scores.append(dict_consistency)
                else:
                    consistency_scores.append(0.0)

            elif isinstance(expected_value, list):
                # List consistency
                if isinstance(actual_value, list):
                    list_consistency = self._evaluate_list_consistency(
                        actual_value, expected_value
                    )
                    consistency_scores.append(list_consistency)
                else:
                    consistency_scores.append(0.0)

        return np.mean(consistency_scores) if consistency_scores else 0.0

    def _evaluate_dict_consistency(self, actual: dict, expected: dict) -> float:
        """Evaluate consistency between two dictionaries."""

        if not actual or not expected:
            return 0.0

        consistency_scores = []

        for key, expected_value in expected.items():
            actual_value = actual.get(key)

            if isinstance(expected_value, (int, float)):
                if isinstance(actual_value, (int, float)):
                    if abs(expected_value) > 1e-10:
                        consistency = 1.0 - abs(actual_value - expected_value) / abs(
                            expected_value
                        )
                    else:
                        consistency = 1.0 if abs(actual_value) < 1e-10 else 0.0
                    consistency_scores.append(max(0.0, consistency))
                else:
                    consistency_scores.append(0.0)
            else:
                consistency_scores.append(
                    1.0 if actual_value == expected_value else 0.0
                )

        return np.mean(consistency_scores)

    def _evaluate_list_consistency(self, actual: list, expected: list) -> float:
        """Evaluate consistency between two lists."""

        if not actual or not expected:
            return 0.0

        # For lists, check if expected items are present
        expected_items = set(expected)
        actual_items = set(actual)

        if not expected_items:
            return 1.0

        intersection = expected_items.intersection(actual_items)
        consistency = len(intersection) / len(expected_items)

        return consistency

    def run_all_tests(
        self, solver_results: dict[str, dict[str, Any]], energy_guard: Any = None
    ) -> list[ParadoxResult]:
        """Run all paradox tests with given solver results."""

        results = []

        for test in self.tests:
            # Find matching solver result
            solver_result = solver_results.get(test.paradox_type.value, {})

            result = self.run_test(test, solver_result, energy_guard)
            results.append(result)

        return results

    def get_statistics(self, results: list[ParadoxResult]) -> dict[str, Any]:
        """Get statistics from paradox test results."""

        if not results:
            return {
                "total_tests": 0,
                "resolved_paradoxes": 0,
                "resolution_rate": 0.0,
                "avg_consistency": 0.0,
                "energy_violations": 0,
                "failure_modes": {},
            }

        resolved_count = sum(1 for r in results if r.resolved)
        energy_violations = sum(1 for r in results if r.energy_violation)
        avg_consistency = np.mean([r.consistency_score for r in results])

        # Count failure modes
        failure_modes = {}
        for result in results:
            if result.failure_mode:
                mode = result.failure_mode.value
                failure_modes[mode] = failure_modes.get(mode, 0) + 1

        return {
            "total_tests": len(results),
            "resolved_paradoxes": resolved_count,
            "resolution_rate": resolved_count / len(results),
            "avg_consistency": avg_consistency,
            "energy_violations": energy_violations,
            "failure_modes": failure_modes,
        }

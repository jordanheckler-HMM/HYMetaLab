"""
Tests for paradox battery validation.
"""

from spacetime.retro.paradox import FailureMode, ParadoxBattery, ParadoxType


def test_paradox_battery_creation():
    """Test that paradox battery creates all test cases."""

    battery = ParadoxBattery()

    assert len(battery.tests) > 0, "Paradox battery should contain tests"

    # Check that we have different types of paradoxes
    paradox_types = {test.paradox_type for test in battery.tests}
    assert (
        ParadoxType.GRANDPARENT in paradox_types
    ), "Should include grandparent paradox"
    assert ParadoxType.BOOTSTRAP in paradox_types, "Should include bootstrap paradox"
    assert (
        ParadoxType.INFORMATION in paradox_types
    ), "Should include information paradox"


def test_grandparent_paradox_consistency():
    """Test grandparent paradox consistency evaluation."""

    battery = ParadoxBattery()

    # Find grandparent paradox test
    grandparent_test = None
    for test in battery.tests:
        if test.paradox_type == ParadoxType.GRANDPARENT:
            grandparent_test = test
            break

    assert grandparent_test is not None, "Grandparent paradox test not found"

    # Test with consistent solver result
    consistent_result = {
        "converged": True,
        "state_past_consistent": {
            "agent_exists": False,
            "agent_age": 0,
            "parent_age": 0,
            "grandparent_age": 0,
            "paradox_resolved": True,
        },
        "iterations": 5,
    }

    result = battery.run_test(grandparent_test, consistent_result)

    assert (
        result.consistency_score > 0.8
    ), f"Consistent result should have high score: {result.consistency_score}"
    assert result.resolved, "Consistent grandparent paradox should be resolved"
    assert (
        result.failure_mode is None
    ), "No failure mode should be identified for resolved paradox"


def test_bootstrap_paradox_consistency():
    """Test bootstrap paradox consistency evaluation."""

    battery = ParadoxBattery()

    # Find bootstrap paradox test
    bootstrap_test = None
    for test in battery.tests:
        if (
            test.paradox_type == ParadoxType.BOOTSTRAP
            and "information" in test.description
        ):
            bootstrap_test = test
            break

    assert bootstrap_test is not None, "Bootstrap paradox test not found"

    # Test with consistent solver result
    consistent_result = {
        "converged": True,
        "state_past_consistent": {
            "knowledge_base": {"fermat_proof": True},
            "information_source": "future_self",
            "proof_string": "Fermat's Last Theorem: a^n + b^n ≠ c^n for n>2",
        },
        "iterations": 10,
    }

    result = battery.run_test(bootstrap_test, consistent_result)

    assert (
        result.consistency_score > 0.8
    ), f"Consistent bootstrap should have high score: {result.consistency_score}"
    assert result.resolved, "Consistent bootstrap paradox should be resolved"


def test_paradox_inconsistent_result():
    """Test paradox evaluation with inconsistent result."""

    battery = ParadoxBattery()
    test = battery.tests[0]  # Use first test

    # Test with inconsistent solver result
    inconsistent_result = {
        "converged": True,
        "state_past_consistent": {
            "agent_exists": True,  # Should be False for grandparent paradox
            "agent_age": 25,
            "parent_age": 50,
            "grandparent_age": 75,
            "paradox_resolved": False,  # Should be True
        },
        "iterations": 15,
    }

    result = battery.run_test(test, inconsistent_result)

    assert (
        result.consistency_score < 0.5
    ), f"Inconsistent result should have low score: {result.consistency_score}"
    assert not result.resolved, "Inconsistent result should not be resolved"
    assert result.failure_mode is not None, "Failure mode should be identified"


def test_paradox_no_solution():
    """Test paradox evaluation when solver doesn't converge."""

    battery = ParadoxBattery()
    test = battery.tests[0]

    # Test with non-convergent result
    no_solution_result = {
        "converged": False,
        "state_past_consistent": {},
        "iterations": 100,
    }

    result = battery.run_test(test, no_solution_result)

    assert result.consistency_score == 0.0, "No solution should have zero consistency"
    assert not result.resolved, "No solution should not be resolved"
    assert (
        result.failure_mode == FailureMode.NO_SOLUTION
    ), "Should identify no solution failure"


def test_paradox_energy_violation():
    """Test paradox evaluation with energy violation."""

    battery = ParadoxBattery()
    test = battery.tests[0]

    # Mock energy guard with violation
    class MockEnergyGuard:
        def get_energy_drift(self):
            return 0.1  # 10% energy drift - violation

    energy_guard = MockEnergyGuard()

    consistent_result = {
        "converged": True,
        "state_past_consistent": test.expected_outcome,
        "iterations": 5,
    }

    result = battery.run_test(test, consistent_result, energy_guard)

    assert result.energy_violation, "Should detect energy violation"
    assert not result.resolved, "Energy violation should prevent resolution"
    assert (
        result.failure_mode == FailureMode.ENERGY_VIOLATION
    ), "Should identify energy violation"


def test_paradox_dict_consistency():
    """Test dictionary consistency evaluation."""

    battery = ParadoxBattery()

    # Test dictionary consistency
    actual = {
        "type": "time_machine",
        "power_requirements": "1.21 Gigawatts",
        "materials": ["plutonium", "flux_capacitor"],
    }

    expected = {
        "type": "time_machine",
        "power_requirements": "1.21 Gigawatts",
        "materials": ["plutonium", "flux_capacitor"],
    }

    consistency = battery._evaluate_dict_consistency(actual, expected)
    assert consistency == 1.0, "Identical dictionaries should have robust consistency"

    # Test with mismatch
    expected_mismatch = {
        "type": "time_machine",
        "power_requirements": "1.22 Gigawatts",  # Slight difference
        "materials": ["plutonium", "flux_capacitor"],
    }

    consistency = battery._evaluate_dict_consistency(actual, expected_mismatch)
    assert consistency < 1.0, "Mismatched dictionaries should have lower consistency"


def test_paradox_list_consistency():
    """Test list consistency evaluation."""

    battery = ParadoxBattery()

    # Test list consistency
    actual = ["plutonium", "flux_capacitor", "deLorean"]
    expected = ["plutonium", "flux_capacitor"]

    consistency = battery._evaluate_list_consistency(actual, expected)
    assert (
        consistency == 1.0
    ), "Actual list containing all expected items should have robust consistency"

    # Test with missing items
    actual_missing = ["plutonium"]  # Missing flux_capacitor
    consistency = battery._evaluate_list_consistency(actual_missing, expected)
    assert consistency == 0.5, "Missing items should reduce consistency"


def test_paradox_battery_statistics():
    """Test paradox battery statistics computation."""

    battery = ParadoxBattery()

    # Create mock results
    from spacetime.retro.paradox import ParadoxResult

    results = [
        ParadoxResult(
            test=battery.tests[0],
            consistency_score=0.9,
            resolved=True,
            iterations_to_resolve=5,
            energy_violation=False,
        ),
        ParadoxResult(
            test=battery.tests[1],
            consistency_score=0.3,
            resolved=False,
            failure_mode=FailureMode.LOGICAL_CONTRADICTION,
            iterations_to_resolve=0,
            energy_violation=False,
        ),
        ParadoxResult(
            test=battery.tests[2],
            consistency_score=0.8,
            resolved=True,
            iterations_to_resolve=10,
            energy_violation=False,
        ),
    ]

    stats = battery.get_statistics(results)

    assert stats["total_tests"] == 3, "Should count total tests"
    assert stats["resolved_paradoxes"] == 2, "Should count resolved paradoxes"
    assert (
        abs(stats["resolution_rate"] - 2 / 3) < 1e-10
    ), "Should compute resolution rate correctly"
    assert (
        stats["avg_consistency"] == (0.9 + 0.3 + 0.8) / 3
    ), "Should compute average consistency"
    assert stats["energy_violations"] == 0, "Should count energy violations"
    assert (
        FailureMode.LOGICAL_CONTRADICTION.value in stats["failure_modes"]
    ), "Should track failure modes"


def test_run_all_tests():
    """Test running all paradox tests."""

    battery = ParadoxBattery()

    # Mock solver results
    solver_results = {}
    for test in battery.tests:
        solver_results[test.paradox_type.value] = {
            "converged": True,
            "state_past_consistent": test.expected_outcome,
            "iterations": 5,
        }

    results = battery.run_all_tests(solver_results)

    assert len(results) == len(battery.tests), "Should run all tests"

    # All results should be resolved (since we provided expected outcomes)
    resolved_count = sum(1 for r in results if r.resolved)
    assert resolved_count > 0, "Some tests should be resolved"


if __name__ == "__main__":
    # Run tests if called directly
    test_paradox_battery_creation()
    test_grandparent_paradox_consistency()
    test_bootstrap_paradox_consistency()
    test_paradox_inconsistent_result()
    test_paradox_no_solution()
    test_paradox_energy_violation()
    test_paradox_dict_consistency()
    test_paradox_list_consistency()
    test_paradox_battery_statistics()
    test_run_all_tests()
    print("All paradox battery tests passed!")

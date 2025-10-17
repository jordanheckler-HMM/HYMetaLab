"""
Tests for energy conservation in retrocausal loops.
"""

from spacetime.physics.energy import EnergyGuard, energy_guard


def test_energy_guard_basic():
    """Test basic energy tracking functionality."""

    guard = EnergyGuard(tolerance=1e-6)

    # Record initial energy state
    guard.record_energy(0, {"total_energy": 100.0})

    # Record energy changes
    guard.record_energy(1, {"total_energy": 99.5})
    guard.record_energy(2, {"total_energy": 100.1})
    guard.record_energy(3, {"total_energy": 100.0})

    # Check conservation
    assert guard.check_conservation(), "Energy should be conserved"
    assert guard.get_energy_drift() < 1e-6, "Energy drift should be minimal"

    # Test violation
    guard.record_energy(4, {"total_energy": 150.0})  # 50% increase
    assert not guard.check_conservation(), "Energy violation should be detected"


def test_energy_guard_violation():
    """Test energy violation detection."""

    guard = EnergyGuard(tolerance=1e-6)

    # Record initial state
    guard.record_energy(0, {"total_energy": 100.0})

    # Large energy increase
    guard.record_energy(1, {"total_energy": 200.0})

    assert (
        not guard.check_conservation()
    ), "Large energy increase should violate conservation"
    assert guard.get_energy_drift() > 0.5, "Energy drift should be significant"
    assert len(guard.violations) > 0, "Violation should be recorded"


def test_loop_energy_check():
    """Test energy conservation in retrocausal loops."""

    guard = EnergyGuard(tolerance=1e-6)

    # Create a loop of events
    loop_events = [
        {"total_energy": 100.0},
        {"total_energy": 99.5},
        {"total_energy": 100.0},
        {"total_energy": 99.5},
    ]

    # Loop should be energy-neutral
    assert guard.check_loop_energy(loop_events), "Energy-neutral loop should pass"

    # Create energy-violating loop
    violating_loop = [
        {"total_energy": 100.0},
        {"total_energy": 150.0},  # Energy creation
        {"total_energy": 100.0},
        {"total_energy": 50.0},  # Energy destruction
    ]

    assert not guard.check_loop_energy(
        violating_loop
    ), "Energy-violating loop should fail"


def test_energy_guard_context_manager():
    """Test energy guard as context manager."""

    with energy_guard(tolerance=1e-6) as guard:
        guard.record_energy(0, {"total_energy": 100.0})
        guard.record_energy(1, {"total_energy": 100.1})

        assert guard.check_conservation(), "Energy should be conserved in context"

    # Guard should be reset after context
    new_guard = guard
    assert len(new_guard.energy_history) == 0, "Energy history should be reset"


def test_energy_drift_calculation():
    """Test energy drift calculation accuracy."""

    guard = EnergyGuard(tolerance=1e-6)

    # Record initial energy
    guard.record_energy(0, {"total_energy": 100.0})

    # Small positive drift
    guard.record_energy(1, {"total_energy": 100.5})
    expected_drift = 0.5 / 100.0  # 0.5% drift
    assert (
        abs(guard.get_energy_drift() - expected_drift) < 1e-10
    ), "Drift calculation incorrect"

    # Reset and test negative drift
    guard.reset()
    guard.record_energy(0, {"total_energy": 100.0})
    guard.record_energy(1, {"total_energy": 99.0})
    expected_drift = 1.0 / 100.0  # 1% negative drift
    assert (
        abs(guard.get_energy_drift() - expected_drift) < 1e-10
    ), "Negative drift calculation incorrect"


def test_energy_components():
    """Test tracking of different energy components."""

    guard = EnergyGuard(tolerance=1e-6)

    # Record state with multiple energy components
    guard.record_energy(
        0,
        {"kinetic_energy": 50.0, "potential_energy": 30.0, "information_energy": 20.0},
    )

    # Check that total energy is computed correctly
    assert (
        guard.energy_history[0].total_energy == 100.0
    ), "Total energy calculation incorrect"

    # Record new state
    guard.record_energy(
        1,
        {"kinetic_energy": 55.0, "potential_energy": 25.0, "information_energy": 20.0},
    )

    assert (
        guard.energy_history[1].total_energy == 100.0
    ), "Total energy should remain constant"
    assert guard.check_conservation(), "Energy should be conserved"


def test_energy_guard_reset():
    """Test energy guard reset functionality."""

    guard = EnergyGuard(tolerance=1e-6)

    # Add some data
    guard.record_energy(0, {"total_energy": 100.0})
    guard.record_energy(1, {"total_energy": 200.0})  # Violation

    assert len(guard.energy_history) == 2, "Energy history should have 2 entries"
    assert len(guard.violations) > 0, "Violation should be recorded"
    assert guard.initial_energy is not None, "Initial energy should be set"

    # Reset
    guard.reset()

    assert len(guard.energy_history) == 0, "Energy history should be empty after reset"
    assert len(guard.violations) == 0, "Violations should be cleared after reset"
    assert guard.initial_energy is None, "Initial energy should be None after reset"


def test_energy_tolerance():
    """Test energy conservation tolerance."""

    # Test with different tolerances
    for tolerance in [1e-3, 1e-6, 1e-9]:
        guard = EnergyGuard(tolerance=tolerance)

        guard.record_energy(0, {"total_energy": 100.0})

        # Small drift within tolerance
        small_drift = tolerance * 0.5
        guard.record_energy(1, {"total_energy": 100.0 * (1 + small_drift)})

        assert (
            guard.check_conservation()
        ), f"Small drift should pass with tolerance {tolerance}"

        # Large drift exceeding tolerance
        large_drift = tolerance * 2
        guard.record_energy(2, {"total_energy": 100.0 * (1 + large_drift)})

        assert (
            not guard.check_conservation()
        ), f"Large drift should fail with tolerance {tolerance}"


if __name__ == "__main__":
    # Run tests if called directly
    test_energy_guard_basic()
    test_energy_guard_violation()
    test_loop_energy_check()
    test_energy_guard_context_manager()
    test_energy_drift_calculation()
    test_energy_components()
    test_energy_guard_reset()
    test_energy_tolerance()
    print("All energy bookkeeping tests passed!")

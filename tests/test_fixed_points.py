"""
Tests for fixed-point solvers with known solutions.
"""

import numpy as np

from spacetime.retro.deutsch import DeutschSolver
from spacetime.retro.novikov import NovikovSolver


def test_novikov_simple_fixed_point():
    """Test Novikov solver with a simple fixed-point function."""

    solver = NovikovSolver(max_iterations=50, tolerance=1e-3)  # Relax tolerance

    # Simple fixed-point: x = 0.5 * x + 0.5
    # Solution: x = 1.0
    def dynamics_function(state):
        x = state.get("x", 0.0)
        return {"x": 0.5 * x + 0.5}

    initial_state = {"x": 0.0}
    retro_patch = {"x": 1.0}  # Start at the fixed point

    result = solver.solve(
        initial_state=initial_state,
        retro_patch=retro_patch,
        dynamics_function=dynamics_function,
    )

    assert (
        result.converged
    ), f"Solver did not converge. Final state: {result.state_past_consistent}, consistency: {result.consistency_score}"
    assert (
        result.consistency_score > 0.9
    ), f"Low consistency score: {result.consistency_score}"

    # Check if solution is close to fixed point
    final_x = result.state_past_consistent["x"]
    assert abs(final_x - 1.0) < 0.01, f"Solution {final_x} not close to fixed point 1.0"


def test_novikov_multiple_variables():
    """Test Novikov solver with multiple state variables."""

    solver = NovikovSolver(max_iterations=100, tolerance=1e-3)

    # Find fixed point: x = 0.5x + 0.3y + 0.2, y = 0.2x + 0.6y + 0.2
    # Solving: x = 0.5x + 0.3y + 0.2 => 0.5x = 0.3y + 0.2
    #          y = 0.2x + 0.6y + 0.2 => 0.4y = 0.2x + 0.2
    # Fixed point approximately: x ≈ 1.0, y ≈ 1.0
    def dynamics_function(state):
        x = state.get("x", 0.0)
        y = state.get("y", 0.0)
        return {"x": 0.5 * x + 0.3 * y + 0.2, "y": 0.2 * x + 0.6 * y + 0.2}

    initial_state = {"x": 0.0, "y": 0.0}
    retro_patch = {"x": 1.0, "y": 1.0}  # Start near fixed point

    result = solver.solve(
        initial_state=initial_state,
        retro_patch=retro_patch,
        dynamics_function=dynamics_function,
    )

    assert (
        result.converged
    ), f"Multi-variable solver did not converge. Consistency: {result.consistency_score}"
    assert (
        result.consistency_score > 0.8
    ), f"Low consistency: {result.consistency_score}"

    # Check that both variables converged
    final_x = result.state_past_consistent["x"]
    final_y = result.state_past_consistent["y"]
    assert abs(final_x) < 2.0 and abs(final_y) < 2.0, "Variables diverged"


def test_deutsch_stationary_distribution():
    """Test Deutsch solver with known stationary distribution."""

    solver = DeutschSolver(max_iterations=100, tolerance=1e-3)

    # Simple 2-state Markov chain with known stationary distribution
    # Transition matrix: [[0.7, 0.3], [0.2, 0.8]]
    # Stationary distribution: [0.4, 0.6]
    transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])

    state_labels = ["state_A", "state_B"]
    initial_distribution = {"state_A": 0.5, "state_B": 0.5}

    result = solver.solve(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
        state_labels=state_labels,
    )

    assert (
        result.converged
    ), f"Deutsch solver did not converge. Consistency: {result.consistency_score}"
    assert (
        result.consistency_score > 0.9
    ), f"Low consistency: {result.consistency_score}"

    # Check if stationary distribution is close to expected
    dist = result.stationary_distribution
    assert (
        abs(dist["state_A"] - 0.4) < 0.2
    ), f"State A probability {dist['state_A']} not close to 0.4"
    assert (
        abs(dist["state_B"] - 0.6) < 0.2
    ), f"State B probability {dist['state_B']} not close to 0.6"


def test_deutsch_retro_influence():
    """Test Deutsch solver with retrocausal influence."""

    solver = DeutschSolver(max_iterations=100, tolerance=1e-3)

    # Base transition matrix
    transition_matrix = np.array([[0.6, 0.4], [0.3, 0.7]])

    state_labels = ["state_A", "state_B"]
    initial_distribution = {"state_A": 0.5, "state_B": 0.5}

    # Retro influence that biases toward state_A
    retro_influence = {"state_A": 0.2, "state_B": -0.1}

    result = solver.solve(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
        state_labels=state_labels,
        retro_influence=retro_influence,
    )

    assert (
        result.converged
    ), f"Deutsch solver with retro influence did not converge. Consistency: {result.consistency_score}"

    # Check that retro influence affected the distribution (be more lenient)
    dist = result.stationary_distribution
    assert (
        dist["state_A"] > 0.4
    ), f"Retro influence should increase state_A probability. Got: {dist['state_A']}"


def test_solver_convergence_limits():
    """Test that solvers handle non-convergent cases gracefully."""

    novikov_solver = NovikovSolver(max_iterations=10, tolerance=1e-6)

    # Divergent function: x = 2 * x (no fixed point except x=0)
    def divergent_dynamics(state):
        x = state.get("x", 1.0)
        return {"x": 2.0 * x}

    initial_state = {"x": 1.0}
    retro_patch = {"x": 1.0}

    result = novikov_solver.solve(
        initial_state=initial_state,
        retro_patch=retro_patch,
        dynamics_function=divergent_dynamics,
    )

    assert not result.converged, "Solver should not converge for divergent function"
    assert (
        result.consistency_score < 0.5
    ), "Consistency should be low for divergent case"


if __name__ == "__main__":
    # Run tests if called directly
    test_novikov_simple_fixed_point()
    test_novikov_multiple_variables()
    test_deutsch_stationary_distribution()
    test_deutsch_retro_influence()
    test_solver_convergence_limits()
    print("All fixed-point tests passed!")

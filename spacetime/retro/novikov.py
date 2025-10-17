"""
Novikov self-consistency principle solver.

Implements fixed-point iteration to find self-consistent solutions
for retrocausal patches under the Novikov self-consistency principle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class NovikovResult:
    """Result of Novikov self-consistency solver."""

    state_past_consistent: dict[str, Any]
    consistency_score: float  # 0..1
    iterations: int
    converged: bool
    entropy: float  # For tie-breaking


class NovikovSolver:
    """
    Novikov self-consistency principle solver.

    Given a proposed retrocausal patch, finds a self-consistent assignment
    by solving F(x) = x where x is the past local state after receiving
    retrocausal information.
    """

    def __init__(
        self, max_iterations: int = 100, tolerance: float = 1e-6, damping: float = 0.5
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

    def solve(
        self,
        initial_state: dict[str, Any],
        retro_patch: dict[str, Any],
        dynamics_function: Callable[[dict[str, Any]], dict[str, Any]],
        entropy_function: Callable[[dict[str, Any]], float] | None = None,
    ) -> NovikovResult:
        """
        Solve for self-consistent state under retrocausal influence.

        Args:
            initial_state: Initial state before retrocausal patch
            retro_patch: Proposed retrocausal modification
            dynamics_function: Function that evolves state forward
            entropy_function: Function to compute entropy (for tie-breaking)

        Returns:
            NovikovResult with self-consistent solution
        """
        if entropy_function is None:
            entropy_function = self._default_entropy

        # Start with initial state
        current_state = initial_state.copy()
        best_state = current_state.copy()
        best_consistency = 0.0
        best_entropy = entropy_function(current_state)

        for iteration in range(self.max_iterations):
            # Apply retrocausal patch
            patched_state = {**current_state, **retro_patch}

            # Evolve forward using dynamics
            evolved_state = dynamics_function(patched_state)

            # Check consistency: how close is evolved state to patched state?
            consistency = self._compute_consistency(patched_state, evolved_state)

            # Update best solution if this is better
            if consistency > best_consistency:
                best_state = patched_state.copy()
                best_consistency = consistency
                best_entropy = entropy_function(patched_state)
            elif abs(consistency - best_consistency) < self.tolerance:
                # Tie-breaking: choose minimum entropy solution
                current_entropy = entropy_function(patched_state)
                if current_entropy < best_entropy:
                    best_state = patched_state.copy()
                    best_entropy = current_entropy

            # Check convergence - if evolved state is very close to patched state
            if consistency > (1.0 - self.tolerance):
                return NovikovResult(
                    state_past_consistent=best_state,
                    consistency_score=consistency,
                    iterations=iteration + 1,
                    converged=True,
                    entropy=best_entropy,
                )

            # Update state with damping for next iteration
            # Use evolved state as the new current state
            for key in current_state:
                if key in evolved_state:
                    current_state[key] = (
                        self.damping * evolved_state[key]
                        + (1 - self.damping) * current_state[key]
                    )

        # Return best solution found
        return NovikovResult(
            state_past_consistent=best_state,
            consistency_score=best_consistency,
            iterations=self.max_iterations,
            converged=False,
            entropy=best_entropy,
        )

    def _compute_consistency(
        self, state1: dict[str, Any], state2: dict[str, Any]
    ) -> float:
        """Compute consistency score between two states."""
        if not state1 or not state2:
            return 0.0

        # For numerical values, compute normalized difference
        total_diff = 0.0
        count = 0

        for key in set(state1.keys()) | set(state2.keys()):
            val1 = state1.get(key, 0.0)
            val2 = state2.get(key, 0.0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical consistency
                if abs(val1) > 1e-10:  # Avoid division by zero
                    diff = abs(val1 - val2) / abs(val1)
                    total_diff += diff
                    count += 1
                elif abs(val2) > 1e-10:
                    diff = abs(val1 - val2) / abs(val2)
                    total_diff += diff
                    count += 1
                else:
                    # Both near zero - robust consistency
                    count += 1
            else:
                # Non-numerical consistency (exact match)
                if val1 == val2:
                    count += 1
                else:
                    total_diff += 1.0
                    count += 1

        if count == 0:
            return 0.0

        # Consistency score: 1.0 = robust, 0.0 = no consistency
        avg_diff = total_diff / count
        return max(0.0, 1.0 - avg_diff)

    def _default_entropy(self, state: dict[str, Any]) -> float:
        """Default entropy function for tie-breaking."""
        entropy = 0.0

        for key, value in state.items():
            if isinstance(value, (int, float)):
                # Use absolute value as a simple entropy measure
                entropy += abs(value)
            elif isinstance(value, str):
                # String length as entropy
                entropy += len(value)
            elif isinstance(value, (list, tuple)):
                # Length of sequence
                entropy += len(value)

        return entropy

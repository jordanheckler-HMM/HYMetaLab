"""
Deutsch CTC (D-CTC) fixed-point solver.

Implements power iteration to find stationary distributions
under the Deutsch closed timelike curve model.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class DeutschResult:
    """Result of Deutsch CTC fixed-point solver."""

    stationary_distribution: dict[str, float]
    consistency_score: float  # 0..1
    iterations: int
    converged: bool
    uniqueness_measure: float  # How unique is the fixed point?


class DeutschSolver:
    """
    Deutsch CTC fixed-point solver.

    Represents local dynamics as stochastic/Markov map on state distribution
    and finds stationary distribution ρ* such that T(ρ*) = ρ*.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(
        self,
        initial_distribution: dict[str, float],
        transition_matrix: np.ndarray,
        state_labels: list,
        retro_influence: dict[str, float] | None = None,
    ) -> DeutschResult:
        """
        Solve for stationary distribution under retrocausal influence.

        Args:
            initial_distribution: Initial probability distribution over states
            transition_matrix: Markov transition matrix T
            state_labels: Labels for states (same order as matrix)
            retro_influence: Optional retrocausal influence on transition

        Returns:
            DeutschResult with stationary distribution
        """
        # Normalize initial distribution
        total = sum(initial_distribution.values())
        if total > 0:
            for key in initial_distribution:
                initial_distribution[key] /= total

        # Convert to numpy array
        n_states = len(state_labels)
        current_dist = np.zeros(n_states)

        for i, label in enumerate(state_labels):
            current_dist[i] = initial_distribution.get(label, 0.0)

        # Apply retrocausal influence to transition matrix if provided
        modified_transition = transition_matrix.copy()
        if retro_influence is not None:
            modified_transition = self._apply_retro_influence(
                modified_transition, retro_influence, state_labels
            )

        # Power iteration to find fixed point
        prev_dist = None

        for iteration in range(self.max_iterations):
            prev_dist = current_dist.copy()

            # Apply transition: ρ_new = T * ρ_current
            current_dist = modified_transition @ current_dist

            # Normalize
            current_dist = current_dist / np.sum(current_dist)

            # Check convergence
            if prev_dist is not None:
                diff = np.linalg.norm(current_dist - prev_dist)
                if diff < self.tolerance:
                    # Compute consistency score
                    consistency = self._compute_consistency(
                        current_dist, modified_transition @ current_dist
                    )

                    # Compute uniqueness measure
                    uniqueness = self._compute_uniqueness(modified_transition)

                    return DeutschResult(
                        stationary_distribution=dict(zip(state_labels, current_dist)),
                        consistency_score=consistency,
                        iterations=iteration + 1,
                        converged=True,
                        uniqueness_measure=uniqueness,
                    )

        # Compute final metrics even if not converged
        consistency = self._compute_consistency(
            current_dist, modified_transition @ current_dist
        )
        uniqueness = self._compute_uniqueness(modified_transition)

        return DeutschResult(
            stationary_distribution=dict(zip(state_labels, current_dist)),
            consistency_score=consistency,
            iterations=self.max_iterations,
            converged=False,
            uniqueness_measure=uniqueness,
        )

    def _apply_retro_influence(
        self,
        transition_matrix: np.ndarray,
        retro_influence: dict[str, float],
        state_labels: list,
    ) -> np.ndarray:
        """Apply retrocausal influence to transition matrix."""
        modified = transition_matrix.copy()

        for state_label, influence in retro_influence.items():
            if state_label in state_labels:
                state_idx = state_labels.index(state_label)

                # Modify transition probabilities from this state
                # This is a simplified model - in practice would be more complex
                for j in range(len(state_labels)):
                    if influence > 0:
                        # Increase probability of certain transitions
                        modified[state_idx, j] *= 1.0 + influence
                    else:
                        # Decrease probability
                        modified[state_idx, j] *= 1.0 + influence

                # Renormalize row
                row_sum = np.sum(modified[state_idx, :])
                if row_sum > 0:
                    modified[state_idx, :] /= row_sum

        return modified

    def _compute_consistency(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Compute consistency between two distributions."""
        # Use Bhattacharyya coefficient as consistency measure
        bc = np.sum(np.sqrt(dist1 * dist2))
        return bc

    def _compute_uniqueness(self, transition_matrix: np.ndarray) -> float:
        """Compute measure of fixed point uniqueness."""
        # Compute eigenvalues of transition matrix
        eigenvals = np.linalg.eigvals(transition_matrix)

        # Sort by absolute value
        eigenvals_sorted = np.sort(np.abs(eigenvals))[::-1]

        if len(eigenvals_sorted) < 2:
            return 1.0

        # Uniqueness is related to the gap between first and second eigenvalues
        # Larger gap = more unique fixed point
        gap = eigenvals_sorted[0] - eigenvals_sorted[1]

        # Normalize to [0, 1]
        return min(1.0, gap / eigenvals_sorted[0]) if eigenvals_sorted[0] > 0 else 0.0

    def create_transition_matrix(
        self, state_labels: list, dynamics_function: Callable[[str], dict[str, float]]
    ) -> np.ndarray:
        """
        Create transition matrix from dynamics function.

        Args:
            state_labels: List of state labels
            dynamics_function: Function that maps current state to next state probabilities

        Returns:
            Transition matrix T where T[i,j] = P(next_state = j | current_state = i)
        """
        n_states = len(state_labels)
        transition_matrix = np.zeros((n_states, n_states))

        for i, current_state in enumerate(state_labels):
            next_state_probs = dynamics_function(current_state)

            for j, next_state in enumerate(state_labels):
                transition_matrix[i, j] = next_state_probs.get(next_state, 0.0)

            # Normalize row
            row_sum = np.sum(transition_matrix[i, :])
            if row_sum > 0:
                transition_matrix[i, :] /= row_sum

        return transition_matrix

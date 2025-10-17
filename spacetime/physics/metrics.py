"""
Spacetime metrics for retrocausality simulations.

Provides special relativity and wormhole metrics as computable toy models
that create controllable CTC opportunities without full GR.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MinkowskiMetric:
    """Special relativity metric for time dilation effects."""

    def __init__(self):
        self.c = 1.0  # Speed of light (normalized)

    def proper_time(self, coordinate_time: float, velocity: float = 0.0) -> float:
        """
        Compute proper time from coordinate time and velocity.

        Args:
            coordinate_time: Time in lab frame
            velocity: Velocity as fraction of c

        Returns:
            Proper time experienced by moving observer
        """
        if velocity >= self.c:
            raise ValueError("Velocity cannot exceed speed of light")

        gamma = 1.0 / np.sqrt(1.0 - (velocity / self.c) ** 2)
        return coordinate_time / gamma

    def time_dilation_factor(self, velocity: float) -> float:
        """Get time dilation factor gamma."""
        if velocity >= self.c:
            return np.inf
        return 1.0 / np.sqrt(1.0 - (velocity / self.c) ** 2)


@dataclass
class WormholeMetric:
    """
    Morris-Thorne wormhole metric approximation.

    Creates controllable CTC opportunities via time offset Δτ
    without implementing full general relativity.
    """

    def __init__(self, mouth_separation: float = 10.0, throat_radius: float = 1.0):
        self.L = mouth_separation  # Distance between wormhole mouths
        self.r0 = throat_radius  # Throat radius
        self.time_offset = 0.0  # Δτ - creates CTC opportunities

    def set_time_offset(self, delta_tau: float) -> None:
        """Set time offset between wormhole mouths."""
        self.time_offset = delta_tau

    def traversal_time(self, velocity: float = 0.5) -> float:
        """
        Compute time for traversal through wormhole.

        Args:
            velocity: Traversal velocity as fraction of c

        Returns:
            Proper time for wormhole traversal
        """
        if velocity <= 0 or velocity >= 1.0:
            raise ValueError("Velocity must be in (0, 1)")

        # Simple approximation: traversal time scales with throat radius
        base_time = self.r0 / velocity
        return base_time

    def ctc_condition(self) -> tuple[bool, float]:
        """
        Check if wormhole configuration creates CTC.

        Returns:
            (is_ctc, time_gain): Whether CTC exists and how much time is gained
        """
        if self.time_offset <= 0:
            return False, 0.0

        # CTC exists if time offset exceeds traversal time
        min_traversal = self.traversal_time(velocity=0.99)  # Near maximum velocity
        time_gain = self.time_offset - min_traversal

        return time_gain > 0, time_gain

    def effective_metric(self, r: float) -> dict[str, float]:
        """
        Get effective metric components at radial coordinate r.

        This is a simplified approximation for simulation purposes.
        """
        if r < self.r0:
            # Inside throat - use flat space approximation
            return {
                "g_tt": -1.0,
                "g_rr": 1.0,
                "g_theta_theta": r**2,
                "g_phi_phi": r**2 * np.sin(np.pi / 4) ** 2,  # Simplified
            }
        else:
            # Outside throat - use Morris-Thorne approximation
            b_r = self.r0**2 / r  # Shape function
            phi_r = -self.r0 / (2 * r)  # Redshift function

            return {
                "g_tt": -np.exp(2 * phi_r),
                "g_rr": 1.0 / (1.0 - b_r / r),
                "g_theta_theta": r**2,
                "g_phi_phi": r**2 * np.sin(np.pi / 4) ** 2,
            }

"""
Energy and entropy bookkeeping for retrocausality simulations.

Tracks energy drift and enforces conservation laws to prevent
"free energy from loops" violations.
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnergyState:
    """Track energy state at a specific time."""

    t: int
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    information_energy: float = 0.0  # Energy cost of information processing
    total_energy: float = 0.0

    def __post_init__(self):
        """Compute total energy."""
        self.total_energy = (
            self.kinetic_energy + self.potential_energy + self.information_energy
        )


@dataclass
class EnergyGuard:
    """
    Energy conservation guard for retrocausality simulations.

    Tracks energy drift and flags violations of conservation laws.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.energy_history: list[EnergyState] = []
        self.violations: list[str] = field(default_factory=list)
        self.initial_energy: float | None = None

    def record_energy(self, t: int, state: dict[str, Any]) -> None:
        """Record energy state at time t."""
        # Extract energy components from state
        kinetic = state.get("kinetic_energy", 0.0)
        potential = state.get("potential_energy", 0.0)
        info_energy = state.get("information_energy", 0.0)

        energy_state = EnergyState(
            t=t,
            kinetic_energy=kinetic,
            potential_energy=potential,
            information_energy=info_energy,
        )

        self.energy_history.append(energy_state)

        # Set initial energy reference
        if self.initial_energy is None:
            self.initial_energy = energy_state.total_energy

    def check_conservation(self) -> bool:
        """
        Check if energy is conserved within tolerance.

        Returns:
            True if energy is conserved, False if violation detected
        """
        if len(self.energy_history) < 2:
            return True

        initial = self.initial_energy
        current = self.energy_history[-1].total_energy

        if initial is None:
            return True

        # Calculate relative drift
        drift = abs(current - initial) / abs(initial) if initial != 0 else 0.0

        if drift > self.tolerance:
            violation_msg = (
                f"Energy conservation violation: drift = {drift:.2e} "
                f"(tolerance = {self.tolerance:.2e}) at t={self.energy_history[-1].t}"
            )
            self.violations.append(violation_msg)
            warnings.warn(violation_msg)
            return False

        return True

    def get_energy_drift(self) -> float:
        """Get current energy drift as fraction of initial energy."""
        if not self.energy_history or self.initial_energy is None:
            return 0.0

        current = self.energy_history[-1].total_energy
        return abs(current - self.initial_energy) / abs(self.initial_energy)

    def check_loop_energy(self, loop_events: list[dict[str, Any]]) -> bool:
        """
        Check if a retrocausal loop creates or destroys energy.

        Args:
            loop_events: List of events in the retrocausal loop

        Returns:
            True if loop is energy-neutral, False if violation
        """
        if len(loop_events) < 2:
            return True

        # Sum energy changes around the loop
        total_delta = 0.0
        for i in range(len(loop_events)):
            current = loop_events[i]
            next_event = loop_events[(i + 1) % len(loop_events)]

            delta = next_event.get("total_energy", 0.0) - current.get(
                "total_energy", 0.0
            )
            total_delta += delta

        # Loop should be energy-neutral
        if abs(total_delta) > self.tolerance:
            violation_msg = f"Loop energy violation: ΔE = {total_delta:.2e}"
            self.violations.append(violation_msg)
            return False

        return True

    def reset(self) -> None:
        """Reset energy tracking."""
        self.energy_history.clear()
        self.violations.clear()
        self.initial_energy = None


# Global energy guard instance
_global_energy_guard = EnergyGuard()


@contextmanager
def energy_guard(tolerance: float = 1e-6):
    """
    Context manager for energy conservation checking.

    Usage:
        with energy_guard():
            # Run retrocausal simulation
            ...
    """
    guard = EnergyGuard(tolerance)
    global _global_energy_guard
    old_guard = _global_energy_guard
    _global_energy_guard = guard

    try:
        yield guard
    finally:
        _global_energy_guard = old_guard


def get_energy_guard() -> EnergyGuard:
    """Get the current global energy guard instance."""
    return _global_energy_guard

#!/usr/bin/env python3
"""
OriginChain v2 - Evolution Engine
Simulates coherence growth in emergent systems with deterministic replay

v2 Engine: Coherence evolution with ≥85% replay consistency
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# v2 Engine: Deterministic seed (overridable via env)
EVOLUTION_SEED = int(os.getenv("EVOLUTION_SEED", "42"))
np.random.seed(EVOLUTION_SEED)


@dataclass
class EvolutionState:
    """State of the evolution at a given timestep"""

    timestep: int
    coherence: float  # 0-1
    emergence_quotient: float  # 0-1 (from OriginChain v1)
    complexity: float  # 0-1
    novelty: float  # 0-1
    interconnectedness: float  # 0-1
    energy: float  # Available energy for evolution
    entropy: float  # System entropy (0-1)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    def state_hash(self) -> str:
        """Generate hash of state for comparison"""
        # Round to 6 decimal places for hash stability
        state_str = f"{self.timestep}:{self.coherence:.6f}:{self.emergence_quotient:.6f}:{self.complexity:.6f}:{self.novelty:.6f}:{self.interconnectedness:.6f}:{self.energy:.6f}:{self.entropy:.6f}"
        return hashlib.md5(state_str.encode()).hexdigest()


class EvolutionEngine:
    """
    Evolution engine for simulating coherence growth in emergent systems

    Coherence evolves based on:
    - Emergence Quotient (EQ) from OriginChain v1
    - Energy dynamics (availability and consumption)
    - Entropy (disorder vs. order)
    - Feedback loops (self-reinforcing coherence)
    """

    def __init__(self, seed: int | None = None, config: dict | None = None):
        """
        Initialize EvolutionEngine

        Args:
            seed: Random seed for determinism
            config: Configuration dictionary
        """
        self.seed = seed if seed is not None else EVOLUTION_SEED
        np.random.seed(self.seed)

        # Configuration
        self.config = config or self._default_config()

        # Evolution parameters
        self.coherence_growth_rate = self.config.get("coherence_growth_rate", 0.1)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.05)
        self.entropy_increase_rate = self.config.get("entropy_increase_rate", 0.03)
        self.feedback_strength = self.config.get("feedback_strength", 0.2)
        self.noise_level = self.config.get("noise_level", 0.01)

        # Simulation state
        self.history: list[EvolutionState] = []
        self.current_state: EvolutionState | None = None

    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            "coherence_growth_rate": 0.1,
            "energy_decay_rate": 0.05,
            "entropy_increase_rate": 0.03,
            "feedback_strength": 0.2,
            "noise_level": 0.01,
            "initial_energy": 1.0,
            "initial_entropy": 0.5,
            "max_timesteps": 100,
        }

    def initialize_state(
        self,
        emergence_quotient: float,
        complexity: float,
        novelty: float,
        interconnectedness: float,
    ) -> EvolutionState:
        """
        Initialize the evolution state

        Args:
            emergence_quotient: EQ from OriginChain v1
            complexity: Complexity score
            novelty: Novelty score
            interconnectedness: Interconnectedness score

        Returns:
            Initial EvolutionState
        """
        # Initial coherence is proportional to EQ
        initial_coherence = emergence_quotient * 0.5  # Start at 50% of EQ

        state = EvolutionState(
            timestep=0,
            coherence=initial_coherence,
            emergence_quotient=emergence_quotient,
            complexity=complexity,
            novelty=novelty,
            interconnectedness=interconnectedness,
            energy=self.config.get("initial_energy", 1.0),
            entropy=self.config.get("initial_entropy", 0.5),
        )

        self.current_state = state
        self.history = [state]

        return state

    def step(self) -> EvolutionState:
        """
        Execute one evolution timestep

        Returns:
            New EvolutionState
        """
        if self.current_state is None:
            raise ValueError("Must initialize state before stepping")

        prev_state = self.current_state

        # Calculate coherence change
        coherence_delta = self._calculate_coherence_delta(prev_state)

        # Calculate energy change
        energy_delta = self._calculate_energy_delta(prev_state)

        # Calculate entropy change
        entropy_delta = self._calculate_entropy_delta(prev_state)

        # Add deterministic noise (seeded)
        np.random.seed(self.seed + prev_state.timestep + 1)
        noise = np.random.normal(0, self.noise_level)

        # Update state
        new_coherence = np.clip(
            prev_state.coherence + coherence_delta + noise, 0.0, 1.0
        )
        new_energy = np.clip(prev_state.energy + energy_delta, 0.0, 2.0)
        new_entropy = np.clip(prev_state.entropy + entropy_delta, 0.0, 1.0)

        new_state = EvolutionState(
            timestep=prev_state.timestep + 1,
            coherence=new_coherence,
            emergence_quotient=prev_state.emergence_quotient,
            complexity=prev_state.complexity,
            novelty=prev_state.novelty,
            interconnectedness=prev_state.interconnectedness,
            energy=new_energy,
            entropy=new_entropy,
        )

        self.current_state = new_state
        self.history.append(new_state)

        return new_state

    def _calculate_coherence_delta(self, state: EvolutionState) -> float:
        """
        Calculate change in coherence

        Coherence increases when:
        - High emergence quotient (EQ)
        - High energy availability
        - Low entropy
        - Positive feedback loops
        """
        # Base growth from EQ
        eq_contribution = state.emergence_quotient * self.coherence_growth_rate

        # Energy modulation (more energy → more growth)
        energy_modulation = state.energy * 0.5

        # Entropy penalty (high entropy → less coherence)
        entropy_penalty = state.entropy * 0.3

        # Feedback loop (coherence begets coherence)
        feedback = state.coherence * self.feedback_strength

        # Interconnectedness bonus
        interconnection_bonus = state.interconnectedness * 0.1

        coherence_delta = (
            eq_contribution * energy_modulation
            - entropy_penalty
            + feedback
            + interconnection_bonus
        )

        return coherence_delta

    def _calculate_energy_delta(self, state: EvolutionState) -> float:
        """
        Calculate change in energy

        Energy decreases over time but can be replenished by novelty
        """
        # Natural decay
        decay = -self.energy_decay_rate

        # Novelty infusion (new patterns bring energy)
        novelty_infusion = state.novelty * 0.05

        # Complexity cost (complex systems require energy)
        complexity_cost = -state.complexity * 0.02

        energy_delta = decay + novelty_infusion + complexity_cost

        return energy_delta

    def _calculate_entropy_delta(self, state: EvolutionState) -> float:
        """
        Calculate change in entropy

        Entropy increases naturally but coherence fights it
        """
        # Natural increase (second law of thermodynamics)
        natural_increase = self.entropy_increase_rate

        # Coherence reduces entropy
        coherence_reduction = -state.coherence * 0.05

        # Interconnection creates order
        order_from_interconnection = -state.interconnectedness * 0.02

        entropy_delta = (
            natural_increase + coherence_reduction + order_from_interconnection
        )

        return entropy_delta

    def simulate(self, num_steps: int) -> list[EvolutionState]:
        """
        Run simulation for multiple timesteps

        Args:
            num_steps: Number of timesteps to simulate

        Returns:
            List of EvolutionStates (full history)
        """
        if self.current_state is None:
            raise ValueError("Must initialize state before simulating")

        for _ in range(num_steps):
            self.step()

        return self.history

    def replay(
        self, initial_state: EvolutionState, num_steps: int
    ) -> list[EvolutionState]:
        """
        Replay simulation from a given initial state

        Args:
            initial_state: Starting state
            num_steps: Number of steps to replay

        Returns:
            List of EvolutionStates from replay
        """
        # Reset seed for deterministic replay
        np.random.seed(self.seed)

        # Set initial state
        self.current_state = initial_state
        self.history = [initial_state]

        # Simulate
        return self.simulate(num_steps)

    def compare_trajectories(
        self,
        trajectory1: list[EvolutionState],
        trajectory2: list[EvolutionState],
        tolerance: float = 1e-6,
    ) -> dict:
        """
        Compare two evolution trajectories for consistency

        Args:
            trajectory1: First trajectory
            trajectory2: Second trajectory
            tolerance: Tolerance for floating point comparison

        Returns:
            Comparison results with consistency metrics
        """
        if len(trajectory1) != len(trajectory2):
            return {
                "consistent": False,
                "consistency_rate": 0.0,
                "error": "Trajectory lengths differ",
            }

        consistent_steps = 0
        total_steps = len(trajectory1)
        max_deviation = 0.0
        deviations = []

        for s1, s2 in zip(trajectory1, trajectory2):
            # Compare key metrics
            coherence_diff = abs(s1.coherence - s2.coherence)
            energy_diff = abs(s1.energy - s2.energy)
            entropy_diff = abs(s1.entropy - s2.entropy)

            max_diff = max(coherence_diff, energy_diff, entropy_diff)
            deviations.append(max_diff)

            if max_diff > max_deviation:
                max_deviation = max_diff

            if max_diff <= tolerance:
                consistent_steps += 1

        consistency_rate = consistent_steps / total_steps if total_steps > 0 else 0.0

        return {
            "consistent": consistency_rate >= 0.85,  # ≥85% threshold
            "consistency_rate": consistency_rate,
            "consistent_steps": consistent_steps,
            "total_steps": total_steps,
            "max_deviation": max_deviation,
            "mean_deviation": np.mean(deviations),
            "std_deviation": np.std(deviations),
        }

    def save_trajectory(self, filepath: Path):
        """Save trajectory to JSON file"""
        trajectory_data = {
            "seed": self.seed,
            "config": self.config,
            "history": [state.to_dict() for state in self.history],
        }

        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2)

    def load_trajectory(self, filepath: Path) -> list[EvolutionState]:
        """Load trajectory from JSON file"""
        with open(filepath) as f:
            trajectory_data = json.load(f)

        self.seed = trajectory_data["seed"]
        self.config = trajectory_data["config"]

        history = []
        for state_dict in trajectory_data["history"]:
            state = EvolutionState(**state_dict)
            history.append(state)

        self.history = history
        if history:
            self.current_state = history[-1]

        return history

    def get_coherence_trajectory(self) -> list[float]:
        """Get coherence values over time"""
        return [state.coherence for state in self.history]

    def get_final_coherence(self) -> float:
        """Get final coherence value"""
        if not self.history:
            return 0.0
        return self.history[-1].coherence

    def analyze_evolution(self) -> dict:
        """
        Analyze the evolution trajectory

        Returns:
            Analysis metrics
        """
        if not self.history:
            return {"error": "No history available"}

        coherence_trajectory = self.get_coherence_trajectory()

        return {
            "initial_coherence": coherence_trajectory[0],
            "final_coherence": coherence_trajectory[-1],
            "mean_coherence": np.mean(coherence_trajectory),
            "max_coherence": np.max(coherence_trajectory),
            "min_coherence": np.min(coherence_trajectory),
            "coherence_growth": coherence_trajectory[-1] - coherence_trajectory[0],
            "total_timesteps": len(self.history),
            "final_energy": self.history[-1].energy,
            "final_entropy": self.history[-1].entropy,
        }


def main():
    """CLI for testing evolution_engine"""
    import argparse

    parser = argparse.ArgumentParser(description="OriginChain v2 Evolution Engine")
    parser.add_argument(
        "command", choices=["test", "replay-test"], help="Command to execute"
    )

    args = parser.parse_args()

    if args.command == "test":
        print("🔄 Testing Evolution Engine...")

        engine = EvolutionEngine(seed=42)

        # Initialize with high emergence
        initial_state = engine.initialize_state(
            emergence_quotient=0.8, complexity=0.7, novelty=0.8, interconnectedness=0.9
        )

        print("\n✅ Initial state:")
        print(f"   Coherence: {initial_state.coherence:.3f}")
        print(f"   EQ: {initial_state.emergence_quotient:.3f}")
        print(f"   Energy: {initial_state.energy:.3f}")
        print(f"   Entropy: {initial_state.entropy:.3f}")

        # Simulate
        history = engine.simulate(num_steps=20)

        print("\n✅ After 20 steps:")
        final_state = history[-1]
        print(f"   Coherence: {final_state.coherence:.3f}")
        print(f"   Growth: {final_state.coherence - initial_state.coherence:.3f}")
        print(f"   Energy: {final_state.energy:.3f}")
        print(f"   Entropy: {final_state.entropy:.3f}")

        # Analyze
        analysis = engine.analyze_evolution()
        print("\n✅ Analysis:")
        print(f"   Mean coherence: {analysis['mean_coherence']:.3f}")
        print(f"   Max coherence: {analysis['max_coherence']:.3f}")
        print(f"   Coherence growth: {analysis['coherence_growth']:.3f}")

    elif args.command == "replay-test":
        print("🔄 Testing Replay Consistency...")

        # First run
        engine1 = EvolutionEngine(seed=42)
        initial_state = engine1.initialize_state(
            emergence_quotient=0.8, complexity=0.7, novelty=0.8, interconnectedness=0.9
        )
        trajectory1 = engine1.simulate(num_steps=50)

        print(f"\n✅ First run: {len(trajectory1)} steps")
        print(f"   Final coherence: {trajectory1[-1].coherence:.6f}")

        # Replay
        engine2 = EvolutionEngine(seed=42)
        trajectory2 = engine2.replay(initial_state, num_steps=50)

        print(f"\n✅ Replay run: {len(trajectory2)} steps")
        print(f"   Final coherence: {trajectory2[-1].coherence:.6f}")

        # Compare
        comparison = engine1.compare_trajectories(trajectory1, trajectory2)

        print("\n✅ Consistency check:")
        print(f"   Consistency rate: {comparison['consistency_rate']*100:.1f}%")
        print(
            f"   Consistent steps: {comparison['consistent_steps']}/{comparison['total_steps']}"
        )
        print(f"   Max deviation: {comparison['max_deviation']:.10f}")
        print(f"   Mean deviation: {comparison['mean_deviation']:.10f}")
        print(f"   Passes ≥85% threshold: {'✅' if comparison['consistent'] else '❌'}")


if __name__ == "__main__":
    main()

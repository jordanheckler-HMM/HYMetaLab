#!/usr/bin/env python3
"""
OriginChain v4 - Adaptivity Trainer
Adaptive parameter learning for evolution optimization

v4 Adaptivity: Learn optimal parameters for improved evolution performance
"""
import json
import os

# Import OriginChain v2 Engine
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from evolution_engine import EvolutionEngine

# v4 Adaptivity: Deterministic seed
ADAPTIVITY_SEED = int(os.getenv("ADAPTIVITY_SEED", "42"))
np.random.seed(ADAPTIVITY_SEED)


@dataclass
class AdaptiveParameters:
    """Adaptive evolution parameters"""

    coherence_growth_rate: float = 0.1
    energy_decay_rate: float = 0.05
    entropy_increase_rate: float = 0.03
    feedback_strength: float = 0.2
    noise_level: float = 0.01

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_config(self) -> dict:
        """Convert to EvolutionEngine config format"""
        return {
            "coherence_growth_rate": self.coherence_growth_rate,
            "energy_decay_rate": self.energy_decay_rate,
            "entropy_increase_rate": self.entropy_increase_rate,
            "feedback_strength": self.feedback_strength,
            "noise_level": self.noise_level,
        }


@dataclass
class TrainingResult:
    """Results from adaptive training"""

    episode: int
    parameters: AdaptiveParameters
    final_coherence: float
    mean_coherence: float
    coherence_growth: float
    total_reward: float
    improvement: float = 0.0  # vs baseline

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "episode": self.episode,
            "parameters": self.parameters.to_dict(),
            "final_coherence": self.final_coherence,
            "mean_coherence": self.mean_coherence,
            "coherence_growth": self.coherence_growth,
            "total_reward": self.total_reward,
            "improvement": self.improvement,
        }


class AdaptivityTrainer:
    """
    Adaptive parameter learning for OriginChain evolution

    Uses gradient-free optimization to learn optimal evolution parameters
    that maximize coherence growth and final coherence.

    Optimization method: Evolution Strategies (ES) / CMA-ES simplified
    """

    def __init__(self, config_path: Path | None = None, seed: int | None = None):
        """
        Initialize AdaptivityTrainer

        Args:
            config_path: Path to adaptive_matrix.yml
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else ADAPTIVITY_SEED
        np.random.seed(self.seed)

        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        # Adaptivity settings
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.exploration_noise = self.config.get("exploration_noise", 0.05)
        self.num_episodes = self.config.get("num_episodes", 20)
        self.steps_per_episode = self.config.get("steps_per_episode", 50)

        # Parameter bounds
        self.param_bounds = self.config.get(
            "parameter_bounds",
            {
                "coherence_growth_rate": (0.05, 0.3),
                "energy_decay_rate": (0.01, 0.1),
                "entropy_increase_rate": (0.01, 0.1),
                "feedback_strength": (0.1, 0.5),
                "noise_level": (0.001, 0.05),
            },
        )

        # Current best parameters
        self.best_params = AdaptiveParameters()
        self.best_reward = -np.inf

        # Training history
        self.history: list[TrainingResult] = []

        # Baseline (v2 fixed parameters)
        self.baseline_params = AdaptiveParameters()
        self.baseline_performance = None

    def _default_config(self) -> dict:
        """Default adaptivity configuration"""
        return {
            "version": "4.0-adaptivity",
            "learning_rate": 0.15,
            "exploration_noise": 0.08,
            "num_episodes": 20,
            "steps_per_episode": 50,
            "reward_weights": {
                "final_coherence": 0.4,
                "mean_coherence": 0.35,
                "coherence_growth": 0.25,
            },
            "parameter_bounds": {
                "coherence_growth_rate": [0.05, 0.3],
                "energy_decay_rate": [0.01, 0.1],
                "entropy_increase_rate": [0.01, 0.1],
                "feedback_strength": [0.1, 0.5],
                "noise_level": [0.001, 0.05],
            },
        }

    def _clip_parameters(self, params: AdaptiveParameters) -> AdaptiveParameters:
        """Clip parameters to valid bounds"""
        clipped = AdaptiveParameters()

        for key, (min_val, max_val) in self.param_bounds.items():
            value = getattr(params, key)
            setattr(clipped, key, np.clip(value, min_val, max_val))

        return clipped

    def _evaluate_parameters(
        self, params: AdaptiveParameters, initial_state_params: dict, seed: int
    ) -> tuple[float, dict]:
        """
        Evaluate a set of parameters by running evolution

        Args:
            params: Parameters to evaluate
            initial_state_params: EQ and components for initialization
            seed: Random seed for this evaluation

        Returns:
            (reward, metrics) tuple
        """
        # Create engine with these parameters
        config = params.to_config()
        config.update(
            {
                "initial_energy": 1.0,
                "initial_entropy": 0.5,
                "max_timesteps": self.steps_per_episode,
            }
        )

        engine = EvolutionEngine(seed=seed, config=config)

        # Initialize and simulate
        engine.initialize_state(**initial_state_params)
        history = engine.simulate(num_steps=self.steps_per_episode)

        # Compute metrics
        analysis = engine.analyze_evolution()

        # Compute reward (weighted combination)
        weights = self.config.get("reward_weights", {})
        reward = (
            weights.get("final_coherence", 0.5) * analysis["final_coherence"]
            + weights.get("mean_coherence", 0.3) * analysis["mean_coherence"]
            + weights.get("coherence_growth", 0.2)
            * max(0, analysis["coherence_growth"])
        )

        return reward, analysis

    def _compute_baseline(self, initial_state_params: dict):
        """Compute baseline performance with v2 fixed parameters"""
        if self.baseline_performance is not None:
            return self.baseline_performance

        reward, metrics = self._evaluate_parameters(
            self.baseline_params, initial_state_params, seed=self.seed
        )

        self.baseline_performance = {
            "reward": reward,
            "final_coherence": metrics["final_coherence"],
            "mean_coherence": metrics["mean_coherence"],
            "coherence_growth": metrics["coherence_growth"],
        }

        return self.baseline_performance

    def train(
        self, initial_state_params: dict, verbose: bool = True
    ) -> list[TrainingResult]:
        """
        Train adaptive parameters

        Args:
            initial_state_params: EQ and components for evolution initialization
            verbose: Print progress

        Returns:
            Training history
        """
        if verbose:
            print("🧠 Starting Adaptive Training...")

        # Compute baseline
        baseline = self._compute_baseline(initial_state_params)
        if verbose:
            print(
                f"   Baseline (v2): Reward {baseline['reward']:.3f}, "
                f"Final Coherence {baseline['final_coherence']:.3f}"
            )

        # Initialize current parameters (start from baseline)
        current_params = AdaptiveParameters()

        for episode in range(self.num_episodes):
            # Reset seed for reproducibility
            np.random.seed(self.seed + episode)

            # Generate parameter variations (exploration)
            param_array = np.array(
                [
                    current_params.coherence_growth_rate,
                    current_params.energy_decay_rate,
                    current_params.entropy_increase_rate,
                    current_params.feedback_strength,
                    current_params.noise_level,
                ]
            )

            # Add exploration noise
            noise = np.random.randn(5) * self.exploration_noise
            perturbed_array = param_array + noise

            # Create perturbed parameters
            perturbed_params = AdaptiveParameters(
                coherence_growth_rate=perturbed_array[0],
                energy_decay_rate=perturbed_array[1],
                entropy_increase_rate=perturbed_array[2],
                feedback_strength=perturbed_array[3],
                noise_level=perturbed_array[4],
            )

            # Clip to bounds
            perturbed_params = self._clip_parameters(perturbed_params)

            # Evaluate perturbed parameters
            reward, metrics = self._evaluate_parameters(
                perturbed_params, initial_state_params, seed=self.seed + episode
            )

            # Update if better
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_params = perturbed_params

                # Update current params toward best (with learning rate)
                for key in [
                    "coherence_growth_rate",
                    "energy_decay_rate",
                    "entropy_increase_rate",
                    "feedback_strength",
                    "noise_level",
                ]:
                    current_val = getattr(current_params, key)
                    best_val = getattr(self.best_params, key)
                    new_val = current_val + self.learning_rate * (
                        best_val - current_val
                    )
                    setattr(current_params, key, new_val)

                current_params = self._clip_parameters(current_params)

            # Compute improvement vs baseline
            improvement = (
                ((reward - baseline["reward"]) / baseline["reward"]) * 100
                if baseline["reward"] > 0
                else 0
            )

            # Record result
            result = TrainingResult(
                episode=episode,
                parameters=perturbed_params,
                final_coherence=metrics["final_coherence"],
                mean_coherence=metrics["mean_coherence"],
                coherence_growth=metrics["coherence_growth"],
                total_reward=reward,
                improvement=improvement,
            )
            self.history.append(result)

            if verbose and episode % 5 == 0:
                print(
                    f"   Episode {episode}: Reward {reward:.3f}, "
                    f"Final Coherence {metrics['final_coherence']:.3f}, "
                    f"Improvement {improvement:.1f}%"
                )

        if verbose:
            print("\n✅ Training Complete!")
            print(f"   Best Reward: {self.best_reward:.3f}")
            print(f"   Best Final Coherence: {self.history[-1].final_coherence:.3f}")
            improvement = (
                (self.best_reward - baseline["reward"]) / baseline["reward"]
            ) * 100
            print(f"   Improvement vs Baseline: {improvement:.1f}%")

        return self.history

    def get_best_parameters(self) -> AdaptiveParameters:
        """Get best learned parameters"""
        return self.best_params

    def get_improvement(self) -> float:
        """Get improvement percentage vs baseline"""
        if self.baseline_performance is None or not self.history:
            return 0.0

        baseline_reward = self.baseline_performance["reward"]
        best_reward = self.best_reward

        return (
            ((best_reward - baseline_reward) / baseline_reward) * 100
            if baseline_reward > 0
            else 0.0
        )

    def save_results(self, output_path: Path):
        """Save training results to JSON"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "version": "4.0-adaptivity",
            "baseline": self.baseline_performance,
            "best_parameters": self.best_params.to_dict(),
            "best_reward": self.best_reward,
            "improvement_percentage": self.get_improvement(),
            "training_history": [r.to_dict() for r in self.history],
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def compare_with_baseline(
        self, initial_state_params: dict, num_trials: int = 10
    ) -> dict:
        """
        Compare best learned parameters with baseline across multiple trials

        Args:
            initial_state_params: EQ and components
            num_trials: Number of trials to average

        Returns:
            Comparison results
        """
        baseline_rewards = []
        adaptive_rewards = []

        for trial in range(num_trials):
            # Baseline
            baseline_reward, _ = self._evaluate_parameters(
                self.baseline_params,
                initial_state_params,
                seed=self.seed + 1000 + trial,
            )
            baseline_rewards.append(baseline_reward)

            # Adaptive
            adaptive_reward, _ = self._evaluate_parameters(
                self.best_params, initial_state_params, seed=self.seed + 1000 + trial
            )
            adaptive_rewards.append(adaptive_reward)

        baseline_mean = np.mean(baseline_rewards)
        adaptive_mean = np.mean(adaptive_rewards)
        improvement = ((adaptive_mean - baseline_mean) / baseline_mean) * 100

        return {
            "baseline_mean": baseline_mean,
            "baseline_std": np.std(baseline_rewards),
            "adaptive_mean": adaptive_mean,
            "adaptive_std": np.std(adaptive_rewards),
            "improvement_percentage": improvement,
            "num_trials": num_trials,
        }


def main():
    """CLI for testing adaptivity_trainer"""
    import argparse

    parser = argparse.ArgumentParser(description="OriginChain v4 Adaptivity Trainer")
    parser.add_argument(
        "command", choices=["test", "train", "compare"], help="Command to execute"
    )

    args = parser.parse_args()

    if args.command == "test":
        print("🧠 Testing Adaptivity Trainer...")

        # Create trainer
        trainer = AdaptivityTrainer(seed=42)

        # Test initial state
        initial_state = {
            "emergence_quotient": 0.8,
            "complexity": 0.7,
            "novelty": 0.8,
            "interconnectedness": 0.9,
        }

        # Get baseline
        baseline = trainer._compute_baseline(initial_state)
        print("\n✅ Baseline (v2 fixed parameters):")
        print(f"   Reward: {baseline['reward']:.3f}")
        print(f"   Final coherence: {baseline['final_coherence']:.3f}")

        # Test single parameter evaluation
        test_params = AdaptiveParameters(
            coherence_growth_rate=0.15, feedback_strength=0.3
        )
        reward, metrics = trainer._evaluate_parameters(
            test_params, initial_state, seed=42
        )
        print("\n✅ Test parameters:")
        print(f"   Reward: {reward:.3f}")
        print(f"   Final coherence: {metrics['final_coherence']:.3f}")

        improvement = ((reward - baseline["reward"]) / baseline["reward"]) * 100
        print(f"   Improvement: {improvement:.1f}%")

    elif args.command == "train":
        print("🧠 Training Adaptive Parameters...")

        trainer = AdaptivityTrainer(seed=42)

        # Use moderate emergence for better differentiation
        initial_state = {
            "emergence_quotient": 0.5,
            "complexity": 0.5,
            "novelty": 0.4,
            "interconnectedness": 0.6,
        }

        # Train
        history = trainer.train(initial_state, verbose=True)

        # Save results
        trainer.save_results(Path("adaptivity_results.json"))
        print("\n✅ Results saved to adaptivity_results.json")

    elif args.command == "compare":
        print("🧠 Comparing Baseline vs Adaptive (10 trials)...")

        trainer = AdaptivityTrainer(seed=42)

        # Use moderate emergence for better differentiation
        initial_state = {
            "emergence_quotient": 0.5,
            "complexity": 0.5,
            "novelty": 0.4,
            "interconnectedness": 0.6,
        }

        # Train first
        print("\n1. Training...")
        trainer.train(initial_state, verbose=False)

        # Compare
        print("\n2. Comparing...")
        comparison = trainer.compare_with_baseline(initial_state, num_trials=10)

        print("\n✅ Comparison Results (10 trials):")
        print(
            f"   Baseline: {comparison['baseline_mean']:.3f} ± {comparison['baseline_std']:.3f}"
        )
        print(
            f"   Adaptive: {comparison['adaptive_mean']:.3f} ± {comparison['adaptive_std']:.3f}"
        )
        print(f"   Improvement: {comparison['improvement_percentage']:.1f}%")
        print(
            f"   Passes ≥25% threshold: {'✅' if comparison['improvement_percentage'] >= 25 else '❌'}"
        )


if __name__ == "__main__":
    main()

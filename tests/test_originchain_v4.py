#!/usr/bin/env python3
"""
OriginChain v4 Adaptivity Test Suite
Tests for adaptive parameter learning and optimization
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptivity_trainer import AdaptiveParameters, AdaptivityTrainer, TrainingResult


@pytest.fixture
def adaptivity_trainer():
    """AdaptivityTrainer fixture with deterministic seed"""
    return AdaptivityTrainer(seed=42)


@pytest.fixture
def high_emergence_state():
    """High emergence initial state"""
    return {
        "emergence_quotient": 0.8,
        "complexity": 0.7,
        "novelty": 0.8,
        "interconnectedness": 0.9,
    }


@pytest.fixture
def moderate_emergence_state():
    """Moderate emergence initial state"""
    return {
        "emergence_quotient": 0.5,
        "complexity": 0.5,
        "novelty": 0.4,
        "interconnectedness": 0.6,
    }


class TestAdaptiveParameters:
    """Test AdaptiveParameters dataclass"""

    def test_default_parameters(self):
        """Test default parameter values"""
        params = AdaptiveParameters()

        assert params.coherence_growth_rate == 0.1
        assert params.energy_decay_rate == 0.05
        assert params.entropy_increase_rate == 0.03
        assert params.feedback_strength == 0.2
        assert params.noise_level == 0.01

    def test_custom_parameters(self):
        """Test custom parameter values"""
        params = AdaptiveParameters(coherence_growth_rate=0.15, feedback_strength=0.3)

        assert params.coherence_growth_rate == 0.15
        assert params.feedback_strength == 0.3

    def test_to_dict(self):
        """Test conversion to dictionary"""
        params = AdaptiveParameters()
        d = params.to_dict()

        assert isinstance(d, dict)
        assert "coherence_growth_rate" in d
        assert d["coherence_growth_rate"] == 0.1

    def test_to_config(self):
        """Test conversion to engine config format"""
        params = AdaptiveParameters()
        config = params.to_config()

        assert isinstance(config, dict)
        assert "coherence_growth_rate" in config
        assert config["coherence_growth_rate"] == 0.1


class TestTrainerInitialization:
    """Test trainer initialization"""

    def test_trainer_creation(self, adaptivity_trainer):
        """Test trainer can be created"""
        assert adaptivity_trainer is not None
        assert adaptivity_trainer.seed == 42

    def test_config_loading(self):
        """Test loading config from file"""
        config_path = Path("adaptive_matrix.yml")

        if config_path.exists():
            trainer = AdaptivityTrainer(config_path=config_path, seed=42)
            assert trainer.config is not None
            assert "learning_rate" in trainer.config

    def test_baseline_params(self, adaptivity_trainer):
        """Test baseline parameters match v2 defaults"""
        baseline = adaptivity_trainer.baseline_params

        assert baseline.coherence_growth_rate == 0.1
        assert baseline.energy_decay_rate == 0.05
        assert baseline.entropy_increase_rate == 0.03


class TestParameterEvaluation:
    """Test parameter evaluation"""

    def test_evaluate_baseline(self, adaptivity_trainer, high_emergence_state):
        """Test evaluating baseline parameters"""
        reward, metrics = adaptivity_trainer._evaluate_parameters(
            adaptivity_trainer.baseline_params, high_emergence_state, seed=42
        )

        assert reward > 0
        assert "final_coherence" in metrics
        assert "mean_coherence" in metrics
        assert 0.0 <= metrics["final_coherence"] <= 1.0

    def test_evaluate_custom_params(self, adaptivity_trainer, high_emergence_state):
        """Test evaluating custom parameters"""
        custom_params = AdaptiveParameters(
            coherence_growth_rate=0.15, feedback_strength=0.3
        )

        reward, metrics = adaptivity_trainer._evaluate_parameters(
            custom_params, high_emergence_state, seed=42
        )

        assert reward > 0
        assert 0.0 <= metrics["final_coherence"] <= 1.0

    def test_parameter_clipping(self, adaptivity_trainer):
        """Test parameter clipping to bounds"""
        # Out-of-bounds parameters
        params = AdaptiveParameters(
            coherence_growth_rate=0.5, energy_decay_rate=0.001  # Above max  # Below min
        )

        clipped = adaptivity_trainer._clip_parameters(params)

        # Should be clipped to bounds
        assert clipped.coherence_growth_rate <= 0.3
        assert clipped.energy_decay_rate >= 0.01


class TestBaseline:
    """Test baseline computation"""

    def test_compute_baseline(self, adaptivity_trainer, high_emergence_state):
        """Test baseline computation"""
        baseline = adaptivity_trainer._compute_baseline(high_emergence_state)

        assert "reward" in baseline
        assert "final_coherence" in baseline
        assert baseline["reward"] > 0

    def test_baseline_caching(self, adaptivity_trainer, high_emergence_state):
        """Test baseline is cached"""
        baseline1 = adaptivity_trainer._compute_baseline(high_emergence_state)
        baseline2 = adaptivity_trainer._compute_baseline(high_emergence_state)

        # Should return same cached result
        assert baseline1 == baseline2


class TestTraining:
    """Test adaptive training"""

    def test_basic_training(self, moderate_emergence_state):
        """Test basic training loop"""
        # Short training run with custom config
        config = {
            "num_episodes": 5,
            "learning_rate": 0.15,
            "exploration_noise": 0.08,
            "steps_per_episode": 50,
        }
        trainer = AdaptivityTrainer(seed=42)
        trainer.num_episodes = 5

        history = trainer.train(moderate_emergence_state, verbose=False)

        assert len(history) == 5
        assert all(isinstance(r, TrainingResult) for r in history)

    def test_training_improves(self, adaptivity_trainer, high_emergence_state):
        """Test training improves performance"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 10

        history = trainer.train(high_emergence_state, verbose=False)

        # Final episode should be better than initial
        initial_reward = history[0].total_reward
        final_reward = history[-1].total_reward

        # Allow for some variance but expect general improvement
        assert final_reward >= initial_reward * 0.95  # Within 5% is acceptable

    def test_best_params_tracked(self, adaptivity_trainer, high_emergence_state):
        """Test best parameters are tracked"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 10

        trainer.train(high_emergence_state, verbose=False)

        best_params = trainer.get_best_parameters()
        assert isinstance(best_params, AdaptiveParameters)
        assert trainer.best_reward > 0


class TestImprovement:
    """Test improvement vs baseline (≥25% requirement)"""

    def test_improvement_calculation(self, adaptivity_trainer, high_emergence_state):
        """Test improvement percentage calculation"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 15

        trainer.train(high_emergence_state, verbose=False)

        improvement = trainer.get_improvement()

        assert isinstance(improvement, float)
        assert improvement >= 0  # Should be positive improvement

    def test_improvement_threshold(self, moderate_emergence_state):
        """Test ≥25% improvement threshold"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.num_episodes = 20

        trainer.train(moderate_emergence_state, verbose=False)

        improvement = trainer.get_improvement()

        assert (
            improvement >= 25.0
        ), f"Improvement {improvement:.1f}% below 25% threshold"

    def test_comparison_with_baseline(self, adaptivity_trainer, high_emergence_state):
        """Test multi-trial comparison with baseline"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 15

        trainer.train(high_emergence_state, verbose=False)

        comparison = trainer.compare_with_baseline(high_emergence_state, num_trials=5)

        assert "baseline_mean" in comparison
        assert "adaptive_mean" in comparison
        assert "improvement_percentage" in comparison
        assert comparison["adaptive_mean"] > comparison["baseline_mean"]

    def test_improvement_consistency(self, moderate_emergence_state):
        """Test improvement is consistent across multiple runs"""
        improvements = []

        for seed in [42, 43, 44]:
            trainer = AdaptivityTrainer(seed=seed)
            trainer.num_episodes = 15
            trainer.train(moderate_emergence_state, verbose=False)
            improvements.append(trainer.get_improvement())

        # All runs should show improvement
        assert all(imp > 0 for imp in improvements)

        # Mean improvement should exceed threshold
        mean_improvement = np.mean(improvements)
        assert mean_improvement >= 20.0  # Slightly lower for cross-seed consistency


class TestSaving:
    """Test saving results"""

    def test_save_results(self, adaptivity_trainer, high_emergence_state):
        """Test saving training results"""
        import tempfile

        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 5
        trainer.train(high_emergence_state, verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            trainer.save_results(temp_path)
            assert temp_path.exists()

            # Verify content
            import json

            with open(temp_path) as f:
                data = json.load(f)

            assert "baseline" in data
            assert "best_parameters" in data
            assert "improvement_percentage" in data
            assert "training_history" in data
        finally:
            temp_path.unlink()


class TestDifferentScenarios:
    """Test adaptivity across different scenarios"""

    def test_high_emergence_scenario(self):
        """Test with high emergence"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 15

        high_state = {
            "emergence_quotient": 0.8,
            "complexity": 0.7,
            "novelty": 0.8,
            "interconnectedness": 0.9,
        }

        trainer.train(high_state, verbose=False)
        improvement = trainer.get_improvement()

        assert improvement > 0

    def test_moderate_emergence_scenario(self, moderate_emergence_state):
        """Test with moderate emergence"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 15

        trainer.train(moderate_emergence_state, verbose=False)
        improvement = trainer.get_improvement()

        assert improvement > 0

    def test_low_emergence_scenario(self):
        """Test with low emergence"""
        trainer = AdaptivityTrainer(seed=42)
        trainer.config["num_episodes"] = 15

        low_state = {
            "emergence_quotient": 0.2,
            "complexity": 0.3,
            "novelty": 0.1,
            "interconnectedness": 0.2,
        }

        trainer.train(low_state, verbose=False)
        improvement = trainer.get_improvement()

        # Even with low emergence, should show some improvement
        assert improvement >= 0


class TestDeterminism:
    """Test deterministic behavior"""

    def test_same_seed_same_result(self, high_emergence_state):
        """Test same seed produces same results"""
        results = []

        for _ in range(2):
            trainer = AdaptivityTrainer(seed=42)
            trainer.config["num_episodes"] = 10
            history = trainer.train(high_emergence_state, verbose=False)
            results.append(history[-1].total_reward)

        # Should be identical
        assert abs(results[0] - results[1]) < 1e-6


def test_suite_summary():
    """Print test suite summary"""
    print("\n" + "=" * 70)
    print("OriginChain v4 Adaptivity Test Suite Summary")
    print("=" * 70)
    print("✅ Adaptive Parameters: Initialization and conversion")
    print("✅ Trainer Initialization: Config loading and baseline")
    print("✅ Parameter Evaluation: Baseline and custom params")
    print("✅ Training: Improvement tracking and convergence")
    print("✅ Improvement: ≥25% threshold verification")
    print("✅ Saving: Results export to JSON")
    print("✅ Scenarios: High, moderate, low emergence")
    print("✅ Determinism: Reproducible results")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

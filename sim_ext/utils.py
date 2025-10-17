"""
Utility functions for extended simulation framework.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_output_dir(run_id: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"discovery_results/{timestamp}_{run_id}"

    subdirs = ["csv", "plots", "reports", "json", "artifacts"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def save_results(data: Any, filepath: str, format: str = "csv") -> None:
    """Save results in specified format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format == "csv" and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif format == "json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "npy" and isinstance(data, np.ndarray):
        np.save(filepath, data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def config_hash(config: dict[str, Any]) -> str:
    """Generate hash for configuration reproducibility."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def validate_energy_conservation(
    energy_history: list[float], tolerance: float = 0.01
) -> bool:
    """Validate energy conservation within tolerance."""
    if len(energy_history) < 100:
        return True

    # Check drift over last 1000 steps
    recent = energy_history[-1000:] if len(energy_history) >= 1000 else energy_history
    drift = abs(recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0

    if drift > tolerance:
        logger.warning(f"Energy drift {drift:.3f} exceeds tolerance {tolerance}")
        return False

    return True


def compute_bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05
) -> tuple:
    """Compute bootstrap confidence intervals."""
    n = len(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    return a / b if b != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def exponential_decay(current: float, target: float, rate: float, dt: float) -> float:
    """Exponential decay towards target."""
    return current + (target - current) * (1 - np.exp(-rate * dt))


def logistic_growth(current: float, capacity: float, rate: float, dt: float) -> float:
    """Logistic growth model."""
    return current + rate * current * (1 - current / capacity) * dt


def quick_tests():
    """Run quick validation tests."""
    logger.info("Running utility tests...")

    # Test energy conservation validation
    energy_history = [1000.0 + i * 0.001 for i in range(1000)]
    assert validate_energy_conservation(energy_history, tolerance=0.01)

    # Test bootstrap CI
    data = np.random.normal(0, 1, 100)
    lower, upper = compute_bootstrap_ci(data)
    assert lower < upper

    # Test safe division
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0

    # Test clamp
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10

    logger.info("All utility tests passed!")


if __name__ == "__main__":
    quick_tests()

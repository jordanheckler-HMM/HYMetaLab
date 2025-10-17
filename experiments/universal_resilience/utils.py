# experiments/universal_resilience/utils.py
"""
Utility functions for Universal Resilience experiment.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_mkdirs(path: str) -> Path:
    """Safely create directory structure."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write DataFrame to CSV with error handling."""
    safe_mkdirs(os.path.dirname(path))
    df.to_csv(path, index=False, **kwargs)


def write_json(data: dict[str, Any], path: str) -> None:
    """Write data to JSON file with error handling."""
    safe_mkdirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def hash_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient from an array of values."""
    if len(values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)

    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0


def calculate_constructiveness(
    severity: float, peak_severity: float = 0.5, asymmetric: bool = False
) -> float:
    """Calculate constructiveness from shock severity with data-driven peak.

    Args:
        severity: Shock severity [0, 1]
        peak_severity: Peak constructiveness severity (fitted from data)
        asymmetric: Whether to use asymmetric fitting
    """
    if asymmetric:
        # Asymmetric constructiveness: systems may tolerate higher shocks better
        if severity <= peak_severity:
            # Linear increase up to peak
            constructiveness = severity / peak_severity
        else:
            # Slower decay after peak (systems more tolerant of high shocks)
            constructiveness = (
                1.0 - (severity - peak_severity) / (1.0 - peak_severity) * 0.5
            )
    else:
        # Symmetric constructiveness around peak
        constructiveness = 1 - abs(severity - peak_severity) / max(
            peak_severity, 1 - peak_severity
        )

    return np.clip(constructiveness, 0.0, 1.0)


def fit_constructiveness_peak(
    data: pd.DataFrame, outcome_col: str = "final_alive_fraction_mean"
) -> dict[str, float]:
    """Fit the optimal peak severity for constructiveness from data.

    Args:
        data: DataFrame with severity and outcome columns
        outcome_col: Column name for the outcome variable

    Returns:
        Dictionary with fitted parameters
    """
    from scipy.optimize import minimize_scalar

    if outcome_col not in data.columns:
        return {"peak_severity": 0.5, "r_squared": 0.0, "asymmetric": False}

    # Remove rows with missing data
    clean_data = data.dropna(subset=["severity", outcome_col])

    if len(clean_data) < 3:
        return {"peak_severity": 0.5, "r_squared": 0.0, "asymmetric": False}

    def objective_symmetric(peak_severity):
        """Objective function for symmetric constructiveness."""
        constructiveness = [
            calculate_constructiveness(s, peak_severity, False)
            for s in clean_data["severity"]
        ]
        correlation = np.corrcoef(constructiveness, clean_data[outcome_col])[0, 1]
        return -abs(correlation) if not np.isnan(correlation) else 0

    def objective_asymmetric(peak_severity):
        """Objective function for asymmetric constructiveness."""
        constructiveness = [
            calculate_constructiveness(s, peak_severity, True)
            for s in clean_data["severity"]
        ]
        correlation = np.corrcoef(constructiveness, clean_data[outcome_col])[0, 1]
        return -abs(correlation) if not np.isnan(correlation) else 0

    # Try symmetric fitting
    try:
        result_sym = minimize_scalar(
            objective_symmetric, bounds=(0.1, 0.9), method="bounded"
        )
        peak_sym = result_sym.x
        r2_sym = -result_sym.fun
    except:
        peak_sym = 0.5
        r2_sym = 0.0

    # Try asymmetric fitting
    try:
        result_asym = minimize_scalar(
            objective_asymmetric, bounds=(0.1, 0.9), method="bounded"
        )
        peak_asym = result_asym.x
        r2_asym = -result_asym.fun
    except:
        peak_asym = 0.5
        r2_asym = 0.0

    # Choose the better fit
    if r2_asym > r2_sym:
        return {"peak_severity": peak_asym, "r_squared": r2_asym, "asymmetric": True}
    else:
        return {"peak_severity": peak_sym, "r_squared": r2_sym, "asymmetric": False}


def fit_ur_formula_exponents(
    data: pd.DataFrame, outcome_col: str = "final_alive_fraction_mean"
) -> dict[str, float]:
    """Fit exponents for UR formula variant: (C^a × K^b) / (G^c).

    Args:
        data: DataFrame with constructiveness, coherence_value, measured_gini_mean, and outcome
        outcome_col: Column name for the outcome variable

    Returns:
        Dictionary with fitted exponents and performance metrics
    """
    from scipy.optimize import minimize

    required_cols = [
        "constructiveness",
        "coherence_value",
        "measured_gini_mean",
        outcome_col,
    ]
    if not all(col in data.columns for col in required_cols):
        return {"a": 1.0, "b": 1.0, "c": 1.0, "r_squared": 0.0}

    # Remove rows with missing data
    clean_data = data.dropna(subset=required_cols)

    if len(clean_data) < 5:
        return {"a": 1.0, "b": 1.0, "c": 1.0, "r_squared": 0.0}

    def objective(exponents):
        """Objective function for UR formula fitting."""
        a, b, c = exponents

        # Calculate UR scores with fitted exponents
        ur_scores = []
        for _, row in clean_data.iterrows():
            c_val = row["constructiveness"]
            k_val = row["coherence_value"]
            g_val = row["measured_gini_mean"] + 1e-6

            ur_score = (c_val**a * k_val**b) / (g_val**c)
            ur_scores.append(ur_score)

        # Calculate correlation with outcome
        try:
            correlation = np.corrcoef(ur_scores, clean_data[outcome_col])[0, 1]
            return -abs(correlation) if not np.isnan(correlation) else 0
        except:
            return 0

    # Initial guess and bounds
    x0 = [1.0, 1.0, 1.0]  # Start with original formula
    bounds = [(0.1, 3.0), (0.1, 3.0), (0.1, 3.0)]  # Reasonable bounds

    try:
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        a, b, c = result.x
        r_squared = -result.fun

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "r_squared": float(r_squared),
        }
    except:
        return {"a": 1.0, "b": 1.0, "c": 1.0, "r_squared": 0.0}


def calculate_ur_score_variant(
    constructiveness: float,
    coherence_value: float,
    gini: float,
    exponents: dict[str, float],
) -> float:
    """Calculate UR score using fitted exponents.

    UR_score = (constructiveness^a × coherence_value^b) / (gini^c)
    """
    a = exponents.get("a", 1.0)
    b = exponents.get("b", 1.0)
    c = exponents.get("c", 1.0)

    return (constructiveness**a * coherence_value**b) / ((gini + 1e-6) ** c)


def calculate_ur_score(
    constructiveness: float, coherence_value: float, gini: float
) -> float:
    """Calculate Universal Resilience score.

    UR_score = (constructiveness * coherence_value) / (gini + 1e-6)
    """
    return (constructiveness * coherence_value) / (gini + 1e-6)


def generate_deterministic_seeds(base_seed: int, n_seeds: int) -> list:
    """Generate deterministic seeds for reproducibility."""
    np.random.seed(base_seed)
    return np.random.randint(0, 2**31, n_seeds).tolist()


def validate_gini_target(
    actual_gini: float, target_gini: float, tolerance: float
) -> bool:
    """Check if actual Gini is within tolerance of target."""
    return abs(actual_gini - target_gini) <= tolerance


def aggregate_cell_results(results: list) -> dict[str, Any]:
    """Aggregate results for a single experimental cell."""
    if not results:
        return {}

    # Extract numeric metrics
    metrics = [
        "final_alive_fraction",
        "time_to_recovery",
        "area_under_survival_curve",
        "collapse_flag",
        "measured_gini",
        "cci_mean",
        "cci_post_shock_mean",
        "constructiveness",
        "ur_score",
        "coherence_value",
    ]

    aggregated = {}

    for metric in metrics:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if values:
            # Ensure values are numeric
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue

            if numeric_values:
                aggregated[f"{metric}_mean"] = np.mean(numeric_values)
                aggregated[f"{metric}_std"] = np.std(numeric_values)
                aggregated[f"{metric}_min"] = np.min(numeric_values)
                aggregated[f"{metric}_max"] = np.max(numeric_values)
        else:
            aggregated[f"{metric}_mean"] = np.nan
            aggregated[f"{metric}_std"] = np.nan
            aggregated[f"{metric}_min"] = np.nan
            aggregated[f"{metric}_max"] = np.nan

    # Add metadata
    aggregated["n_replicates"] = len(results)

    # Add cell parameters (from first result)
    if results:
        first_result = results[0]
        for param in ["severity", "target_gini", "coherence_level", "n_agents"]:
            aggregated[param] = first_result.get(param)

    return aggregated

"""
Statistical analysis for Universal Resilience experiment - Patch v3.
Single source of truth for model metrics + UR fallbacks with variance guard.
"""

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def constructiveness(series_severity: np.ndarray, peak: float) -> np.ndarray:
    """Calculate constructiveness with given peak."""
    denom = max(peak, 1.0 - peak)
    return np.clip(1.0 - np.abs(series_severity - peak) / denom, 0.0, 1.0)


def r2_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Simple R² calculation with intercept."""
    try:
        if len(x) < 2 or len(y) < 2:
            return 0.0

        # Add intercept column
        X = np.column_stack([np.ones(len(x)), x])

        # Solve normal equations
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return 0.0

            return 1.0 - (ss_res / ss_tot)
        except np.linalg.LinAlgError:
            return 0.0
    except:
        return 0.0


def fit_constructiveness_peak(
    df_cells: pd.DataFrame, peaks: list[float], y_col: str
) -> dict[str, Any]:
    """Fit constructiveness peak using grid search."""
    best = {"peak": None, "r2": -1.0}

    for p in peaks:
        c = constructiveness(df_cells["severity"].values, p)
        r2 = r2_simple(c, df_cells[y_col].values)
        if r2 > best["r2"]:
            best = {"peak": p, "r2": r2}

    return best


def fit_ur_exponents(
    df_cells: pd.DataFrame,
    peak: float,
    eps: float,
    lam: float,
    init_grid: dict[str, list[float]],
    y_col: str,
) -> dict[str, Any]:
    """Fit UR exponents using grid search + log-OLS refinement."""

    if (
        df_cells.empty
        or y_col not in df_cells.columns
        or "severity" not in df_cells.columns
        or "coherence_value_mean" not in df_cells.columns
        or "measured_gini_mean" not in df_cells.columns
    ):
        return {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "r2_ur": 0.0,
            "r2_coherence": 0.0,
            "r2_gini": 0.0,
            "r2_constructiveness": 0.0,
            "p_star_used": peak,
        }

    C = constructiveness(df_cells["severity"].values, peak)
    K = np.clip(df_cells["coherence_value_mean"].values, eps, None)
    G = np.clip(df_cells["measured_gini_mean"].values, eps, None)
    Y = np.clip(df_cells[y_col].values, eps, None)

    # Coarse grid initialization
    best = {"a": 1.0, "b": 1.0, "c": 1.0, "r2": -1.0}

    for a, b, c in product(init_grid["a"], init_grid["b"], init_grid["c"]):
        ur = (C**a) * (K**b) / (G**c)
        r2 = r2_simple(ur, Y)
        if r2 > best["r2"]:
            best = {"a": a, "b": b, "c": c, "r2": r2}

    # Refine via log-OLS with ridge
    try:
        X = np.column_stack([np.log(C + eps), np.log(K + eps), -np.log(G + eps)])
        y = np.log(Y + eps)

        # Closed form ridge: beta = (X^T X + lam I)^-1 X^T y
        XtX = X.T @ X
        XtX += lam * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y)

        a_hat, b_hat, c_hat = beta.tolist()
        ur_refined = np.exp(X @ beta)
        r2_refined = r2_simple(ur_refined, Y)

        # Choose better of grid vs refined
        if r2_refined > best["r2"]:
            best = {"a": a_hat, "b": b_hat, "c": c_hat, "r2": r2_refined}
    except Exception as e:
        print(f"    Warning: UR refinement failed: {e}")

    # Calculate baseline R² for single factors
    r2_coherence = r2_simple(K, Y)
    r2_gini = r2_simple(1.0 / G, Y)  # Inverse Gini
    r2_constructiveness = r2_simple(C, Y)

    return {
        "a": best["a"],
        "b": best["b"],
        "c": best["c"],
        "r2_ur": best["r2"],
        "r2_coherence": r2_coherence,
        "r2_gini": r2_gini,
        "r2_constructiveness": r2_constructiveness,
        "p_star_used": peak,
    }


class UniversalResilienceAnalyzer:
    """Analyzes Universal Resilience experiment results with single source of truth."""

    def __init__(self):
        self.model_results = {}
        self.fitted_params = {}

    def analyze_results(
        self, cell_aggregates: list[dict[str, Any]], config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Analyze aggregated results with variance guard and UR fallbacks."""

        if not cell_aggregates:
            return {"error": "No data to analyze"}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(cell_aggregates)

        # Check variance guard
        Y = df["final_alive_fraction_mean"].values
        Y = np.array([float(y) if y is not None else 0.0 for y in Y])  # Ensure numeric
        min_target_variance = (
            float(config.get("ur_learning", {}).get("min_target_variance", 2e-2))
            if config
            else 2e-2
        )

        if np.var(Y) < min_target_variance:
            print(
                f"  ⚠️ Variance guard triggered: var(Y)={np.var(Y):.6f} < {min_target_variance}"
            )
            ur_fit = {
                "skipped": True,
                "reason": "low variance",
                "r2_ur": 0.0,
                "a": None,
                "b": None,
                "c": None,
                "peak": None,
            }

            # Calculate baseline models only
            model_metrics = self._calculate_baseline_models(df)
            model_metrics["ur"] = ur_fit

            # Save single source of truth
            self._save_model_fits(model_metrics, config)

            return {
                "model_metrics": model_metrics,
                "variance_guard_triggered": True,
                "target_variance": np.var(Y),
                "min_target_variance": min_target_variance,
            }

        # Normal analysis with UR fitting
        print("  🔍 Fitting UR parameters...")

        # A. Learn constructiveness peak (p*)
        ur_learning_config = config.get("ur_learning", {}) if config else {}
        peak_grid = ur_learning_config.get(
            "peak_grid", [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        )
        p_star_params = fit_constructiveness_peak(
            df, peak_grid, "final_alive_fraction_mean"
        )
        print(
            f"    ✓ Learned p*: {p_star_params.get('peak', 0.5):.3f} (R²={p_star_params.get('r2', 0.0):.3f})"
        )

        # B. Fit UR exponents (a,b,c)
        eps = float(ur_learning_config.get("log_smoothing_eps", 1e-3))
        lam = float(ur_learning_config.get("ridge_lambda", 1e-4))
        exponent_grid = ur_learning_config.get(
            "exponent_grid",
            {
                "a": [0.5, 1.0, 1.5, 2.0],
                "b": [0.5, 1.0, 1.5, 2.0],
                "c": [0.5, 1.0, 1.5, 2.0],
            },
        )

        ur_exponents_params = fit_ur_exponents(
            df,
            p_star_params.get("peak", 0.5),
            eps,
            lam,
            exponent_grid,
            "final_alive_fraction_mean",
        )
        print(
            f"    ✓ Learned exponents: a={ur_exponents_params.get('a', 1.0):.3f}, b={ur_exponents_params.get('b', 1.0):.3f}, c={ur_exponents_params.get('c', 1.0):.3f} (R²={ur_exponents_params.get('r2_ur', 0.0):.3f})"
        )

        # Calculate all model metrics
        model_metrics = self._calculate_all_models(
            df, p_star_params, ur_exponents_params
        )

        # Save single source of truth
        self._save_model_fits(model_metrics, config)

        print("✓ Completed statistical analysis")

        return {"model_metrics": model_metrics, "variance_guard_triggered": False}

    def _calculate_baseline_models(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate baseline models when UR fitting is skipped."""

        Y = df["final_alive_fraction_mean"].values

        # Single factor models
        r2_constructiveness = r2_simple(df["constructiveness_mean"].values, Y)
        r2_coherence = r2_simple(df["coherence_value_mean"].values, Y)
        r2_inv_gini = r2_simple(1.0 / df["measured_gini_mean"].values, Y)

        # Interaction model
        X_interaction = np.column_stack(
            [
                df["constructiveness_mean"].values,
                df["coherence_value_mean"].values,
                df["measured_gini_mean"].values,
            ]
        )

        try:
            model = LinearRegression()
            model.fit(X_interaction, Y)
            y_pred = model.predict(X_interaction)
            r2_interaction = r2_score(Y, y_pred)
        except:
            r2_interaction = 0.0

        return {
            "r2_constructiveness": r2_constructiveness,
            "r2_coherence": r2_coherence,
            "r2_inv_gini": r2_inv_gini,
            "r2_interaction": r2_interaction,
        }

    def _calculate_all_models(
        self,
        df: pd.DataFrame,
        p_star_params: dict[str, Any],
        ur_exponents_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate all models including UR."""

        Y = df["final_alive_fraction_mean"].values

        # Single factor models
        r2_constructiveness = r2_simple(df["constructiveness_mean"].values, Y)
        r2_coherence = r2_simple(df["coherence_value_mean"].values, Y)
        r2_inv_gini = r2_simple(1.0 / df["measured_gini_mean"].values, Y)

        # UR model with learned parameters
        peak = p_star_params.get("peak", 0.5)
        C = constructiveness(df["severity"].values, peak)
        K = df["coherence_value_mean"].values
        G = df["measured_gini_mean"].values

        a = ur_exponents_params.get("a", 1.0)
        b = ur_exponents_params.get("b", 1.0)
        c = ur_exponents_params.get("c", 1.0)

        UR_score = (C**a) * (K**b) / (G**c)
        r2_ur = r2_simple(UR_score, Y)

        # Interaction model
        X_interaction = np.column_stack(
            [
                df["constructiveness_mean"].values,
                df["coherence_value_mean"].values,
                df["measured_gini_mean"].values,
            ]
        )

        try:
            model = LinearRegression()
            model.fit(X_interaction, Y)
            y_pred = model.predict(X_interaction)
            r2_interaction = r2_score(Y, y_pred)
        except:
            r2_interaction = 0.0

        return {
            "r2_constructiveness": r2_constructiveness,
            "r2_coherence": r2_coherence,
            "r2_inv_gini": r2_inv_gini,
            "r2_ur": r2_ur,
            "r2_interaction": r2_interaction,
            "ur": {"skipped": False, "peak": peak, "a": a, "b": b, "c": c, "r2": r2_ur},
        }

    def _save_model_fits(
        self, model_metrics: dict[str, Any], config: dict[str, Any] = None
    ):
        """Save model fits as single source of truth."""

        # Save as JSON (single source of truth)
        output_dir = (
            config.get("outputs", {}).get(
                "root_dir", "discovery_results/universal_resilience"
            )
            if config
            else "discovery_results/universal_resilience"
        )

        # This will be called from run.py with the proper output directory
        self.model_metrics = model_metrics

        # Also create flat CSV for bar chart
        csv_data = []

        # Add baseline models
        csv_data.append(
            {
                "model_name": "constructiveness",
                "r_squared": model_metrics.get("r2_constructiveness", 0.0),
                "model_type": "single_factor",
            }
        )
        csv_data.append(
            {
                "model_name": "coherence",
                "r_squared": model_metrics.get("r2_coherence", 0.0),
                "model_type": "single_factor",
            }
        )
        csv_data.append(
            {
                "model_name": "inv_gini",
                "r_squared": model_metrics.get("r2_inv_gini", 0.0),
                "model_type": "single_factor",
            }
        )

        # Add UR model if not skipped
        if not model_metrics.get("ur", {}).get("skipped", False):
            csv_data.append(
                {
                    "model_name": "ur",
                    "r_squared": model_metrics.get("r2_ur", 0.0),
                    "model_type": "ur_formula",
                }
            )

        # Add interaction model
        csv_data.append(
            {
                "model_name": "interaction",
                "r_squared": model_metrics.get("r2_interaction", 0.0),
                "model_type": "interaction",
            }
        )

        self.csv_data = csv_data

    def calculate_key_findings(
        self, analysis_results: dict[str, Any], cell_aggregates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate key findings for the report."""
        findings = {}

        # Get model metrics
        model_metrics = analysis_results.get("model_metrics", {})

        # Best R²
        best_r2 = max(
            model_metrics.get("r2_constructiveness", 0.0),
            model_metrics.get("r2_coherence", 0.0),
            model_metrics.get("r2_inv_gini", 0.0),
            model_metrics.get("r2_ur", 0.0),
            model_metrics.get("r2_interaction", 0.0),
        )
        findings["best_r_squared"] = best_r2

        # UR Score R²
        ur_r2 = model_metrics.get("r2_ur", 0.0)
        findings["ur_score_r_squared"] = ur_r2

        # Collapse rate
        df_agg = pd.DataFrame(cell_aggregates)
        if "collapsed_flag_rate" in df_agg.columns:
            collapse_rate = df_agg["collapsed_flag_rate"].mean()
            findings["catastrophic_collapse_rate"] = collapse_rate

        # Optimal shock
        if "final_alive_fraction_mean" in df_agg.columns:
            best_resilience_idx = df_agg["final_alive_fraction_mean"].idxmax()
            optimal_severity = df_agg.loc[best_resilience_idx, "severity"]
            findings["optimal_shock_severity"] = optimal_severity

        # Coherence effect
        if (
            "coherence_value_mean" in df_agg.columns
            and "final_alive_fraction_mean" in df_agg.columns
        ):
            try:
                correlation = df_agg["coherence_value_mean"].corr(
                    df_agg["final_alive_fraction_mean"]
                )
                findings["coherence_effect_size"] = correlation
            except:
                findings["coherence_effect_size"] = None

        return findings


def create_analyzer() -> UniversalResilienceAnalyzer:
    """Create an analyzer instance."""
    return UniversalResilienceAnalyzer()

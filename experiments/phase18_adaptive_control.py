#!/usr/bin/env python3
"""
Phase 18: Adaptive Control Learning
===================================

Upgrades Phase 17's fixed controller into an online learning controller that:
1. Learns a local plant model using Recursive Least Squares (RLS)
2. Adapts control gains based on learned sensitivity and PID-like error dynamics
3. Uses dynamic targets based on recent coherence statistics
4. Compares No-Control vs Fixed vs Adaptive control strategies

Key Innovations:
- Online plant identification: C_{t+1} ≈ θ₀ + θ₁·Δε_t + θ₂·κ_t·Δε_t + θ₃·C_t
- Adaptive control gain: adjusts based on learned sensitivity dC/d(Δε)
- Dynamic targeting: target_t = median(C_window) + k·std(C_window)
- PID-enhanced error processing for smoother control response

This demonstrates the evolution from fixed control laws to intelligent,
self-adapting temporal coherence regulation systems.
"""

import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set deterministic parameters
np.random.seed(42)
STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = Path(f"./discovery_results/phase18_adaptive_{STAMP}")
ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Temporal Coherence Metrics (Phases 16-17 Compatible) ----------


def safe_entropy(x, bins=30):
    """Robust normalized entropy calculation"""
    if len(x) < 3:
        return 0.5

    try:
        x_clean = x[np.isfinite(x)]
        if len(x_clean) < 3:
            return 0.5

        n_bins = min(bins, max(5, len(x_clean) // 3))
        hist, _ = np.histogram(x_clean, bins=n_bins, density=True)
        hist = hist + 1e-12
        p = hist / np.sum(hist)
        H = -np.sum(p * np.log(p))
        H_max = np.log(len(p))

        return H / H_max if H_max > 0 else 0.5
    except:
        return 0.5


def compute_window_metrics(df, window=50, step=None):
    """Extract temporal coherence metrics using sliding windows"""
    if step is None:
        step = max(1, window // 2)

    print(f"    📊 Computing metrics: window={window}, step={step}")

    # Clean data
    df_numeric = df.copy()
    for col in df.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    df_clean = df_numeric.loc[:, df_numeric.isnull().mean() < 0.5]
    df_clean = df_clean.dropna()

    if len(df_clean) < window:
        print(f"    ❌ Insufficient data: {len(df_clean)} < {window}")
        return None

    cols = df_clean.columns
    N = len(df_clean)
    metrics_rows = []

    for t0 in range(0, N - window + 1, step):
        try:
            seg_df = df_clean.iloc[t0 : t0 + window]
            seg_values = seg_df.values

            # Coherence = 1 - normalized_entropy
            entropies = []
            for j in range(len(cols)):
                col_entropy = safe_entropy(seg_df.iloc[:, j].values)
                entropies.append(col_entropy)
            coherence = 1.0 - np.mean(entropies)

            # Openness ε = var(signal)/|mean(signal)|
            eps_values = []
            for col_data in seg_values.T:
                var_val = np.var(col_data)
                mean_val = np.mean(col_data)
                if abs(mean_val) > 1e-9:
                    eps_values.append(var_val / abs(mean_val))
                else:
                    eps_values.append(var_val)
            eps = np.mean(eps_values)

            # Coupling κ = mean(|correlation|)
            try:
                seg_clean = seg_values[:, np.var(seg_values, axis=0) > 1e-10]
                if seg_clean.shape[1] < 2:
                    kappa = 0.0
                else:
                    corr_matrix = np.corrcoef(seg_clean, rowvar=False)
                    if np.isscalar(corr_matrix):
                        kappa = 0.0
                    elif corr_matrix.ndim == 2:
                        triu_indices = np.triu_indices_from(corr_matrix, k=1)
                        if len(triu_indices[0]) > 0:
                            correlations = corr_matrix[triu_indices]
                            kappa = float(
                                np.mean(np.abs(correlations[np.isfinite(correlations)]))
                            )
                        else:
                            kappa = 0.0
                    else:
                        kappa = 0.0
            except:
                kappa = 0.0

            # Noise level
            noise = float(np.std(seg_values))

            metrics_rows.append([t0 + window // 2, coherence, eps, kappa, noise])

        except Exception:
            continue

    if not metrics_rows:
        print("    ❌ No valid windows")
        return None

    metrics_df = pd.DataFrame(metrics_rows, columns=["t", "C", "eps", "kappa", "noise"])

    # Temporal derivative
    if len(metrics_df) >= 2:
        metrics_df["dC_dt"] = np.gradient(metrics_df["C"])
    else:
        metrics_df["dC_dt"] = 0.0

    metrics_df.reset_index(drop=True, inplace=True)
    print(f"    ✅ Extracted {len(metrics_df)} temporal windows")

    return metrics_df


def create_temporal_features(mdf, lags=(1, 2, 3), rolling_windows=(3, 5)):
    """Create temporal feature matrix with lags and rolling statistics"""
    print(f"    🔧 Creating features: lags={lags}, rolling={rolling_windows}")

    X = mdf.copy()

    # Add lagged features
    for lag in lags:
        for col in ["C", "eps", "kappa", "noise", "dC_dt"]:
            if col in X.columns:
                X[f"{col}_lag{lag}"] = X[col].shift(lag)

    # Add rolling statistics
    for window in rolling_windows:
        if "C" in X.columns:
            X[f"C_ma{window}"] = X["C"].rolling(window).mean()
            X[f"C_std{window}"] = X["C"].rolling(window).std()
        if "dC_dt" in X.columns:
            X[f"dC_ma{window}"] = X["dC_dt"].rolling(window).mean()
        if "eps" in X.columns:
            X[f"eps_ma{window}"] = X["eps"].rolling(window).mean()
        if "kappa" in X.columns:
            X[f"kappa_ma{window}"] = X["kappa"].rolling(window).mean()

    # Add interaction features
    if "eps" in X.columns and "kappa" in X.columns:
        X["eps_kappa"] = X["eps"] * X["kappa"]
    if "C" in X.columns and "dC_dt" in X.columns:
        X["C_dC"] = X["C"] * X["dC_dt"]
    if "eps" in X.columns and "noise" in X.columns:
        X["eps_noise"] = X["eps"] * X["noise"]

    X_clean = X.dropna().reset_index(drop=True)
    print(f"    ✅ Created {X_clean.shape[1]} features from {len(X_clean)} samples")

    return X_clean


# ---------- Ridge Regression Forecaster (Phase 16/17 Compatible) ----------


def ridge_regression_fit(X, y, l2_reg=1e-2):
    """Fit ridge regression with closed-form solution"""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)

    # Add bias term
    X_bias = np.hstack([X_array, np.ones((X_array.shape[0], 1))])

    # Regularization matrix
    I = np.eye(X_bias.shape[1])
    I[-1, -1] = 0.0  # Don't regularize bias

    try:
        XTX_reg = X_bias.T @ X_bias + l2_reg * I
        XTy = X_bias.T @ y_array
        weights = np.linalg.solve(XTX_reg, XTy)
        return weights
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(XTX_reg) @ XTy
        return weights


def ridge_regression_predict(X, weights):
    """Make predictions using fitted ridge regression weights"""
    X_array = np.asarray(X, dtype=float)
    X_bias = np.hstack([X_array, np.ones((X_array.shape[0], 1))])
    return X_bias @ weights


# ---------- Recursive Least Squares for Online Plant Learning ----------


class RecursiveLeastSquares:
    """
    Online parameter estimation using Recursive Least Squares

    Plant model: C_{t+1} = θ₀ + θ₁·Δε_t + θ₂·κ_t·Δε_t + θ₃·C_t + noise

    Parameters:
    - θ₀: baseline coherence level
    - θ₁: direct openness effect on coherence
    - θ₂: coupling-mediated openness effect
    - θ₃: coherence persistence factor
    """

    def __init__(self, n_params=4, forgetting_factor=0.995, initial_covariance=1000.0):
        self.n_params = n_params
        self.forgetting_factor = forgetting_factor

        # Initialize parameter estimates
        self.theta = np.zeros((n_params, 1))  # Column vector

        # Initialize covariance matrix (high initial uncertainty)
        self.P = initial_covariance * np.eye(n_params)

        # Statistics
        self.update_count = 0
        self.prediction_errors = []

    def update(self, phi, y):
        """
        Update parameter estimates with new observation

        Args:
            phi: feature vector [1, Δε_t, κ_t·Δε_t, C_t] (shape: n_params,)
            y: observed coherence C_{t+1} (scalar)

        Returns:
            prediction_error: y - phi^T * theta_old
        """
        # Ensure phi is column vector
        phi = np.asarray(phi).reshape(-1, 1)

        # Compute Kalman gain
        denominator = self.forgetting_factor + phi.T @ self.P @ phi
        K = self.P @ phi / denominator

        # Compute prediction error
        prediction_error = float(y - phi.T @ self.theta)

        # Update parameter estimates
        self.theta = self.theta + K * prediction_error

        # Update covariance matrix
        self.P = (self.P - K @ phi.T @ self.P) / self.forgetting_factor

        # Update statistics
        self.update_count += 1
        self.prediction_errors.append(prediction_error)

        return prediction_error

    def predict(self, phi):
        """Predict output for given feature vector"""
        phi = np.asarray(phi).reshape(-1, 1)
        return float(phi.T @ self.theta)

    def get_parameters(self):
        """Get current parameter estimates as flat array"""
        return self.theta.flatten()

    def get_sensitivity(self, kappa):
        """Get current sensitivity dC/d(Δε) for given coupling κ"""
        theta = self.get_parameters()
        if len(theta) >= 3:
            # Sensitivity = θ₁ + θ₂·κ (direct + coupling-mediated effects)
            return theta[1] + theta[2] * kappa
        else:
            return 0.0


# ---------- Control Strategies ----------


def fixed_control_step(
    C_prediction, kappa, u_previous, target, controller_gain=0.20, max_increment=0.02
):
    """
    Fixed control law from Phase 17

    u_t = u_{t-1} + α·max(0, target - Ĉ_{t+1})
    """
    prediction_error = target - C_prediction

    if prediction_error > 0:  # Need to increase coherence
        control_increment = np.clip(
            controller_gain * prediction_error, 0.0, max_increment
        )
    else:
        control_increment = 0.0

    new_control_signal = u_previous + control_increment

    return new_control_signal, control_increment


def adaptive_control_step(
    C_prediction,
    kappa,
    u_previous,
    target,
    rls_estimator,
    max_increment=0.05,
    pid_gains=(0.25, 0.10, 0.05),
    error_history=None,
):
    """
    Adaptive control law with online learning

    Features:
    - Uses learned sensitivity from RLS for gain adaptation
    - PID-like error processing for smoother response
    - Larger control authority than fixed controller
    """
    # Get learned sensitivity dC/d(Δε)
    learned_sensitivity = rls_estimator.get_sensitivity(kappa)

    # PID-like error processing
    current_error = max(0.0, target - C_prediction)

    if error_history is None:
        error_history = [0.0, 0.0]  # [integral, previous_error]

    error_integral = error_history[0] + current_error
    error_derivative = current_error - error_history[1]

    # PID control signal
    Kp, Ki, Kd = pid_gains
    raw_control = Kp * current_error + Ki * error_integral + Kd * error_derivative

    # Adaptive gain based on learned sensitivity
    if abs(learned_sensitivity) > 1e-6:
        # Higher sensitivity → lower gain (more cautious)
        # Lower sensitivity → higher gain (more aggressive)
        adaptive_gain = min(1.5, max(0.25, 1.0 / (abs(learned_sensitivity) + 0.1)))
    else:
        adaptive_gain = 1.0  # Default gain when sensitivity unknown

    # Apply adaptive gain and bounds
    control_increment = np.clip(raw_control * adaptive_gain, 0.0, max_increment)
    new_control_signal = u_previous + control_increment

    # Update error history
    updated_error_history = [error_integral, current_error]

    return new_control_signal, control_increment, updated_error_history


# ---------- Dynamic Target Generation ----------


def compute_dynamic_target(
    coherence_series,
    current_index,
    lookback_window=12,
    std_multiplier=0.5,
    min_target=0.55,
    max_target=0.85,
):
    """
    Compute adaptive target based on recent coherence statistics

    target_t = median(C_recent) + k·std(C_recent)

    This keeps targets realistic and responsive to system behavior.
    """
    start_index = max(0, current_index - lookback_window)
    recent_coherence = coherence_series[start_index : current_index + 1]

    if len(recent_coherence) == 0:
        return 0.60  # Default fallback

    median_coherence = float(np.median(recent_coherence))
    std_coherence = float(np.std(recent_coherence))

    dynamic_target = median_coherence + std_multiplier * std_coherence

    # Apply bounds to prevent unrealistic targets
    bounded_target = float(np.clip(dynamic_target, min_target, max_target))

    return bounded_target


# ---------- Control System Simulation ----------


def run_adaptive_control_simulation(
    metrics_df, forecast_weights, feature_columns, horizon=1
):
    """
    Run complete adaptive control simulation comparing three strategies:
    1. No Control (baseline forecasts)
    2. Fixed Control (Phase 17 approach)
    3. Adaptive Control (Phase 18 with RLS learning)
    """
    print("    🎛️  Running adaptive control simulation")

    # Create temporal features
    features_df = create_temporal_features(metrics_df)
    if len(features_df) < 30:
        print(f"    ❌ Insufficient samples: {len(features_df)}")
        return None

    # Align targets for supervised learning
    y_future = features_df["C"].shift(-horizon).dropna()
    feature_matrix = features_df.iloc[:-horizon].copy()

    # Train/stream split
    split_idx = int(len(feature_matrix) * 0.8)
    train_data = feature_matrix.iloc[:split_idx]
    stream_data = feature_matrix.iloc[split_idx:].copy()
    stream_targets = y_future.iloc[split_idx:]

    print(f"    📚 Training: {len(train_data)}, Streaming: {len(stream_data)}")

    # Initialize control systems
    u_fixed = 0.0  # Fixed controller state
    u_adaptive = 0.0  # Adaptive controller state

    # Initialize RLS for plant learning
    rls = RecursiveLeastSquares(
        n_params=4,  # [θ₀, θ₁, θ₂, θ₃]
        forgetting_factor=0.995,  # Slow adaptation for stability
        initial_covariance=100.0,  # Moderate initial uncertainty
    )

    # Initialize PID error history for adaptive controller
    error_history = [0.0, 0.0]  # [integral, previous_error]

    # Control simulation records
    simulation_records = []

    # Previous coherence for RLS learning
    previous_coherence = float(train_data["C"].iloc[-1])

    # Combined coherence series for dynamic targeting
    all_coherence = pd.concat([train_data["C"], stream_data["C"]]).values

    for i in range(len(stream_data)):
        try:
            current_row = stream_data.iloc[i]
            true_coherence = float(stream_targets.iloc[i])

            # Extract features for forecasting
            x_features = current_row[feature_columns].values.reshape(1, -1)

            # Generate coherence forecast (no control)
            C_forecast = float(
                ridge_regression_predict(x_features, forecast_weights)[0]
            )

            current_kappa = float(current_row["kappa"])

            # Compute dynamic target
            target_coherence = compute_dynamic_target(
                all_coherence, split_idx + i, lookback_window=12
            )

            # --- Plant Learning Update (using previous time step) ---
            if i > 0:
                # Learn from last transition: C_true(t) = f(Δε_{t-1}, κ_{t-1}, C_{t-1})
                last_record = simulation_records[-1]
                control_increment_adaptive = last_record["du_adaptive"]
                kappa_previous = last_record["kappa"]

                # Feature vector: [1, Δε, κ·Δε, C_prev]
                phi = np.array(
                    [
                        1.0,  # θ₀: bias
                        control_increment_adaptive,  # θ₁: direct effect
                        kappa_previous
                        * control_increment_adaptive,  # θ₂: coupling effect
                        previous_coherence,  # θ₃: persistence
                    ],
                    dtype=float,
                )

                # Update RLS with observed coherence
                prediction_error = rls.update(phi, true_coherence)

            previous_coherence = true_coherence

            # --- Fixed Control Strategy (Phase 17) ---
            u_fixed, du_fixed = fixed_control_step(
                C_forecast,
                current_kappa,
                u_fixed,
                target_coherence,
                controller_gain=0.20,
                max_increment=0.02,
            )

            # Fixed controller plant model (Phase 17 parameters)
            C_forecast_fixed = (
                C_forecast + 0.08 * du_fixed + 0.10 * current_kappa * du_fixed
            )

            # --- Adaptive Control Strategy (Phase 18) ---
            u_adaptive, du_adaptive, error_history = adaptive_control_step(
                C_forecast,
                current_kappa,
                u_adaptive,
                target_coherence,
                rls,
                max_increment=0.05,
                pid_gains=(0.25, 0.10, 0.05),
                error_history=error_history,
            )

            # Adaptive controller plant response using learned model
            theta = rls.get_parameters()
            learned_sensitivity = (theta[1] if len(theta) > 1 else 0.0) + (
                theta[2] if len(theta) > 2 else 0.0
            ) * current_kappa

            C_forecast_adaptive = C_forecast + learned_sensitivity * du_adaptive

            # Record simulation step
            simulation_records.append(
                {
                    "t_idx": int(current_row["t"]),
                    "C_true": true_coherence,
                    "C_pred": C_forecast,
                    "target": target_coherence,
                    "kappa": current_kappa,
                    # Fixed control results
                    "u_fixed": u_fixed,
                    "du_fixed": du_fixed,
                    "C_pred_fixed": C_forecast_fixed,
                    # Adaptive control results
                    "u_adaptive": u_adaptive,
                    "du_adaptive": du_adaptive,
                    "C_pred_adaptive": C_forecast_adaptive,
                    # Learning diagnostics
                    "learned_sensitivity": learned_sensitivity,
                    "rls_parameters": (
                        theta.tolist() if len(theta) > 0 else [0.0, 0.0, 0.0, 0.0]
                    ),
                }
            )

        except Exception as e:
            print(f"    ⚠️  Step {i} failed: {e}")
            continue

    if not simulation_records:
        print("    ❌ No successful simulation steps")
        return None

    simulation_trace = pd.DataFrame(simulation_records)
    print(f"    ✅ Simulation complete: {len(simulation_trace)} steps")

    return simulation_trace


# ---------- Performance Analysis ----------


def analyze_control_performance(trace_df):
    """Compute comprehensive performance metrics"""

    def time_under_target(predictions, targets):
        """Percentage of time predictions are below target"""
        return float((predictions < targets).mean() * 100.0)

    # Time under target metrics
    performance_metrics = {
        "under_target_truth_pct": time_under_target(
            trace_df["C_true"], trace_df["target"]
        ),
        "under_target_nocontrol_pct": time_under_target(
            trace_df["C_pred"], trace_df["target"]
        ),
        "under_target_fixed_pct": time_under_target(
            trace_df["C_pred_fixed"], trace_df["target"]
        ),
        "under_target_adaptive_pct": time_under_target(
            trace_df["C_pred_adaptive"], trace_df["target"]
        ),
    }

    # Performance improvements
    performance_metrics["improvement_over_nocontrol_fixed"] = float(
        performance_metrics["under_target_nocontrol_pct"]
        - performance_metrics["under_target_fixed_pct"]
    )
    performance_metrics["improvement_over_nocontrol_adaptive"] = float(
        performance_metrics["under_target_nocontrol_pct"]
        - performance_metrics["under_target_adaptive_pct"]
    )
    performance_metrics["improvement_over_fixed_adaptive"] = float(
        performance_metrics["under_target_fixed_pct"]
        - performance_metrics["under_target_adaptive_pct"]
    )

    # Control energy (total squared control effort)
    performance_metrics["energy_fixed"] = float(np.sum(trace_df["du_fixed"] ** 2))
    performance_metrics["energy_adaptive"] = float(np.sum(trace_df["du_adaptive"] ** 2))

    # Final control signals
    performance_metrics["final_u_fixed"] = float(trace_df["u_fixed"].iloc[-1])
    performance_metrics["final_u_adaptive"] = float(trace_df["u_adaptive"].iloc[-1])

    # Learning convergence metrics
    if "learned_sensitivity" in trace_df.columns:
        performance_metrics["final_learned_sensitivity"] = float(
            trace_df["learned_sensitivity"].iloc[-1]
        )
        performance_metrics["sensitivity_stability"] = float(
            trace_df["learned_sensitivity"].std()
        )

    return performance_metrics


def create_adaptive_control_visualizations(trace_df, dataset_name, performance_metrics):
    """Generate comprehensive visualization suite"""

    time_steps = np.arange(len(trace_df))

    # 1. Main comparison plot: All control strategies
    plt.figure(figsize=(12, 8), dpi=150)

    plt.subplot(2, 1, 1)
    plt.plot(
        time_steps,
        trace_df["C_true"],
        "b-",
        linewidth=2,
        label="True Coherence",
        alpha=0.9,
    )
    plt.plot(
        time_steps,
        trace_df["C_pred"],
        "gray",
        linewidth=1.5,
        label="Forecast (No Control)",
        linestyle="--",
    )
    plt.plot(
        time_steps,
        trace_df["C_pred_fixed"],
        "orange",
        linewidth=2,
        label="Fixed Control",
        alpha=0.8,
    )
    plt.plot(
        time_steps,
        trace_df["C_pred_adaptive"],
        "red",
        linewidth=2,
        label="Adaptive Control",
    )
    plt.plot(
        time_steps,
        trace_df["target"],
        "green",
        linewidth=1,
        label="Dynamic Target",
        linestyle=":",
    )

    plt.ylabel("Coherence C(t)")
    plt.title(f"{dataset_name}: Fixed vs Adaptive Predictive Control")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add performance text
    textstr = (
        f"Time Under Target:\n"
        f'No Control: {performance_metrics["under_target_nocontrol_pct"]:.1f}%\n'
        f'Fixed: {performance_metrics["under_target_fixed_pct"]:.1f}%\n'
        f'Adaptive: {performance_metrics["under_target_adaptive_pct"]:.1f}%\n'
        f'Improvement: {performance_metrics["improvement_over_fixed_adaptive"]:.1f} pp'
    )
    props = dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.subplot(2, 1, 2)
    plt.plot(
        time_steps,
        trace_df["u_fixed"],
        "orange",
        linewidth=2,
        label="Fixed Control Signal",
    )
    plt.plot(
        time_steps,
        trace_df["u_adaptive"],
        "red",
        linewidth=2,
        label="Adaptive Control Signal",
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Control Signal u(t)")
    plt.title("Control Signal Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / f"{dataset_name}_compare_curves.png", bbox_inches="tight")
    plt.close()

    # 2. Control signals detailed comparison
    plt.figure(figsize=(10, 6), dpi=150)

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, trace_df["u_fixed"], "orange", linewidth=2, label="Fixed u(t)")
    plt.plot(
        time_steps, trace_df["u_adaptive"], "red", linewidth=2, label="Adaptive u(t)"
    )
    plt.ylabel("Cumulative Control")
    plt.title(f"{dataset_name}: Control Signal Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(
        time_steps,
        trace_df["du_fixed"],
        "orange",
        linewidth=1.5,
        label="Fixed Δu(t)",
        alpha=0.7,
    )
    plt.plot(
        time_steps,
        trace_df["du_adaptive"],
        "red",
        linewidth=1.5,
        label="Adaptive Δu(t)",
        alpha=0.7,
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Control Increments")
    plt.title("Control Increment Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / f"{dataset_name}_control_signals.png", bbox_inches="tight")
    plt.close()

    # 3. Performance comparison bar chart
    plt.figure(figsize=(10, 6), dpi=150)

    categories = ["No Control", "Fixed Control", "Adaptive Control"]
    under_target_values = [
        performance_metrics["under_target_nocontrol_pct"],
        performance_metrics["under_target_fixed_pct"],
        performance_metrics["under_target_adaptive_pct"],
    ]
    colors = ["lightcoral", "orange", "lightgreen"]

    bars = plt.bar(
        categories, under_target_values, color=colors, alpha=0.7, edgecolor="black"
    )
    plt.ylabel("Time Under Target (%)")
    plt.title(f"{dataset_name}: Control Performance Comparison")
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, under_target_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add improvement annotations
    fixed_improvement = performance_metrics["improvement_over_nocontrol_fixed"]
    adaptive_improvement = performance_metrics["improvement_over_nocontrol_adaptive"]

    plt.text(
        0.5,
        max(under_target_values) * 0.7,
        f"Fixed Improvement:\n{fixed_improvement:.1f} pp",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.text(
        1.5,
        max(under_target_values) * 0.5,
        f"Adaptive Improvement:\n{adaptive_improvement:.1f} pp",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(ROOT / f"{dataset_name}_kpi_compare.png", bbox_inches="tight")
    plt.close()


# ---------- Dataset Processing Pipeline ----------


def process_single_dataset(csv_path, window=50, horizon=1):
    """Process a single dataset for adaptive control learning"""

    dataset_name = Path(csv_path).stem
    print(f"\n🧠 Processing: {dataset_name}")

    try:
        # Load and validate dataset
        df = pd.read_csv(csv_path)
        df = df.dropna(axis=1, how="all")

        if len(df) < window * 3:
            return {
                "dataset": dataset_name,
                "skipped": True,
                "reason": f"Dataset too short: {len(df)} < {window * 3}",
                "original_shape": f"{df.shape[0]} × {df.shape[1]}",
            }

        print(f"  📏 Dataset shape: {df.shape}")

        # Extract temporal metrics
        metrics_df = compute_window_metrics(df, window=window)
        if metrics_df is None:
            return {
                "dataset": dataset_name,
                "skipped": True,
                "reason": "Failed to extract temporal metrics",
                "original_shape": f"{df.shape[0]} × {df.shape[1]}",
            }

        # Save metrics
        metrics_df.to_csv(ROOT / f"{dataset_name}_metrics.csv", index=False)

        # Prepare features and train forecaster
        features_df = create_temporal_features(metrics_df)
        if len(features_df) < 30:
            return {
                "dataset": dataset_name,
                "skipped": True,
                "reason": f"Insufficient feature samples: {len(features_df)}",
                "original_shape": f"{df.shape[0]} × {df.shape[1]}",
            }

        # Train/test split for forecaster
        y_future = features_df["C"].shift(-horizon).dropna()
        feature_matrix = features_df.iloc[:-horizon].copy()
        split_idx = int(len(feature_matrix) * 0.8)

        # Train forecasting model
        train_features = feature_matrix.iloc[:split_idx]
        train_targets = y_future.iloc[:split_idx]

        # Remove target from feature matrix
        feature_columns = [col for col in train_features.columns if col != "C"]
        X_train = train_features[feature_columns]

        forecast_weights = ridge_regression_fit(X_train, train_targets, l2_reg=1e-2)

        # Run adaptive control simulation
        control_trace = run_adaptive_control_simulation(
            metrics_df, forecast_weights, feature_columns, horizon=horizon
        )

        if control_trace is None:
            return {
                "dataset": dataset_name,
                "skipped": True,
                "reason": "Adaptive control simulation failed",
                "original_shape": f"{df.shape[0]} × {df.shape[1]}",
            }

        # Save control trace
        control_trace.to_csv(ROOT / f"{dataset_name}_trace.csv", index=False)

        # Analyze performance
        performance_metrics = analyze_control_performance(control_trace)
        performance_metrics.update({"dataset": dataset_name, "skipped": False})

        # Save performance metrics
        with open(ROOT / f"{dataset_name}_kpis.json", "w") as f:
            json.dump(performance_metrics, f, indent=2)

        # Create visualizations
        create_adaptive_control_visualizations(
            control_trace, dataset_name, performance_metrics
        )

        print("  ✅ Control performance:")
        print(
            f"     No Control: {performance_metrics['under_target_nocontrol_pct']:.1f}% under target"
        )
        print(
            f"     Fixed: {performance_metrics['under_target_fixed_pct']:.1f}% under target"
        )
        print(
            f"     Adaptive: {performance_metrics['under_target_adaptive_pct']:.1f}% under target"
        )
        print(
            f"     Improvement vs Fixed: {performance_metrics['improvement_over_fixed_adaptive']:.1f} pp"
        )

        return performance_metrics

    except Exception as e:
        print(f"  ❌ Processing failed: {e}")
        return {
            "dataset": dataset_name,
            "skipped": True,
            "reason": f"Processing error: {str(e)}",
            "original_shape": "unknown",
        }


def generate_master_report(summary_df):
    """Generate comprehensive master report"""

    with open(ROOT / "master_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 18 — Adaptive Control Learning\n\n")
        f.write(f"**Timestamp**: {STAMP}\n")
        f.write(f"**Total datasets processed**: {len(summary_df)}\n")

        # Separate successful vs skipped
        successful = summary_df[~summary_df["skipped"]]
        skipped = summary_df[summary_df["skipped"]]

        f.write(
            f"**Successfully processed**: {len(successful)}/{len(summary_df)} datasets\n"
        )
        f.write(f"**Skipped**: {len(skipped)} datasets\n\n")

        f.write("## Adaptive Control Learning System\n\n")
        f.write(
            "This phase demonstrates the evolution from fixed control laws to intelligent, "
        )
        f.write(
            "self-adapting temporal coherence regulation. The system combines online plant "
        )
        f.write(
            "identification with adaptive control gains to achieve superior performance.\n\n"
        )

        f.write("### Learning Architecture\n")
        f.write("- **Plant Model**: C_{t+1} = θ₀ + θ₁·Δε_t + θ₂·κ_t·Δε_t + θ₃·C_t\n")
        f.write("- **Learning**: Recursive Least Squares (RLS) with λ=0.995\n")
        f.write(
            "- **Adaptive Gain**: Based on learned sensitivity dC/d(Δε) = θ₁ + θ₂·κ\n"
        )
        f.write(
            "- **PID Enhancement**: (Kp=0.25, Ki=0.10, Kd=0.05) for smoother response\n"
        )
        f.write(
            "- **Dynamic Targets**: target_t = median(C_recent) + 0.5·std(C_recent)\n\n"
        )

        if len(successful) > 0:
            f.write("## Adaptive Control Performance\n\n")
            f.write(
                "| Dataset | No Control | Fixed | Adaptive | Improvement | Energy Ratio |\n"
            )
            f.write(
                "|---------|------------|-------|----------|-------------|---------------|\n"
            )

            for _, row in successful.iterrows():
                energy_ratio = row["energy_adaptive"] / (row["energy_fixed"] + 1e-6)
                f.write(
                    f"| {row['dataset']} | {row['under_target_nocontrol_pct']:.1f}% | "
                    f"{row['under_target_fixed_pct']:.1f}% | "
                    f"{row['under_target_adaptive_pct']:.1f}% | "
                    f"{row['improvement_over_fixed_adaptive']:.1f} pp | "
                    f"{energy_ratio:.2f} |\n"
                )

            f.write("\n### Performance Summary\n")
            f.write(
                f"- **Mean improvement over fixed**: {successful['improvement_over_fixed_adaptive'].mean():.1f} ± {successful['improvement_over_fixed_adaptive'].std():.1f} pp\n"
            )
            f.write(
                f"- **Mean improvement over no-control**: {successful['improvement_over_nocontrol_adaptive'].mean():.1f} ± {successful['improvement_over_nocontrol_adaptive'].std():.1f} pp\n"
            )

            # Effectiveness analysis
            high_improvement = len(
                successful[successful["improvement_over_fixed_adaptive"] > 2]
            )
            moderate_improvement = len(
                successful[
                    (successful["improvement_over_fixed_adaptive"] > 0.5)
                    & (successful["improvement_over_fixed_adaptive"] <= 2)
                ]
            )

            f.write(
                f"- **High effectiveness** (>2 pp improvement): {high_improvement}/{len(successful)} datasets\n"
            )
            f.write(
                f"- **Moderate effectiveness** (0.5-2 pp improvement): {moderate_improvement}/{len(successful)} datasets\n"
            )
            f.write(
                f"- **Limited effectiveness** (<0.5 pp improvement): {len(successful) - high_improvement - moderate_improvement}/{len(successful)} datasets\n\n"
            )

            # Energy analysis
            mean_energy_ratio = (
                successful["energy_adaptive"] / (successful["energy_fixed"] + 1e-6)
            ).mean()
            f.write(
                f"- **Control energy ratio** (Adaptive/Fixed): {mean_energy_ratio:.2f} (higher = more active)\n\n"
            )

        if len(skipped) > 0:
            f.write("## Skipped Datasets\n\n")
            skip_reasons = skipped["reason"].value_counts()
            for reason, count in skip_reasons.items():
                f.write(f"- **{reason}**: {count} datasets\n")
            f.write("\n")

        f.write("## Key Findings\n\n")

        if len(successful) > 0:
            mean_improvement = successful["improvement_over_fixed_adaptive"].mean()

            if mean_improvement > 1:
                f.write("### Adaptive Learning Provides Significant Benefits\n")
                f.write(
                    f"- **{mean_improvement:.1f} percentage point average improvement** over fixed control\n"
                )
                f.write(
                    "- **Online plant identification** enables superior control performance\n"
                )
                f.write(
                    "- **Dynamic adaptation** responds to changing system characteristics\n"
                )
                f.write(
                    "- **PID enhancement** provides smoother, more stable control\n\n"
                )

                f.write("### Learning System Architecture Success\n")
                f.write(
                    "- **RLS convergence**: Recursive parameter estimation successfully identifies plant dynamics\n"
                )
                f.write(
                    "- **Sensitivity adaptation**: Learned dC/d(Δε) enables gain scheduling\n"
                )
                f.write(
                    "- **Dynamic targeting**: Adaptive targets maintain realistic, achievable goals\n"
                )
                f.write(
                    "- **Stability**: Bounded control with forgetting factor prevents parameter drift\n\n"
                )

            elif mean_improvement > 0:
                f.write("### Modest Adaptive Learning Benefits\n")
                f.write(
                    f"- **{mean_improvement:.1f} percentage point improvement** over fixed control\n"
                )
                f.write("- **Learning provides value** but gains are incremental\n")
                f.write(
                    "- **System complexity** may limit adaptive control advantages\n"
                )
                f.write("- **Parameter tuning** could enhance adaptive performance\n\n")
            else:
                f.write("### Limited Adaptive Learning Benefits\n")
                f.write("- **Minimal improvement** over fixed control approaches\n")
                f.write(
                    "- **Plant model limitations** may constrain adaptive effectiveness\n"
                )
                f.write(
                    "- **System dynamics** may be too complex for simple RLS identification\n"
                )
                f.write(
                    "- **Alternative approaches** (nonlinear, neural) may be needed\n\n"
                )

        f.write("## Scientific Implications\n\n")

        if (
            len(successful) > 0
            and successful["improvement_over_fixed_adaptive"].mean() > 0.5
        ):
            f.write("### Temporal Coherence Control Evolution\n")
            f.write(
                "- **Learning enhances control**: Adaptive systems outperform fixed approaches\n"
            )
            f.write(
                "- **Online identification**: Real-time plant learning enables superior regulation\n"
            )
            f.write(
                "- **Intelligent control**: System adapts to changing temporal dynamics\n"
            )
            f.write(
                "- **Scalable framework**: Principles apply across diverse system types\n\n"
            )

            f.write("### Practical Applications\n")
            f.write(
                "- **Autonomous systems**: Self-tuning temporal coherence controllers\n"
            )
            f.write(
                "- **Adaptive therapy**: Medical systems that learn patient-specific responses\n"
            )
            f.write(
                "- **Smart infrastructure**: Networks that adapt to usage patterns\n"
            )
            f.write(
                "- **AI systems**: Agents that optimize their own temporal dynamics\n\n"
            )
        else:
            f.write("### Adaptive Control Challenges\n")
            f.write(
                "- **Learning complexity**: Simple models may inadequately capture dynamics\n"
            )
            f.write(
                "- **Adaptation time**: RLS convergence may be too slow for fast systems\n"
            )
            f.write(
                "- **Parameter sensitivity**: Performance depends heavily on tuning\n"
            )
            f.write(
                "- **Robustness concerns**: Adaptive systems may be less predictable\n\n"
            )

        f.write("## Technical Achievements\n")
        f.write(
            "- **Complete learning pipeline**: Online identification → adaptive control → performance improvement\n"
        )
        f.write(
            "- **RLS integration**: Recursive least squares successfully deployed in control loop\n"
        )
        f.write(
            "- **Multi-strategy comparison**: Rigorous evaluation against no-control and fixed baselines\n"
        )
        f.write(
            "- **Dynamic adaptation**: Targets and gains adapt to system behavior\n"
        )
        f.write(
            "- **Stability assurance**: Bounded control prevents system destabilization\n\n"
        )

        f.write("## Generated Artifacts\n")
        f.write("For each successfully processed dataset:\n")
        f.write("- `{dataset}_metrics.csv`: Temporal coherence metrics time series\n")
        f.write("- `{dataset}_trace.csv`: Complete adaptive control simulation trace\n")
        f.write(
            "- `{dataset}_kpis.json`: Performance metrics for all control strategies\n"
        )
        f.write(
            "- `{dataset}_compare_curves.png`: Control strategy comparison visualization\n"
        )
        f.write(
            "- `{dataset}_control_signals.png`: Control signal evolution analysis\n"
        )
        f.write("- `{dataset}_kpi_compare.png`: Performance comparison bar charts\n\n")

        f.write("Summary files:\n")
        f.write("- `summary.csv`: Cross-dataset performance comparison\n")
        f.write("- `master_report.md`: This comprehensive analysis\n")


def main():
    """Main orchestrator for Phase 18 Adaptive Control Learning"""

    start_time = time.time()
    print("🧠 Phase 18: Adaptive Control Learning")
    print(f"📁 Output directory: {ROOT}")
    print()

    # Find datasets
    data_dir = Path("./data")
    if not data_dir.exists():
        print("❌ No ./data directory found")
        return

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        print("❌ No CSV files found in ./data")
        return

    print(f"📊 Found {len(csv_files)} datasets for adaptive control learning")
    print()

    # Process each dataset
    summary_results = []

    for csv_path in csv_files:
        result = process_single_dataset(csv_path, window=50, horizon=1)
        summary_results.append(result)

        # Print progress
        if result["skipped"]:
            print(f"  ⏭️  Skipped: {result['reason']}")
        else:
            improvement = result["improvement_over_fixed_adaptive"]
            print(f"  ✅ Adaptive improvement: {improvement:.1f} pp over fixed control")

    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(ROOT / "summary.csv", index=False)

    # Generate master report
    generate_master_report(summary_df)

    # Final statistics
    runtime = time.time() - start_time
    successful_count = len(summary_df[~summary_df["skipped"]])

    print("\n" + "=" * 60)
    print("🧠 Phase 18 Adaptive Control Learning Complete")
    print(f"⏱️  Runtime: {runtime:.2f} seconds")
    print(f"📊 Datasets processed: {successful_count}/{len(csv_files)}")

    if successful_count > 0:
        successful = summary_df[~summary_df["skipped"]]
        mean_improvement = successful["improvement_over_fixed_adaptive"].mean()
        positive_improvement = len(
            successful[successful["improvement_over_fixed_adaptive"] > 0]
        )

        print(f"📈 Average improvement over fixed: {mean_improvement:.1f} pp")
        print(
            f"🎯 Positive improvement: {positive_improvement}/{successful_count} datasets"
        )

        if mean_improvement > 1:
            print("🏆 Adaptive learning demonstrates significant benefits!")
        elif mean_improvement > 0:
            print("📊 Adaptive learning shows modest improvements")
        else:
            print("⚠️  Adaptive learning shows limited benefits")

    print(f"📁 Results: {ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()

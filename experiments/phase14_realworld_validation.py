#!/usr/bin/env python3
"""
Phase 14: Real-World Validation of Temporal Coherence Laws
===========================================================

Tests whether real datasets obey the discovered universal temporal law:
dC/dt ≈ A·(ε - ε_opt) + B·κ - C·noise

Where:
- C(t): Normalized coherence = 1 - H(t)/H_max
- ε(t): Openness = var(signal)/|mean(signal)|
- κ(t): Coupling = mean(|correlation_ij|) across variables
- noise: System noise level

Analyzes any CSV files in ./data/ directory and fits the temporal coherence model
to determine if real-world systems follow the same laws as our simulations.
"""

import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import scipy for advanced statistics, fall back to numpy if unavailable
try:
    from scipy import optimize
    from scipy.stats import entropy, linregress

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not available, using numpy fallbacks")

# Set deterministic seed and global parameters
np.random.seed(42)
STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = Path(f"./discovery_results/phase14_validation_{STAMP}")
ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Helper Functions ----------


def sliding_windows(arr, window_size, overlap=0.5):
    """Generate sliding windows with specified overlap"""
    n = len(arr)
    step = max(1, int(window_size * (1 - overlap)))
    windows = []
    for i in range(0, n - window_size + 1, step):
        windows.append(arr[i : i + window_size])
    return windows


def safe_entropy(x, bins=30):
    """Compute normalized entropy with fallbacks"""
    if len(x) < 2:
        return 0.0

    if HAS_SCIPY:
        # Use scipy's entropy function
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        if len(hist) == 0:
            return 0.0
        return entropy(hist) / np.log(len(hist))
    else:
        # Numpy fallback
        hist, _ = np.histogram(x, bins=bins)
        hist = hist + 1e-10  # Small constant to avoid log(0)
        p = hist / np.sum(hist)
        H = -np.sum(p * np.log(p))
        return H / np.log(len(p))


def robust_correlation(data):
    """Compute robust correlation matrix with error handling"""
    try:
        # Remove constant columns
        data_clean = data[:, np.var(data, axis=0) > 1e-10]
        if data_clean.shape[1] < 2:
            return 0.0

        corr_matrix = np.corrcoef(data_clean, rowvar=False)

        # Handle NaN values
        if np.isnan(corr_matrix).any():
            return 0.0

        # Extract upper triangular correlations (excluding diagonal)
        if corr_matrix.ndim == 2 and corr_matrix.shape[0] > 1:
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix[triu_indices]
            return np.mean(np.abs(correlations))
        else:
            return 0.0
    except:
        return 0.0


def compute_openness(data):
    """Compute openness ε = var(signal)/|mean(signal)| with robustness"""
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    openness_values = []
    for col in range(data.shape[1]):
        col_data = data[:, col]
        var_val = np.var(col_data)
        mean_val = np.mean(col_data)

        if abs(mean_val) < 1e-10:  # Avoid division by very small numbers
            openness_values.append(var_val)  # Use variance directly if mean ≈ 0
        else:
            openness_values.append(var_val / abs(mean_val))

    return np.mean(openness_values)


def compute_feedback_strength(data):
    """Compute α_est = mean(|Δsignal|)/|signal| as proxy for feedback"""
    if len(data) < 2:
        return 0.0

    diff_data = np.diff(data, axis=0)
    mean_change = np.mean(np.abs(diff_data))
    mean_signal = np.mean(np.abs(data))

    if mean_signal < 1e-10:
        return 0.0
    return mean_change / mean_signal


# ---------- Core Analysis Functions ----------


def analyze_dataset(csv_path, window_size=50):
    """
    Analyze a single dataset to extract temporal coherence metrics

    Returns:
        dataset_name: str
        metrics_df: DataFrame with time-series metrics
    """
    print(f"📊 Analyzing {csv_path.name}...")

    try:
        # Load and clean data
        df = pd.read_csv(csv_path)

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Convert to numeric, replacing non-numeric with NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove columns with too many NaNs (>50%)
        df = df.loc[:, df.isnull().mean() < 0.5]

        # Forward fill remaining NaNs
        df = df.fillna(method="ffill").fillna(method="bfill")

        if df.empty or len(df) < window_size:
            print("  ⚠️  Dataset too small or empty after cleaning")
            return csv_path.stem, None

        print(f"  📏 Data shape: {df.shape}, Window size: {window_size}")

        # Extract time-series windows
        data_array = df.values
        n_samples, n_variables = data_array.shape

        metrics = []
        window_times = []

        # Sliding window analysis
        step = max(1, window_size // 4)  # 75% overlap for smooth analysis
        for start_idx in range(0, n_samples - window_size + 1, step):
            end_idx = start_idx + window_size
            window_data = data_array[start_idx:end_idx]

            # Time index (center of window)
            time_center = start_idx + window_size // 2

            # Compute metrics for this window
            try:
                # 1. Coherence = 1 - normalized_entropy
                entropies = []
                for col_idx in range(n_variables):
                    col_entropy = safe_entropy(window_data[:, col_idx])
                    entropies.append(col_entropy)

                mean_entropy = np.mean(entropies)
                coherence = 1.0 - mean_entropy

                # 2. Openness ε
                openness = compute_openness(window_data)

                # 3. Coupling κ
                coupling = robust_correlation(window_data)

                # 4. Feedback strength α_est
                feedback = compute_feedback_strength(window_data)

                # 5. Noise level
                noise_level = np.std(window_data)

                # 6. Additional diagnostics
                signal_power = np.mean(np.var(window_data, axis=0))
                dynamic_range = np.ptp(window_data)  # Peak-to-peak

                metrics.append(
                    {
                        "time_index": time_center,
                        "coherence": coherence,
                        "openness": openness,
                        "coupling": coupling,
                        "feedback": feedback,
                        "noise": noise_level,
                        "signal_power": signal_power,
                        "dynamic_range": dynamic_range,
                    }
                )

            except Exception as e:
                print(f"    ⚠️  Window {start_idx}-{end_idx} failed: {e}")
                continue

        if not metrics:
            print("  ❌ No valid windows extracted")
            return csv_path.stem, None

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Compute temporal derivative dC/dt using gradient
        metrics_df["dC_dt"] = np.gradient(metrics_df["coherence"])

        print(f"  ✅ Extracted {len(metrics_df)} windows with metrics")
        return csv_path.stem, metrics_df

    except Exception as e:
        print(f"  ❌ Failed to analyze {csv_path.name}: {e}")
        return csv_path.stem, None


def fit_temporal_law(metrics_df):
    """
    Fit the temporal coherence law: dC/dt = A·(ε - ε_opt) + B·κ - C·noise

    Returns:
        coefficients: dict with A, B, C values
        r_squared: goodness of fit
        predictions: model predictions
        diagnostics: additional fit statistics
    """
    if metrics_df is None or len(metrics_df) < 10:
        return None, 0.0, None, {}

    try:
        # Prepare variables
        coherence_rate = metrics_df["dC_dt"].values
        openness = metrics_df["openness"].values
        coupling = metrics_df["coupling"].values
        noise = metrics_df["noise"].values

        # Remove any infinite or NaN values
        valid_mask = (
            np.isfinite(coherence_rate)
            & np.isfinite(openness)
            & np.isfinite(coupling)
            & np.isfinite(noise)
        )

        if np.sum(valid_mask) < 5:  # Need minimum data points
            return None, 0.0, None, {}

        y = coherence_rate[valid_mask]

        # Center openness around its mean (ε - ε_opt approximation)
        openness_centered = openness[valid_mask] - np.mean(openness[valid_mask])
        coupling_vals = coupling[valid_mask]
        noise_vals = noise[valid_mask]

        # Build design matrix X = [ε-ε_opt, κ, -noise]
        X = np.column_stack([openness_centered, coupling_vals, -noise_vals])

        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(y)), X])

        # Linear regression using least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(
                X_with_intercept, y, rcond=None
            )

            # Extract coefficients
            intercept = coeffs[0]
            A, B, C = (
                coeffs[1],
                coeffs[2],
                coeffs[3],
            )  # Note: C is already negative from -noise

            # Predictions and R²
            y_pred = X_with_intercept @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Diagnostics
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            mean_abs_error = np.mean(np.abs(y - y_pred))

            coefficients = {
                "A": float(A),
                "B": float(B),
                "C": float(C),
                "intercept": float(intercept),
            }
            diagnostics = {
                "rmse": float(rmse),
                "mae": float(mean_abs_error),
                "n_points": int(np.sum(valid_mask)),
                "condition_number": float(np.linalg.cond(X_with_intercept)),
            }

            return coefficients, float(r_squared), y_pred, diagnostics

        except np.linalg.LinAlgError:
            print("    ⚠️  Linear algebra error in fitting")
            return None, 0.0, None, {}

    except Exception as e:
        print(f"    ⚠️  Fitting error: {e}")
        return None, 0.0, None, {}


def create_visualizations(
    dataset_name, metrics_df, coefficients, r_squared, predictions, output_dir
):
    """Create comprehensive visualizations for the dataset analysis"""

    if metrics_df is None:
        return

    print("  📈 Creating visualizations...")

    try:
        # 1. Coherence time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150)

        ax1.plot(
            metrics_df["time_index"],
            metrics_df["coherence"],
            "b-",
            linewidth=2,
            alpha=0.8,
        )
        ax1.set_ylabel("Coherence C(t)")
        ax1.set_title(f"{dataset_name}: Temporal Coherence Evolution")
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            metrics_df["time_index"],
            metrics_df["dC_dt"],
            "r-",
            linewidth=1.5,
            alpha=0.7,
        )
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("dC/dt")
        ax2.set_title("Coherence Rate of Change")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{dataset_name}_coherence_evolution.png", bbox_inches="tight"
        )
        plt.close()

        # 2. Openness-Coupling phase space
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        scatter = ax.scatter(
            metrics_df["openness"],
            metrics_df["coupling"],
            c=metrics_df["coherence"],
            cmap="viridis",
            alpha=0.7,
            s=50,
        )
        ax.set_xlabel("Openness ε(t)")
        ax.set_ylabel("Coupling κ(t)")
        ax.set_title(f"{dataset_name}: ε-κ Phase Space (colored by Coherence)")
        cbar = plt.colorbar(scatter, ax=ax, label="Coherence C(t)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_phase_space.png", bbox_inches="tight")
        plt.close()

        # 3. Model fit visualization
        if coefficients is not None and predictions is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

            # Observed vs Predicted scatter
            observed = metrics_df["dC_dt"].values
            valid_mask = (
                np.isfinite(observed)
                & np.isfinite(metrics_df["openness"])
                & np.isfinite(metrics_df["coupling"])
                & np.isfinite(metrics_df["noise"])
            )
            obs_valid = observed[valid_mask]

            if len(predictions) == len(obs_valid):
                ax1.scatter(obs_valid, predictions, alpha=0.6, s=30)

                # robust fit line
                min_val, max_val = min(obs_valid.min(), predictions.min()), max(
                    obs_valid.max(), predictions.max()
                )
                ax1.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                )

                ax1.set_xlabel("Observed dC/dt")
                ax1.set_ylabel("Predicted dC/dt")
                ax1.set_title(f"Model Fit (R² = {r_squared:.3f})")
                ax1.grid(True, alpha=0.3)

            # Residuals plot
            if len(predictions) == len(obs_valid):
                residuals = obs_valid - predictions
                ax2.scatter(predictions, residuals, alpha=0.6, s=30)
                ax2.axhline(0, color="red", linestyle="--", alpha=0.8)
                ax2.set_xlabel("Predicted dC/dt")
                ax2.set_ylabel("Residuals")
                ax2.set_title("Residual Analysis")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_dir / f"{dataset_name}_model_fit.png", bbox_inches="tight"
            )
            plt.close()

        # 4. Multi-variable time series
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        axes = axes.flatten()

        variables = ["openness", "coupling", "feedback", "noise"]
        colors = ["orange", "green", "purple", "red"]

        for i, (var, color) in enumerate(zip(variables, colors)):
            if var in metrics_df.columns:
                axes[i].plot(
                    metrics_df["time_index"],
                    metrics_df[var],
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                )
                axes[i].set_title(f"{var.capitalize()}")
                axes[i].set_ylabel(var)
                axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Index")
        axes[-2].set_xlabel("Time Index")
        plt.suptitle(f"{dataset_name}: System Variables Evolution")
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_variables.png", bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"    ⚠️  Visualization error: {e}")


def validate_temporal_law(dataset_name, coefficients, r_squared, diagnostics):
    """Assess whether the dataset follows the universal temporal law"""

    if coefficients is None:
        return {
            "follows_law": False,
            "confidence": "None",
            "reason": "Fit failed",
            "assessment": "Unable to fit temporal coherence law",
        }

    # Validation criteria
    r2_threshold = 0.3  # Minimum R² for reasonable fit
    coeff_significance = 0.01  # Minimum coefficient magnitude

    # Check R² quality
    if r_squared < r2_threshold:
        confidence = "Low"
        follows_law = False
        reason = f"Poor fit quality (R² = {r_squared:.3f} < {r2_threshold})"
    elif r_squared < 0.6:
        confidence = "Moderate"
        follows_law = True
        reason = f"Moderate fit quality (R² = {r_squared:.3f})"
    else:
        confidence = "High"
        follows_law = True
        reason = f"Good fit quality (R² = {r_squared:.3f})"

    # Assess coefficient patterns
    A, B, C = coefficients["A"], coefficients["B"], coefficients["C"]

    # Expected signs: A can be ±, B should be +, C should be +
    pattern_analysis = []
    if abs(A) > coeff_significance:
        pattern_analysis.append(f"Openness effect: A = {A:.3f}")
    if abs(B) > coeff_significance:
        pattern_analysis.append(
            f"Coupling effect: B = {B:.3f}" + (" ✓" if B > 0 else " (unexpected sign)")
        )
    if abs(C) > coeff_significance:
        pattern_analysis.append(
            f"Noise effect: C = {C:.3f}" + (" ✓" if C > 0 else " (unexpected sign)")
        )

    if not pattern_analysis:
        follows_law = False
        reason += "; No significant effects detected"

    # Overall assessment
    if follows_law and r_squared > 0.5:
        assessment = f"Dataset shows evidence of temporal coherence law with {confidence.lower()} confidence"
    elif follows_law:
        assessment = f"Weak evidence of temporal coherence law with {confidence.lower()} confidence"
    else:
        assessment = "Dataset does not clearly follow the temporal coherence law"

    return {
        "follows_law": follows_law,
        "confidence": confidence,
        "reason": reason,
        "assessment": assessment,
        "pattern_analysis": pattern_analysis,
        "coefficients_magnitude": {"A": abs(A), "B": abs(B), "C": abs(C)},
    }


# ---------- Main Orchestrator ----------


def run_validation_suite():
    """Main orchestrator for Phase 14 validation"""

    start_time = time.time()
    print("🚀 Phase 14: Real-World Validation of Temporal Coherence Laws")
    print(f"📁 Output directory: {ROOT}")
    print()

    # Find datasets
    data_dir = Path("./data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("📂 Created ./data/ directory")
        print("   Please add CSV datasets and rerun")
        return

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in ./data/ directory")
        print("   Please add datasets with time-series data:")
        print("   - Rows = time samples")
        print("   - Columns = variables/sensors")
        print("   - Format: standard CSV with headers")
        return

    print(f"📊 Found {len(csv_files)} datasets:")
    for csv_file in csv_files:
        print(f"   • {csv_file.name}")
    print()

    # Process each dataset
    all_results = []

    for csv_path in csv_files:
        print(f"🔍 Processing {csv_path.name}...")

        # Analyze dataset
        dataset_name, metrics_df = analyze_dataset(csv_path, window_size=50)

        if metrics_df is None:
            print(f"  ❌ Skipping {dataset_name} - analysis failed")
            continue

        # Fit temporal law
        coefficients, r_squared, predictions, diagnostics = fit_temporal_law(metrics_df)

        # Validate against universal law
        validation = validate_temporal_law(
            dataset_name, coefficients, r_squared, diagnostics
        )

        # Create visualizations
        create_visualizations(
            dataset_name, metrics_df, coefficients, r_squared, predictions, ROOT
        )

        # Export results
        metrics_df.to_csv(ROOT / f"{dataset_name}_metrics.csv", index=False)

        # Compile summary
        result_summary = {
            "dataset": dataset_name,
            "file_path": str(csv_path),
            "n_windows": len(metrics_df),
            "data_shape": f"{len(pd.read_csv(csv_path))} × {len(pd.read_csv(csv_path).columns)}",
            "coefficients": (
                coefficients if coefficients else {"A": None, "B": None, "C": None}
            ),
            "r_squared": r_squared,
            "diagnostics": diagnostics,
            "validation": validation,
            "mean_coherence": float(metrics_df["coherence"].mean()),
            "coherence_std": float(metrics_df["coherence"].std()),
            "mean_openness": float(metrics_df["openness"].mean()),
            "mean_coupling": float(metrics_df["coupling"].mean()),
            "temporal_law_evidence": validation["follows_law"],
        }

        all_results.append(result_summary)

        # Individual dataset summary
        with open(ROOT / f"{dataset_name}_analysis.json", "w") as f:
            json.dump(result_summary, f, indent=2, default=str)

        print(
            f"  ✅ {dataset_name}: R² = {r_squared:.3f}, Law evidence: {validation['follows_law']}"
        )
        print()

    # Generate master summary
    if all_results:
        results_df = pd.DataFrame(
            [
                {
                    "dataset": r["dataset"],
                    "r_squared": r["r_squared"],
                    "follows_law": r["temporal_law_evidence"],
                    "confidence": r["validation"]["confidence"],
                    "coeff_A": r["coefficients"]["A"] if r["coefficients"]["A"] else 0,
                    "coeff_B": r["coefficients"]["B"] if r["coefficients"]["B"] else 0,
                    "coeff_C": r["coefficients"]["C"] if r["coefficients"]["C"] else 0,
                    "mean_coherence": r["mean_coherence"],
                    "n_windows": r["n_windows"],
                }
                for r in all_results
            ]
        )

        results_df.to_csv(ROOT / "validation_summary.csv", index=False)

        # Master report
        with open(ROOT / "master_report.md", "w") as f:
            f.write("# Phase 14 — Real-World Validation of Temporal Coherence Laws\n\n")
            f.write(f"**Timestamp**: {STAMP}\n")
            f.write(f"**Datasets analyzed**: {len(all_results)}\n\n")

            f.write("## Universal Law Being Tested\n")
            f.write("```\ndC/dt = A·(ε - ε_opt) + B·κ - C·noise\n```\n")
            f.write("Where:\n")
            f.write("- **C(t)**: Normalized coherence = 1 - H(t)/H_max\n")
            f.write("- **ε(t)**: Openness = var(signal)/|mean(signal)|\n")
            f.write("- **κ(t)**: Coupling = mean(|correlation|) across variables\n")
            f.write("- **A, B, C**: Fitted coefficients\n\n")

            f.write("## Results Summary\n\n")

            law_followers = sum(1 for r in all_results if r["temporal_law_evidence"])
            f.write(
                f"**Datasets following temporal law**: {law_followers}/{len(all_results)} ({law_followers/len(all_results)*100:.1f}%)\n\n"
            )

            for result in all_results:
                f.write(f"### {result['dataset']}\n")
                f.write(f"- **Model fit**: R² = {result['r_squared']:.3f}\n")
                f.write(
                    f"- **Follows law**: {result['validation']['follows_law']} ({result['validation']['confidence']} confidence)\n"
                )
                if result["coefficients"]["A"] is not None:
                    f.write(
                        f"- **Coefficients**: A = {result['coefficients']['A']:.3f}, B = {result['coefficients']['B']:.3f}, C = {result['coefficients']['C']:.3f}\n"
                    )
                f.write(f"- **Assessment**: {result['validation']['assessment']}\n")
                f.write(
                    f"- **Data**: {result['n_windows']} windows from {result['data_shape']} raw data\n\n"
                )

            f.write("## Scientific Interpretation\n\n")
            if law_followers > 0:
                f.write(
                    "✅ **Universal law validation**: Evidence found in real-world data\n\n"
                )
                f.write(
                    "The temporal coherence law dC/dt = A·(ε-ε_opt) + B·κ - C·noise shows predictive power "
                )
                f.write(
                    "in actual datasets, suggesting our simulated discoveries reflect genuine "
                )
                f.write("universal principles of temporal emergence.\n\n")
            else:
                f.write(
                    "⚠️  **Limited validation**: Law evidence weak in current datasets\n\n"
                )
                f.write(
                    "The temporal coherence law may require specific conditions or dataset types "
                )
                f.write("to manifest clearly. Consider testing with:\n")
                f.write(
                    "- Neural/EEG recordings\n- Financial time series\n- Climate data\n- Biological rhythms\n\n"
                )

            f.write("## Methodology\n")
            f.write(
                "- **Window analysis**: 50-sample sliding windows with 75% overlap\n"
            )
            f.write("- **Coherence**: 1 - normalized_entropy across variables\n")
            f.write(
                "- **Openness**: Variance-to-mean ratio indicating system flexibility\n"
            )
            f.write("- **Coupling**: Mean absolute correlation between variables\n")
            f.write(
                "- **Model fitting**: Linear regression of dC/dt vs (ε, κ, noise)\n\n"
            )

            f.write("## Generated Artifacts\n")
            f.write("- `validation_summary.csv`: Cross-dataset comparison\n")
            f.write("- `{dataset}_metrics.csv`: Per-dataset temporal metrics\n")
            f.write("- `{dataset}_analysis.json`: Complete analysis results\n")
            f.write("- `{dataset}_*.png`: Visualization suite per dataset\n\n")

        # Overall statistics
        print("=" * 60)
        print("🎯 Phase 14 Validation Complete")
        print(f"⏱️  Runtime: {time.time() - start_time:.2f} seconds")
        print(f"📊 Datasets processed: {len(all_results)}")
        print(f"✅ Datasets showing temporal law: {law_followers}/{len(all_results)}")

        if law_followers > 0:
            avg_r2 = np.mean(
                [r["r_squared"] for r in all_results if r["temporal_law_evidence"]]
            )
            print(f"📈 Average R² for law followers: {avg_r2:.3f}")

        print(f"📁 Results: {ROOT}")
        print("=" * 60)

    else:
        print("❌ No datasets successfully processed")


if __name__ == "__main__":
    run_validation_suite()

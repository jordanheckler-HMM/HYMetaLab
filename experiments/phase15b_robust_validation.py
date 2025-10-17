#!/usr/bin/env python3
"""
Phase 15B: Robust Comparative Empirical Validation
=================================================

Enhanced version that handles smaller datasets and provides better diagnostics.
Compares temporal coherence law expression across system categories with
adaptive windowing and fallback analysis methods.
"""

import datetime as dt
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import entropy, ttest_ind

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

np.random.seed(42)
STAMP = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = Path(f"./discovery_results/phase15b_validation_{STAMP}")
ROOT.mkdir(parents=True, exist_ok=True)


def safe_entropy(x, bins=20):
    """Robust entropy calculation"""
    if len(x) < 5:
        return 0.5  # Default moderate entropy for tiny samples

    try:
        # Remove NaN/inf values
        x_clean = x[np.isfinite(x)]
        if len(x_clean) < 3:
            return 0.5

        # Adaptive binning based on data size
        n_bins = min(bins, max(3, len(x_clean) // 5))

        hist, _ = np.histogram(x_clean, bins=n_bins)
        hist = hist + 1e-10  # Regularization
        p = hist / hist.sum()

        H = -np.sum(p * np.log(p))
        H_max = np.log(len(p))

        return H / H_max if H_max > 0 else 0.5
    except:
        return 0.5


def categorize_dataset(name):
    """Enhanced dataset categorization with more patterns"""
    name_lower = name.lower()

    # Biological/temporal patterns
    if any(kw in name_lower for kw in ["pulse", "eoi", "entropy"]):
        return "biological"

    # Social/system patterns
    if any(kw in name_lower for kw in ["runs", "phase"]):
        return "social"

    # Event/dynamics patterns
    if any(kw in name_lower for kw in ["events", "cycles", "recovery"]):
        return "biological"

    # Frequency/spectrum patterns
    if any(kw in name_lower for kw in ["freq", "spectrum"]):
        return "physical"

    return "unknown"


def analyze_dataset_adaptive(csv_path):
    """Adaptive analysis that handles variable dataset sizes"""
    name = Path(csv_path).stem
    category = categorize_dataset(name)

    print(f"  📊 {name} → {category}")

    try:
        # Load data with robust error handling
        df = pd.read_csv(csv_path)
        original_shape = df.shape

        # Basic cleaning
        df = df.dropna(axis=1, how="all")  # Remove empty columns

        # Convert to numeric where possible
        numeric_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > len(df) * 0.3:  # At least 30% valid data
                    numeric_cols.append(col)
            except:
                continue

        if len(numeric_cols) == 0:
            print("    ❌ No numeric columns found")
            return None

        df_numeric = df[numeric_cols].dropna()
        if len(df_numeric) < 10:
            print(f"    ❌ Too few numeric rows: {len(df_numeric)}")
            return None

        print(f"    📏 {original_shape} → {df_numeric.shape} (cleaned)")

        # Adaptive window size based on data length
        N = len(df_numeric)
        if N < 50:
            window_size = max(5, N // 4)
            method = "small_dataset"
        elif N < 200:
            window_size = max(10, N // 8)
            method = "medium_dataset"
        else:
            window_size = 50
            method = "large_dataset"

        print(f"    🔧 Using {method} approach, window={window_size}")

        # Extract metrics with adaptive windowing
        data_array = df_numeric.values
        n_samples, n_vars = data_array.shape

        # Compute global metrics first (fallback)
        global_coherence = compute_coherence_robust(data_array)
        global_openness = compute_openness_robust(data_array)
        global_coupling = compute_coupling_robust(data_array)
        global_noise = np.std(data_array)

        if N < 20:  # Very small dataset - use global metrics only
            print(f"    📊 Global analysis only (N={N})")
            return {
                "dataset": name,
                "category": category,
                "method": "global",
                "A": np.nan,  # Can't compute derivatives
                "B": np.nan,
                "C": np.nan,
                "R2": np.nan,
                "n_windows": 1,
                "mean_coherence": float(global_coherence),
                "mean_openness": float(global_openness),
                "mean_coupling": float(global_coupling),
                "mean_noise": float(global_noise),
                "data_shape": f"{N} × {n_vars}",
                "analysis_notes": f"Dataset too small for windowing (N={N})",
            }

        # Windowed analysis for larger datasets
        metrics = []
        step_size = max(1, window_size // 4)  # 75% overlap

        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            window_data = data_array[start:end]

            try:
                coherence = compute_coherence_robust(window_data)
                openness = compute_openness_robust(window_data)
                coupling = compute_coupling_robust(window_data)
                noise = np.std(window_data)

                metrics.append(
                    {
                        "time": start + window_size // 2,
                        "coherence": coherence,
                        "openness": openness,
                        "coupling": coupling,
                        "noise": noise,
                    }
                )
            except Exception:
                continue

        if len(metrics) < 5:
            print(f"    ⚠️  Too few valid windows: {len(metrics)}")
            # Fall back to global analysis
            return {
                "dataset": name,
                "category": category,
                "method": "global_fallback",
                "A": np.nan,
                "B": np.nan,
                "C": np.nan,
                "R2": np.nan,
                "n_windows": len(metrics),
                "mean_coherence": float(global_coherence),
                "mean_openness": float(global_openness),
                "mean_coupling": float(global_coupling),
                "mean_noise": float(global_noise),
                "data_shape": f"{N} × {n_vars}",
                "analysis_notes": f"Insufficient windows ({len(metrics)}), used global metrics",
            }

        # Convert to DataFrame and fit temporal law
        mdf = pd.DataFrame(metrics)

        # Compute temporal derivative with smoothing
        coherence_vals = mdf["coherence"].values
        if len(coherence_vals) < 5:
            dC_dt = np.zeros_like(coherence_vals)
        else:
            # Use central differences with edge handling
            dC_dt = np.gradient(coherence_vals)

        mdf["dC_dt"] = dC_dt

        # Fit temporal coherence law: dC/dt = A·(ε-ε_opt) + B·κ - C·noise
        try:
            openness_centered = mdf["openness"] - mdf["openness"].mean()
            coupling_vals = mdf["coupling"]
            noise_vals = mdf["noise"]

            X = np.column_stack([openness_centered, coupling_vals, -noise_vals])
            y = mdf["dC_dt"]

            # Check for sufficient variation
            if np.std(y) < 1e-10:
                print("    ⚠️  No coherence variation detected")
                coeffs = [0.0, 0.0, 0.0]
                r2 = 0.0
            else:
                # Robust linear regression
                coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coeffs

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        except Exception as e:
            print(f"    ⚠️  Fitting failed: {e}")
            A, B, C, r2 = 0.0, 0.0, 0.0, 0.0

        result = {
            "dataset": name,
            "category": category,
            "method": method,
            "A": A,
            "B": B,
            "C": C,
            "R2": float(r2),
            "n_windows": len(metrics),
            "mean_coherence": float(mdf["coherence"].mean()),
            "std_coherence": float(mdf["coherence"].std()),
            "mean_openness": float(mdf["openness"].mean()),
            "mean_coupling": float(mdf["coupling"].mean()),
            "mean_noise": float(mdf["noise"].mean()),
            "data_shape": f"{N} × {n_vars}",
            "analysis_notes": f"Successfully analyzed with {len(metrics)} windows",
        }

        print(f"    ✅ Method: {method}, R² = {r2:.3f}, Windows: {len(metrics)}")
        return result

    except Exception as e:
        print(f"    ❌ Analysis failed: {e}")
        return None


def compute_coherence_robust(data):
    """Robust coherence computation"""
    if data.size == 0:
        return 0.5

    try:
        # Compute entropy across variables
        entropies = []
        for col in range(data.shape[1]):
            col_data = data[:, col]
            if len(col_data) > 0:
                entropies.append(safe_entropy(col_data))

        if not entropies:
            return 0.5

        mean_entropy = np.mean(entropies)
        coherence = 1.0 - mean_entropy
        return np.clip(coherence, 0.0, 1.0)
    except:
        return 0.5


def compute_openness_robust(data):
    """Robust openness computation"""
    try:
        if data.size == 0:
            return 1.0

        openness_vals = []
        for col in range(data.shape[1]):
            col_data = data[:, col]
            if len(col_data) > 1:
                var_val = np.var(col_data)
                mean_val = np.mean(col_data)

                if abs(mean_val) > 1e-8:
                    openness_vals.append(var_val / abs(mean_val))
                else:
                    openness_vals.append(var_val)  # Use variance if mean ≈ 0

        if not openness_vals:
            return 1.0

        return np.mean(openness_vals)
    except:
        return 1.0


def compute_coupling_robust(data):
    """Robust coupling computation"""
    try:
        if data.shape[1] < 2:
            return 0.0

        # Remove constant columns
        col_vars = np.var(data, axis=0)
        active_cols = col_vars > 1e-10

        if np.sum(active_cols) < 2:
            return 0.0

        data_active = data[:, active_cols]
        corr_matrix = np.corrcoef(data_active, rowvar=False)

        if corr_matrix.ndim != 2:
            return 0.0

        # Extract upper triangular correlations
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        if len(triu_indices[0]) == 0:
            return 0.0

        correlations = corr_matrix[triu_indices]
        valid_corrs = correlations[np.isfinite(correlations)]

        if len(valid_corrs) == 0:
            return 0.0

        return np.mean(np.abs(valid_corrs))
    except:
        return 0.0


def create_robust_visualizations(df):
    """Create visualizations with error handling for sparse data"""
    print("\n📈 Creating robust visualizations...")

    try:
        # 1. Coefficient comparison by category
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        coeffs = ["A", "B", "C", "R2"]
        titles = [
            "Openness Effect (A)",
            "Coupling Effect (B)",
            "Noise Effect (C)",
            "Model Fit (R²)",
        ]

        for i, (coef, title) in enumerate(zip(coeffs, titles)):
            ax = axes[i // 2, i % 2]

            categories = df["category"].unique()
            for j, cat in enumerate(categories):
                cat_data = df[df["category"] == cat][coef]
                valid_data = cat_data[np.isfinite(cat_data)]

                if len(valid_data) > 0:
                    # Jitter x-coordinates for visibility
                    x_coords = np.random.normal(j, 0.1, len(valid_data))
                    ax.scatter(
                        x_coords,
                        valid_data,
                        alpha=0.7,
                        s=50,
                        label=cat if i == 0 else "",
                    )

                    # Add mean line if we have data
                    if len(valid_data) > 0:
                        ax.hlines(
                            valid_data.mean(),
                            j - 0.3,
                            j + 0.3,
                            colors="red",
                            linewidth=2,
                        )

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.set_title(title)

            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.savefig(ROOT / "coefficient_comparison.png", bbox_inches="tight")
        plt.close()

        # 2. System properties by category
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

        properties = ["mean_coherence", "mean_openness", "mean_coupling", "mean_noise"]
        prop_titles = ["Mean Coherence", "Mean Openness", "Mean Coupling", "Mean Noise"]

        for i, (prop, title) in enumerate(zip(properties, prop_titles)):
            ax = axes[i // 2, i % 2]

            categories = df["category"].unique()
            for j, cat in enumerate(categories):
                cat_data = df[df["category"] == cat][prop]
                valid_data = cat_data[np.isfinite(cat_data)]

                if len(valid_data) > 0:
                    x_coords = np.random.normal(j, 0.1, len(valid_data))
                    ax.scatter(x_coords, valid_data, alpha=0.7, s=50)
                    ax.hlines(
                        valid_data.mean(), j - 0.3, j + 0.3, colors="red", linewidth=2
                    )

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.set_title(title)

        plt.tight_layout()
        plt.savefig(ROOT / "system_properties.png", bbox_inches="tight")
        plt.close()

        # 3. Dataset analysis method breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        # Methods used
        method_counts = df["method"].value_counts()
        ax1.pie(
            method_counts.values,
            labels=method_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("Analysis Methods Used")

        # Category distribution
        cat_counts = df["category"].value_counts()
        ax2.pie(
            cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", startangle=90
        )
        ax2.set_title("System Categories")

        plt.tight_layout()
        plt.savefig(ROOT / "analysis_breakdown.png", bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"    ⚠️  Visualization error: {e}")


def run_robust_comparative_validation():
    """Main orchestrator with robust analysis"""

    start_time = time.time()
    print("🚀 Phase 15B: Robust Comparative Empirical Validation")
    print(f"📁 Output directory: {ROOT}")
    print()

    # Find datasets
    data_dir = Path("./data")
    csvs = list(data_dir.glob("*.csv"))

    if not csvs:
        print("❌ No CSV files found")
        return

    print(f"📊 Found {len(csvs)} datasets for robust analysis")
    print()

    # Analyze with adaptive methods
    print("🔍 Robust analysis by category:")
    results = []

    for csv_path in csvs:
        result = analyze_dataset_adaptive(csv_path)
        if result:
            results.append(result)

    if not results:
        print("❌ No datasets successfully analyzed")
        return

    print(f"\n✅ Successfully analyzed {len(results)} datasets")

    # Create results DataFrame
    df = pd.DataFrame(results)
    df.to_csv(ROOT / "robust_comparative_results.csv", index=False)

    # Category summary
    print("\n📊 Category Summary:")
    category_summary = []

    for category, group in df.groupby("category"):
        n_total = len(group)
        n_valid_fit = len(group[np.isfinite(group["R2"])])

        # Compute statistics for available data
        stats = {
            "category": category,
            "n_datasets": n_total,
            "n_valid_fits": n_valid_fit,
            "mean_coherence": float(group["mean_coherence"].mean()),
            "std_coherence": float(group["mean_coherence"].std()),
            "mean_openness": float(group["mean_openness"].mean()),
            "mean_coupling": float(group["mean_coupling"].mean()),
            "mean_noise": float(group["mean_noise"].mean()),
        }

        # Add coefficient stats if we have valid fits
        if n_valid_fit > 0:
            valid_group = group[np.isfinite(group["R2"])]
            for coef in ["A", "B", "C", "R2"]:
                valid_coef = valid_group[coef][np.isfinite(valid_group[coef])]
                if len(valid_coef) > 0:
                    stats[f"{coef}_mean"] = float(valid_coef.mean())
                    stats[f"{coef}_std"] = float(valid_coef.std())
                else:
                    stats[f"{coef}_mean"] = np.nan
                    stats[f"{coef}_std"] = np.nan

        category_summary.append(stats)
        print(f"  📊 {category}: {n_total} datasets, {n_valid_fit} valid fits")

    summary_df = pd.DataFrame(category_summary)
    summary_df.to_csv(ROOT / "category_summary.csv", index=False)

    # Create visualizations
    create_robust_visualizations(df)

    # Generate master report
    with open(ROOT / "master_report.md", "w") as f:
        f.write("# Phase 15B — Robust Comparative Empirical Validation\n\n")
        f.write(f"**Timestamp**: {STAMP}\n")
        f.write(f"**Datasets analyzed**: {len(df)}\n")
        f.write(f"**Categories**: {', '.join(df['category'].unique())}\n\n")

        f.write("## Temporal Coherence Law\n")
        f.write("```\ndC/dt = A·(ε - ε_opt) + B·κ - C·noise\n```\n\n")

        f.write("## Robust Analysis Results\n\n")
        f.write("### Analysis Methods Used\n")
        method_counts = df["method"].value_counts()
        for method, count in method_counts.items():
            f.write(f"- **{method}**: {count} datasets\n")
        f.write("\n")

        f.write("### Category Performance\n\n")
        f.write(
            "| Category | Datasets | Valid Fits | Mean Coherence | Mean Coupling |\n"
        )
        f.write(
            "|----------|----------|------------|----------------|---------------|\n"
        )

        for _, row in summary_df.iterrows():
            f.write(
                f"| {row['category']} | {row['n_datasets']} | {row['n_valid_fits']} | {row['mean_coherence']:.3f} | {row['mean_coupling']:.3f} |\n"
            )

        f.write("\n## Key Findings\n\n")

        # Analysis findings
        total_valid = len(df[np.isfinite(df["R2"])])
        f.write("### Method Robustness\n")
        f.write(
            f"- **Successfully analyzed**: {len(df)}/{len(csvs)} datasets ({len(df)/len(csvs)*100:.1f}%)\n"
        )
        f.write(f"- **Valid temporal law fits**: {total_valid}/{len(df)} datasets\n")
        f.write(
            f"- **Adaptive windowing**: Handled datasets from {df['data_shape'].str.extract('(\d+)').astype(int).min().iloc[0]} to {df['data_shape'].str.extract('(\d+)').astype(int).max().iloc[0]} samples\n\n"
        )

        # Category patterns
        for category in df["category"].unique():
            cat_data = df[df["category"] == category]
            f.write(f"### {category.title()} Systems\n")
            f.write(f"- **Datasets**: {len(cat_data)}\n")
            f.write(
                f"- **Mean coherence**: {cat_data['mean_coherence'].mean():.3f} ± {cat_data['mean_coherence'].std():.3f}\n"
            )
            f.write(
                f"- **Mean coupling**: {cat_data['mean_coupling'].mean():.3f} ± {cat_data['mean_coupling'].std():.3f}\n"
            )

            valid_cat = cat_data[np.isfinite(cat_data["R2"])]
            if len(valid_cat) > 0:
                f.write(
                    f"- **Temporal law fits**: {len(valid_cat)}/{len(cat_data)} datasets\n"
                )
                f.write(f"- **Mean R²**: {valid_cat['R2'].mean():.3f}\n")
            f.write("\n")

        f.write("## Methodology\n")
        f.write(
            "- **Adaptive windowing**: Variable window sizes based on dataset length\n"
        )
        f.write(
            "- **Robust metrics**: Error-tolerant entropy, openness, and coupling calculations\n"
        )
        f.write(
            "- **Fallback analysis**: Global metrics for datasets too small for windowing\n"
        )
        f.write(
            "- **Multi-method approach**: Large, medium, and small dataset strategies\n\n"
        )

        f.write("## Generated Artifacts\n")
        f.write("- `robust_comparative_results.csv`: Complete analysis results\n")
        f.write("- `category_summary.csv`: Statistical summary by system category\n")
        f.write(
            "- `coefficient_comparison.png`: Coefficient distributions by category\n"
        )
        f.write("- `system_properties.png`: System properties by category\n")
        f.write("- `analysis_breakdown.png`: Method and category distributions\n")

    # Final summary
    runtime = time.time() - start_time

    print("\n" + "=" * 60)
    print("🎯 Phase 15B Robust Comparative Validation Complete")
    print(f"⏱️  Runtime: {runtime:.2f} seconds")
    print(f"📊 Datasets analyzed: {len(df)}/{len(csvs)}")

    for category in df["category"].unique():
        count = len(df[df["category"] == category])
        valid_fits = len(df[(df["category"] == category) & np.isfinite(df["R2"])])
        print(f"   • {category}: {count} datasets, {valid_fits} valid fits")

    print(f"📁 Results: {ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    run_robust_comparative_validation()

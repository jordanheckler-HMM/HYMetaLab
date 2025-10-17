#!/usr/bin/env python3
"""
HYMetaLab Auto-Theorist 2.0
ML-based parameter optimization and experiment queue generation
Features: Linear/Ridge/GradientBoost regression, ranked experiments, auto-YAML generation
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

# Try to import sklearn for ML models
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not available. Using correlation-based fallback.")

SUMMARY_GLOB = "discovery_results/*/summary.json"
OUT_DIR = Path("tools/autotheorist_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_KEYS = [
    "epsilon",
    "rho",
    "rho_star",
    "trust_delta",
    "meaning_delta",
    "hope_delta",
    "truth_delta",
    "coupling",
    "c_eff",
    "lambda",
    "lambda_star",
]
METRIC_KEYS = [
    "delta_cci",
    "delta_hazard",
    "delta_risk",
    "delta_survival",
    "openlaws_score",
]


def _read_summary(p: Path) -> dict[str, Any]:
    try:
        d = json.loads(p.read_text())
        d["_source_dir"] = str(p.parent)
        return d
    except Exception as e:
        return {"_error": str(e), "_source_dir": str(p.parent)}


def _flatten(d: dict[str, Any], prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common key variants"""
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "dcci" in lc or "delta_cci" in lc or "Δcci" in c or "mean_cci_gain" in lc:
            col_map[c] = "delta_cci"
        elif "hazard" in lc and ("delta" in lc or "Δ" in c):
            col_map[c] = "delta_hazard"
        elif lc.endswith("epsilon") or lc == "epsilon" or ".eps" in lc:
            col_map[c] = "epsilon"
        elif re.search(r"(^|[._])rho([._]|$)", lc):
            col_map[c] = "rho"
        elif "trust_delta" in lc:
            col_map[c] = "trust_delta"
        elif "meaning_delta" in lc:
            col_map[c] = "meaning_delta"
        elif "hope_delta" in lc:
            col_map[c] = "hope_delta"
        elif "truth_delta" in lc:
            col_map[c] = "truth_delta"
        elif "openlaws_score" in lc:
            col_map[c] = "openlaws_score"
        elif "lambda" in lc and "star" not in lc:
            col_map[c] = "lambda"
        elif "lambda_star" in lc:
            col_map[c] = "lambda_star"
    df = df.rename(columns=col_map)
    return df


def load_df() -> pd.DataFrame:
    rows = []
    for p in Path(".").glob(SUMMARY_GLOB):
        d = _read_summary(p)
        flat = _flatten(d)
        rows.append(flat)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _coerce(df)

    # Keep only useful columns
    keep = ["_source_dir"] + [
        c for c in df.columns if c in set(PARAM_KEYS + METRIC_KEYS)
    ]
    keep = [c for c in keep if c in df.columns]
    if not keep:
        return pd.DataFrame()

    df = df[keep].copy()

    # Numeric coercion - be defensive
    for c in list(df.columns):
        if c != "_source_dir":
            try:
                df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")
            except:
                pass

    return df.dropna(how="all", axis=1)


def rank_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations between params and metrics"""
    params = list(set([c for c in df.columns if c in PARAM_KEYS]))
    metrics = list(set([c for c in df.columns if c in METRIC_KEYS]))
    rows = []

    for p in params:
        for m in metrics:
            try:
                s = df[[p, m]].dropna()
                if len(s) >= 5:
                    corr = s[p].corr(s[m])
                    if not np.isnan(corr):
                        rows.append(
                            {
                                "param": p,
                                "metric": m,
                                "corr": float(corr),
                                "n": len(s),
                                "abs_corr": abs(float(corr)),
                            }
                        )
            except:
                pass

    if not rows:
        # Return empty DataFrame with correct dtypes
        return pd.DataFrame(
            {
                "param": pd.Series(dtype=str),
                "metric": pd.Series(dtype=str),
                "corr": pd.Series(dtype=float),
                "n": pd.Series(dtype=int),
                "abs_corr": pd.Series(dtype=float),
            }
        )

    r = pd.DataFrame(rows).sort_values(by="abs_corr", ascending=False)
    return r


def train_ml_models(df: pd.DataFrame) -> dict[str, Any]:
    """
    Train ML models to predict delta_cci and delta_hazard
    Returns model performance and optimal parameter bands
    """
    if not ML_AVAILABLE:
        return {"status": "ML not available", "fallback": "using correlation analysis"}

    results = {}

    for target in ["delta_cci", "delta_hazard"]:
        if target not in df.columns:
            continue

        # Prepare data
        param_cols = [c for c in df.columns if c in PARAM_KEYS]
        X = df[param_cols].dropna(how="all")
        y = df.loc[X.index, target].dropna()
        X = X.loc[y.index]

        if len(X) < 10:
            results[target] = {"status": "insufficient_data", "n_samples": len(X)}
            continue

        # Train three models
        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "GradientBoost": GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            ),
        }

        model_scores = {}
        best_model = None
        best_score = -np.inf

        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X, y, cv=min(5, len(X) // 2), scoring="r2"
                )
                mean_score = scores.mean()
                model_scores[name] = {
                    "r2": round(mean_score, 4),
                    "r2_std": round(scores.std(), 4),
                }

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = (name, model)
            except:
                model_scores[name] = {"error": "training_failed"}

        # Train best model on full data
        if best_model:
            model_name, model = best_model
            model.fit(X, y)

            # Extract feature importances or coefficients
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            else:
                importances = np.ones(len(param_cols))

            param_importance = {
                param_cols[i]: round(float(importances[i]), 4)
                for i in range(len(param_cols))
            }
            param_importance = dict(
                sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            )

            # Predict optimal bands (maximize for delta_cci, minimize for delta_hazard)
            optimization_direction = "maximize" if target == "delta_cci" else "minimize"

            results[target] = {
                "best_model": model_name,
                "r2_score": round(best_score, 4),
                "n_samples": len(X),
                "n_features": len(param_cols),
                "model_scores": model_scores,
                "parameter_importance": param_importance,
                "optimization": optimization_direction,
            }

    return results


def generate_optimal_sweeps(df: pd.DataFrame, ml_results: dict) -> dict[str, list]:
    """Generate optimal parameter sweeps based on ML insights"""
    sweeps = {}

    # Get top parameters for delta_cci
    if "delta_cci" in ml_results and "parameter_importance" in ml_results["delta_cci"]:
        top_params = list(ml_results["delta_cci"]["parameter_importance"].keys())[:4]
    else:
        # Fallback to correlation-based
        ranks = rank_relationships(df)
        cci_ranks = ranks[ranks["metric"] == "delta_cci"].head(4)
        top_params = cci_ranks["param"].tolist()

    # Generate sweeps for each parameter
    for p in top_params:
        if p not in df.columns:
            continue

        series = df[p].dropna()
        if len(series) < 3:
            continue

        # Use quantiles with smart padding
        q = np.quantile(series, [0.1, 0.25, 0.5, 0.75, 0.9])

        # Create 5-point sweep around optimal region
        if "delta_cci" in ml_results and "optimization" in ml_results["delta_cci"]:
            # Bias toward upper quantiles for maximization
            vals = [q[2], q[3], q[4], q[3] * 1.1, q[4] * 1.05]
        else:
            # Balanced sweep
            vals = [q[1], q[2], q[3], q[2] * 1.05, q[3] * 1.05]

        sweeps[p] = sorted(set([round(float(v), 6) for v in vals if not np.isnan(v)]))[
            :5
        ]

    return sweeps


def generate_experiment_queue(
    df: pd.DataFrame, ml_results: dict, ranks: pd.DataFrame
) -> list[dict]:
    """Generate ranked queue of proposed experiments"""
    queue = []

    sweeps = generate_optimal_sweeps(df, ml_results)

    # Top 5 experiments based on ML importance or correlation
    if "delta_cci" in ml_results and "parameter_importance" in ml_results["delta_cci"]:
        top_params = list(ml_results["delta_cci"]["parameter_importance"].items())[:5]
        experiments_data = [
            (param, importance, "ML_importance") for param, importance in top_params
        ]
    else:
        # Fallback to correlation
        top_corr = ranks.nlargest(5, "abs_corr")
        experiments_data = [
            (row["param"], row["corr"], "correlation") for _, row in top_corr.iterrows()
        ]

    for i, (param, score, method) in enumerate(experiments_data, 1):
        if param not in sweeps:
            continue

        experiment = {
            "rank": i,
            "id": f"autotheorist_exp{i}_{int(time.time())}",
            "title": f"Auto-Theorist Experiment {i}: {param.title()} Optimization",
            "hypothesis": f"Optimizing {param} is predicted to improve ΔCCI (method: {method}, score: {score:.3f})",
            "primary_parameter": param,
            "parameter_sweep": {param: sweeps[param]},
            "method": method,
            "prediction_score": round(float(score), 4),
            "preregistered": True,
            "thresholds": {
                "delta_cci_min": 0.03,
                "delta_hazard_max": -0.01,
                "rationale": f"Auto-generated based on {method} analysis",
                "source": "tools/auto_theorist.py v2.0",
                "version": "2.0.0",
            },
            "seeds": [11, 17, 23, 29],
            "data_source": "SIMULATION_ONLY",
            "depends_on": [],
            "adapter": "adapters/autotheorist_adapter.py",
        }

        queue.append(experiment)

    return queue


def export_experiment_yamls(queue: list[dict]):
    """Export top experiments as preregistered YAML files"""
    studies_dir = Path("studies")
    studies_dir.mkdir(exist_ok=True)

    generated = []

    for exp in queue[:3]:  # Export top 3
        yaml_path = studies_dir / f"{exp['id']}.yml"

        # Clean up for YAML export
        yaml_content = {
            "study_id": exp["id"],
            "title": exp["title"],
            "version": "2.0",
            "preregistered": exp["preregistered"],
            "hypothesis": exp["hypothesis"],
            "thresholds": exp["thresholds"],
            "seeds": exp["seeds"],
            "sweeps": exp["parameter_sweep"],
            "adapter": exp["adapter"],
            "data_source": exp["data_source"],
            "depends_on": exp["depends_on"],
            "metadata": {
                "generated_by": "auto_theorist v2.0",
                "method": exp["method"],
                "prediction_score": exp["prediction_score"],
                "rank": exp["rank"],
            },
        }

        yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False))
        generated.append(str(yaml_path))

    return generated


def main():
    print("🤖 Auto-Theorist v2.0 - ML-Powered Hypothesis Generation")
    print("=" * 70)

    # Load data
    print("📊 Loading discovery results...")
    df = load_df()

    if df.empty:
        print("❌ No summaries found. Run studies first to generate discovery_results/")
        sys.exit(0)

    print(f"✅ Loaded {len(df)} study results")
    print(f"   Parameters: {[c for c in df.columns if c in PARAM_KEYS]}")
    print(f"   Metrics: {[c for c in df.columns if c in METRIC_KEYS]}")

    # Correlation analysis
    print("\n🔍 Computing parameter-metric correlations...")
    ranks = rank_relationships(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(OUT_DIR / "param_metric_correlations.csv", index=False)
    print(f"   Exported: {OUT_DIR/'param_metric_correlations.csv'}")

    # ML model training
    print("\n🧠 Training ML prediction models...")
    ml_results = train_ml_models(df)
    (OUT_DIR / "ml_model_results.json").write_text(json.dumps(ml_results, indent=2))

    if ML_AVAILABLE:
        for target, results in ml_results.items():
            if "best_model" in results:
                print(
                    f"   {target}: {results['best_model']} (R² = {results['r2_score']:.3f})"
                )

    # Generate hypotheses
    print("\n💡 Generating hypotheses...")
    hyps = []
    for _, row in ranks.head(10).iterrows():
        direction = "increase" if row["corr"] > 0 else "decrease"
        confidence = (
            "high"
            if abs(row["corr"]) > 0.7
            else "moderate" if abs(row["corr"]) > 0.4 else "low"
        )

        hyps.append(
            {
                "rank": len(hyps) + 1,
                "text": f"{direction.title()} {row['param']} likely improves {row['metric']} (corr={row['corr']:.3f}, n={int(row['n'])}, confidence={confidence}).",
                "param": row["param"],
                "metric": row["metric"],
                "corr": float(row["corr"]),
                "n": int(row["n"]),
                "confidence": confidence,
            }
        )

    (OUT_DIR / "hypotheses.json").write_text(
        json.dumps({"generated": hyps, "version": "2.0"}, indent=2)
    )
    print(f"   Exported: {OUT_DIR/'hypotheses.json'} ({len(hyps)} hypotheses)")

    # Generate experiment queue
    print("\n📋 Generating ranked experiment queue...")
    queue = generate_experiment_queue(df, ml_results, ranks)
    (OUT_DIR / "autotheorist_queue.json").write_text(json.dumps(queue, indent=2))
    print(
        f"   Exported: {OUT_DIR/'autotheorist_queue.json'} ({len(queue)} experiments)"
    )

    # Export top experiments as YAML
    print("\n📝 Generating preregistered study YAMLs...")
    yaml_files = export_experiment_yamls(queue)
    for yf in yaml_files:
        print(f"   ✅ {yf}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Auto-Theorist 2.0 Complete")
    print(f"   📊 Analyzed: {len(df)} studies")
    print(f"   💡 Generated: {len(hyps)} hypotheses")
    print(f"   🎯 Queued: {len(queue)} experiments")
    print(f"   📝 YAMLs: {len(yaml_files)} preregistered studies")
    print("\n📁 Outputs:")
    print(f"   - {OUT_DIR/'param_metric_correlations.csv'}")
    print(f"   - {OUT_DIR/'ml_model_results.json'}")
    print(f"   - {OUT_DIR/'hypotheses.json'}")
    print(f"   - {OUT_DIR/'autotheorist_queue.json'}")
    for yf in yaml_files:
        print(f"   - {yf}")
    print("=" * 70)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from .tuning import logistic


def _safe_median(series: pd.Series, default: float) -> float:
    try:
        return (
            float(np.median(series.dropna().values))
            if len(series.dropna())
            else default
        )
    except Exception:
        return default


def map_survival(df: pd.DataFrame) -> dict:
    if "hazard" in df.columns:
        base_hazard = float(np.clip(df["hazard"].median(), 1e-6, 0.1))
    else:
        cols = [c for c in df.columns if "mort" in c.lower() or "hazard" in c.lower()]
        base_hazard = (
            0.01 if not cols else float(np.clip(df[cols[0]].median(), 1e-6, 0.1))
        )
    return {"biological": {"baseline_hazard": base_hazard}}


def map_shocks(df: pd.DataFrame) -> dict:
    severity = 0.2
    if "magnitude" in df.columns:
        q = df["magnitude"].quantile(0.9)
        severity = float(np.interp(q, [2.5, 7.5], [0.2, 0.8]))
    return {"ai": {"shock_severity": severity, "regrowth_rate": 0.1}}


def map_goals(df: pd.DataFrame, tuning: dict | None = None) -> dict:
    tuning = tuning or {}
    tcfg = tuning.get(
        "collapse_logistic", {"center": 0.30, "k": 40.0, "floor": 0.02, "cap": 0.98}
    )
    g = _safe_median(df.get("gini", pd.Series([0.26])), 0.26)
    goals = int(np.clip(_safe_median(df.get("goal_count", pd.Series([4])), 4), 2, 5))
    pop = int(_safe_median(df.get("population", pd.Series([300])), 300))
    social_weight = 0.6

    # Soft collapse risk via logistic curve around Gini center
    risk = logistic(
        g,
        x0=float(tcfg["center"]),
        k=float(tcfg["k"]),
        floor=float(tcfg["floor"]),
        cap=float(tcfg["cap"]),
    )
    collapse_model = {
        "type": "logistic",
        "center": float(tcfg["center"]),
        "k": float(tcfg["k"]),
        "floor": float(tcfg["floor"]),
        "cap": float(tcfg["cap"]),
        "effective_risk_at_gini": float(risk),
    }
    return {
        "social": {
            "gini": float(g),
            "goal_diversity": goals,
            "population": pop,
            "social_weight": social_weight,
            "collapse_risk_model": collapse_model,
        }
    }


def map_calibration(df: pd.DataFrame, tuning: dict | None = None) -> dict:
    """
    Accepts either the open demo RT dataset, a local domain-specific CSV if present, or synthetic.
    Expects columns:
    - reported_confidence in [0,1]
    - correct in {0,1}
    Smoothing: Beta(α,β) prior on accuracy to avoid extreme/noisy estimates.

    Uses stabilized CCI computation with epsilon guards to prevent division by zero.
    """
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[0].parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from core.cci_math import CCIConfig, compute_cci_from_data

    tuning = tuning or {}
    ccfg = tuning.get(
        "cci", {"min_noise": 0.05, "beta_prior_success": 2.0, "beta_prior_fail": 2.0}
    )
    have_cols = {"reported_confidence", "correct"}.issubset(df.columns)

    if have_cols:
        conf = float(np.clip(df["reported_confidence"].median(), 0.0, 1.0))
        # Beta prior smoothing for accuracy
        s = float(df["correct"].sum())
        n = float(len(df["correct"]))
        alpha = float(ccfg["beta_prior_success"])
        beta = float(ccfg["beta_prior_fail"])
        acc = float((s + alpha) / (n + alpha + beta)) if n > 0 else 0.5

        # Use stabilized CCI computation
        config = CCIConfig(
            EPSILON=float(ccfg["min_noise"]),
            BETA_PRIOR_SUCCESS=alpha,
            BETA_PRIOR_FAIL=beta,
        )

        cci_value, cci_metadata = compute_cci_from_data(
            reported_confidence=df["reported_confidence"].values,
            correct=df["correct"].values,
            config=config,
        )

        # Extract noise with epsilon guard applied
        noise = cci_metadata.get("effective_noise", float(ccfg["min_noise"]))

        return {
            "cci": {
                "reported_confidence_med": conf,
                "accuracy": acc,
                "noise": noise,
                "cci_value": cci_value,
                "epsilon_guard_applied": cci_metadata.get(
                    "epsilon_guard_applied", False
                ),
                "calibration_semantics": cci_metadata.get(
                    "calibration_semantics", "standard_calibration"
                ),
            }
        }
    else:
        conf, acc = 0.6, 0.55
        # Noise estimate with floor
        noise = float(np.clip(1.0 - acc, ccfg["min_noise"], 0.5))
        return {
            "cci": {"reported_confidence_med": conf, "accuracy": acc, "noise": noise}
        }


def _estimate_orbital_period(a_series: pd.Series) -> float:
    """
    Estimate a characteristic orbital period T from semi-major axis 'a' (normalized units).
    We assume T ∝ a^(3/2). Return median T in arbitrary time units.
    """
    a_med = _safe_median(a_series, 3.0)
    return float(np.power(max(a_med, 1e-6), 1.5))


def map_gravity(df: pd.DataFrame, tuning: dict | None = None) -> dict:
    tuning = tuning or {}
    gcfg = tuning.get(
        "gravity", {"dt_period_divisor": 1200, "eps_min": 0.01, "eps_max": 0.12}
    )

    ecc_med = _safe_median(df.get("eccentricity", pd.Series([0.3])), 0.3)
    mass_med = _safe_median(df.get("mass", pd.Series([1.0])), 1.0)
    a_med = _safe_median(df.get("a", pd.Series([3.0])), 3.0)

    # ε selection: more eccentric and more massive systems tend to need more softening for stability.
    eps = float(
        np.interp(
            0.5 * ecc_med + 0.5 * np.tanh(mass_med),
            [0.0, 1.0],
            [gcfg["eps_max"], gcfg["eps_min"]],
        )
    )
    # dt from characteristic orbital period
    T = _estimate_orbital_period(df.get("a", pd.Series([a_med])))
    dt = float(max(T / float(gcfg["dt_period_divisor"]), 1e-5))

    return {"gravity": {"softening": eps, "n": 50, "dt": dt}}


def build_param_pack(surv, shocks, goals, calib, grav) -> dict:
    return {
        "survival_params": surv,
        "shock_params": shocks,
        "goals_params": goals,
        "calibration_params": calib,
        "gravity_params": grav,
    }

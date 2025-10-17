"""
Statistical models for validating fear-violence dynamics.
"""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ModelFitter:
    """Fits statistical models to validate fear-violence hypotheses."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize model fitter with configuration.

        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.min_samples = self.config.get("min_samples", 30)
        self.vif_threshold = self.config.get("vif_threshold", 10.0)
        self.align_timescales = self.config.get("align_timescales", True)
        self.check_gini_variance = self.config.get("check_gini_variance", True)

    def fit_all_models(
        self, features_df: pd.DataFrame, city_name: str
    ) -> dict[str, Any]:
        """
        Fit all models for a city.

        Args:
            features_df: DataFrame with engineered features
            city_name: Name of the city

        Returns:
            Dictionary with model results
        """
        results = {
            "city": city_name,
            "n_observations": len(features_df),
            "date_range": {
                "start": features_df["date"].min().isoformat(),
                "end": features_df["date"].max().isoformat(),
            },
        }

        # Check data availability
        if len(features_df) < self.min_samples:
            results["error"] = (
                f"Insufficient data: {len(features_df)} < {self.min_samples}"
            )
            return results

        # Check time scale alignment if enabled
        if self.align_timescales:
            alignment_check = self._check_time_scale_alignment(features_df)
            if not alignment_check["aligned"]:
                results["error"] = (
                    f"Time scale misalignment: {alignment_check['message']}"
                )
                return results
            results["time_scale_alignment"] = alignment_check

        # Fit each model type
        try:
            results["aggression_model"] = self._fit_aggression_model(features_df)
        except Exception as e:
            results["aggression_model"] = {"error": str(e)}

        try:
            results["collapse_model"] = self._fit_collapse_model(features_df)
        except Exception as e:
            results["collapse_model"] = {"error": str(e)}

        try:
            results["event_study"] = self._fit_event_study(features_df)
        except Exception as e:
            results["event_study"] = {"error": str(e)}

        try:
            results["survival_model"] = self._fit_survival_model(features_df)
        except Exception as e:
            results["survival_model"] = {"error": str(e)}

        # Calculate replication metrics
        results["replication_metrics"] = self._calculate_replication_metrics(results)

        return results

    def _check_time_scale_alignment(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check if time scales are properly aligned for modeling.

        Args:
            df: Features DataFrame with date column

        Returns:
            Dictionary with alignment check results
        """
        # Check Gini variance if enabled
        if self.check_gini_variance and "gini" in df.columns:
            gini_values = df["gini"].dropna()
            if len(gini_values) > 0:
                gini_variance = gini_values.var()
                if gini_variance <= 0:
                    return {
                        "aligned": False,
                        "message": f"Gini variance is {gini_variance:.6f} (≤ 0) - insufficient variation for modeling",
                        "gini_variance": gini_variance,
                        "gini_n_obs": len(gini_values),
                    }

        # Check for sufficient date coverage
        date_range = (df["date"].max() - df["date"].min()).days
        if date_range < 365:  # Less than 1 year
            return {
                "aligned": False,
                "message": f"Insufficient date coverage: {date_range} days < 365 days",
                "date_range_days": date_range,
            }

        # Check for regular date intervals (weekly frequency expected)
        if len(df) > 1:
            date_diffs = df["date"].diff().dropna()
            median_diff = date_diffs.median()
            if median_diff.days < 6 or median_diff.days > 8:  # Not weekly
                return {
                    "aligned": False,
                    "message": f"Irregular date intervals: median {median_diff.days} days (expected 7)",
                    "median_interval_days": median_diff.days,
                }

        return {
            "aligned": True,
            "message": "Time scales properly aligned for modeling",
            "date_range_days": date_range,
            "n_observations": len(df),
        }

    def _fit_aggression_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit aggression ~ Fear + CCI + Fear*CCI + controls model."""
        # Prepare features
        feature_cols = ["fear_index", "cci_proxy", "fear_x_cci"]
        control_cols = ["gini", "trend", "month_sin", "month_cos"]

        # Check data availability
        available_features = [col for col in feature_cols if col in df.columns]
        available_controls = [col for col in control_cols if col in df.columns]

        if len(available_features) < 2:
            return {"error": "Insufficient features for aggression model"}

        # Prepare data
        X_cols = available_features + available_controls
        X = df[X_cols].fillna(0)
        y = df["crime_count"] if "crime_count" in df.columns else df["fear_index"]

        # Remove rows with missing target
        valid_mask = ~(y.isna() | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < self.min_samples:
            return {"error": f"Insufficient valid observations: {len(X)}"}

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit OLS model
        X_with_const = sm.add_constant(X_scaled)
        model = sm.OLS(y, X_with_const).fit()

        # Calculate VIF for multicollinearity
        vif_data = []
        for i in range(1, X_scaled.shape[1] + 1):
            vif = variance_inflation_factor(X_scaled, i - 1)
            vif_data.append({"feature": X_cols[i - 1], "vif": vif})

        # Extract key coefficients
        coefficients = {}
        p_values = {}

        for i, col in enumerate(X_cols):
            if i + 1 < len(model.params):  # +1 for constant term
                coefficients[col] = model.params.iloc[i + 1]
                p_values[col] = model.pvalues.iloc[i + 1]

        return {
            "model_type": "OLS",
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "coefficients": coefficients,
            "p_values": p_values,
            "vif": vif_data,
            "n_observations": len(X),
            "aic": model.aic,
            "bic": model.bic,
        }

    def _fit_collapse_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit Collapse ~ Gini + CCI + Gini*CCI + controls model."""
        # Check if collapse flag exists
        if "collapse_flag" not in df.columns:
            return {"error": "No collapse flag available"}

        # Prepare features
        feature_cols = ["gini", "cci_proxy", "gini_x_cci"]
        control_cols = ["fear_index", "trend", "month_sin", "month_cos"]

        available_features = [col for col in feature_cols if col in df.columns]
        available_controls = [col for col in control_cols if col in df.columns]

        if len(available_features) < 2:
            return {"error": "Insufficient features for collapse model"}

        # Prepare data
        X_cols = available_features + available_controls
        X = df[X_cols].fillna(0)
        y = df["collapse_flag"].astype(int)

        # Remove rows with missing target
        valid_mask = ~(y.isna() | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < self.min_samples:
            return {"error": f"Insufficient valid observations: {len(X)}"}

        # Check if we have both classes
        if y.nunique() < 2:
            return {"error": "No variation in collapse flag"}

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        X_with_const = sm.add_constant(X_scaled)
        model = sm.Logit(y, X_with_const).fit(disp=0)

        # Calculate AUC
        y_pred_proba = model.predict(X_with_const)
        auc = roc_auc_score(y, y_pred_proba)

        # Extract coefficients
        coefficients = {}
        p_values = {}

        for i, col in enumerate(X_cols):
            if i + 1 < len(model.params):
                coefficients[col] = model.params.iloc[i + 1]
                p_values[col] = model.pvalues.iloc[i + 1]

        return {
            "model_type": "Logistic",
            "auc": auc,
            "pseudo_r_squared": model.prsquared,
            "coefficients": coefficients,
            "p_values": p_values,
            "n_observations": len(X),
            "collapse_rate": y.mean(),
            "aic": model.aic,
            "bic": model.bic,
        }

    def _fit_event_study(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit event study around shocks comparing recovery patterns."""
        if "shock_severity" not in df.columns:
            return {"error": "No shock severity data available"}

        # Identify shock events
        moderate_shocks = df["shock_bin"] == "moderate"
        high_shocks = df["shock_bin"] == "high"

        if not (moderate_shocks.any() or high_shocks.any()):
            return {"error": "No shock events found"}

        # Calculate recovery metrics
        recovery_results = {}

        for shock_type, shock_mask in [
            ("moderate", moderate_shocks),
            ("high", high_shocks),
        ]:
            if not shock_mask.any():
                continue

            shock_dates = df[shock_mask]["date"]
            recovery_times = []

            for shock_date in shock_dates:
                # Look at 4 weeks before and after
                window_start = shock_date - pd.Timedelta(weeks=4)
                window_end = shock_date + pd.Timedelta(weeks=4)

                window_data = df[
                    (df["date"] >= window_start) & (df["date"] <= window_end)
                ]

                if len(window_data) < 5:  # Need at least 5 observations
                    continue

                # Calculate recovery time (time to return to pre-shock level)
                pre_shock_level = window_data[window_data["date"] < shock_date][
                    "fear_index"
                ].mean()

                if pd.isna(pre_shock_level):
                    continue

                post_shock_data = window_data[window_data["date"] > shock_date]
                recovery_time = None

                for i, (_, row) in enumerate(post_shock_data.iterrows()):
                    if row["fear_index"] <= pre_shock_level:
                        recovery_time = i + 1  # weeks
                        break

                if recovery_time is not None:
                    recovery_times.append(recovery_time)

            if recovery_times:
                recovery_results[shock_type] = {
                    "n_events": len(recovery_times),
                    "mean_recovery_time": np.mean(recovery_times),
                    "median_recovery_time": np.median(recovery_times),
                    "std_recovery_time": np.std(recovery_times),
                }

        # Compare recovery times
        comparison = {}
        if "moderate" in recovery_results and "high" in recovery_results:
            moderate_mean = recovery_results["moderate"]["mean_recovery_time"]
            high_mean = recovery_results["high"]["mean_recovery_time"]

            # Statistical test
            moderate_times = [
                recovery_results["moderate"]["mean_recovery_time"]
            ] * recovery_results["moderate"]["n_events"]
            high_times = [
                recovery_results["high"]["mean_recovery_time"]
            ] * recovery_results["high"]["n_events"]

            if len(moderate_times) > 1 and len(high_times) > 1:
                t_stat, p_value = stats.ttest_ind(moderate_times, high_times)
                comparison = {
                    "moderate_faster": moderate_mean < high_mean,
                    "difference": high_mean - moderate_mean,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                }

        return {"recovery_results": recovery_results, "comparison": comparison}

    def _fit_survival_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit survival model if suitable data is available."""
        # This is a placeholder for survival analysis
        # In practice, you'd need duration data (e.g., time to recovery, time to collapse)

        if "collapse_flag" not in df.columns:
            return {"error": "No survival data available"}

        # Simple analysis: time between collapse events
        collapse_events = df[df["collapse_flag"]]["date"]

        if len(collapse_events) < 2:
            return {"error": "Insufficient collapse events for survival analysis"}

        # Calculate inter-event times
        inter_event_times = collapse_events.diff().dt.days.dropna()

        if len(inter_event_times) == 0:
            return {"error": "No inter-event times calculated"}

        # Fit exponential distribution (simplest survival model)
        try:
            # Estimate rate parameter
            rate = 1 / inter_event_times.mean()

            # Calculate alpha (shape parameter) - should be close to 1 for exponential
            alpha = 1.0  # Exponential distribution has alpha = 1

            return {
                "model_type": "Exponential",
                "rate": rate,
                "alpha": alpha,
                "mean_survival_time": inter_event_times.mean(),
                "n_events": len(collapse_events),
                "alpha_in_range": 0.3
                <= alpha
                <= 0.5,  # Check if alpha is in expected range
            }
        except Exception as e:
            return {"error": f"Survival model failed: {str(e)}"}

    def _calculate_replication_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Calculate replication metrics for success criteria."""
        metrics = {}

        # Fear × CCI interaction
        if "aggression_model" in results and "error" not in results["aggression_model"]:
            aggression_model = results["aggression_model"]
            if "fear_x_cci" in aggression_model.get("coefficients", {}):
                fear_cci_coef = aggression_model["coefficients"]["fear_x_cci"]
                fear_cci_pval = aggression_model["p_values"].get("fear_x_cci", 1.0)

                metrics["fear_cci_interaction_negative"] = fear_cci_coef < 0
                metrics["fear_cci_interaction_significant"] = fear_cci_pval < 0.05
                metrics["fear_cci_interaction_replicated"] = (
                    fear_cci_coef < 0 and fear_cci_pval < 0.05
                )

        # Gini → Collapse
        if "collapse_model" in results and "error" not in results["collapse_model"]:
            collapse_model = results["collapse_model"]
            if "gini" in collapse_model.get("coefficients", {}):
                gini_coef = collapse_model["coefficients"]["gini"]
                gini_pval = collapse_model["p_values"].get("gini", 1.0)

                metrics["gini_collapse_positive"] = gini_coef > 0
                metrics["gini_collapse_significant"] = gini_pval < 0.05
                metrics["gini_collapse_replicated"] = gini_coef > 0 and gini_pval < 0.05

        # Shock recovery patterns
        if "event_study" in results and "error" not in results["event_study"]:
            event_study = results["event_study"]
            if "comparison" in event_study:
                comparison = event_study["comparison"]
                metrics["moderate_shocks_faster_recovery"] = comparison.get(
                    "moderate_faster", False
                )
                metrics["recovery_difference_significant"] = (
                    comparison.get("p_value", 1.0) < 0.05
                )
                metrics["shock_recovery_replicated"] = (
                    comparison.get("moderate_faster", False)
                    and comparison.get("p_value", 1.0) < 0.05
                )

        # Survival alpha
        if "survival_model" in results and "error" not in results["survival_model"]:
            survival_model = results["survival_model"]
            if "alpha" in survival_model:
                alpha = survival_model["alpha"]
                metrics["survival_alpha_in_range"] = 0.3 <= alpha <= 0.5

        return metrics

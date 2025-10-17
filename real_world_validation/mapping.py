# real_world_validation/mapping.py
"""
Mapping module for converting real-world data to simulation constructs.
Maps external data to shocks, survival, collapse, and CCI metrics.
"""

from typing import Any

import numpy as np
import pandas as pd

from .utils import robust_rolling_window, zscore


class DataMapper:
    """Maps real-world data to simulation constructs."""

    def __init__(self, scenario_config: dict[str, Any]):
        self.scenario_config = scenario_config
        self.metrics_config = scenario_config.get("metrics", {})
        self.scenario_kind = scenario_config.get("kind", "unknown")

    def map_to_simulation_constructs(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Map harmonized data to simulation constructs."""
        results = {}

        print(
            f"\nMapping data to simulation constructs for {self.scenario_kind} scenario..."
        )

        # Map shocks
        shocks_df = self._map_shocks(df)
        if not shocks_df.empty:
            results["shocks"] = shocks_df
            print(f"✓ Mapped {len(shocks_df)} shock events")

        # Map survival
        survival_df = self._map_survival(df)
        if not survival_df.empty:
            results["survival"] = survival_df
            print(f"✓ Mapped survival data: {len(survival_df)} periods")

        # Map collapse
        collapse_df = self._map_collapse(df)
        if not collapse_df.empty:
            results["collapse"] = collapse_df
            print(f"✓ Mapped collapse data: {len(collapse_df)} periods")

        # Map CCI (if enabled)
        cci_config = self.metrics_config.get("cci", {})
        if cci_config.get("enabled", False):
            cci_df = self._map_cci(df)
            if not cci_df.empty:
                results["cci"] = cci_df
                print(f"✓ Mapped CCI data: {len(cci_df)} periods")
        else:
            print("ℹ CCI mapping disabled for this scenario")

        return results

    def _map_shocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map data to shock events."""
        shocks_config = self.metrics_config.get("shocks", {})
        method = shocks_config.get("method", "drawdown_zscore")

        if method == "drawdown_zscore":
            return self._map_drawdown_shocks(df, shocks_config)
        elif method == "spike_zscore":
            return self._map_spike_shocks(df, shocks_config)
        elif method == "gdp_drawdown":
            return self._map_gdp_shocks(df, shocks_config)
        else:
            print(f"Unknown shock mapping method: {method}")
            return pd.DataFrame()

    def _map_drawdown_shocks(
        self, df: pd.DataFrame, config: dict[str, Any]
    ) -> pd.DataFrame:
        """Map market drawdowns to shock events."""
        # Look for drawdown data
        drawdown_cols = [col for col in df.columns if "drawdown" in col.lower()]
        if not drawdown_cols:
            print("No drawdown data found for shock mapping")
            return pd.DataFrame()

        drawdown_col = drawdown_cols[0]
        shocks_df = df[["date", drawdown_col]].copy()

        # Calculate z-score of drawdowns
        shocks_df["drawdown_zscore"] = zscore(shocks_df[drawdown_col], window=30)

        # Identify shock events (z-score < -2)
        shock_threshold = config.get("threshold", -2.0)
        shocks_df["is_shock"] = shocks_df["drawdown_zscore"] < shock_threshold

        # Classify shock severity
        shocks_df["shock_severity"] = np.where(
            shocks_df["is_shock"],
            np.clip(-shocks_df["drawdown_zscore"] / 2, 0, 1),  # Normalize to [0,1]
            0,
        )

        # Classify shock types based on severity
        shocks_df["shock_class"] = np.where(
            shocks_df["shock_severity"] == 0,
            "none",
            np.where(
                shocks_df["shock_severity"] < 0.5,
                "constructive",
                np.where(
                    shocks_df["shock_severity"] < 0.7, "transition", "destructive"
                ),
            ),
        )

        # Keep only shock events
        shock_events = shocks_df[shocks_df["is_shock"]].copy()

        if not shock_events.empty:
            shock_events = shock_events.drop("is_shock", axis=1)
            shock_events["shock_id"] = range(len(shock_events))

        return shock_events

    def _map_spike_shocks(
        self, df: pd.DataFrame, config: dict[str, Any]
    ) -> pd.DataFrame:
        """Map epidemic spikes to shock events."""
        series_name = config.get("series", "new_cases_smoothed")

        # Look for the series in the data
        series_cols = [col for col in df.columns if series_name in col.lower()]
        if not series_cols:
            print(f"No {series_name} data found for spike shock mapping")
            return pd.DataFrame()

        series_col = series_cols[0]
        shocks_df = df[["date", series_col]].copy()

        # Calculate z-score of the series
        shocks_df["spike_zscore"] = zscore(shocks_df[series_col], window=30)

        # Identify shock events (z-score > 2)
        shock_threshold = config.get("threshold", 2.0)
        shocks_df["is_shock"] = shocks_df["spike_zscore"] > shock_threshold

        # Classify shock severity
        shocks_df["shock_severity"] = np.where(
            shocks_df["is_shock"],
            np.clip(shocks_df["spike_zscore"] / 4, 0, 1),  # Normalize to [0,1]
            0,
        )

        # Classify shock types
        shocks_df["shock_class"] = np.where(
            shocks_df["shock_severity"] == 0,
            "none",
            np.where(
                shocks_df["shock_severity"] < 0.5,
                "constructive",
                np.where(
                    shocks_df["shock_severity"] < 0.7, "transition", "destructive"
                ),
            ),
        )

        # Keep only shock events
        shock_events = shocks_df[shocks_df["is_shock"]].copy()

        if not shock_events.empty:
            shock_events = shock_events.drop("is_shock", axis=1)
            shock_events["shock_id"] = range(len(shock_events))

        return shock_events

    def _map_gdp_shocks(self, df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        """Map GDP growth drops to shock events."""
        # Look for GDP growth data
        gdp_cols = [
            col
            for col in df.columns
            if "gdp" in col.lower() and "growth" in col.lower()
        ]
        if not gdp_cols:
            print("No GDP growth data found for shock mapping")
            return pd.DataFrame()

        gdp_col = gdp_cols[0]
        shocks_df = df[["date", gdp_col]].copy()

        # Calculate z-score of GDP growth (negative values are shocks)
        shocks_df["gdp_zscore"] = zscore(shocks_df[gdp_col], window=30)

        # Identify shock events (z-score < -1.5 for negative growth)
        shock_threshold = config.get("threshold", -1.5)
        shocks_df["is_shock"] = (shocks_df["gdp_zscore"] < shock_threshold) & (
            shocks_df[gdp_col] < 0
        )

        # Classify shock severity
        shocks_df["shock_severity"] = np.where(
            shocks_df["is_shock"],
            np.clip(-shocks_df["gdp_zscore"] / 3, 0, 1),  # Normalize to [0,1]
            0,
        )

        # Classify shock types
        shocks_df["shock_class"] = np.where(
            shocks_df["shock_severity"] == 0,
            "none",
            np.where(
                shocks_df["shock_severity"] < 0.5,
                "constructive",
                np.where(
                    shocks_df["shock_severity"] < 0.7, "transition", "destructive"
                ),
            ),
        )

        # Keep only shock events
        shock_events = shocks_df[shocks_df["is_shock"]].copy()

        if not shock_events.empty:
            shock_events = shock_events.drop("is_shock", axis=1)
            shock_events["shock_id"] = range(len(shock_events))

        return shock_events

    def _map_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map data to survival curves."""
        survival_config = self.metrics_config.get("survival", {})
        method = survival_config.get("method", "trough_to_baseline")

        if method == "trough_to_baseline":
            return self._map_trough_to_baseline_survival(df, survival_config)
        else:
            print(f"Unknown survival mapping method: {method}")
            return pd.DataFrame()

    def _map_trough_to_baseline_survival(
        self, df: pd.DataFrame, config: dict[str, Any]
    ) -> pd.DataFrame:
        """Map trough-to-baseline recovery to survival curves."""
        baseline_window_days = config.get("baseline_window_days", 120)
        series_name = config.get("series", None)

        # Determine the series to analyze
        if series_name:
            series_cols = [col for col in df.columns if series_name in col.lower()]
            if not series_cols:
                print(f"No {series_name} data found for survival mapping")
                return pd.DataFrame()
            series_col = series_cols[0]
        else:
            # Default to first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [
                col
                for col in numeric_cols
                if col not in ["year", "month", "day_of_year", "week_of_year"]
            ]
            if not numeric_cols:
                print("No numeric data found for survival mapping")
                return pd.DataFrame()
            series_col = numeric_cols[0]

        survival_df = df[["date", series_col]].copy()
        survival_df = survival_df.dropna(subset=[series_col])

        if len(survival_df) < baseline_window_days:
            print(
                f"Insufficient data for survival mapping: {len(survival_df)} < {baseline_window_days}"
            )
            return pd.DataFrame()

        # Calculate baseline (rolling mean)
        survival_df["baseline"] = robust_rolling_window(
            survival_df[series_col], baseline_window_days
        ).mean()

        # Identify troughs (local minima)
        survival_df["is_trough"] = (
            survival_df[series_col] < survival_df[series_col].shift(1)
        ) & (survival_df[series_col] < survival_df[series_col].shift(-1))

        # Calculate recovery periods
        trough_indices = survival_df[survival_df["is_trough"]].index
        recovery_periods = []

        for trough_idx in trough_indices:
            trough_value = survival_df.loc[trough_idx, series_col]
            baseline_value = survival_df.loc[trough_idx, "baseline"]

            if pd.isna(trough_value) or pd.isna(baseline_value):
                continue

            # Find recovery point (first point after trough that reaches baseline)
            recovery_idx = None
            for i in range(trough_idx + 1, len(survival_df)):
                if survival_df.loc[i, series_col] >= baseline_value:
                    recovery_idx = i
                    break

            if recovery_idx is not None:
                recovery_days = recovery_idx - trough_idx
                recovery_periods.append(
                    {
                        "trough_date": survival_df.loc[trough_idx, "date"],
                        "recovery_date": survival_df.loc[recovery_idx, "date"],
                        "recovery_days": recovery_days,
                        "trough_value": trough_value,
                        "baseline_value": baseline_value,
                        "recovery_depth": (baseline_value - trough_value)
                        / baseline_value,
                    }
                )

        if not recovery_periods:
            print("No recovery periods found")
            return pd.DataFrame()

        survival_result = pd.DataFrame(recovery_periods)
        survival_result["survival_id"] = range(len(survival_result))

        return survival_result

    def _map_collapse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map data to collapse risk."""
        collapse_config = self.metrics_config.get("collapse", {})
        threshold = collapse_config.get("threshold", 0.3)

        # Look for Gini data
        gini_cols = [col for col in df.columns if "gini" in col.lower()]

        if gini_cols:
            return self._map_gini_collapse(df, gini_cols[0], threshold)
        else:
            # Use proxy method
            proxy_method = collapse_config.get("gini_proxy", "return_dispersion")
            return self._map_proxy_collapse(df, proxy_method, threshold)

    def _map_gini_collapse(
        self, df: pd.DataFrame, gini_col: str, threshold: float
    ) -> pd.DataFrame:
        """Map Gini coefficient to collapse risk."""
        collapse_df = df[["date", gini_col]].copy()
        collapse_df = collapse_df.dropna(subset=[gini_col])

        if collapse_df.empty:
            print("No Gini data available for collapse mapping")
            return pd.DataFrame()

        # Calculate collapse risk
        collapse_df["gini"] = collapse_df[gini_col]
        collapse_df["collapse_risk"] = np.where(
            collapse_df["gini"] >= threshold, 1.0, collapse_df["gini"] / threshold
        )

        # Identify collapse events
        collapse_df["is_collapse"] = collapse_df["gini"] >= threshold

        # Add complexity and coordination proxies (simplified)
        collapse_df["complexity_proxy"] = 0.5  # Default moderate complexity
        collapse_df["coordination_proxy"] = 0.7  # Default moderate coordination

        # Calculate collapse law: risk ∝ (inequality × complexity) / coordination
        collapse_df["collapse_law"] = (
            collapse_df["gini"] * collapse_df["complexity_proxy"]
        ) / collapse_df["coordination_proxy"]

        return collapse_df

    def _map_proxy_collapse(
        self, df: pd.DataFrame, proxy_method: str, threshold: float
    ) -> pd.DataFrame:
        """Map proxy measures to collapse risk."""
        if proxy_method == "return_dispersion":
            # Use market return volatility as inequality proxy
            return_cols = [col for col in df.columns if "return" in col.lower()]
            if not return_cols:
                print("No return data found for collapse proxy")
                return pd.DataFrame()

            return_col = return_cols[0]
            collapse_df = df[["date", return_col]].copy()
            collapse_df = collapse_df.dropna(subset=[return_col])

            # Calculate rolling volatility as inequality proxy
            collapse_df["volatility"] = robust_rolling_window(
                collapse_df[return_col], 30
            ).std()

            # Normalize volatility to [0,1] range
            vol_min, vol_max = (
                collapse_df["volatility"].min(),
                collapse_df["volatility"].max(),
            )
            if vol_max > vol_min:
                collapse_df["gini_proxy"] = (collapse_df["volatility"] - vol_min) / (
                    vol_max - vol_min
                )
            else:
                collapse_df["gini_proxy"] = 0.5

            # Calculate collapse risk
            collapse_df["collapse_risk"] = np.where(
                collapse_df["gini_proxy"] >= threshold,
                1.0,
                collapse_df["gini_proxy"] / threshold,
            )

            collapse_df["is_collapse"] = collapse_df["gini_proxy"] >= threshold

            return collapse_df

        else:
            print(f"Unknown collapse proxy method: {proxy_method}")
            return pd.DataFrame()

    def _map_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map data to CCI (Consciousness Calibration Index)."""
        # This is a placeholder implementation
        # In practice, CCI requires predictions vs outcomes data
        print("ℹ CCI mapping requires prediction vs outcome data - skipping for now")
        return pd.DataFrame()


def map_data_to_simulation(
    scenario_config: dict[str, Any], df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Map harmonized data to simulation constructs."""
    mapper = DataMapper(scenario_config)
    return mapper.map_to_simulation_constructs(df)

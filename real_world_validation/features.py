"""
Feature engineering module for creating fear-violence indicators.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates fear-violence features from cleaned data."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize feature engineer with configuration.

        Args:
            config: Feature engineering configuration
        """
        self.config = config or {}
        self.collapse_quantile = self.config.get("collapse_quantile", 0.9)
        self.collapse_min_weeks = self.config.get("collapse_min_weeks", 3)
        self.cci_window = self.config.get(
            "cci_window", 12
        )  # 12 weeks for CCI calculation
        self.align_timescales = self.config.get("align_timescales", True)
        self.gini_resample_method = self.config.get(
            "gini_resample_method", "ffill"
        )  # 'ffill', 'cubic', 'linear'

    def create_features(self, cleaned_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create all features from cleaned data sources.

        Args:
            cleaned_data: Dictionary of cleaned DataFrames

        Returns:
            DataFrame with all engineered features
        """
        # Start with date range from all sources
        all_dates = set()
        for df in cleaned_data.values():
            if not df.empty and "date" in df.columns:
                all_dates.update(df["date"].dt.date)

        if not all_dates:
            return pd.DataFrame()

        # Create base DataFrame with all dates
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = pd.date_range(start=min_date, end=max_date, freq="W")

        features_df = pd.DataFrame({"date": date_range})

        # Merge each data source
        for source_name, df in cleaned_data.items():
            if not df.empty and "date" in df.columns:
                features_df = self._merge_source_data(features_df, df, source_name)

        # Create engineered features
        features_df = self._create_fear_index(features_df)
        features_df = self._create_cci_proxy(features_df)
        features_df = self._create_shock_severity(features_df)
        features_df = self._create_collapse_flag(features_df)
        features_df = self._create_interaction_terms(features_df)
        features_df = self._create_temporal_features(features_df)

        return features_df

    def _merge_source_data(
        self, base_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge a data source into the base features DataFrame."""
        if source_df.empty:
            return base_df

        # Ensure date column is datetime
        source_df = source_df.copy()
        source_df["date"] = pd.to_datetime(source_df["date"])

        # Merge based on source type
        if source_name == "crime":
            base_df = base_df.merge(
                source_df[["date", "count"]],
                on="date",
                how="left",
                suffixes=("", f"_{source_name}"),
            )
            base_df = base_df.rename(columns={"count": "crime_count"})

        elif source_name == "gini":
            # Handle time scale alignment for Gini data
            if self.align_timescales:
                source_df = self._resample_gini(source_df, target_freq="W")

            base_df = base_df.merge(source_df[["date", "gini"]], on="date", how="left")

        elif source_name == "trends":
            base_df = base_df.merge(
                source_df[["date", "fear_index"]],
                on="date",
                how="left",
                suffixes=("", "_raw"),
            )

        elif source_name == "events":
            base_df = base_df.merge(
                source_df[["date", "magnitude"]], on="date", how="left"
            )
            base_df = base_df.rename(columns={"magnitude": "event_magnitude"})

        return base_df

    def _resample_gini(
        self, gini_df: pd.DataFrame, target_freq: str = "W"
    ) -> pd.DataFrame:
        """
        Resample Gini data to target frequency using step-hold (forward-fill) or interpolation.

        Args:
            gini_df: DataFrame with 'date' and 'gini' columns
            target_freq: Target frequency ('W' for weekly, 'D' for daily, etc.)

        Returns:
            Resampled DataFrame
        """
        if gini_df.empty or "gini" not in gini_df.columns:
            return gini_df

        # Ensure date column is datetime
        gini_df = gini_df.copy()
        gini_df["date"] = pd.to_datetime(gini_df["date"])

        # Remove duplicates and sort by date
        gini_df = gini_df.drop_duplicates(subset=["date"]).sort_values("date")

        # Set date as index for resampling
        gini_df = gini_df.set_index("date")

        # Detect original frequency
        original_freq = pd.infer_freq(gini_df.index)
        if original_freq is None:
            # Try to infer from data
            time_diffs = gini_df.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            if median_diff.days >= 365:
                original_freq = "A"  # Annual
            elif median_diff.days >= 30:
                original_freq = "M"  # Monthly
            elif median_diff.days >= 7:
                original_freq = "W"  # Weekly
            else:
                original_freq = "D"  # Daily

        # Check if resampling is needed
        if original_freq == target_freq:
            return gini_df.reset_index()

        # Resample based on method
        if self.gini_resample_method == "ffill":
            # Forward-fill (step-hold)
            resampled = gini_df.resample(target_freq).ffill()
            warnings.warn(
                f"Gini data resampled from {original_freq} to {target_freq} using forward-fill"
            )

        elif self.gini_resample_method in ["cubic", "linear"]:
            # Interpolation methods
            if len(gini_df) < 4 and self.gini_resample_method == "cubic":
                # Fall back to linear for insufficient points
                self.gini_resample_method = "linear"
                warnings.warn(
                    "Insufficient data for cubic interpolation, falling back to linear"
                )

            # Create continuous index
            start_date = gini_df.index.min()
            end_date = gini_df.index.max()
            target_index = pd.date_range(
                start=start_date, end=end_date, freq=target_freq
            )

            # Interpolate
            interpolated = gini_df["gini"].interpolate(method=self.gini_resample_method)
            resampled = interpolated.reindex(
                target_index, method=self.gini_resample_method
            )
            resampled = pd.DataFrame({"gini": resampled})

            warnings.warn(
                f"Gini data resampled from {original_freq} to {target_freq} using {self.gini_resample_method} interpolation"
            )

        else:
            raise ValueError(f"Unknown resample method: {self.gini_resample_method}")

        # Remove NaN values at the beginning
        resampled = resampled.dropna()

        return resampled.reset_index()

    def _create_fear_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized fear index from trends data."""
        if "fear_index" not in df.columns:
            df["fear_index"] = 0.0
            return df

        # Z-score the fear index
        fear_values = df["fear_index"].fillna(0)
        df["fear_index"] = (fear_values - fear_values.mean()) / fear_values.std()

        # Clip to reasonable range
        df["fear_index"] = np.clip(df["fear_index"], -3, 3)

        return df

    def _create_cci_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create CCI proxy from volatility of crime and fear."""
        df["cci_proxy"] = 0.5  # Default value

        # Calculate rolling volatility of crime and fear
        if "crime_count" in df.columns:
            crime_volatility = (
                df["crime_count"].rolling(window=self.cci_window, min_periods=1).std()
            )

            if "fear_index" in df.columns:
                fear_volatility = (
                    df["fear_index"]
                    .rolling(window=self.cci_window, min_periods=1)
                    .std()
                )

                # Combine volatilities
                combined_volatility = (crime_volatility + fear_volatility) / 2
            else:
                combined_volatility = crime_volatility
        else:
            # Use only fear volatility if no crime data
            if "fear_index" in df.columns:
                combined_volatility = (
                    df["fear_index"]
                    .rolling(window=self.cci_window, min_periods=1)
                    .std()
                )
            else:
                return df

        # Convert volatility to CCI proxy (higher volatility = lower CCI)
        max_volatility = combined_volatility.max()
        if max_volatility > 0:
            df["cci_proxy"] = 1.0 - (combined_volatility / max_volatility)
            df["cci_proxy"] = np.clip(df["cci_proxy"], 0.0, 1.0)

        return df

    def _create_shock_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create shock severity from events data."""
        df["shock_severity"] = 0.0
        df["shock_bin"] = "none"

        if "event_magnitude" in df.columns:
            # Z-score event magnitudes
            magnitudes = df["event_magnitude"].fillna(0)
            if magnitudes.std() > 0:
                z_scores = (magnitudes - magnitudes.mean()) / magnitudes.std()

                # Clip to [0,1] range
                df["shock_severity"] = np.clip(z_scores, 0, 1)

                # Create bins
                df["shock_bin"] = pd.cut(
                    df["shock_severity"],
                    bins=[-np.inf, 0.5, 0.8, np.inf],
                    labels=["low", "moderate", "high"],
                )

        return df

    def _create_collapse_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create collapse flag based on multiple indicators."""
        df["collapse_flag"] = False

        # Calculate collapse indicators
        collapse_indicators = []

        # High crime indicator
        if "crime_count" in df.columns:
            crime_90th = df["crime_count"].quantile(self.collapse_quantile)
            high_crime = df["crime_count"] > crime_90th
            collapse_indicators.append(high_crime)

        # High fear indicator
        if "fear_index" in df.columns:
            fear_90th = df["fear_index"].quantile(self.collapse_quantile)
            high_fear = df["fear_index"] > fear_90th
            collapse_indicators.append(high_fear)

        # High inequality indicator
        if "gini" in df.columns:
            gini_90th = df["gini"].quantile(self.collapse_quantile)
            high_gini = df["gini"] > gini_90th
            collapse_indicators.append(high_gini)

        # Low CCI indicator
        if "cci_proxy" in df.columns:
            cci_10th = df["cci_proxy"].quantile(1 - self.collapse_quantile)
            low_cci = df["cci_proxy"] < cci_10th
            collapse_indicators.append(low_cci)

        if collapse_indicators:
            # Collapse if majority of indicators are high
            collapse_score = np.mean(collapse_indicators, axis=0)
            df["collapse_score"] = collapse_score

            # Flag collapse if score is high for consecutive periods
            high_score = collapse_score > 0.5
            df["collapse_flag"] = self._detect_consecutive_periods(
                high_score, self.collapse_min_weeks
            )

        return df

    def _detect_consecutive_periods(
        self, series: pd.Series, min_periods: int
    ) -> pd.Series:
        """Detect consecutive periods where condition is True."""
        result = pd.Series(False, index=series.index)

        # Find runs of True values
        runs = []
        current_run_start = None

        for i, value in enumerate(series):
            if value and current_run_start is None:
                current_run_start = i
            elif not value and current_run_start is not None:
                runs.append((current_run_start, i - 1))
                current_run_start = None

        # Handle run that goes to the end
        if current_run_start is not None:
            runs.append((current_run_start, len(series) - 1))

        # Flag runs that meet minimum length
        for start, end in runs:
            if end - start + 1 >= min_periods:
                result.iloc[start : end + 1] = True

        return result

    def _create_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction terms for modeling."""
        # Fear × CCI interaction
        if "fear_index" in df.columns and "cci_proxy" in df.columns:
            df["fear_x_cci"] = df["fear_index"] * df["cci_proxy"]

        # Gini × CCI interaction
        if "gini" in df.columns and "cci_proxy" in df.columns:
            df["gini_x_cci"] = df["gini"] * df["cci_proxy"]

        # Fear × Gini interaction
        if "fear_index" in df.columns and "gini" in df.columns:
            df["fear_x_gini"] = df["fear_index"] * df["gini"]

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features for seasonality and trends."""
        if df.empty:
            return df

        # Extract temporal components
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["week_of_year"] = df["date"].dt.isocalendar().week

        # Create cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # Linear trend
        df["trend"] = range(len(df))

        return df

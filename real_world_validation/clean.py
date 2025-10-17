"""
Data cleaning and preprocessing module.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd


class DataCleaner:
    """Handles data cleaning, resampling, and quality control."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize data cleaner with configuration.

        Args:
            config: Cleaning configuration dictionary
        """
        self.config = config or {}
        self.outlier_z_threshold = self.config.get("outlier_z_threshold", 4.0)
        self.max_gap_periods = self.config.get("max_gap_periods", 2)
        self.aggregation_freq = self.config.get(
            "aggregation_freq", "W"
        )  # Weekly by default

    def clean_city_data(
        self,
        city_data: dict[str, pd.DataFrame],
        timezone: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Clean all data sources for a city.

        Args:
            city_data: Dictionary of source DataFrames
            timezone: City timezone
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned_data = {}

        for source_name, df in city_data.items():
            try:
                cleaned_df = self.clean_source_data(
                    df, source_name, timezone, start_date, end_date
                )
                cleaned_data[source_name] = cleaned_df
                print(f"  ✓ Cleaned {source_name}: {len(cleaned_df)} rows")
            except Exception as e:
                warnings.warn(f"Failed to clean {source_name}: {str(e)}")
                continue

        return cleaned_data

    def clean_source_data(
        self,
        df: pd.DataFrame,
        source_name: str,
        timezone: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Clean a single data source.

        Args:
            df: Input DataFrame
            source_name: Name of the data source
            timezone: Target timezone
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Ensure date column exists and is datetime
        if "date" not in cleaned_df.columns:
            raise ValueError(f"No 'date' column found in {source_name}")

        cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])

        # Localize to city timezone
        if cleaned_df["date"].dt.tz is None:
            cleaned_df["date"] = cleaned_df["date"].dt.tz_localize(timezone)
        else:
            cleaned_df["date"] = cleaned_df["date"].dt.tz_convert(timezone)

        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize(timezone)
            cleaned_df = cleaned_df[cleaned_df["date"] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize(timezone)
            cleaned_df = cleaned_df[cleaned_df["date"] <= end_dt]

        # Remove duplicates
        cleaned_df = self._remove_duplicates(cleaned_df)

        # Handle outliers
        cleaned_df = self._cap_outliers(cleaned_df, source_name)

        # Resample to consistent frequency
        cleaned_df = self._resample_data(cleaned_df, source_name)

        # Fill gaps
        cleaned_df = self._fill_gaps(cleaned_df)

        # Add quality flags
        cleaned_df = self._add_quality_flags(cleaned_df, source_name)

        return cleaned_df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on date."""
        if df.empty:
            return df

        # Sort by date first
        df_sorted = df.sort_values("date")

        # Remove duplicates, keeping first occurrence
        df_deduped = df_sorted.drop_duplicates(subset=["date"], keep="first")

        return df_deduped

    def _cap_outliers(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Cap outliers using z-score threshold."""
        if df.empty:
            return df

        df_capped = df.copy()

        # Identify numeric columns to cap
        numeric_cols = df_capped.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "date"]

        for col in numeric_cols:
            if col in df_capped.columns:
                # Calculate z-scores
                z_scores = np.abs(
                    (df_capped[col] - df_capped[col].mean()) / df_capped[col].std()
                )

                # Cap outliers
                outlier_mask = z_scores > self.outlier_z_threshold
                if outlier_mask.any():
                    # Cap to threshold
                    upper_bound = (
                        df_capped[col].mean()
                        + self.outlier_z_threshold * df_capped[col].std()
                    )
                    lower_bound = (
                        df_capped[col].mean()
                        - self.outlier_z_threshold * df_capped[col].std()
                    )

                    df_capped.loc[outlier_mask, col] = np.clip(
                        df_capped.loc[outlier_mask, col], lower_bound, upper_bound
                    )

        return df_capped

    def _resample_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Resample data to consistent frequency."""
        if df.empty:
            return df

        df_resampled = df.copy()

        # Set date as index for resampling
        df_resampled = df_resampled.set_index("date")

        # Resample based on source type
        if source_name == "crime":
            # Sum counts over the period
            df_resampled = df_resampled.resample(self.aggregation_freq).sum()
        elif source_name == "gini":
            # Forward fill Gini (annual data)
            df_resampled = df_resampled.resample(self.aggregation_freq).ffill()
        elif source_name == "trends":
            # Mean of fear index over the period
            df_resampled = df_resampled.resample(self.aggregation_freq).mean()
        elif source_name == "events":
            # Sum magnitude over the period
            df_resampled = df_resampled.resample(self.aggregation_freq).sum()
        else:
            # Default to mean
            df_resampled = df_resampled.resample(self.aggregation_freq).mean()

        # Reset index to get date column back
        df_resampled = df_resampled.reset_index()

        return df_resampled

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small gaps in the data."""
        if df.empty:
            return df

        df_filled = df.copy()

        # Identify numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "date"]

        for col in numeric_cols:
            if col in df_filled.columns:
                # Forward fill small gaps
                df_filled[col] = df_filled[col].fillna(
                    method="ffill", limit=self.max_gap_periods
                )

                # Backward fill remaining gaps
                df_filled[col] = df_filled[col].fillna(
                    method="bfill", limit=self.max_gap_periods
                )

        return df_filled

    def _add_quality_flags(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Add data quality flags."""
        if df.empty:
            return df

        df_flagged = df.copy()

        # Missing data flag
        df_flagged["missing_data"] = df_flagged.isnull().any(axis=1)

        # Outlier flag (for numeric columns)
        numeric_cols = df_flagged.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in ["date", "missing_data"]
        ]

        outlier_flags = []
        for col in numeric_cols:
            if col in df_flagged.columns:
                z_scores = np.abs(
                    (df_flagged[col] - df_flagged[col].mean()) / df_flagged[col].std()
                )
                outlier_flags.append(z_scores > self.outlier_z_threshold)

        if outlier_flags:
            df_flagged["has_outliers"] = np.any(outlier_flags, axis=0)
        else:
            df_flagged["has_outliers"] = False

        # Data completeness score
        df_flagged["completeness_score"] = 1.0 - df_flagged.isnull().sum(axis=1) / len(
            df_flagged.columns
        )

        return df_flagged

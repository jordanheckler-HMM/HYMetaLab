# real_world_validation/etl.py
"""
ETL pipeline for real-world validation data.
Handles data harmonization, cleaning, and validation.
"""

from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    log_missingness,
    pct_change,
    robust_rolling_window,
    validate_schema,
    zscore,
)


class ETLPipeline:
    """ETL pipeline for real-world data processing."""

    def __init__(self, scenario_config: dict[str, Any]):
        self.scenario_config = scenario_config
        self.window = scenario_config.get("window", {})
        self.start_date = pd.to_datetime(self.window.get("start", "2000-01-01"))
        self.end_date = pd.to_datetime(self.window.get("end", "2024-12-31"))

    def process_raw_data(
        self, raw_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Process raw data into clean, harmonized format."""
        processed_data = {}

        for source_name, df in raw_data.items():
            print(f"\nProcessing {source_name}...")

            if source_name == "owid":
                processed_df = self._process_owid_data(df)
            elif source_name.startswith("world_bank"):
                processed_df = self._process_worldbank_data(df, source_name)
            elif source_name == "market":
                processed_df = self._process_market_data(df)
            else:
                print(f"Unknown data source: {source_name}")
                continue

            if processed_df is not None and not processed_df.empty:
                processed_data[source_name] = processed_df
                print(f"✓ Processed {source_name}: {len(processed_df)} rows")
            else:
                print(f"❌ Failed to process {source_name}")

        return processed_data

    def _process_owid_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Process OWID COVID-19 data."""
        try:
            # Validate required columns
            required_cols = [
                "date",
                "iso_code",
                "new_cases_smoothed",
                "new_deaths_smoothed",
                "stringency_index",
            ]
            if not validate_schema(df, required_cols):
                return None

            # Convert date column
            df["date"] = pd.to_datetime(df["date"])

            # Filter by date range
            df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]

            # Get country-specific data if specified
            data_config = self.scenario_config.get("data", {}).get("owid", {})
            iso3 = data_config.get("iso3")

            if iso3:
                df = df[df["iso_code"] == iso3].copy()
                if df.empty:
                    print(f"No data found for ISO3: {iso3}")
                    return None

            # Select relevant columns
            columns_to_keep = ["date", "iso_code"]
            if "columns" in data_config:
                columns_to_keep.extend(data_config["columns"])
            else:
                columns_to_keep.extend(
                    ["new_cases_smoothed", "new_deaths_smoothed", "stringency_index"]
                )

            # Keep only available columns
            available_cols = [col for col in columns_to_keep if col in df.columns]
            df = df[available_cols].copy()

            # Remove rows where all numeric columns are NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df.dropna(subset=numeric_cols, how="all")

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            log_missingness(df, f"OWID {iso3 or 'Global'}")
            return df

        except Exception as e:
            print(f"Error processing OWID data: {e}")
            return None

    def _process_worldbank_data(
        self, df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame | None:
        """Process World Bank data."""
        try:
            # World Bank CSV has metadata rows at the top
            # Find the actual data start
            data_start = None
            for i, row in df.iterrows():
                if isinstance(row.iloc[0], str) and row.iloc[0].startswith(
                    "Country Name"
                ):
                    data_start = i
                    break

            if data_start is None:
                print("Could not find data start in World Bank CSV")
                return None

            # Extract data portion
            df_data = df.iloc[data_start:].copy()
            df_data.columns = df_data.iloc[0]
            df_data = df_data.iloc[1:].reset_index(drop=True)

            # Convert to long format
            country_cols = [
                "Country Name",
                "Country Code",
                "Indicator Name",
                "Indicator Code",
            ]
            year_cols = [col for col in df_data.columns if col not in country_cols]

            # Melt to long format
            df_long = pd.melt(
                df_data,
                id_vars=country_cols,
                value_vars=year_cols,
                var_name="year",
                value_name="value",
            )

            # Convert year to date (use January 1st)
            df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")
            df_long = df_long.dropna(subset=["year"])
            df_long["date"] = pd.to_datetime(
                df_long["year"].astype(int).astype(str) + "-01-01"
            )

            # Filter by date range
            df_long = df_long[
                (df_long["date"] >= self.start_date)
                & (df_long["date"] <= self.end_date)
            ]

            # Convert value to numeric
            df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
            df_long = df_long.dropna(subset=["value"])

            # Pivot to wide format if multiple indicators
            if df_long["Indicator Name"].nunique() > 1:
                df_wide = df_long.pivot_table(
                    index=["date", "Country Name", "Country Code"],
                    columns="Indicator Name",
                    values="value",
                ).reset_index()
                df_wide.columns.name = None
            else:
                df_wide = df_long[
                    ["date", "Country Name", "Country Code", "value"]
                ].copy()
                indicator_name = df_long["Indicator Name"].iloc[0]
                df_wide[indicator_name] = df_wide["value"]
                df_wide = df_wide.drop("value", axis=1)

            # Sort by date
            df_wide = df_wide.sort_values("date").reset_index(drop=True)

            log_missingness(df_wide, f"World Bank {source_name}")
            return df_wide

        except Exception as e:
            print(f"Error processing World Bank data: {e}")
            return None

    def _process_market_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Process market data from Stooq."""
        try:
            # Validate required columns
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if not validate_schema(df, required_cols):
                return None

            # Convert date column
            df["date"] = pd.to_datetime(df["Date"])

            # Filter by date range
            df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]

            # Calculate returns
            df["close_price"] = pd.to_numeric(df["Close"], errors="coerce")
            df["returns"] = pct_change(df["close_price"])

            # Calculate drawdown
            df["cumulative_returns"] = (1 + df["returns"]).cumprod()
            df["running_max"] = df["cumulative_returns"].expanding().max()
            df["drawdown"] = (df["cumulative_returns"] - df["running_max"]) / df[
                "running_max"
            ]

            # Select relevant columns
            df = df[["date", "close_price", "returns", "drawdown", "Volume"]].copy()

            # Remove rows with NaN returns
            df = df.dropna(subset=["returns"]).reset_index(drop=True)

            log_missingness(df, "Market Data")
            return df

        except Exception as e:
            print(f"Error processing market data: {e}")
            return None

    def harmonize_data(self, processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Harmonize all processed data into a single time series."""
        harmonized_dfs = []

        for source_name, df in processed_data.items():
            if df.empty:
                continue

            # Ensure date column exists
            if "date" not in df.columns:
                print(f"No date column in {source_name}")
                continue

            # Create a base DataFrame with date
            base_df = df[["date"]].copy()

            # Add source-specific columns with prefixes
            for col in df.columns:
                if col != "date":
                    base_df[f"{source_name}_{col}"] = df[col]

            harmonized_dfs.append(base_df)

        if not harmonized_dfs:
            print("No data to harmonize")
            return pd.DataFrame()

        # Merge all DataFrames on date
        harmonized_df = harmonized_dfs[0]
        for df in harmonized_dfs[1:]:
            harmonized_df = harmonized_df.merge(df, on="date", how="outer")

        # Sort by date and fill missing values
        harmonized_df = harmonized_df.sort_values("date").reset_index(drop=True)

        # Forward fill missing values for up to 7 days
        harmonized_df = harmonized_df.fillna(method="ffill", limit=7)

        print(
            f"\nHarmonized data: {len(harmonized_df)} rows, {len(harmonized_df.columns)} columns"
        )
        log_missingness(harmonized_df, "Harmonized Data")

        return harmonized_df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis."""
        df = df.copy()

        # Add time-based features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week

        # Add rolling statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col
            for col in numeric_cols
            if col not in ["year", "month", "day_of_year", "week_of_year"]
        ]

        for col in numeric_cols:
            if df[col].notna().sum() > 30:  # Only if we have enough data
                # 30-day rolling mean and std
                df[f"{col}_rolling_mean_30d"] = robust_rolling_window(
                    df[col], 30
                ).mean()
                df[f"{col}_rolling_std_30d"] = robust_rolling_window(df[col], 30).std()

                # Z-score
                df[f"{col}_zscore"] = zscore(df[col], window=30)

        return df

    def validate_final_data(self, df: pd.DataFrame) -> bool:
        """Validate final harmonized data."""
        if df.empty:
            print("❌ Final dataset is empty")
            return False

        # Check date range
        if df["date"].min() > self.end_date or df["date"].max() < self.start_date:
            print("❌ Data does not cover the required date range")
            return False

        # Check for sufficient data points
        if len(df) < 30:
            print(f"❌ Insufficient data points: {len(df)}")
            return False

        print(
            f"✓ Final validation passed: {len(df)} rows from {df['date'].min()} to {df['date'].max()}"
        )
        return True


def run_etl_pipeline(
    scenario_config: dict[str, Any], raw_data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Run the complete ETL pipeline."""
    print("\n" + "=" * 50)
    print("ETL PIPELINE")
    print("=" * 50)

    pipeline = ETLPipeline(scenario_config)

    # Process raw data
    processed_data = pipeline.process_raw_data(raw_data)

    if not processed_data:
        print("❌ No data processed successfully")
        return pd.DataFrame()

    # Harmonize data
    harmonized_df = pipeline.harmonize_data(processed_data)

    if harmonized_df.empty:
        print("❌ Data harmonization failed")
        return pd.DataFrame()

    # Create derived features
    final_df = pipeline.create_derived_features(harmonized_df)

    # Validate final data
    if not pipeline.validate_final_data(final_df):
        return pd.DataFrame()

    print("✓ ETL pipeline completed successfully")
    return final_df

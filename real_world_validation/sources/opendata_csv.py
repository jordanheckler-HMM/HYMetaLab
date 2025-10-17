"""
OpenData CSV source adapter for loading CSV files.
"""

import os
from typing import Any

import pandas as pd

from .base import SourceAdapter


class OpenDataCSVSource(SourceAdapter):
    """CSV source adapter for open data files."""

    def validate_config(self) -> None:
        """Validate CSV source configuration."""
        required_keys = ["path"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Check if file exists
        if not os.path.exists(self.config["path"]):
            raise FileNotFoundError(f"Data file not found: {self.config['path']}")

    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
            DataFrame with standardized columns
        """
        try:
            # Load CSV with flexible parsing
            df = pd.read_csv(
                self.config["path"],
                parse_dates=True,
                infer_datetime_format=True,
                low_memory=False,
            )

            # Map columns based on config
            column_mapping = {}

            # Date column mapping
            if "date_col" in self.config:
                column_mapping[self.config["date_col"]] = "date"

            # Value column mapping
            if "count_col" in self.config:
                column_mapping[self.config["count_col"]] = "count"
            elif "value_col" in self.config:
                column_mapping[self.config["value_col"]] = "value"

            # Additional column mappings
            if "mag_col" in self.config:
                column_mapping[self.config["mag_col"]] = "magnitude"
            if "type_col" in self.config:
                column_mapping[self.config["type_col"]] = "event_type"
            if "term_col" in self.config:
                column_mapping[self.config["term_col"]] = "term"

            # Apply column mapping
            df = df.rename(columns=column_mapping)

            # Standardize columns
            df = self._standardize_columns(df)

            # Drop rows with missing critical data
            critical_cols = ["date"]
            df = df.dropna(subset=critical_cols)

            return df

        except Exception as e:
            raise RuntimeError(
                f"Failed to load CSV from {self.config['path']}: {str(e)}"
            )

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the loaded data.

        Returns:
            Dictionary with metadata information
        """
        try:
            df = self.load()
            return {
                "source_type": "csv",
                "file_path": self.config["path"],
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": (
                        df["date"].min().isoformat() if "date" in df.columns else None
                    ),
                    "end": (
                        df["date"].max().isoformat() if "date" in df.columns else None
                    ),
                },
                "data_types": df.dtypes.to_dict(),
            }
        except Exception as e:
            return {
                "source_type": "csv",
                "file_path": self.config["path"],
                "error": str(e),
            }

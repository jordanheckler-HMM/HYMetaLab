"""
Google Trends CSV source adapter for fear-related search terms.
"""

from typing import Any

import pandas as pd

from .base import SourceAdapter


class TrendsCSVSource(SourceAdapter):
    """CSV source adapter for Google Trends fear data."""

    def validate_config(self) -> None:
        """Validate Trends CSV source configuration."""
        required_keys = ["path"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def load(self) -> pd.DataFrame:
        """
        Load and merge multiple fear terms from Trends CSV.

        Returns:
            DataFrame with standardized fear index
        """
        try:
            # Load CSV
            df = pd.read_csv(
                self.config["path"],
                parse_dates=True,
                infer_datetime_format=True,
                low_memory=False,
            )

            # Map columns
            column_mapping = {}
            if "date_col" in self.config:
                column_mapping[self.config["date_col"]] = "date"
            if "value_col" in self.config:
                column_mapping[self.config["value_col"]] = "value"
            if "term_col" in self.config:
                column_mapping[self.config["term_col"]] = "term"

            df = df.rename(columns=column_mapping)
            df = self._standardize_columns(df)

            # If multiple terms, merge them
            if "term" in df.columns and df["term"].nunique() > 1:
                df = self._merge_fear_terms(df)

            # Ensure we have a fear_index column
            if "fear_index" not in df.columns and "value" in df.columns:
                df["fear_index"] = df["value"]

            return df[["date", "fear_index"]].dropna()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Trends CSV from {self.config['path']}: {str(e)}"
            )

    def _merge_fear_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple fear terms by z-scoring and averaging.

        Args:
            df: DataFrame with multiple fear terms

        Returns:
            DataFrame with merged fear index
        """
        # Pivot to get terms as columns
        pivot_df = df.pivot_table(
            index="date", columns="term", values="value", aggfunc="mean"
        ).fillna(0)

        # Z-score each term
        z_scored = pivot_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # Average across terms to get fear index
        fear_index = z_scored.mean(axis=1)

        # Convert back to long format
        result_df = pd.DataFrame(
            {"date": fear_index.index, "fear_index": fear_index.values}
        )

        return result_df

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the loaded trends data.

        Returns:
            Dictionary with metadata information
        """
        try:
            df = self.load()
            return {
                "source_type": "trends_csv",
                "file_path": self.config["path"],
                "rows": len(df),
                "date_range": {
                    "start": df["date"].min().isoformat(),
                    "end": df["date"].max().isoformat(),
                },
                "fear_index_stats": {
                    "mean": float(df["fear_index"].mean()),
                    "std": float(df["fear_index"].std()),
                    "min": float(df["fear_index"].min()),
                    "max": float(df["fear_index"].max()),
                },
            }
        except Exception as e:
            return {
                "source_type": "trends_csv",
                "file_path": self.config["path"],
                "error": str(e),
            }

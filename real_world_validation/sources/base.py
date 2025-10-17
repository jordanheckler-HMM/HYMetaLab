"""
Base source adapter abstract class.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class SourceAdapter(ABC):
    """Abstract base class for data source adapters."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize source adapter with configuration.

        Args:
            config: Source configuration dictionary
        """
        self.config = config
        self.validate_config()

    @abstractmethod
    def validate_config(self) -> None:
        """Validate the source configuration."""
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from the source.

        Returns:
            DataFrame with standardized columns
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the loaded data.

        Returns:
            Dictionary with metadata information
        """
        pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and types.

        Args:
            df: Input DataFrame

        Returns:
            Standardized DataFrame
        """
        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Ensure numeric columns are numeric
        numeric_cols = ["count", "value", "magnitude", "gini"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

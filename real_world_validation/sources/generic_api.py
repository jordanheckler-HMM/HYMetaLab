"""
Generic API source adapter for future API integrations.
"""

from typing import Any

import pandas as pd
import requests

from .base import SourceAdapter


class GenericAPISource(SourceAdapter):
    """Generic API source adapter for JSON endpoints."""

    def validate_config(self) -> None:
        """Validate API source configuration."""
        required_keys = ["url"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def load(self) -> pd.DataFrame:
        """
        Load data from API endpoint.

        Returns:
            DataFrame with standardized columns
        """
        try:
            # Make API request
            headers = self.config.get("headers", {})
            params = self.config.get("params", {})

            response = requests.get(
                self.config["url"], headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                # Try to extract data from nested structure
                df = pd.json_normalize(data)

            # Apply column mapping if specified
            if "column_mapping" in self.config:
                df = df.rename(columns=self.config["column_mapping"])

            # Standardize columns
            df = self._standardize_columns(df)

            return df

        except Exception as e:
            raise RuntimeError(
                f"Failed to load data from API {self.config['url']}: {str(e)}"
            )

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the API source.

        Returns:
            Dictionary with metadata information
        """
        return {
            "source_type": "api",
            "url": self.config["url"],
            "headers": self.config.get("headers", {}),
            "params": self.config.get("params", {}),
            "status": "configured",
        }

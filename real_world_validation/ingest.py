"""
Data ingestion orchestrator for loading city data from multiple sources.
"""

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .sources import GenericAPISource, OpenDataCSVSource, TrendsCSVSource


class DataIngester:
    """Orchestrates data loading from multiple sources for a city."""

    def __init__(self, registry_path: str = "real_world_validation/registry.yaml"):
        """
        Initialize data ingester with registry configuration.

        Args:
            registry_path: Path to registry YAML file
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.source_factories = {
            "csv": OpenDataCSVSource,
            "trends_csv": TrendsCSVSource,
            "api": GenericAPISource,
        }

    def _load_registry(self) -> dict[str, Any]:
        """Load registry configuration from YAML file."""
        try:
            with open(self.registry_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load registry from {self.registry_path}: {str(e)}"
            )

    def get_city_config(self, city_key: str) -> dict[str, Any] | None:
        """
        Get configuration for a specific city.

        Args:
            city_key: City identifier (e.g., 'nyc', 'chi')

        Returns:
            City configuration dictionary or None if not found
        """
        for group_name, group_config in self.registry["groups"].items():
            for city in group_config["cities"]:
                if city["key"] == city_key:
                    return {
                        "city": city,
                        "group": group_name,
                        "group_config": group_config,
                    }
        return None

    def get_group_cities(self, group_name: str) -> list[str]:
        """
        Get list of city keys in a group.

        Args:
            group_name: Group name (e.g., 'us_core', 'world_sample')

        Returns:
            List of city keys
        """
        if group_name not in self.registry["groups"]:
            return []

        return [city["key"] for city in self.registry["groups"][group_name]["cities"]]

    def load_city_data(self, city_key: str) -> dict[str, pd.DataFrame]:
        """
        Load all data sources for a city.

        Args:
            city_key: City identifier

        Returns:
            Dictionary mapping source names to DataFrames
        """
        city_config = self.get_city_config(city_key)
        if not city_config:
            raise ValueError(f"City '{city_key}' not found in registry")

        city_data = {}
        city_info = city_config["city"]

        print(f"Loading data for {city_info['name']} ({city_key})...")

        for source_name, source_config in city_info.get("fetch", {}).items():
            try:
                # For fetch types, load from CSV files
                if source_config["type"] in [
                    "socrata",
                    "census_acs",
                    "pytrends",
                    "fema",
                    "gini_worldbank",
                ]:
                    # Map source names to CSV file names
                    csv_filename = f"{source_name}.csv"
                    if source_name == "trends":
                        csv_filename = "trends_fear.csv"
                    elif source_name == "gini_worldbank":
                        csv_filename = "gini.csv"

                    csv_path = f"data/{city_key}/{csv_filename}"

                    if Path(csv_path).exists():
                        # Load CSV directly
                        df = pd.read_csv(csv_path)
                        city_data[source_name] = df
                        print(
                            f"  ✓ Loaded {source_name}: {len(df)} rows from {csv_filename}"
                        )
                    else:
                        print(f"  ❌ Missing data file: {csv_filename}")
                        continue
                else:
                    # For other types, use source adapter
                    source_type = source_config["type"]
                    if source_type not in self.source_factories:
                        warnings.warn(
                            f"Unknown source type '{source_type}' for {source_name}"
                        )
                        continue

                    source_adapter = self.source_factories[source_type](source_config)
                    df = source_adapter.load()
                    city_data[source_name] = df
                    print(f"  ✓ Loaded {source_name}: {len(df)} rows")

            except Exception as e:
                warnings.warn(f"Failed to load {source_name} for {city_key}: {str(e)}")
                continue

        return city_data

    def load_group_data(self, group_name: str) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load data for all cities in a group.

        Args:
            group_name: Group name (e.g., 'us_core', 'world_sample')

        Returns:
            Dictionary mapping city keys to their data sources
        """
        city_keys = self.get_group_cities(group_name)
        if not city_keys:
            raise ValueError(f"Group '{group_name}' not found in registry")

        group_data = {}

        print(f"Loading data for group '{group_name}' ({len(city_keys)} cities)...")

        for city_key in city_keys:
            try:
                city_data = self.load_city_data(city_key)
                group_data[city_key] = city_data
                print(f"✓ Completed {city_key}")
            except Exception as e:
                warnings.warn(f"Failed to load data for {city_key}: {str(e)}")
                continue

        return group_data

    def get_success_criteria(self) -> dict[str, Any]:
        """Get success criteria thresholds from registry."""
        return self.registry.get("success_criteria", {})

    def validate_data_availability(self, city_key: str) -> dict[str, bool]:
        """
        Check which data sources are available for a city.

        Args:
            city_key: City identifier

        Returns:
            Dictionary mapping source names to availability status
        """
        city_config = self.get_city_config(city_key)
        if not city_config:
            return {}

        availability = {}
        city_info = city_config["city"]

        for source_name, source_config in city_info.get("fetch", {}).items():
            try:
                source_type = source_config["type"]
                if source_type == "csv":
                    # Check if file exists
                    file_path = source_config["path"]
                    availability[source_name] = Path(file_path).exists()
                elif source_type in [
                    "socrata",
                    "census_acs",
                    "pytrends",
                    "fema",
                    "gini_worldbank",
                ]:
                    # For fetch types, check if corresponding CSV exists
                    csv_path = f"data/{city_key}/{source_name}.csv"
                    if source_name == "trends":
                        csv_path = f"data/{city_key}/trends_fear.csv"
                    elif source_name == "gini_worldbank":
                        csv_path = f"data/{city_key}/gini.csv"
                    availability[source_name] = Path(csv_path).exists()
                else:
                    # For other types, assume available if config is valid
                    availability[source_name] = True
            except Exception:
                availability[source_name] = False

        return availability

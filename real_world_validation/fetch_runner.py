"""
Fetch runner orchestrator for collecting data from multiple sources.
"""

import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from real_world_validation.fetch.census_acs import CensusACSClient
from real_world_validation.fetch.fema_disasters import FEMAClient
from real_world_validation.fetch.socrata_client import SocrataClient
from real_world_validation.fetch.trends_pytrends import TrendsClient
from real_world_validation.fetch.worldbank_gini import WorldBankClient


class FetchRunner:
    """Orchestrates data fetching from multiple sources."""

    def __init__(self, registry_path: str = "real_world_validation/registry.yaml"):
        """
        Initialize fetch runner.

        Args:
            registry_path: Path to registry YAML file
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()

        # Initialize clients
        self.socrata_client = SocrataClient(os.getenv("SOCRATA_APP_TOKEN"))
        self.census_client = CensusACSClient(os.getenv("CENSUS_API_KEY"))
        self.trends_client = TrendsClient()
        self.fema_client = FEMAClient(os.getenv("FEMA_API_KEY"))
        self.worldbank_client = WorldBankClient()

    def _load_registry(self) -> dict[str, Any]:
        """Load registry configuration."""
        try:
            with open(self.registry_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to load registry from {self.registry_path}: {str(e)}"
            )

    def fetch_city(self, city_key: str) -> dict[str, pd.DataFrame]:
        """
        Fetch all data for a single city.

        Args:
            city_key: City identifier

        Returns:
            Dictionary of DataFrames by data type
        """
        # Find city configuration
        city_config = None
        for group_name, group_config in self.registry["groups"].items():
            for city in group_config.get("cities", []):
                if city["key"] == city_key:
                    city_config = city
                    group_name_found = group_name
                    break
            if city_config:
                break

        if not city_config:
            raise ValueError(f"City '{city_key}' not found in registry")

        # Get group-level settings
        group_config = self.registry["groups"][group_name_found]
        start_date = group_config.get("start", "2019-01-01")
        end_date = group_config.get("end", "2025-06-30")

        # Create output directory
        output_dir = Path(f"data/{city_key}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch data from each source
        results = {}

        fetch_config = city_config.get("fetch", {})

        # Crime data (Socrata)
        if "crime" in fetch_config:
            crime_config = fetch_config["crime"]
            if crime_config.get("type") == "socrata":
                try:
                    crime_df = self._fetch_crime_data(
                        crime_config, start_date, end_date
                    )
                    results["crime"] = crime_df
                    self._save_csv(crime_df, output_dir / "crime.csv")
                except Exception as e:
                    warnings.warn(
                        f"Failed to fetch crime data for {city_key}: {str(e)}"
                    )
                    results["crime"] = pd.DataFrame(columns=["date", "count"])

        # Gini data (Census ACS)
        if "gini" in fetch_config:
            gini_config = fetch_config["gini"]
            if gini_config.get("type") == "census_acs":
                try:
                    gini_df = self._fetch_census_gini(gini_config)
                    results["gini"] = gini_df
                    self._save_csv(gini_df, output_dir / "gini.csv")
                except Exception as e:
                    warnings.warn(f"Failed to fetch Gini data for {city_key}: {str(e)}")
                    results["gini"] = pd.DataFrame(columns=["date", "gini"])

        # Trends data (Google Trends)
        if "trends" in fetch_config:
            trends_config = fetch_config["trends"]
            if trends_config.get("type") == "pytrends":
                try:
                    trends_df = self._fetch_trends_data(
                        trends_config, start_date, end_date
                    )
                    results["trends"] = trends_df
                    self._save_csv(trends_df, output_dir / "trends_fear.csv")
                except Exception as e:
                    warnings.warn(
                        f"Failed to fetch trends data for {city_key}: {str(e)}"
                    )
                    results["trends"] = pd.DataFrame(columns=["date", "term", "value"])

        # Events data (FEMA)
        if "events" in fetch_config:
            events_config = fetch_config["events"]
            if events_config.get("type") == "fema":
                try:
                    events_df = self._fetch_fema_events(city_key, start_date, end_date)
                    results["events"] = events_df
                    self._save_csv(events_df, output_dir / "events.csv")
                except Exception as e:
                    warnings.warn(
                        f"Failed to fetch events data for {city_key}: {str(e)}"
                    )
                    results["events"] = pd.DataFrame(
                        columns=["date", "event_type", "magnitude"]
                    )

        # World Bank Gini (for international cities)
        if "gini_worldbank" in fetch_config:
            wb_config = fetch_config["gini_worldbank"]
            try:
                wb_gini_df = self._fetch_worldbank_gini(wb_config)
                results["gini"] = wb_gini_df
                self._save_csv(wb_gini_df, output_dir / "gini.csv")
            except Exception as e:
                warnings.warn(
                    f"Failed to fetch World Bank Gini data for {city_key}: {str(e)}"
                )
                results["gini"] = pd.DataFrame(columns=["date", "gini"])

        return results

    def _fetch_crime_data(
        self, config: dict[str, Any], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch crime data from Socrata."""
        domain = config["domain"]
        dataset_id = config.get("dataset_id")
        date_col = config.get("date_col")

        return self.socrata_client.fetch_city_crime_data(
            domain=domain,
            dataset_id=dataset_id,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
        )

    def _fetch_census_gini(self, config: dict[str, Any]) -> pd.DataFrame:
        """Fetch Gini data from Census ACS."""
        geo = config["geo"]
        state_fips = config["state_fips"]
        county_fips = config["county_fips"]
        years = config["years"]

        if geo == "county":
            return self.census_client.fetch_gini_county(state_fips, county_fips, years)
        elif geo == "state":
            return self.census_client.fetch_state_gini(state_fips, years)
        else:
            raise ValueError(f"Unknown geo type: {geo}")

    def _fetch_trends_data(
        self, config: dict[str, Any], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch trends data from Google Trends."""
        geo = config["geo"]
        terms = config["terms"]

        return self.trends_client.fetch_city_trends(
            geo=geo, terms=terms, start_date=start_date, end_date=end_date
        )

    def _fetch_fema_events(
        self, city_key: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch FEMA events data."""
        # Map city keys to state codes
        city_state_map = {
            "nyc": "NY",
            "la": "CA",
            "chi": "IL",
            "hou": "TX",
            "phx": "AZ",
            "sea": "WA",
            "mia": "FL",
            "atl": "GA",
            "den": "CO",
            "msp": "MN",
        }

        state = city_state_map.get(city_key)
        if state:
            return self.fema_client.fetch_state_events(state, start_date, end_date)
        else:
            return pd.DataFrame(columns=["date", "event_type", "magnitude"])

    def _fetch_worldbank_gini(self, config: dict[str, Any]) -> pd.DataFrame:
        """Fetch Gini data from World Bank."""
        country_code = config["country_code"]
        return self.worldbank_client.fetch_country_gini(country_code)

    def _save_csv(self, df: pd.DataFrame, filepath: Path):
        """Save DataFrame to CSV."""
        if df.empty:
            # Create empty CSV with headers
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)

    def fetch_group(self, group_key: str) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch data for all cities in a group.

        Args:
            group_key: Group identifier

        Returns:
            Dictionary of city results
        """
        if group_key not in self.registry["groups"]:
            raise ValueError(f"Group '{group_key}' not found in registry")

        group_config = self.registry["groups"][group_key]
        cities = group_config.get("cities", [])

        results = {}

        for city in cities:
            city_key = city["key"]
            city_name = city["name"]

            print(f"Fetching data for {city_name} ({city_key})...")

            try:
                city_results = self.fetch_city(city_key)
                results[city_key] = city_results
                print(f"  ✓ Successfully fetched data for {city_key}")

            except Exception as e:
                print(f"  ✗ Failed to fetch data for {city_key}: {str(e)}")
                results[city_key] = {}

        return results

    def fetch_all(self) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
        """
        Fetch data for all groups.

        Returns:
            Dictionary of group results
        """
        all_results = {}

        for group_key in self.registry["groups"].keys():
            print(f"Fetching data for group: {group_key}")
            group_results = self.fetch_group(group_key)
            all_results[group_key] = group_results

        return all_results

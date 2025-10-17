"""
Census ACS client for fetching Gini coefficient data.
"""

import warnings
from datetime import datetime

import pandas as pd
import requests


class CensusACSClient:
    """Client for fetching ACS data from the US Census Bureau."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Census ACS client.

        Args:
            api_key: Census API key (optional but recommended for higher rate limits)
        """
        self.api_key = api_key
        self.base_url = "https://api.census.gov/data"
        self.session = requests.Session()

    def fetch_gini_county(
        self, state_fips: str, county_fips: str, years: list[int]
    ) -> pd.DataFrame:
        """
        Fetch Gini coefficient data for a county.

        Args:
            state_fips: State FIPS code (2 digits)
            county_fips: County FIPS code (3 digits)
            years: List of years to fetch

        Returns:
            DataFrame with columns [date, gini]
        """
        results = []

        for year in years:
            try:
                # ACS 5-year estimates are available for years ending in 0-9
                # We'll use the most recent 5-year estimate for each year
                acs_year = self._get_acs_year(year)

                # Build API URL
                url = f"{self.base_url}/{acs_year}/acs/acs5"

                params = {
                    "get": "B19083_001E",  # Gini coefficient
                    "for": f"county:{county_fips}",
                    "in": f"state:{state_fips}",
                    "key": self.api_key,
                }

                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}

                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if len(data) > 1:  # Has data beyond headers
                    gini_value = data[1][0]  # First data row, first column

                    if gini_value and gini_value != "-666666666":  # Valid value
                        # Convert to float and normalize to 0-1 range
                        gini_float = float(gini_value) / 100.0

                        # Create date (January 1st of the year)
                        date = datetime(year, 1, 1)

                        results.append({"date": date, "gini": gini_float})

            except Exception as e:
                warnings.warn(
                    f"Failed to fetch Gini data for {state_fips}-{county_fips} in {year}: {str(e)}"
                )
                continue

        if not results:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=["date", "gini"])

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by date
        df = df.sort_values("date")

        return df

    def _get_acs_year(self, target_year: int) -> int:
        """
        Get the appropriate ACS year for a target year.

        ACS 5-year estimates are released for years ending in 0-9.
        We use the most recent 5-year estimate available.
        """
        # ACS 5-year estimates are available for years ending in 0-9
        # For years 2019-2025, we'll use 2019 ACS (most recent available)
        if target_year >= 2019:
            return 2019
        elif target_year >= 2014:
            return 2014
        elif target_year >= 2009:
            return 2009
        else:
            return 2009  # Earliest available

    def fetch_state_gini(self, state_fips: str, years: list[int]) -> pd.DataFrame:
        """
        Fetch Gini coefficient data for a state.

        Args:
            state_fips: State FIPS code (2 digits)
            years: List of years to fetch

        Returns:
            DataFrame with columns [date, gini]
        """
        results = []

        for year in years:
            try:
                acs_year = self._get_acs_year(year)

                # Build API URL
                url = f"{self.base_url}/{acs_year}/acs/acs5"

                params = {
                    "get": "B19083_001E",  # Gini coefficient
                    "for": f"state:{state_fips}",
                    "key": self.api_key,
                }

                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}

                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if len(data) > 1:  # Has data beyond headers
                    gini_value = data[1][0]  # First data row, first column

                    if gini_value and gini_value != "-666666666":  # Valid value
                        # Convert to float and normalize to 0-1 range
                        gini_float = float(gini_value) / 100.0

                        # Create date (January 1st of the year)
                        date = datetime(year, 1, 1)

                        results.append({"date": date, "gini": gini_float})

            except Exception as e:
                warnings.warn(
                    f"Failed to fetch state Gini data for {state_fips} in {year}: {str(e)}"
                )
                continue

        if not results:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=["date", "gini"])

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by date
        df = df.sort_values("date")

        return df

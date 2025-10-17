"""
World Bank client for fetching international Gini coefficient data.
"""

import warnings
from datetime import datetime

import pandas as pd
import requests


class WorldBankClient:
    """Client for fetching World Bank data."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize World Bank client.

        Args:
            api_key: World Bank API key (optional)
        """
        self.api_key = api_key
        self.base_url = "https://api.worldbank.org/v2"
        self.session = requests.Session()

    def fetch_country_gini(
        self, country_code: str, start_year: int = 2019, end_year: int = 2025
    ) -> pd.DataFrame:
        """
        Fetch Gini coefficient data for a country.

        Args:
            country_code: ISO country code (e.g., 'USA', 'GBR')
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with columns [date, gini]
        """
        results = []

        try:
            # World Bank Gini coefficient indicator
            indicator = "SI.POV.GINI"

            # Build API URL
            url = f"{self.base_url}/country/{country_code}/indicator/{indicator}"

            params = {
                "date": f"{start_year}:{end_year}",
                "format": "json",
                "per_page": 1000,
            }

            if self.api_key:
                params["key"] = self.api_key

            # Make request
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Process results
            if len(data) > 1 and data[1]:  # Has data
                for item in data[1]:
                    try:
                        year = item.get("date")
                        gini_value = item.get("value")

                        if year and gini_value is not None:
                            # Convert to float and normalize to 0-1 range
                            gini_float = float(gini_value) / 100.0

                            # Create date (January 1st of the year)
                            date = datetime(int(year), 1, 1)

                            results.append({"date": date, "gini": gini_float})

                    except Exception as e:
                        warnings.warn(
                            f"Failed to process World Bank data point: {str(e)}"
                        )
                        continue

        except Exception as e:
            warnings.warn(
                f"Failed to fetch World Bank Gini data for {country_code}: {str(e)}"
            )
            return pd.DataFrame(columns=["date", "gini"])

        if not results:
            return pd.DataFrame(columns=["date", "gini"])

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by date
        df = df.sort_values("date")

        return df

    def fetch_multiple_countries_gini(
        self, country_codes: list[str], start_year: int = 2019, end_year: int = 2025
    ) -> pd.DataFrame:
        """
        Fetch Gini coefficient data for multiple countries.

        Args:
            country_codes: List of ISO country codes
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with columns [date, gini, country_code]
        """
        all_results = []

        for country_code in country_codes:
            try:
                country_data = self.fetch_country_gini(
                    country_code, start_year, end_year
                )
                if not country_data.empty:
                    country_data["country_code"] = country_code
                    all_results.append(country_data)

            except Exception as e:
                warnings.warn(f"Failed to fetch Gini data for {country_code}: {str(e)}")
                continue

        if not all_results:
            return pd.DataFrame(columns=["date", "gini", "country_code"])

        return pd.concat(all_results, ignore_index=True)

    def get_country_code_mapping(self) -> dict[str, str]:
        """
        Get mapping of common country names to ISO codes.

        Returns:
            Dictionary mapping country names to ISO codes
        """
        return {
            "united states": "USA",
            "united kingdom": "GBR",
            "france": "FRA",
            "japan": "JPN",
            "india": "IND",
            "nigeria": "NGA",
            "brazil": "BRA",
            "australia": "AUS",
            "canada": "CAN",
            "germany": "DEU",
            "china": "CHN",
            "russia": "RUS",
            "mexico": "MEX",
            "south korea": "KOR",
            "italy": "ITA",
            "spain": "ESP",
            "netherlands": "NLD",
            "sweden": "SWE",
            "norway": "NOR",
            "denmark": "DNK",
        }

    def fetch_country_by_name(
        self, country_name: str, start_year: int = 2019, end_year: int = 2025
    ) -> pd.DataFrame:
        """
        Fetch Gini data for a country by name.

        Args:
            country_name: Country name
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with Gini data
        """
        mapping = self.get_country_code_mapping()
        country_code = mapping.get(country_name.lower())

        if not country_code:
            warnings.warn(f"Unknown country: {country_name}")
            return pd.DataFrame(columns=["date", "gini"])

        return self.fetch_country_gini(country_code, start_year, end_year)

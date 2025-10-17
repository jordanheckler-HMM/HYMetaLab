"""
FEMA client for fetching disaster events data.
"""

import warnings
from datetime import datetime
from typing import Any

import pandas as pd
import requests


class FEMAClient:
    """Client for fetching FEMA disaster data."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize FEMA client.

        Args:
            api_key: FEMA API key (optional)
        """
        self.api_key = api_key
        self.base_url = "https://www.fema.gov/api/open/v1"
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def fetch_events(
        self, start_date: str, end_date: str, states: list[str] = None
    ) -> pd.DataFrame:
        """
        Fetch disaster events from FEMA.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            states: List of state codes to filter by

        Returns:
            DataFrame with columns [date, event_type, magnitude]
        """
        results = []

        try:
            # Build API URL
            url = f"{self.base_url}/DisasterDeclarationsSummaries"

            # Build parameters
            params = {
                "$filter": f"declarationDate ge '{start_date}' and declarationDate le '{end_date}'",
                "$top": 1000,  # Limit results
                "$format": "json",
            }

            # Add state filter if provided
            if states:
                state_filter = " or ".join([f"state eq '{state}'" for state in states])
                params["$filter"] += f" and ({state_filter})"

            # Make request
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Process results
            for item in data.get("DisasterDeclarationsSummaries", []):
                try:
                    # Extract event information
                    declaration_date = item.get("declarationDate")
                    incident_type = item.get("incidentType", "Unknown")

                    if not declaration_date:
                        continue

                    # Convert date
                    date = datetime.strptime(declaration_date, "%Y-%m-%d")

                    # Calculate magnitude based on available data
                    magnitude = self._calculate_magnitude(item)

                    results.append(
                        {
                            "date": date,
                            "event_type": incident_type,
                            "magnitude": magnitude,
                        }
                    )

                except Exception as e:
                    warnings.warn(f"Failed to process FEMA event: {str(e)}")
                    continue

        except Exception as e:
            warnings.warn(f"Failed to fetch FEMA events: {str(e)}")
            return pd.DataFrame(columns=["date", "event_type", "magnitude"])

        if not results:
            return pd.DataFrame(columns=["date", "event_type", "magnitude"])

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by date
        df = df.sort_values("date")

        return df

    def _calculate_magnitude(self, event_data: dict[str, Any]) -> float:
        """
        Calculate event magnitude from FEMA event data.

        Args:
            event_data: Raw FEMA event data

        Returns:
            Magnitude value (0-1)
        """
        # Base magnitude by incident type
        incident_type = event_data.get("incidentType", "").lower()

        type_magnitudes = {
            "hurricane": 0.9,
            "tornado": 0.8,
            "flood": 0.7,
            "wildfire": 0.8,
            "earthquake": 0.9,
            "severe storm": 0.6,
            "snow": 0.5,
            "ice": 0.5,
            "fire": 0.6,
            "drought": 0.4,
            "freeze": 0.4,
            "other": 0.5,
        }

        base_magnitude = type_magnitudes.get(incident_type, 0.5)

        # Adjust based on available severity indicators
        if "damageLevelId" in event_data:
            damage_level = event_data["damageLevelId"]
            if damage_level == "1":  # Major
                base_magnitude += 0.2
            elif damage_level == "2":  # Minor
                base_magnitude -= 0.1

        # Ensure magnitude is in [0, 1] range
        return max(0.0, min(1.0, base_magnitude))

    def fetch_state_events(
        self, state: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch disaster events for a specific state.

        Args:
            state: State code
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with disaster events
        """
        return self.fetch_events(start_date, end_date, states=[state])

    def fetch_city_events(
        self, city_state: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch disaster events for a city (approximated by state).

        Args:
            city_state: City, State format
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with disaster events
        """
        # Extract state from city_state
        if "," in city_state:
            state = city_state.split(",")[-1].strip()
        else:
            # Try to map common city names to states
            state = self._get_state_from_city(city_state)

        if state:
            return self.fetch_state_events(state, start_date, end_date)
        else:
            warnings.warn(f"Could not determine state for city: {city_state}")
            return pd.DataFrame(columns=["date", "event_type", "magnitude"])

    def _get_state_from_city(self, city: str) -> str | None:
        """Map city names to state codes."""
        city_state_map = {
            "new york": "NY",
            "los angeles": "CA",
            "chicago": "IL",
            "houston": "TX",
            "phoenix": "AZ",
            "seattle": "WA",
            "miami": "FL",
            "atlanta": "GA",
            "denver": "CO",
            "minneapolis": "MN",
        }

        city_lower = city.lower().strip()
        return city_state_map.get(city_lower)

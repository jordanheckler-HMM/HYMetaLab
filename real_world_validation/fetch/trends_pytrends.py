"""
Google Trends client for fetching fear-related search data.
"""

import time
import warnings
from datetime import datetime

import pandas as pd
from pytrends.request import TrendReq


class TrendsClient:
    """Client for fetching Google Trends data."""

    def __init__(self, hl: str = "en-US", tz: int = 360):
        """
        Initialize Trends client.

        Args:
            hl: Host language
            tz: Timezone offset
        """
        self.hl = hl
        self.tz = tz
        self.pytrends = None

    def _get_pytrends(self):
        """Get or create pytrends instance."""
        if self.pytrends is None:
            self.pytrends = TrendReq(hl=self.hl, tz=self.tz)
        return self.pytrends

    def fetch_weekly_terms(
        self, geo: str, terms: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch weekly trends data for multiple terms.

        Args:
            geo: Geographic location (e.g., 'US-NY-501' for NYC)
            terms: List of search terms
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns [date, term, value]
        """
        results = []

        try:
            pytrends = self._get_pytrends()

            # Convert dates to datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # Build payload
            pytrends.build_payload(
                terms,
                cat=0,  # All categories
                timeframe=f"{start_date} {end_date}",
                geo=geo,
                gprop="",
            )

            # Get interest over time
            interest_df = pytrends.interest_over_time()

            if interest_df.empty:
                warnings.warn(f"No trends data found for {geo} with terms {terms}")
                return pd.DataFrame(columns=["date", "term", "value"])

            # Convert to long format
            for term in terms:
                if term in interest_df.columns:
                    term_data = interest_df[term].reset_index()
                    term_data.columns = ["date", "value"]
                    term_data["term"] = term
                    results.append(term_data)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            warnings.warn(f"Failed to fetch trends data for {geo}: {str(e)}")
            return pd.DataFrame(columns=["date", "term", "value"])

        if not results:
            return pd.DataFrame(columns=["date", "term", "value"])

        # Combine all results
        df = pd.concat(results, ignore_index=True)

        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Sort by date and term
        df = df.sort_values(["date", "term"])

        return df

    def fetch_city_trends(
        self, geo: str, terms: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch trends data for a city with error handling.

        Args:
            geo: Geographic location
            terms: Search terms
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with trends data
        """
        try:
            return self.fetch_weekly_terms(geo, terms, start_date, end_date)
        except Exception as e:
            warnings.warn(f"Failed to fetch city trends for {geo}: {str(e)}")
            return pd.DataFrame(columns=["date", "term", "value"])

    def fetch_multiple_geos(
        self, geo_term_pairs: list[tuple], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch trends data for multiple geographic locations.

        Args:
            geo_term_pairs: List of (geo, terms) tuples
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with trends data
        """
        all_results = []

        for geo, terms in geo_term_pairs:
            try:
                geo_data = self.fetch_city_trends(geo, terms, start_date, end_date)
                if not geo_data.empty:
                    geo_data["geo"] = geo
                    all_results.append(geo_data)

                # Rate limiting between requests
                time.sleep(2)

            except Exception as e:
                warnings.warn(f"Failed to fetch trends for {geo}: {str(e)}")
                continue

        if not all_results:
            return pd.DataFrame(columns=["date", "term", "value", "geo"])

        return pd.concat(all_results, ignore_index=True)

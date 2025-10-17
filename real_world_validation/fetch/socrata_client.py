"""
Socrata client for fetching city crime data.
"""

import warnings

import pandas as pd
import requests


class SocrataClient:
    """Client for fetching data from Socrata open data portals."""

    def __init__(self, app_token: str | None = None):
        """
        Initialize Socrata client.

        Args:
            app_token: Socrata application token (optional but recommended)
        """
        self.app_token = app_token
        self.session = requests.Session()
        if app_token:
            self.session.headers.update({"X-App-Token": app_token})

    def auto_discover_dataset(
        self, domain: str, keywords: list[str] = None
    ) -> str | None:
        """
        Auto-discover the best crime dataset for a domain.

        Args:
            domain: Socrata domain (e.g., 'data.cityofnewyork.us')
            keywords: Keywords to search for

        Returns:
            Dataset ID or None if not found
        """
        if keywords is None:
            keywords = ["crime", "incident", "offense", "arrest"]

        try:
            # Search the catalog
            search_url = f"https://{domain}/api/catalog/v1"
            params = {"q": " ".join(keywords), "limit": 50}

            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()

            results = response.json().get("results", [])

            # Find the best dataset
            best_dataset = None
            best_score = 0

            for result in results:
                resource = result.get("resource", {})
                dataset_id = resource.get("id")

                if not dataset_id:
                    continue

                # Check if it has date columns
                columns = resource.get("columns_field_name", [])
                has_date = any("date" in col.lower() for col in columns)

                if not has_date:
                    continue

                # Score based on size and relevance
                row_count = resource.get("row_count", 0)
                name = resource.get("name", "").lower()

                score = row_count
                for keyword in keywords:
                    if keyword in name:
                        score *= 2

                if score > best_score:
                    best_score = score
                    best_dataset = dataset_id

            return best_dataset

        except Exception as e:
            warnings.warn(f"Failed to auto-discover dataset for {domain}: {str(e)}")
            return None

    def fetch_daily_counts(
        self,
        domain: str,
        dataset_id: str,
        date_col: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch daily crime counts from a Socrata dataset.

        Args:
            domain: Socrata domain
            dataset_id: Dataset identifier
            date_col: Name of the date column
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns [date, count]
        """
        try:
            # Build query URL
            base_url = f"https://{domain}/resource/{dataset_id}.json"

            # Build date filter
            where_clause = f"{date_col} is not null"
            if start_date:
                where_clause += f" and {date_col} >= '{start_date}'"
            if end_date:
                where_clause += f" and {date_col} <= '{end_date}'"

            params = {
                "$where": where_clause,
                "$select": f"{date_col}, count(*) as count",
                "$group": date_col,
                "$order": date_col,
                "$limit": 50000,
            }

            # Make request
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                # Return empty DataFrame with correct structure
                return pd.DataFrame(columns=["date", "count"])

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Rename columns
            df = df.rename(columns={date_col: "date"})

            # Convert date column
            df["date"] = pd.to_datetime(df["date"])

            # Convert count to integer
            df["count"] = pd.to_numeric(df["count"], errors="coerce")

            # Remove rows with invalid dates or counts
            df = df.dropna(subset=["date", "count"])

            # Sort by date
            df = df.sort_values("date")

            return df

        except Exception as e:
            warnings.warn(
                f"Failed to fetch crime data from {domain}/{dataset_id}: {str(e)}"
            )
            return pd.DataFrame(columns=["date", "count"])

    def fetch_city_crime_data(
        self,
        domain: str,
        dataset_id: str = None,
        date_col: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch crime data for a city, with auto-discovery if needed.

        Args:
            domain: Socrata domain
            dataset_id: Dataset ID (optional, will auto-discover if None)
            date_col: Date column name (optional)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with crime data
        """
        # Auto-discover dataset if not provided
        if not dataset_id:
            dataset_id = self.auto_discover_dataset(domain)
            if not dataset_id:
                warnings.warn(f"No crime dataset found for {domain}")
                return pd.DataFrame(columns=["date", "count"])

        # Auto-discover date column if not provided
        if not date_col:
            date_col = self._discover_date_column(domain, dataset_id)
            if not date_col:
                warnings.warn(f"No date column found for {domain}/{dataset_id}")
                return pd.DataFrame(columns=["date", "count"])

        # Fetch the data
        return self.fetch_daily_counts(
            domain, dataset_id, date_col, start_date, end_date
        )

    def _discover_date_column(self, domain: str, dataset_id: str) -> str | None:
        """Discover the best date column in a dataset."""
        try:
            # Get dataset metadata
            metadata_url = f"https://{domain}/api/views/{dataset_id}/columns.json"
            response = self.session.get(metadata_url, timeout=30)
            response.raise_for_status()

            columns = response.json()

            # Look for date columns
            date_columns = []
            for col in columns:
                col_name = col.get("name", "")
                col_type = col.get("dataTypeName", "").lower()

                if "date" in col_type or "date" in col_name.lower():
                    date_columns.append(col_name)

            # Return the first date column found
            return date_columns[0] if date_columns else None

        except Exception as e:
            warnings.warn(
                f"Failed to discover date column for {domain}/{dataset_id}: {str(e)}"
            )
            return None

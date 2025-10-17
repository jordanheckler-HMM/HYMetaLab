# real_world_validation/fetchers.py
"""
Data fetchers for real-world validation.
Fetches data from external sources with local caching.
"""

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .utils import hash_file, now_iso, safe_mkdirs


class DataFetcher:
    """Base class for data fetching with caching."""

    def __init__(self, cache_dir: str = ".cache/real_world"):
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw"
        safe_mkdirs(self.raw_dir)

    def _get_cache_path(self, source_name: str, filename: str) -> Path:
        """Get cache file path."""
        return self.raw_dir / source_name / filename

    def _fetch_with_cache(
        self, url: str, source_name: str, filename: str, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch data with caching."""
        cache_path = self._get_cache_path(source_name, filename)
        safe_mkdirs(cache_path.parent)

        metadata = {
            "source_url": url,
            "fetched_at": now_iso(),
            "cache_path": str(cache_path),
        }

        # Check if cached file exists and is recent (within 24 hours)
        if not force_refresh and cache_path.exists():
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age < 86400:  # 24 hours
                print(f"Using cached data: {cache_path}")
                metadata["cached"] = True
                metadata["file_hash"] = hash_file(str(cache_path))
                return pd.read_csv(cache_path), metadata

        # Fetch fresh data
        print(f"Fetching data from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Handle different content types
            if url.endswith(".csv") or "csv" in response.headers.get(
                "content-type", ""
            ):
                df = pd.read_csv(io.StringIO(response.text))
            elif url.endswith(".zip") or "zip" in response.headers.get(
                "content-type", ""
            ):
                # Handle ZIP files (World Bank format)
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                    else:
                        raise ValueError("No CSV files found in ZIP")
            else:
                # Try to parse as CSV anyway
                df = pd.read_csv(io.StringIO(response.text))

            # Save to cache
            df.to_csv(cache_path, index=False)
            metadata["cached"] = False
            metadata["file_hash"] = hash_file(str(cache_path))
            metadata["etag"] = response.headers.get("etag")
            metadata["last_modified"] = response.headers.get("last-modified")

            print(f"Data cached to: {cache_path}")
            return df, metadata

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            raise


class OWIDFetcher(DataFetcher):
    """Fetcher for Our World in Data COVID-19 dataset."""

    def fetch_covid_data(
        self, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch OWID COVID-19 data."""
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        return self._fetch_with_cache(url, "owid", "covid_data.csv", force_refresh)


class WorldBankFetcher(DataFetcher):
    """Fetcher for World Bank data."""

    def fetch_gini(
        self, iso3: str, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch Gini coefficient data for a country."""
        url = f"http://api.worldbank.org/v2/country/{iso3}/indicator/SI.POV.GINI?downloadformat=csv"
        filename = f"gini_{iso3}.csv"
        return self._fetch_with_cache(url, "worldbank", filename, force_refresh)

    def fetch_gdp_growth(
        self, iso3: str, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch GDP growth data for a country."""
        url = f"http://api.worldbank.org/v2/country/{iso3}/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=csv"
        filename = f"gdp_growth_{iso3}.csv"
        return self._fetch_with_cache(url, "worldbank", filename, force_refresh)

    def fetch_multiple_countries(
        self, countries: list, indicators: list, force_refresh: bool = False
    ) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
        """Fetch data for multiple countries and indicators."""
        results = {}

        for country in countries:
            for indicator in indicators:
                if indicator == "gini":
                    df, metadata = self.fetch_gini(country, force_refresh)
                elif indicator == "gdp_growth":
                    df, metadata = self.fetch_gdp_growth(country, force_refresh)
                else:
                    continue

                key = f"{country}_{indicator}"
                results[key] = (df, metadata)

        return results


class StooqFetcher(DataFetcher):
    """Fetcher for Stooq market data."""

    def fetch_market_data(
        self, ticker: str, interval: str = "d", force_refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch market data from Stooq."""
        url = f"https://stooq.com/q/d/l/?s={ticker}&i={interval}"
        filename = f"market_{ticker}_{interval}.csv"
        return self._fetch_with_cache(url, "stooq", filename, force_refresh)

    def fetch_with_fallback(
        self,
        primary_ticker: str,
        fallback_ticker: str,
        interval: str = "d",
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch market data with fallback ticker."""
        try:
            print(f"Trying primary ticker: {primary_ticker}")
            return self.fetch_market_data(primary_ticker, interval, force_refresh)
        except Exception as e:
            print(f"Primary ticker failed: {e}")
            print(f"Trying fallback ticker: {fallback_ticker}")
            try:
                return self.fetch_market_data(fallback_ticker, interval, force_refresh)
            except Exception as e2:
                print(f"Fallback ticker also failed: {e2}")
                raise Exception(f"Both {primary_ticker} and {fallback_ticker} failed")


def fetch_scenario_data(
    scenario_config: dict[str, Any], force_refresh: bool = False
) -> dict[str, Any]:
    """Fetch all data required for a scenario."""
    results = {"data": {}, "metadata": {}, "errors": []}

    # Initialize fetchers
    owid_fetcher = OWIDFetcher()
    worldbank_fetcher = WorldBankFetcher()
    stooq_fetcher = StooqFetcher()

    data_config = scenario_config.get("data", {})

    # Fetch OWID data
    if "owid" in data_config:
        try:
            owid_config = data_config["owid"]
            df, metadata = owid_fetcher.fetch_covid_data(force_refresh)
            results["data"]["owid"] = df
            results["metadata"]["owid"] = metadata
            print(f"✓ Fetched OWID data: {len(df)} rows")
        except Exception as e:
            error_msg = f"Failed to fetch OWID data: {e}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    # Fetch World Bank data
    if "world_bank" in data_config:
        try:
            wb_config = data_config["world_bank"]

            if "gini" in wb_config and wb_config["gini"]:
                if "iso3" in wb_config["gini"]:
                    # Single country
                    iso3 = wb_config["gini"]["iso3"]
                    df, metadata = worldbank_fetcher.fetch_gini(iso3, force_refresh)
                    results["data"]["world_bank_gini"] = df
                    results["metadata"]["world_bank_gini"] = metadata
                    print(f"✓ Fetched World Bank Gini for {iso3}: {len(df)} rows")
                elif "countries" in wb_config:
                    # Multiple countries
                    countries = wb_config["countries"]
                    indicators = ["gini"] if wb_config.get("gini") else []
                    if wb_config.get("gdp_growth"):
                        indicators.append("gdp_growth")

                    multi_results = worldbank_fetcher.fetch_multiple_countries(
                        countries, indicators, force_refresh
                    )

                    for key, (df, metadata) in multi_results.items():
                        results["data"][f"world_bank_{key}"] = df
                        results["metadata"][f"world_bank_{key}"] = metadata
                        print(f"✓ Fetched World Bank {key}: {len(df)} rows")

            if "gdp_growth" in wb_config and wb_config["gdp_growth"]:
                if "iso3" in wb_config["gdp_growth"]:
                    iso3 = wb_config["gdp_growth"]["iso3"]
                    df, metadata = worldbank_fetcher.fetch_gdp_growth(
                        iso3, force_refresh
                    )
                    results["data"]["world_bank_gdp_growth"] = df
                    results["metadata"]["world_bank_gdp_growth"] = metadata
                    print(f"✓ Fetched World Bank GDP growth for {iso3}: {len(df)} rows")

        except Exception as e:
            error_msg = f"Failed to fetch World Bank data: {e}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    # Fetch Stooq market data
    if "market" in data_config:
        try:
            market_config = data_config["market"]
            ticker_primary = market_config.get("ticker_primary", "^spx")
            ticker_fallback = market_config.get("ticker_fallback", "spy.us")
            interval = market_config.get("interval", "d")

            df, metadata = stooq_fetcher.fetch_with_fallback(
                ticker_primary, ticker_fallback, interval, force_refresh
            )
            results["data"]["market"] = df
            results["metadata"]["market"] = metadata
            print(f"✓ Fetched market data: {len(df)} rows")

        except Exception as e:
            error_msg = f"Failed to fetch market data: {e}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

    return results

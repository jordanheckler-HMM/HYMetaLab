#!/usr/bin/env python3
"""
Simple data validation script to show what data we currently have.
"""

from pathlib import Path

import pandas as pd


def validate_current_data():
    """Show what data we currently have."""
    print("🔍 Current Data Validation")
    print("=" * 50)

    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ No data directory found")
        return

    print(f"📁 Data directory: {data_dir}")

    # Check each city
    cities = ["nyc", "la", "chi", "hou", "phx", "sea", "mia", "atl", "den", "msp"]

    for city in cities:
        city_dir = data_dir / city
        if not city_dir.exists():
            continue

        print(f"\n🏙️  {city.upper()}:")

        # Check each data source
        sources = {
            "crime": "crime.csv",
            "gini": "gini.csv",
            "trends": "trends_fear.csv",
            "events": "events.csv",
        }

        for source_name, filename in sources.items():
            file_path = city_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"  ✅ {source_name}: {len(df)} rows")

                    # Show sample data
                    if len(df) > 0:
                        print(f"      Sample: {df.iloc[0].to_dict()}")

                except Exception as e:
                    print(f"  ❌ {source_name}: Error reading file - {e}")
            else:
                print(f"  ❌ {source_name}: Missing file")

    # Show NYC data in detail
    print("\n📊 NYC Data Details:")
    nyc_dir = data_dir / "nyc"

    if nyc_dir.exists():
        for source_name, filename in sources.items():
            file_path = nyc_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"\n{source_name.upper()} ({len(df)} rows):")
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

                    if source_name == "crime":
                        print(f"  Total crimes: {df['count'].sum()}")
                        print(f"  Avg weekly crimes: {df['count'].mean():.1f}")
                    elif source_name == "gini":
                        print(
                            f"  Gini range: {df['gini'].min():.3f} to {df['gini'].max():.3f}"
                        )
                    elif source_name == "trends":
                        print(f"  Fear terms: {df['term'].unique()}")
                        print(f"  Avg fear score: {df['value'].mean():.1f}")
                    elif source_name == "events":
                        print(f"  Event types: {df['event_type'].unique()}")
                        print(f"  Avg magnitude: {df['magnitude'].mean():.3f}")

                except Exception as e:
                    print(f"  Error: {e}")


def show_data_summary():
    """Show a summary of available data."""
    print("\n📈 Data Summary:")
    print("=" * 30)

    data_dir = Path("data")
    if not data_dir.exists():
        print("No data available")
        return

    total_cities = 0
    total_sources = 0

    for city_dir in data_dir.iterdir():
        if city_dir.is_dir():
            total_cities += 1
            sources = ["crime.csv", "gini.csv", "trends_fear.csv", "events.csv"]
            for source in sources:
                if (city_dir / source).exists():
                    total_sources += 1

    print(f"Cities with data: {total_cities}")
    print(f"Total data sources: {total_sources}")
    print(f"Max possible sources: {total_cities * 4}")
    print(f"Coverage: {total_sources / (total_cities * 4) * 100:.1f}%")

    if total_cities > 0:
        print(f"\n✅ Ready for analysis with {total_cities} cities!")
        print("💡 Run: python rv_cli.py run --city nyc")
    else:
        print("\n❌ No cities have data yet")
        print("💡 Run: python rv_cli.py fetch --city nyc")


if __name__ == "__main__":
    validate_current_data()
    show_data_summary()

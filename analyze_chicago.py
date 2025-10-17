#!/usr/bin/env python3
"""
Analysis script for Chicago real crime data.
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def analyze_chicago_data():
    """Analyze the Chicago real crime data."""
    print("🔬 Chicago Real Crime Data Analysis")
    print("=" * 45)

    # Load Chicago data
    data_dir = Path("data/chi")

    # Load crime data
    crime_df = pd.read_csv(data_dir / "crime.csv")
    crime_df["date"] = pd.to_datetime(crime_df["date"])

    # Load Gini data
    gini_df = pd.read_csv(data_dir / "gini.csv")
    gini_df["date"] = pd.to_datetime(gini_df["date"])

    print("📊 Chicago Data:")
    print(f"  Crime records: {len(crime_df):,}")
    print(f"  Date range: {crime_df['date'].min()} to {crime_df['date'].max()}")
    print(f"  Gini records: {len(gini_df)}")

    # Aggregate crime to daily counts
    daily_crime = (
        crime_df.groupby(crime_df["date"].dt.date)["count"].sum().reset_index()
    )
    daily_crime["date"] = pd.to_datetime(daily_crime["date"])
    daily_crime.columns = ["date", "daily_crimes"]

    print("\n📈 Daily Crime Analysis:")
    print(f"  Days with data: {len(daily_crime)}")
    print(f"  Total crimes: {daily_crime['daily_crimes'].sum():,}")
    print(f"  Avg daily crimes: {daily_crime['daily_crimes'].mean():.1f}")
    print(f"  Max daily crimes: {daily_crime['daily_crimes'].max():,}")
    print(f"  Min daily crimes: {daily_crime['daily_crimes'].min():,}")

    # Weekly aggregation
    daily_crime["week"] = daily_crime["date"].dt.isocalendar().week
    daily_crime["year"] = daily_crime["date"].dt.year
    weekly_crime = (
        daily_crime.groupby(["year", "week"])["daily_crimes"].sum().reset_index()
    )
    weekly_crime["date"] = pd.to_datetime(
        weekly_crime["year"].astype(str)
        + "-"
        + weekly_crime["week"].astype(str)
        + "-1",
        format="%Y-%W-%w",
    )

    print("\n📅 Weekly Crime Analysis:")
    print(f"  Weeks with data: {len(weekly_crime)}")
    print(f"  Avg weekly crimes: {weekly_crime['daily_crimes'].mean():.1f}")
    print(
        f"  Crime trend: {weekly_crime['daily_crimes'].iloc[-1] - weekly_crime['daily_crimes'].iloc[0]:+.0f} crimes/week"
    )

    # Seasonal analysis
    daily_crime["month"] = daily_crime["date"].dt.month
    monthly_avg = daily_crime.groupby("month")["daily_crimes"].mean()

    print("\n🌡️ Seasonal Patterns:")
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for i, month in enumerate(months, 1):
        if i in monthly_avg.index:
            print(f"  {month}: {monthly_avg[i]:.1f} crimes/day")

    # Find peak crime periods
    peak_days = daily_crime.nlargest(5, "daily_crimes")
    print("\n🔥 Peak Crime Days:")
    for _, day in peak_days.iterrows():
        print(f"  {day['date'].strftime('%Y-%m-%d')}: {day['daily_crimes']:,} crimes")

    # Gini analysis
    print("\n💰 Inequality Analysis:")
    print(f"  Gini range: {gini_df['gini'].min():.4f} to {gini_df['gini'].max():.4f}")
    print(f"  Gini trend: {gini_df['gini'].iloc[-1] - gini_df['gini'].iloc[0]:+.4f}")

    # Correlation analysis
    # Merge crime and gini data
    analysis_df = weekly_crime.merge(gini_df, on="date", how="left")
    analysis_df["gini"] = analysis_df["gini"].fillna(method="ffill")

    if len(analysis_df) > 1:
        corr_gini_crime = analysis_df["gini"].corr(analysis_df["daily_crimes"])
        print("\n🔗 Correlation Analysis:")
        print(f"  Gini ↔ Crime: {corr_gini_crime:.3f}")

        if abs(corr_gini_crime) > 0.3:
            print("  💡 Strong correlation between inequality and crime!")
        elif abs(corr_gini_crime) > 0.1:
            print("  💡 Moderate correlation between inequality and crime")
        else:
            print("  💡 Weak correlation between inequality and crime")

    print("\n🎉 Chicago Analysis Complete!")
    print("✅ Real crime data successfully analyzed")
    print(f"📊 {len(crime_df):,} individual crime records processed")
    print(f"📈 {len(weekly_crime)} weeks of aggregated data")

    return daily_crime, weekly_crime


if __name__ == "__main__":
    try:
        daily, weekly = analyze_chicago_data()
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()

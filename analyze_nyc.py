#!/usr/bin/env python3
"""
Simple analysis script for the NYC data we have.
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def analyze_nyc_data():
    """Analyze the NYC data we have."""
    print("🔬 NYC Fear-Violence Analysis")
    print("=" * 40)

    # Load data
    data_dir = Path("data/nyc")

    # Load crime data
    crime_df = pd.read_csv(data_dir / "crime.csv")
    crime_df["date"] = pd.to_datetime(crime_df["date"])

    # Load Gini data
    gini_df = pd.read_csv(data_dir / "gini.csv")
    gini_df["date"] = pd.to_datetime(gini_df["date"])

    # Load trends data
    trends_df = pd.read_csv(data_dir / "trends_fear.csv")
    trends_df["date"] = pd.to_datetime(trends_df["date"])

    # Load events data
    events_df = pd.read_csv(data_dir / "events.csv")
    events_df["date"] = pd.to_datetime(events_df["date"])

    print("📊 Data loaded:")
    print(f"  Crime: {len(crime_df)} weeks")
    print(f"  Gini: {len(gini_df)} years")
    print(f"  Trends: {len(trends_df)} observations")
    print(f"  Events: {len(events_df)} events")

    # Create Fear Index from trends
    print("\n🧠 Creating Fear Index...")

    # Calculate weekly fear index (average of all terms per week)
    weekly_fear = trends_df.groupby("date")["value"].mean().reset_index()
    weekly_fear.columns = ["date", "fear_index"]

    # Normalize fear index (z-score)
    weekly_fear["fear_index"] = (
        weekly_fear["fear_index"] - weekly_fear["fear_index"].mean()
    ) / weekly_fear["fear_index"].std()

    print(
        f"  Fear Index range: {weekly_fear['fear_index'].min():.2f} to {weekly_fear['fear_index'].max():.2f}"
    )

    # Merge crime and fear data
    analysis_df = crime_df.merge(weekly_fear, on="date", how="inner")

    # Add Gini (forward fill annual data)
    analysis_df = analysis_df.merge(gini_df, on="date", how="left")
    analysis_df["gini"] = analysis_df["gini"].fillna(method="ffill")

    print(f"\n📈 Analysis Dataset: {len(analysis_df)} weeks")

    # Basic correlations
    print("\n🔗 Correlations:")
    corr_fear_crime = analysis_df["fear_index"].corr(analysis_df["count"])
    corr_gini_crime = analysis_df["gini"].corr(analysis_df["count"])
    corr_fear_gini = analysis_df["fear_index"].corr(analysis_df["gini"])

    print(f"  Fear Index ↔ Crime: {corr_fear_crime:.3f}")
    print(f"  Gini ↔ Crime: {corr_gini_crime:.3f}")
    print(f"  Fear Index ↔ Gini: {corr_fear_gini:.3f}")

    # Test the hypothesis
    print("\n🎯 Hypothesis Testing:")
    print(
        "Hypothesis: Fear leads to aggression (crime), moderated by inequality (Gini)"
    )

    # Simple regression: Crime ~ Fear + Gini + Fear*Gini
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # Prepare features
    X = analysis_df[["fear_index", "gini"]].copy()
    X["fear_gini_interaction"] = X["fear_index"] * X["gini"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Target: crime count
    y = analysis_df["count"].values

    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Get coefficients
    coef_fear = model.coef_[0]
    coef_gini = model.coef_[1]
    coef_interaction = model.coef_[2]
    r_squared = model.score(X_scaled, y)

    print("\n📊 Regression Results:")
    print(f"  Fear Index coefficient: {coef_fear:.3f}")
    print(f"  Gini coefficient: {coef_gini:.3f}")
    print(f"  Fear×Gini interaction: {coef_interaction:.3f}")
    print(f"  R²: {r_squared:.3f}")

    # Interpret results
    print("\n💡 Interpretation:")
    if coef_fear > 0:
        print("  ✅ Fear Index positively predicts crime (supports hypothesis)")
    else:
        print("  ❌ Fear Index negatively predicts crime (contradicts hypothesis)")

    if coef_interaction < 0:
        print("  ✅ Negative interaction: Gini moderates fear→crime relationship")
    else:
        print("  ❌ Positive interaction: Gini amplifies fear→crime relationship")

    # Event analysis
    print("\n⚡ Event Analysis:")

    # Find weeks with events
    event_weeks = []
    for _, event in events_df.iterrows():
        # Find closest week to event
        time_diff = abs(analysis_df["date"] - event["date"])
        closest_week_idx = time_diff.idxmin()
        event_weeks.append(
            {
                "week_idx": closest_week_idx,
                "event_type": event["event_type"],
                "magnitude": event["magnitude"],
                "date": event["date"],
            }
        )

    if event_weeks:
        print(f"  Found {len(event_weeks)} events near our data period")

        # Compare crime before/after events
        for event in event_weeks[:3]:  # Show first 3 events
            week_idx = event["week_idx"]
            if week_idx > 2 and week_idx < len(analysis_df) - 2:
                before_crime = analysis_df.iloc[week_idx - 2 : week_idx]["count"].mean()
                after_crime = analysis_df.iloc[week_idx : week_idx + 2]["count"].mean()

                print(
                    f"    {event['event_type']} ({event['date'].strftime('%Y-%m-%d')}):"
                )
                print(f"      Before: {before_crime:.1f} crimes/week")
                print(f"      After: {after_crime:.1f} crimes/week")
                print(
                    f"      Change: {((after_crime - before_crime) / before_crime * 100):+.1f}%"
                )

    # Summary
    print("\n🎉 Summary:")
    print(f"  Data covers {len(analysis_df)} weeks in 2019")
    print(f"  Fear Index shows {corr_fear_crime:.1%} correlation with crime")
    print(f"  Model explains {r_squared:.1%} of crime variation")

    if coef_fear > 0 and coef_interaction < 0:
        print("  ✅ Results SUPPORT the fear-violence hypothesis!")
    else:
        print("  ❌ Results do NOT support the fear-violence hypothesis")

    return analysis_df


if __name__ == "__main__":
    try:
        results_df = analyze_nyc_data()
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()

# real_world_validation/utils.py
"""
Utility functions for real-world validation module.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def safe_mkdirs(path: str) -> Path:
    """Safely create directory structure."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write DataFrame to CSV with error handling."""
    safe_mkdirs(os.path.dirname(path))
    df.to_csv(path, index=False, **kwargs)


def write_json(data: dict[str, Any], path: str) -> None:
    """Write data to JSON file with error handling."""
    safe_mkdirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def hash_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def robust_rolling_window(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Robust rolling window calculation with proper handling of NaNs."""
    if min_periods is None:
        min_periods = max(1, window // 2)

    return series.rolling(window=window, min_periods=min_periods)


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling z-score."""
    rolling_mean = robust_rolling_window(series, window).mean()
    rolling_std = robust_rolling_window(series, window).std()
    return (series - rolling_mean) / rolling_std


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage change."""
    return series.pct_change(periods=periods)


def resample_daily(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Resample to daily frequency."""
    df_resampled = df.set_index(date_col).resample("D").mean()
    return df_resampled.reset_index()


def resample_weekly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Resample to weekly frequency."""
    df_resampled = df.set_index(date_col).resample("W").mean()
    return df_resampled.reset_index()


def resample_monthly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Resample to monthly frequency."""
    df_resampled = df.set_index(date_col).resample("M").mean()
    return df_resampled.reset_index()


def validate_schema(
    df: pd.DataFrame, required_cols: list, log_missing: bool = True
) -> bool:
    """Validate DataFrame schema."""
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        if log_missing:
            print(f"Warning: Missing required columns: {missing_cols}")
        return False

    return True


def log_missingness(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Log missing data statistics."""
    total_rows = len(df)
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / total_rows * 100).round(2)

    print(f"\n{name} Missing Data Report:")
    print(f"Total rows: {total_rows}")

    for col in df.columns:
        missing_count = missing_stats[col]
        missing_percent = missing_pct[col]
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_percent}%)")
        else:
            print(f"  {col}: No missing data")


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_code_version() -> str:
    """Get code version information."""
    try:
        import real_world_validation

        return real_world_validation.__version__
    except AttributeError:
        return "unknown"

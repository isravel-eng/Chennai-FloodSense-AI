"""
feature_engineering.py
-----------------------
Builds the processed datasets used by Model 2 (flood-risk classifier).

V2 feature additions are intentionally lightweight and use only information
already present in master_dataset.csv. No future target information is used.

Added features:
  - month_sin/month_cos and day_of_year_sin/day_of_year_cos: cyclical seasonality
  - rainfall_lag_1/2/3/7: previous recorded rainfall for each locality
  - rainfall_change_1d: change from the previous recorded rainfall
  - rainfall_7d_per_day / rainfall_30d_per_day: normalized accumulation
  - rainfall_7d_ratio_30d: recent rainfall concentration

The script is idempotent and regenerates the processed CSVs from the raw data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 7]


def build_locality_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per locality with its latest known coordinates/elevation."""
    lookup = (
        df.sort_values("date")
        .groupby("locality", as_index=False)
        .agg(
            latitude=("latitude", "last"),
            longitude=("longitude", "last"),
            elevation_m_approx=("elevation_m_approx", "last"),
        )
        .sort_values("locality")
        .reset_index(drop=True)
    )
    return lookup


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_rainfall_lags(df: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """Previous recorded rainfall for each locality, ordered by date."""
    df = df.sort_values(["locality", "date"]).reset_index(drop=True)
    for lag in lags:
        col = f"rainfall_lag_{lag}"
        df[col] = df.groupby("locality")["rainfall_mm"].shift(lag)
        df[col] = df[col].fillna(df["rainfall_mm"])
    return df


def add_rainfall_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized accumulation and short-term rainfall-change features."""
    df["rainfall_change_1d"] = df["rainfall_mm"] - df["rainfall_lag_1"]
    df["rainfall_7d_per_day"] = df["rainfall_7d_mm"] / 7.0
    df["rainfall_30d_per_day"] = df["rainfall_30d_mm"] / 30.0
    df["rainfall_7d_ratio_30d"] = df["rainfall_7d_mm"] / (df["rainfall_30d_mm"] + 1e-6)
    return df


def build_model2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_cyclical_features(df.copy())
    df = add_rainfall_lags(df)
    df = add_rainfall_dynamics(df)

    ordered_cols = [
        "date", "locality", "latitude", "longitude", "elevation_m_approx",
        "rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm",
        "year", "month", "month_sin", "month_cos", "day_of_year",
        "day_of_year_sin", "day_of_year_cos", "is_northeast_monsoon",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_7",
        "rainfall_change_1d", "rainfall_7d_per_day", "rainfall_30d_per_day",
        "rainfall_7d_ratio_30d", "flood_occurred_documented",
    ]
    return df[ordered_cols]


def build_monthly_rainfall_citywide(df: pd.DataFrame) -> pd.DataFrame:
    """City-wide average monthly rainfall series used by Model 1 (SARIMA)."""
    daily_citywide = (
        df.groupby("date", as_index=False)["rainfall_mm"].mean()
        .rename(columns={"rainfall_mm": "avg_rainfall_mm"})
    )
    daily_citywide["date"] = pd.to_datetime(daily_citywide["date"])
    monthly = (
        daily_citywide.set_index("date")["avg_rainfall_mm"]
        .resample("MS")
        .mean()
        .to_frame("avg_rainfall_mm")
        .reset_index()
        .rename(columns={"date": "month_start"})
    )
    return monthly


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])

    lookup = build_locality_lookup(df)
    lookup_path = PROCESSED_DIR / "locality_lookup.csv"
    lookup.to_csv(lookup_path, index=False)
    print(f"Wrote {lookup_path} ({len(lookup)} localities)")

    features = build_model2_features(df)
    features_path = PROCESSED_DIR / "model2_features.csv"
    features.to_csv(features_path, index=False)
    print(f"Wrote {features_path} ({len(features)} rows, {features.shape[1]} cols)")

    monthly = build_monthly_rainfall_citywide(df)
    monthly_path = PROCESSED_DIR / "monthly_rainfall_citywide.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Wrote {monthly_path} ({len(monthly)} months)")


if __name__ == "__main__":
    main()

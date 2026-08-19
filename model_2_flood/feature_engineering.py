"""
feature_engineering.py
-----------------------
Builds the processed datasets used by Model 2 (XGBoost flood-risk classifier):

  data/processed/locality_lookup.csv   -> one row per locality (lat/lon/elevation)
  data/processed/model2_features.csv   -> master_dataset.csv + engineered features

Engineered features added on top of the raw master_dataset columns:
  - month_sin, month_cos      : cyclical encoding of calendar month
  - rainfall_lag_1/2/3/7      : rainfall (mm) N days *before* the current record,
                                 computed per locality on the locality's own
                                 observation sequence (the dataset is not daily
                                 for every locality, so "lag" = previous
                                 recorded reading for that locality, not
                                 strictly previous calendar day)

This script is idempotent - re-running it regenerates both CSVs from the raw
master_dataset.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 7]


def build_locality_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per locality with its coordinates + elevation (most recent value used)."""
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


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_rainfall_lags(df: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """Per-locality lag of rainfall_mm over the locality's own record sequence."""
    df = df.sort_values(["locality", "date"]).reset_index(drop=True)
    for lag in lags:
        col = f"rainfall_lag_{lag}"
        df[col] = df.groupby("locality")["rainfall_mm"].shift(lag)
    # Fill leading NaNs (start of each locality's series) using that locality's
    # own rainfall_mm at time 0 (no rainfall history yet -> assume same as
    # current reading, a conservative "no prior signal" fallback).
    lag_cols = [f"rainfall_lag_{lag}" for lag in lags]
    for col in lag_cols:
        df[col] = df[col].fillna(df["rainfall_mm"])
    return df


def build_model2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_cyclical_month(df.copy())
    df = add_rainfall_lags(df)
    ordered_cols = [
        "date", "locality", "latitude", "longitude", "elevation_m_approx",
        "rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm",
        "year", "month", "month_sin", "month_cos", "day_of_year",
        "is_northeast_monsoon",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_7",
        "flood_occurred_documented",
    ]
    return df[ordered_cols]


def build_monthly_rainfall_citywide(df: pd.DataFrame) -> pd.DataFrame:
    """City-wide average monthly rainfall series, used to train Model 1 (SARIMA)."""
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
    df = pd.read_csv(RAW_PATH)

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

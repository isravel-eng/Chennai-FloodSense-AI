"""
preprocessing.py
----------------
Builds the serving metadata artifact for Model 2.

FEATURE_ORDER is the single source of truth for the trained model's input
schema. Live inference imports this order so training and inference cannot
silently drift apart.
"""

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_ORDER = [
    "rainfall_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "latitude",
    "longitude",
    "elevation_m_approx",
    "month",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_northeast_monsoon",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_7",
    "rainfall_change_1d",
    "rainfall_7d_per_day",
    "rainfall_30d_per_day",
    "rainfall_7d_ratio_30d",
]

TARGET_COL = "flood_occurred_documented"


def build_preprocessing(df: pd.DataFrame, locality_stats: dict | None = None) -> dict:
    if locality_stats is None:
        locality_stats = (
            df.groupby("locality")[["latitude", "longitude", "elevation_m_approx"]]
            .last()
            .to_dict(orient="index")
        )

    return {
        "feature_order": FEATURE_ORDER,
        "target_col": TARGET_COL,
        "locality_stats": locality_stats,
        "scaling": None,
        "model_type": "tree_classifier",
        "model_version": "2.0",
        "notes": (
            "V2 adds cyclical day-of-year features and rainfall dynamics. "
            "No target-derived feature is used. Forecast rainfall is only "
            "passed through the existing live forecast path as a documented "
            "near-term proxy."
        ),
    }


def main():
    df = pd.read_csv(FEATURES_PATH)
    missing = [c for c in FEATURE_ORDER + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"model2_features.csv is missing columns: {missing}")

    preprocessing = build_preprocessing(df)
    out_path = MODELS_DIR / "flood_preprocessing.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(preprocessing, f)
    print(f"Saved {out_path}")
    print(f"Feature order ({len(FEATURE_ORDER)} features): {FEATURE_ORDER}")


if __name__ == "__main__":
    main()

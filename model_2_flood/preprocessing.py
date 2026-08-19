"""
preprocessing.py
-----------------
Builds and saves the preprocessing artifact for Model 2 (flood-risk
XGBoost classifier): models/flood_preprocessing.pkl

This is a plain Python dict (not an sklearn Pipeline object) so that it
unpickles portably regardless of sklearn version drift between training
and inference machines - see the "pickle portability" fix noted in
BUILD_LOG.md. It stores:

  - feature_order : the exact ordered list of column names XGBoost expects.
                     live/live_features.py and model_2_flood/predict_flood.py
                     both import this to guarantee the live feature vector
                     matches training order exactly.
  - locality_stats : per-locality latitude/longitude/elevation, used by the
                     live layer as a fallback lookup.
  - scaling        : None - XGBoost is tree-based and does not require
                     feature scaling; kept as an explicit field so future
                     model swaps (e.g. logistic regression) have a documented
                     place to add a fitted StandardScaler.

Run standalone: python model_2_flood/preprocessing.py
"""

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "model2_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical feature order - Model 2 (XGBoost) is trained and served with
# EXACTLY this column order. live/live_features.py must reproduce it.
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
    "is_northeast_monsoon",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_7",
]

TARGET_COL = "flood_occurred_documented"


def build_preprocessing(df: pd.DataFrame) -> dict:
    locality_stats = (
        df.groupby("locality")[["latitude", "longitude", "elevation_m_approx"]]
        .last()
        .to_dict(orient="index")
    )
    preprocessing = {
        "feature_order": FEATURE_ORDER,
        "target_col": TARGET_COL,
        "locality_stats": locality_stats,
        "scaling": None,  # tree model - no scaling needed
        "model_type": "xgboost.XGBClassifier",
        "notes": (
            "feature_order is authoritative. live/live_features.py builds a "
            "dict keyed by these exact names, in this exact order, before "
            "calling model.predict_proba()."
        ),
    }
    return preprocessing


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

"""
predict_flood.py
-----------------
Thin, stable prediction interface around the saved Model 2 artifacts.
This is the ONLY module both `predict_end_to_end.py` (historical/manual
use) and `live/live_prediction.py` (live use) should import from - it
guarantees the feature vector is built in the exact order flood_model.pkl
was trained on, every time.

Usage:
    from model_2_flood.predict_flood import predict_flood, RISK_BANDS

    result = predict_flood({
        "rainfall_mm": 42.0,
        "rainfall_3d_mm": 88.0,
        "rainfall_7d_mm": 143.0,
        "rainfall_30d_mm": 310.0,
        "latitude": 12.901,
        "longitude": 80.2279,
        "elevation_m_approx": 5,
        "month": 11,
        "month_sin": ...,
        "month_cos": ...,
        "is_northeast_monsoon": 1,
        "rainfall_lag_1": 12.0,
        "rainfall_lag_2": 8.0,
        "rainfall_lag_3": 5.0,
        "rainfall_lag_7": 3.0,
    })
    # -> {"probability": 0.72, "risk_band": "HIGH", "threshold_used": 0.34}
"""

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Ordered LOW -> HIGH. Bands are intentionally more conservative than the
# "best-F1" threshold from evaluation.py (0.763) because in a public safety
# tool, false negatives (missed flood) are far more costly than false
# positives (unnecessary caution) - so MEDIUM/HIGH trigger earlier.
RISK_BANDS = [
    (0.00, 0.15, "LOW"),
    (0.15, 0.40, "MEDIUM"),
    (0.40, 1.01, "HIGH"),
]

_model = None
_preprocessing = None


def _load_artifacts():
    global _model, _preprocessing
    if _model is None:
        with open(MODELS_DIR / "flood_model.pkl", "rb") as f:
            _model = pickle.load(f)
    if _preprocessing is None:
        with open(MODELS_DIR / "flood_preprocessing.pkl", "rb") as f:
            _preprocessing = pickle.load(f)
    return _model, _preprocessing


def band_for(probability: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= probability < hi:
            return label
    return "HIGH"


def predict_flood(features: dict) -> dict:
    """
    features: dict containing every key in flood_preprocessing.pkl's
    feature_order. Extra keys are ignored; missing keys raise KeyError
    with a clear message (fail loudly rather than silently defaulting).
    """
    model, preprocessing = _load_artifacts()
    feature_order = preprocessing["feature_order"]

    missing = [c for c in feature_order if c not in features]
    if missing:
        raise KeyError(
            f"predict_flood() is missing required feature(s): {missing}. "
            f"Full required order is: {feature_order}"
        )

    row = np.array([[features[col] for col in feature_order]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])

    return {
        "probability": round(probability, 4),
        "risk_band": band_for(probability),
        "feature_order": feature_order,
    }


if __name__ == "__main__":
    # Smoke test using a plausible November (monsoon) Sholinganallur reading.
    import math

    demo = {
        "rainfall_mm": 42.0,
        "rainfall_3d_mm": 88.0,
        "rainfall_7d_mm": 143.0,
        "rainfall_30d_mm": 310.0,
        "latitude": 12.901,
        "longitude": 80.2279,
        "elevation_m_approx": 5,
        "month": 11,
        "month_sin": math.sin(2 * math.pi * 11 / 12),
        "month_cos": math.cos(2 * math.pi * 11 / 12),
        "is_northeast_monsoon": 1,
        "rainfall_lag_1": 55.0,
        "rainfall_lag_2": 30.0,
        "rainfall_lag_3": 20.0,
        "rainfall_lag_7": 5.0,
    }
    print(predict_flood(demo))

"""
predict_flood.py
-----------------
Stable inference interface for Model 2.

The probability comes directly from the trained model. The application keeps
simple LOW / MEDIUM / HIGH bands for a stable UI, while the validation-selected
threshold is returned separately as `threshold_used` for alert logic/reporting.
"""

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_model = None
_preprocessing = None
_metadata = None

# Stable UI bands promised by the project presentation.
RISK_BANDS = [
    (0.00, 0.15, "LOW"),
    (0.15, 0.40, "MEDIUM"),
    (0.40, 1.01, "HIGH"),
]


def _load_artifacts():
    global _model, _preprocessing, _metadata
    if _model is None:
        with open(MODELS_DIR / "flood_model.pkl", "rb") as f:
            _model = pickle.load(f)
    if _preprocessing is None:
        with open(MODELS_DIR / "flood_preprocessing.pkl", "rb") as f:
            _preprocessing = pickle.load(f)
    if _metadata is None:
        path = MODELS_DIR / "flood_model_selection.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _metadata = pickle.load(f)
        else:
            _metadata = {"decision_threshold": 0.5}
    return _model, _preprocessing, _metadata


def band_for(probability: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= probability < hi:
            return label
    return "HIGH"


def predict_flood(features: dict) -> dict:
    model, preprocessing, metadata = _load_artifacts()
    feature_order = preprocessing["feature_order"]
    missing = [c for c in feature_order if c not in features]
    if missing:
        raise KeyError(f"predict_flood() is missing required feature(s): {missing}")

    row = np.array([[features[col] for col in feature_order]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    return {
        "probability": round(probability, 4),
        "risk_band": band_for(probability),
        "threshold_used": round(threshold, 4),
        "feature_order": feature_order,
    }

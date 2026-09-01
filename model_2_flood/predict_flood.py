import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "flood_model.pkl"

FEATURES = [
    "rainfall_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "latitude",
    "longitude",
]

RISK_BANDS = [
    (0.00, 0.30, "LOW"),
    (0.30, 0.70, "MEDIUM"),
    (0.70, 1.01, "HIGH"),
]

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def band_for(probability):
    for low, high, label in RISK_BANDS:
        if low <= probability < high:
            return label
    return "HIGH"


def predict_flood(values):
    missing = [name for name in FEATURES if name not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([{name: values[name] for name in FEATURES}])
    probability = float(load_model().predict_proba(row)[0, 1])

    return {
        "probability": round(probability, 4),
        "risk_band": band_for(probability),
    }

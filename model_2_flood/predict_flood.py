import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RISK_BANDS = [(0.00, 0.15, "LOW"), (0.15, 0.40, "MEDIUM"), (0.40, 1.01, "HIGH")]
_model = None
_preprocessing = None

def _load_artifacts():
    global _model, _preprocessing
    if _model is None:
        with open(MODELS_DIR / "flood_model.pkl", "rb") as f: _model = pickle.load(f)
    if _preprocessing is None:
        with open(MODELS_DIR / "flood_preprocessing.pkl", "rb") as f: _preprocessing = pickle.load(f)
    return _model, _preprocessing

def band_for(probability: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= probability < hi: return label
    return "HIGH"

def predict_flood(features: dict) -> dict:
    model, preprocessing = _load_artifacts()
    order = preprocessing["feature_order"]
    missing = [c for c in order if c not in features]
    if missing: raise KeyError(f"predict_flood() is missing required feature(s): {missing}")
    row = np.array([[features[c] for c in order]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    return {"probability": round(probability, 4), "risk_band": band_for(probability), "feature_order": order}

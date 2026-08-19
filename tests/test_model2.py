"""
test_model2.py
---------------
Sanity checks for the saved flood-risk model (models/flood_model.pkl)
and the predict_flood() interface.
Run: python -m pytest tests/test_model2.py -v
  or: python tests/test_model2.py
"""

import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_2_flood.predict_flood import predict_flood, band_for, RISK_BANDS  # noqa: E402

MODELS_DIR = ROOT / "models"

DEMO_LOW = {
    "rainfall_mm": 0.0, "rainfall_3d_mm": 0.0, "rainfall_7d_mm": 2.0,
    "rainfall_30d_mm": 10.0, "latitude": 13.0604, "longitude": 80.2496,
    "elevation_m_approx": 8, "month": 3,
    "month_sin": math.sin(2 * math.pi * 3 / 12), "month_cos": math.cos(2 * math.pi * 3 / 12),
    "is_northeast_monsoon": 0,
    "rainfall_lag_1": 0.0, "rainfall_lag_2": 0.0, "rainfall_lag_3": 0.0, "rainfall_lag_7": 0.0,
}

DEMO_HIGH = {
    "rainfall_mm": 250.0, "rainfall_3d_mm": 400.0, "rainfall_7d_mm": 600.0,
    "rainfall_30d_mm": 900.0, "latitude": 13.0067, "longitude": 80.2,
    "elevation_m_approx": 9, "month": 11,
    "month_sin": math.sin(2 * math.pi * 11 / 12), "month_cos": math.cos(2 * math.pi * 11 / 12),
    "is_northeast_monsoon": 1,
    "rainfall_lag_1": 180.0, "rainfall_lag_2": 90.0, "rainfall_lag_3": 60.0, "rainfall_lag_7": 30.0,
}


def test_model_files_exist():
    assert (MODELS_DIR / "flood_model.pkl").exists()
    assert (MODELS_DIR / "flood_preprocessing.pkl").exists()


def test_preprocessing_feature_order():
    with open(MODELS_DIR / "flood_preprocessing.pkl", "rb") as f:
        meta = pickle.load(f)
    assert len(meta["feature_order"]) == 15
    assert meta["feature_order"][0] == "rainfall_mm"


def test_predict_returns_expected_shape():
    result = predict_flood(DEMO_LOW)
    assert "probability" in result
    assert "risk_band" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_missing_feature_raises():
    bad = dict(DEMO_LOW)
    del bad["rainfall_mm"]
    try:
        predict_flood(bad)
        raised = False
    except KeyError:
        raised = True
    assert raised, "predict_flood should raise KeyError on missing features"


def test_dry_month_scores_lower_than_extreme_monsoon_event():
    low_result = predict_flood(DEMO_LOW)
    high_result = predict_flood(DEMO_HIGH)
    assert low_result["probability"] < high_result["probability"], (
        "A dry March reading should score a lower flood probability than "
        "an extreme November monsoon event with heavy rainfall history."
    )


def test_band_boundaries():
    assert band_for(0.0) == "LOW"
    assert band_for(0.99) == "HIGH"
    assert band_for(0.20) == "MEDIUM"


if __name__ == "__main__":
    test_model_files_exist()
    test_preprocessing_feature_order()
    test_predict_returns_expected_shape()
    test_missing_feature_raises()
    test_dry_month_scores_lower_than_extreme_monsoon_event()
    test_band_boundaries()
    print("All Model 2 tests passed.")

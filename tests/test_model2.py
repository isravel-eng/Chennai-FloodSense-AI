import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_2_flood.predict_flood import predict_flood, band_for


DEMO_LOW = {
    "rainfall_mm": 0.0,
    "rainfall_3d_mm": 0.0,
    "rainfall_7d_mm": 2.0,
    "rainfall_30d_mm": 10.0,
    "latitude": 13.0604,
    "longitude": 80.2496,
}

DEMO_HIGH = {
    "rainfall_mm": 250.0,
    "rainfall_3d_mm": 400.0,
    "rainfall_7d_mm": 600.0,
    "rainfall_30d_mm": 900.0,
    "latitude": 13.0067,
    "longitude": 80.2,
}


def test_model_file_exists():
    assert (ROOT / "models" / "flood_model.pkl").exists()


def test_predict_returns_probability_and_band():
    result = predict_flood(DEMO_LOW)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_missing_feature_raises():
    bad = dict(DEMO_LOW)
    del bad["rainfall_mm"]
    try:
        predict_flood(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_risk_bands():
    assert band_for(0.0) == "LOW"
    assert band_for(0.5) == "MEDIUM"
    assert band_for(0.9) == "HIGH"


if __name__ == "__main__":
    test_model_file_exists()
    test_predict_returns_probability_and_band()
    test_missing_feature_raises()
    test_risk_bands()
    print("All Model 2 tests passed.")

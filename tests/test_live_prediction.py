"""
test_live_prediction.py
------------------------
Tests the live prediction pipeline WITHOUT hitting the real weather API,
by feeding a mocked weather dict directly into live_features.py +
predict_flood.py (the same path live_prediction.py uses internally).
This keeps the test suite runnable offline / in CI.

Run: python -m pytest tests/test_live_prediction.py -v
  or: python tests/test_live_prediction.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.rainfall_history import get_recent_rainfall, get_locality_climatology  # noqa: E402
from live.live_features import build_current_features, build_forecast_24h_features  # noqa: E402
from model_2_flood.predict_flood import predict_flood  # noqa: E402

MOCK_WEATHER = {
    "current_precipitation_mm": 15.0,
    "forecast_next_24h_precipitation_mm": 95.0,
}
MOCK_LOCATION = {"latitude": 12.901, "longitude": 80.2279, "elevation_m_approx": 5}


def test_climatology_fallback_has_required_keys():
    clim = get_locality_climatology("Sholinganallur", month=11)
    for key in ["rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm",
                "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_7"]:
        assert key in clim


def test_get_recent_rainfall_uses_climatology_when_log_empty():
    history = get_recent_rainfall("Sholinganallur", month=11)
    assert history["source"] == "climatology_fallback"


def test_build_features_produce_15_keys_minus_target():
    history = get_recent_rainfall("Sholinganallur", month=11)
    features = build_current_features(MOCK_WEATHER, history, MOCK_LOCATION, month=11)
    expected_keys = {
        "rainfall_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_30d_mm",
        "latitude", "longitude", "elevation_m_approx", "month", "month_sin",
        "month_cos", "is_northeast_monsoon",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_7",
    }
    assert set(features.keys()) == expected_keys


def test_current_vs_forecast_use_different_rainfall_mm():
    history = get_recent_rainfall("Sholinganallur", month=11)
    current = build_current_features(MOCK_WEATHER, history, MOCK_LOCATION, month=11)
    forecast = build_forecast_24h_features(MOCK_WEATHER, history, MOCK_LOCATION, month=11)
    assert current["rainfall_mm"] == 15.0
    assert forecast["rainfall_mm"] == 95.0


def test_full_pipeline_produces_valid_prediction():
    history = get_recent_rainfall("Sholinganallur", month=11)
    features = build_forecast_24h_features(MOCK_WEATHER, history, MOCK_LOCATION, month=11)
    result = predict_flood(features)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}


def test_monsoon_month_flag_correct():
    history = get_recent_rainfall("Sholinganallur", month=11)
    features = build_current_features(MOCK_WEATHER, history, MOCK_LOCATION, month=11)
    assert features["is_northeast_monsoon"] == 1
    features_march = build_current_features(MOCK_WEATHER, history, MOCK_LOCATION, month=3)
    assert features_march["is_northeast_monsoon"] == 0


if __name__ == "__main__":
    test_climatology_fallback_has_required_keys()
    test_get_recent_rainfall_uses_climatology_when_log_empty()
    test_build_features_produce_15_keys_minus_target()
    test_current_vs_forecast_use_different_rainfall_mm()
    test_full_pipeline_produces_valid_prediction()
    test_monsoon_month_flag_correct()
    print("All live prediction tests passed.")

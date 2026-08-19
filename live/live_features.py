"""
live_features.py
-----------------
Converts (weather API data) + (recent rainfall history) + (locality info)
into the EXACT feature dict Model 2 (XGBoost) expects, in the order
recorded in models/flood_preprocessing.pkl.

Feature order (do not change without retraining):
  rainfall_mm, rainfall_3d_mm, rainfall_7d_mm, rainfall_30d_mm,
  latitude, longitude, elevation_m_approx,
  month, month_sin, month_cos, is_northeast_monsoon,
  rainfall_lag_1, rainfall_lag_2, rainfall_lag_3, rainfall_lag_7

Two builders are provided, matching the two predictions the live app
shows (see live_prediction.py):

  build_current_features()  -> "rainfall_mm" = today's actual/observed
                                precipitation so far (current conditions)
  build_forecast_24h_features() -> "rainfall_mm" = next-24h FORECAST
                                precipitation from the weather API

Model 2 was trained on ACTUAL daily rainfall only (never on forecast
data, and never on temperature/humidity/wind - see README.md "Version 1
vs Version 2"). Using forecast precipitation in the same "rainfall_mm"
slot is a deliberate, documented approximation for the 24h-ahead
prediction - it is the best available near-term proxy without retraining
the model on forecast-vs-actual data, which is future work (Version 2).
"""

import math
from datetime import datetime


def _month_cyclical(month: int) -> dict:
    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def _is_northeast_monsoon(month: int) -> int:
    # Chennai's northeast monsoon: October-December (matches master_dataset.csv encoding)
    return 1 if month in (10, 11, 12) else 0


def _base_features(location: dict, history: dict, month: int) -> dict:
    cyc = _month_cyclical(month)
    return {
        "rainfall_3d_mm": history["rainfall_3d_mm"],
        "rainfall_7d_mm": history["rainfall_7d_mm"],
        "rainfall_30d_mm": history["rainfall_30d_mm"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "elevation_m_approx": location["elevation_m_approx"],
        "month": month,
        "month_sin": cyc["month_sin"],
        "month_cos": cyc["month_cos"],
        "is_northeast_monsoon": _is_northeast_monsoon(month),
        "rainfall_lag_1": history["rainfall_lag_1"],
        "rainfall_lag_2": history["rainfall_lag_2"],
        "rainfall_lag_3": history["rainfall_lag_3"],
        "rainfall_lag_7": history["rainfall_lag_7"],
    }


def build_current_features(weather: dict, history: dict, location: dict, month: int = None) -> dict:
    """Prediction A - current risk, based on rainfall observed so far today."""
    month = month or datetime.now().month
    features = _base_features(location, history, month)
    features["rainfall_mm"] = float(weather.get("current_precipitation_mm", 0.0) or 0.0)
    return features


def build_forecast_24h_features(weather: dict, history: dict, location: dict, month: int = None) -> dict:
    """Prediction B - next-24h risk, based on next-24h forecast rainfall."""
    month = month or datetime.now().month
    features = _base_features(location, history, month)
    features["rainfall_mm"] = float(weather.get("forecast_next_24h_precipitation_mm", 0.0) or 0.0)
    return features

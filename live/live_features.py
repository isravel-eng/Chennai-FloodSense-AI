"""
live_features.py
-----------------
Builds the exact V2 Model 2 feature vector from live weather + recent rainfall.

The live layer uses only features that are also present in the training data.
Open-Meteo temperature/humidity/wind are intentionally not passed to the
model because the historical training dataset does not contain those fields.
"""

import math
from datetime import datetime


def _month_cyclical(month: int) -> dict:
    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def _day_cyclical(day_of_year: int) -> dict:
    return {
        "day_of_year_sin": math.sin(2 * math.pi * day_of_year / 365.25),
        "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 365.25),
    }


def _is_northeast_monsoon(month: int) -> int:
    return 1 if month in (10, 11, 12) else 0


def _base_features(location: dict, history: dict, month: int, day_of_year: int) -> dict:
    cyc = _month_cyclical(month)
    day_cyc = _day_cyclical(day_of_year)
    rainfall_mm = float(history.get("rainfall_mm", 0.0) or 0.0)
    lag_1 = float(history.get("rainfall_lag_1", 0.0) or 0.0)
    return {
        "rainfall_3d_mm": float(history["rainfall_3d_mm"]),
        "rainfall_7d_mm": float(history["rainfall_7d_mm"]),
        "rainfall_30d_mm": float(history["rainfall_30d_mm"]),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "elevation_m_approx": float(location["elevation_m_approx"]),
        "month": month,
        "month_sin": cyc["month_sin"],
        "month_cos": cyc["month_cos"],
        "day_of_year_sin": day_cyc["day_of_year_sin"],
        "day_of_year_cos": day_cyc["day_of_year_cos"],
        "is_northeast_monsoon": _is_northeast_monsoon(month),
        "rainfall_lag_1": lag_1,
        "rainfall_lag_2": float(history["rainfall_lag_2"]),
        "rainfall_lag_3": float(history["rainfall_lag_3"]),
        "rainfall_lag_7": float(history["rainfall_lag_7"]),
        "rainfall_change_1d": rainfall_mm - lag_1,
        "rainfall_7d_per_day": float(history["rainfall_7d_mm"]) / 7.0,
        "rainfall_30d_per_day": float(history["rainfall_30d_mm"]) / 30.0,
        "rainfall_7d_ratio_30d": float(history["rainfall_7d_mm"]) / (float(history["rainfall_30d_mm"]) + 1e-6),
    }


def build_current_features(weather: dict, history: dict, location: dict, month: int = None, day_of_year: int = None) -> dict:
    """Current-risk vector using observed precipitation."""
    now = datetime.now()
    month = month or now.month
    day_of_year = day_of_year or now.timetuple().tm_yday
    features = _base_features(location, history, month, day_of_year)
    features["rainfall_mm"] = float(weather.get("current_precipitation_mm", 0.0) or 0.0)
    features["rainfall_change_1d"] = features["rainfall_mm"] - features["rainfall_lag_1"]
    return features


def build_forecast_24h_features(weather: dict, history: dict, location: dict, month: int = None, day_of_year: int = None) -> dict:
    """Next-24h vector using Open-Meteo forecast precipitation as the rainfall input."""
    now = datetime.now()
    month = month or now.month
    day_of_year = day_of_year or now.timetuple().tm_yday
    features = _base_features(location, history, month, day_of_year)
    features["rainfall_mm"] = float(weather.get("forecast_next_24h_precipitation_mm", 0.0) or 0.0)
    features["rainfall_change_1d"] = features["rainfall_mm"] - features["rainfall_lag_1"]
    return features

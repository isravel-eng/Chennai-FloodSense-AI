import math
from datetime import date, datetime


def _month_cyclical(month: int) -> dict:
    return {"month_sin": math.sin(2 * math.pi * month / 12), "month_cos": math.cos(2 * math.pi * month / 12)}


def _day_cyclical(day_of_year: int) -> dict:
    return {"day_of_year_sin": math.sin(2 * math.pi * day_of_year / 365.25), "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 365.25)}


def _is_northeast_monsoon(month: int) -> int:
    return 1 if month in (10, 11, 12) else 0


def _base_features(location: dict, history: dict, month: int, day_of_year: int) -> dict:
    cyc = _month_cyclical(month)
    day_cyc = _day_cyclical(day_of_year)
    rainfall_mm = float(history.get("rainfall_mm", 0.0) or 0.0)
    lag_1 = float(history.get("rainfall_lag_1", 0.0) or 0.0)
    return {
        "rainfall_3d_mm": float(history["rainfall_3d_mm"]), "rainfall_7d_mm": float(history["rainfall_7d_mm"]),
        "rainfall_30d_mm": float(history["rainfall_30d_mm"]), "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]), "elevation_m_approx": float(location["elevation_m_approx"]),
        "month": month, "month_sin": cyc["month_sin"], "month_cos": cyc["month_cos"],
        "day_of_year_sin": day_cyc["day_of_year_sin"], "day_of_year_cos": day_cyc["day_of_year_cos"],
        "is_northeast_monsoon": _is_northeast_monsoon(month), "rainfall_lag_1": lag_1,
        "rainfall_lag_2": float(history["rainfall_lag_2"]), "rainfall_lag_3": float(history["rainfall_lag_3"]),
        "rainfall_lag_7": float(history["rainfall_lag_7"]), "rainfall_change_1d": rainfall_mm - lag_1,
        "rainfall_7d_per_day": float(history["rainfall_7d_mm"]) / 7.0,
        "rainfall_30d_per_day": float(history["rainfall_30d_mm"]) / 30.0,
        "rainfall_7d_ratio_30d": float(history["rainfall_7d_mm"]) / (float(history["rainfall_30d_mm"]) + 1e-6),
    }


def build_current_features(weather: dict, history: dict, location: dict, month: int = None, day_of_year: int = None) -> dict:
    now = datetime.now(); month = month or now.month; day_of_year = day_of_year or now.timetuple().tm_yday
    features = _base_features(location, history, month, day_of_year)
    features["rainfall_mm"] = float(weather.get("current_precipitation_mm", 0.0) or 0.0)
    features["rainfall_change_1d"] = features["rainfall_mm"] - features["rainfall_lag_1"]
    return features


def build_forecast_24h_features(weather: dict, history: dict, location: dict, month: int = None, day_of_year: int = None) -> dict:
    now = datetime.now(); month = month or now.month; day_of_year = day_of_year or now.timetuple().tm_yday
    features = _base_features(location, history, month, day_of_year)
    features["rainfall_mm"] = float(weather.get("forecast_next_24h_precipitation_mm", 0.0) or 0.0)
    features["rainfall_change_1d"] = features["rainfall_mm"] - features["rainfall_lag_1"]
    return features


def build_future_day_features(weather: dict, history: dict, location: dict, date: date, day_index: int = 0) -> dict:
    features = _base_features(location, history, date.month, date.timetuple().tm_yday)
    daily = weather.get("daily_forecast", [])
    values = [float(x.get("rainfall_mm", 0.0) or 0.0) for x in daily[: day_index + 1]]
    rainfall_mm = values[-1] if values else 0.0
    prior = values[:-1]
    previous_day = prior[-1] if prior else float(history.get("rainfall_lag_1", 0.0) or 0.0)
    features["rainfall_mm"] = rainfall_mm
    features["rainfall_lag_1"] = previous_day
    features["rainfall_change_1d"] = rainfall_mm - previous_day
    features["rainfall_3d_mm"] = float(history["rainfall_3d_mm"]) + sum(prior[-2:])
    features["rainfall_7d_mm"] = float(history["rainfall_7d_mm"]) + sum(prior[-6:])
    features["rainfall_30d_mm"] = float(history["rainfall_30d_mm"]) + sum(prior[-29:])
    features["rainfall_lag_2"] = float(prior[-2] if len(prior) >= 2 else history["rainfall_lag_1"])
    features["rainfall_lag_3"] = float(prior[-3] if len(prior) >= 3 else history["rainfall_lag_2"])
    features["rainfall_lag_7"] = float(history["rainfall_lag_7"])
    features["rainfall_7d_per_day"] = features["rainfall_7d_mm"] / 7.0
    features["rainfall_30d_per_day"] = features["rainfall_30d_mm"] / 30.0
    features["rainfall_7d_ratio_30d"] = features["rainfall_7d_mm"] / (features["rainfall_30d_mm"] + 1e-6)
    return features

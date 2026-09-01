def build_current_features(weather: dict, history: dict, location: dict) -> dict:
    return {
        "rainfall_mm": float(weather.get("current_precipitation_mm", 0.0) or 0.0),
        "rainfall_3d_mm": float(history.get("rainfall_3d_mm", 0.0)),
        "rainfall_7d_mm": float(history.get("rainfall_7d_mm", 0.0)),
        "rainfall_30d_mm": float(history.get("rainfall_30d_mm", 0.0)),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
    }


def build_forecast_24h_features(weather: dict, history: dict, location: dict) -> dict:
    return {
        "rainfall_mm": float(weather.get("forecast_next_24h_precipitation_mm", 0.0) or 0.0),
        "rainfall_3d_mm": float(history.get("rainfall_3d_mm", 0.0)),
        "rainfall_7d_mm": float(history.get("rainfall_7d_mm", 0.0)),
        "rainfall_30d_mm": float(history.get("rainfall_30d_mm", 0.0)),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
    }


def build_future_day_features(weather: dict, history: dict, location: dict, date, day_index: int = 0) -> dict:
    daily = weather.get("daily_forecast", [])
    rainfall = float(daily[day_index].get("rainfall_mm", 0.0) or 0.0) if day_index < len(daily) else 0.0

    return {
        "rainfall_mm": rainfall,
        "rainfall_3d_mm": float(history.get("rainfall_3d_mm", 0.0)),
        "rainfall_7d_mm": float(history.get("rainfall_7d_mm", 0.0)),
        "rainfall_30d_mm": float(history.get("rainfall_30d_mm", 0.0)),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
    }

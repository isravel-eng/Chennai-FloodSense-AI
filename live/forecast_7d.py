"""Seven-day live rainfall and flood-risk forecasting using Open-Meteo."""

from datetime import datetime
from typing import Callable

from live.live_features import build_future_day_features


def forecast_next_7_days(weather: dict, history: dict, location: dict, predict_flood: Callable[[dict], dict]) -> list[dict]:
    results = []
    for index, item in enumerate(weather.get("daily_forecast", [])[:7]):
        date = datetime.strptime(item["date"], "%Y-%m-%d").date()
        features = build_future_day_features(weather, history, location, date, index)
        prediction = predict_flood(features)
        results.append({
            "date": item["date"],
            "rainfall_mm": round(float(item["rainfall_mm"]), 2),
            "probability": prediction["probability"],
            "risk_band": prediction["risk_band"],
        })
    return results

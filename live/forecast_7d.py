"""
Seven-day live rainfall + flood-risk forecasting.

Open-Meteo supplies the observed/current conditions and the next seven
calendar days of precipitation. Model 2 is evaluated once per forecast day
using the same feature schema used by the trained classifier.

This module deliberately does not pretend that the historical locality data
supports a fitted daily ARIMA/SARIMA model. ARIMA/SARIMA remains the research
and long-horizon rainfall component; the live 1-7 day rainfall source is the
weather forecast API.
"""

from datetime import datetime
from typing import Callable

from live.live_features import build_future_day_features


def forecast_next_7_days(
    weather: dict,
    history: dict,
    location: dict,
    predict_flood: Callable[[dict], dict],
) -> list[dict]:
    """Return daily rainfall and flood-risk predictions for the next 7 days."""
    daily = weather.get("daily_forecast", [])
    results: list[dict] = []

    for index, item in enumerate(daily[:7]):
        date_text = item["date"]
        date = datetime.strptime(date_text, "%Y-%m-%d").date()
        features = build_future_day_features(
            weather,
            history,
            location,
            date=date,
            day_index=index,
        )
        prediction = predict_flood(features)
        results.append(
            {
                "date": date_text,
                "rainfall_mm": round(float(item["rainfall_mm"]), 2),
                "probability": prediction["probability"],
                "risk_band": prediction["risk_band"],
                "threshold_used": prediction["threshold_used"],
            }
        )

    return results

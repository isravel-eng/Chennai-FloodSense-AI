"""
live_prediction.py
-------------------
Live FloodSense prediction entry point.

Produces current risk, next-24-hour risk, and a seven-calendar-day risk
forecast. Daily rainfall for days 1-7 comes from Open-Meteo because the
historical locality series is irregular and is not sufficient to claim a
reliable fitted daily ARIMA/SARIMA forecast.
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.forecast_7d import forecast_next_7_days  # noqa: E402
from live.weather_api import get_weather_for_locality  # noqa: E402
from live.rainfall_history import get_recent_rainfall  # noqa: E402
from live.live_features import build_current_features, build_forecast_24h_features  # noqa: E402
from model_2_flood.predict_flood import predict_flood  # noqa: E402


def predict_live_flood(locality: str) -> dict:
    weather = get_weather_for_locality(locality)
    location = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "elevation_m_approx": weather["elevation_m_approx"],
    }

    month = datetime.now().month
    history = get_recent_rainfall(locality, month=month)

    current_features = build_current_features(weather, history, location, month)
    forecast_features = build_forecast_24h_features(weather, history, location, month)

    current_result = predict_flood(current_features)
    forecast_result = predict_flood(forecast_features)
    next_7_days = forecast_next_7_days(weather, history, location, predict_flood)

    return {
        "locality": weather["locality"],
        "updated_at": weather["fetched_at"],
        "current": {
            "rainfall_input_mm": current_features["rainfall_mm"],
            "probability": current_result["probability"],
            "risk_band": current_result["risk_band"],
        },
        "next_24h": {
            "forecast_rainfall_mm": forecast_features["rainfall_mm"],
            "probability": forecast_result["probability"],
            "risk_band": forecast_result["risk_band"],
        },
        "next_7_days": next_7_days,
        "context": {
            "rainfall_last_7d_mm": history["rainfall_7d_mm"],
            "rainfall_last_30d_mm": history["rainfall_30d_mm"],
            "rainfall_history_source": history["source"],
            "is_northeast_monsoon": month in (10, 11, 12),
        },
    }


def _pretty_print(result: dict):
    print(f"\n{result['locality'].upper()}")
    print(f"Updated: {result['updated_at']}\n")
    print("CURRENT RISK")
    print(f"  rainfall so far today : {result['current']['rainfall_input_mm']} mm")
    print(f"  probability           : {result['current']['probability']}")
    print(f"  risk band             : {result['current']['risk_band']}\n")
    print("NEXT 24H RISK")
    print(f"  forecast rainfall     : {result['next_24h']['forecast_rainfall_mm']} mm")
    print(f"  probability           : {result['next_24h']['probability']}")
    print(f"  risk band             : {result['next_24h']['risk_band']}\n")
    print("NEXT 7 DAYS")
    for day in result["next_7_days"]:
        print(
            f"  {day['date']} : {day['rainfall_mm']:.2f} mm | "
            f"probability={day['probability']} | {day['risk_band']}"
        )
    print("\nCONTEXT")
    print(f"  rainfall last 7 days  : {result['context']['rainfall_last_7d_mm']} mm")
    print(f"  rainfall last 30 days : {result['context']['rainfall_last_30d_mm']} mm")
    print(f"  history source        : {result['context']['rainfall_history_source']}")
    print(f"  monsoon season        : {result['context']['is_northeast_monsoon']}")


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    result = predict_live_flood(locality)
    _pretty_print(result)

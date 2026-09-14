import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.forecast_7d import forecast_next_7_days
from live.live_features import build_current_features, build_forecast_24h_features
from live.rainfall_history import get_recent_rainfall
from live.weather_api import get_weather_for_locality
from model_2_flood.predict_flood import predict_flood


def predict_live_flood(locality: str) -> dict:
    weather = get_weather_for_locality(locality)
    location = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
    }
    history = get_recent_rainfall(locality)

    current_features = build_current_features(weather, history, location)
    forecast_features = build_forecast_24h_features(weather, history, location)

    current_result = predict_flood(current_features)
    forecast_result = predict_flood(forecast_features)
    next_7_days = forecast_next_7_days(weather, history, location, predict_flood)

    return {
        "locality": weather["locality"],
        "updated_at": weather["fetched_at"],
        "current": current_result,
        "next_24h": forecast_result,
        "next_7_days": next_7_days,
        "context": {
            "rainfall_last_7d_mm": history.get("rainfall_7d_mm", 0),
            "rainfall_last_30d_mm": history.get("rainfall_30d_mm", 0),
        },
    }


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    print(predict_live_flood(locality))

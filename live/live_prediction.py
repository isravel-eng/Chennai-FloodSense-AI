import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.forecast_7d import forecast_next_7_days
from live.weather_api import get_weather_for_locality
from live.rainfall_history import get_recent_rainfall
from live.live_features import build_current_features, build_forecast_24h_features
from model_2_flood.predict_flood import predict_flood


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
        "next_7_days": forecast_next_7_days(weather, history, location, predict_flood),
        "context": {
            "rainfall_last_7d_mm": history["rainfall_7d_mm"],
            "rainfall_last_30d_mm": history["rainfall_30d_mm"],
            "rainfall_history_source": history["source"],
            "is_northeast_monsoon": month in (10, 11, 12),
        },
    }


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    print(predict_live_flood(locality))

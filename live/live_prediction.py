"""Live flood prediction pipeline for Chennai FloodSense AI.

Orchestrates:
  1. Live weather fetch (Open-Meteo)
  2. Rainfall observation persistence (PostgreSQL or CSV)
  3. Recent rainfall history retrieval (PostgreSQL or CSV + climatology fallback)
  4. Feature engineering
  5. Flood model inference (current + 24h forecast + 7-day forecast)
  6. Response assembly

The return dict shape is unchanged so all existing API endpoints continue
to work without modification.
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.forecast_7d import forecast_next_7_days
from live.weather_api import get_weather_for_locality
from live.rainfall_history import get_recent_rainfall, log_observation
from live.live_features import build_current_features, build_forecast_24h_features
from model_2_flood.predict_flood import predict_flood, RISK_BANDS


def predict_live_flood(locality: str) -> dict:
    """Run the full live flood prediction pipeline for a single locality.

    After fetching weather, today's current precipitation is persisted so the
    rainfall log grows automatically with each prediction request.
    """
    weather = get_weather_for_locality(locality)
    location = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "elevation_m_approx": weather["elevation_m_approx"],
    }
    month = datetime.now().month

    # Persist today's current precipitation observation.
    # This is fire-and-forget: a DB failure must not break the prediction.
    try:
        log_observation(
            locality=weather["locality"],
            rainfall_mm=float(weather.get("current_precipitation_mm") or 0.0),
        )
    except Exception:
        pass  # non-fatal — prediction continues with whatever history exists

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
            "risk_band_thresholds": {label: f"{lo:.2f} to {hi:.2f}" for lo, hi, label in RISK_BANDS},
        },
    }


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    print(predict_live_flood(locality))

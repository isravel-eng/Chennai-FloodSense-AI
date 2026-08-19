"""
live_prediction.py
-------------------
The main live entry point:

    from live.live_prediction import predict_live_flood
    result = predict_live_flood("Sholinganallur")

Produces TWO predictions, matching the architecture in README.md:

  current   : flood risk based on rainfall observed so far today
  next_24h  : flood risk based on next-24h forecast rainfall

Both share the same recent-rainfall history (3d/7d/30d + lags) and
location context; only the "rainfall_mm" slot differs between them.

This module requires internet access to Open-Meteo (see weather_api.py).
If you're running inside a sandboxed environment with restricted egress,
this call will raise a requests.exceptions.ConnectionError - that's a
network policy issue, not a bug in this code. Run it on a machine with
normal internet access.

Run standalone: python live/live_prediction.py "Sholinganallur"
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.weather_api import get_weather_for_locality  # noqa: E402
from live.rainfall_history import get_recent_rainfall  # noqa: E402
from live.live_features import build_current_features, build_forecast_24h_features  # noqa: E402
from model_2_flood.predict_flood import predict_flood  # noqa: E402


def predict_live_flood(locality: str) -> dict:
    # 1 & 2. Live weather (also gives us locality lat/lon/elevation)
    weather = get_weather_for_locality(locality)
    location = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "elevation_m_approx": weather["elevation_m_approx"],
    }

    # 3. Recent actual rainfall (log-backed, climatology fallback)
    month = datetime.now().month
    history = get_recent_rainfall(locality, month=month)

    # 4. Build both feature vectors
    current_features = build_current_features(weather, history, location, month)
    forecast_features = build_forecast_24h_features(weather, history, location, month)

    # 5. Run existing XGBoost model on each
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
    print("CONTEXT")
    print(f"  rainfall last 7 days  : {result['context']['rainfall_last_7d_mm']} mm")
    print(f"  rainfall last 30 days : {result['context']['rainfall_last_30d_mm']} mm")
    print(f"  history source        : {result['context']['rainfall_history_source']}")
    print(f"  monsoon season        : {result['context']['is_northeast_monsoon']}")


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    result = predict_live_flood(locality)
    _pretty_print(result)

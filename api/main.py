import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from live.live_prediction import predict_live_flood
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"

app = FastAPI(title="Chennai FloodSense AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/localities")
def get_localities():
    df = pd.read_csv(LOOKUP_PATH)
    localities = [
        {
            "name": row["locality"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "elevation_m_approx": float(row["elevation_m_approx"]),
        }
        for _, row in df.iterrows()
    ]
    return {"count": len(localities), "localities": localities}


@app.get("/api/v1/flood-risk/{locality}")
def flood_risk(locality: str):
    try:
        result = predict_live_flood(locality)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise HTTPException(status_code=503, detail="Weather service unavailable, try again shortly.")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/v1/flood-risk-all")
def flood_risk_all():
    """
    Returns flood risk for ALL localities in a single response.
    Runs the individual predictions in parallel threads so it's much
    faster than the frontend calling /flood-risk/{locality} 30 times.
    """
    df = pd.read_csv(LOOKUP_PATH)
    names = df["locality"].tolist()

    results = []

    def fetch_one(name):
        try:
            result = predict_live_flood(name)
            return {"name": name, "ok": True, "data": result}
        except Exception as e:
            return {"name": name, "ok": False, "error": str(e)}

    # Run up to 10 predictions at the same time instead of one-by-one
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_one, name) for name in names]
        for future in as_completed(futures):
            results.append(future.result())

    return {"count": len(results), "results": results}
RAINFALL_MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"


@app.get("/api/v1/rainfall-forecast")
def rainfall_forecast(months: int = 6):
    """
    City-wide (not per-locality) monthly rainfall research forecast,
    using Model 1 (SARIMA). This is NOT the same model used for
    flood-risk predictions - see model_1_rainfall/sarima.py notes.
    """
    if months < 1 or months > 24:
        raise HTTPException(status_code=422, detail="months must be between 1 and 24")

    try:
        model = SARIMAXResults.load(str(RAINFALL_MODEL_PATH))
        forecast = model.get_forecast(steps=months)
        mean = forecast.predicted_mean.round(1)
        ci = forecast.conf_int(alpha=0.05).round(1)

        predictions = []
        for i, (date, value) in enumerate(mean.items(), start=1):
            predictions.append({
                "month": i,
                "period": str(date.date()),
                "forecast_rainfall_mm": float(value),
                "lower_95_mm": float(ci.iloc[i - 1, 0]),
                "upper_95_mm": float(ci.iloc[i - 1, 1]),
            })

        return {
            "scope": "city_wide",
            "months_requested": months,
            "predictions": predictions,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Rainfall model file not found on server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")
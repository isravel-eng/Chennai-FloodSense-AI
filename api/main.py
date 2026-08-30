import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.live_prediction import predict_live_flood
from model_1_rainfall.locality_forecast import forecast_locality

LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"
RAINFALL_MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"

app = FastAPI(title="Chennai FloodSense AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/localities")
def get_localities():
    df = pd.read_csv(LOOKUP_PATH)
    localities = [{"name": row["locality"], "latitude": float(row["latitude"]), "longitude": float(row["longitude"]), "elevation_m_approx": float(row["elevation_m_approx"])} for _, row in df.iterrows()]
    return {"count": len(localities), "localities": localities}


def _predict(name: str):
    try:
        return {"name": name, "ok": True, "data": predict_live_flood(name)}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


@app.get("/api/v1/flood-risk/{locality}")
def flood_risk(locality: str):
    try:
        return predict_live_flood(locality)
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
    names = pd.read_csv(LOOKUP_PATH)["locality"].tolist()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_predict, names))
    return {"count": len(results), "results": results}


@app.get("/api/v1/daily-forecast/{locality}")
def daily_forecast(locality: str):
    """Live daily rainfall + flood-risk forecast for the next seven days."""
    result = flood_risk(locality)
    return {
        "locality": result["locality"],
        "updated_at": result["updated_at"],
        "forecast_source": "Open-Meteo daily forecast",
        "days": result.get("next_7_days", []),
    }


@app.get("/api/v1/rainfall-forecast/locality/{locality}")
def locality_rainfall_forecast(locality: str, months: int = 12):
    if months not in (12, 24, 36):
        raise HTTPException(status_code=422, detail="months must be 12, 24, or 36")
    try:
        return forecast_locality(locality, months)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Locality forecast failed: {str(e)}")


@app.get("/api/v1/rainfall-forecast")
def rainfall_forecast(months: int = 6):
    if months < 1 or months > 36:
        raise HTTPException(status_code=422, detail="months must be between 1 and 36")
    try:
        model = SARIMAXResults.load(str(RAINFALL_MODEL_PATH))
        forecast = model.get_forecast(steps=months)
        mean = forecast.predicted_mean.round(1)
        ci = forecast.conf_int(alpha=0.05).round(1)
        predictions = [{"month": i, "period": str(date.date()), "forecast_rainfall_mm": float(value), "lower_95_mm": float(ci.iloc[i - 1, 0]), "upper_95_mm": float(ci.iloc[i - 1, 1])} for i, (date, value) in enumerate(mean.items(), start=1)]
        return {"scope": "city_wide", "months_requested": months, "predictions": predictions}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Rainfall model file not found on server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

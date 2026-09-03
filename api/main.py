from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.live_prediction import predict_live_flood
from model_1_rainfall.locality_forecast import forecast_locality

LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"
RAINFALL_MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"

app = FastAPI(title="Chennai FloodSense AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _known_localities() -> list[str]:
    if not LOOKUP_PATH.exists():
        return []
    return pd.read_csv(LOOKUP_PATH)["locality"].dropna().astype(str).tolist()


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/localities")
def get_localities():
    if not LOOKUP_PATH.exists():
        raise HTTPException(status_code=500, detail="Locality lookup not found on server")
    try:
        df = pd.read_csv(LOOKUP_PATH)
        required = {"locality", "latitude", "longitude", "elevation_m_approx"}
        missing = required.difference(df.columns)
        if missing:
            raise HTTPException(status_code=500, detail=f"Locality lookup missing columns: {sorted(missing)}")
        localities = [
            {
                "name": str(row["locality"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "elevation_m_approx": float(row["elevation_m_approx"]),
            }
            for _, row in df.iterrows()
        ]
        return {"count": len(localities), "localities": localities}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load localities: {exc}")


def _predict(name: str):
    try:
        return {"name": name, "ok": True, "data": predict_live_flood(name)}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


@app.get("/api/v1/flood-risk/{locality}")
def flood_risk(locality: str):
    try:
        return predict_live_flood(locality)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise HTTPException(status_code=503, detail="Weather service unavailable. Check internet access and try again.")
    except requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Weather service request failed: {exc}")
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {str(exc)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.get("/api/v1/flood-risk-all")
def flood_risk_all():
    names = _known_localities()
    if not names:
        raise HTTPException(status_code=500, detail="No localities configured")
    with ThreadPoolExecutor(max_workers=min(8, len(names))) as executor:
        results = list(executor.map(_predict, names))
    ok = sum(1 for item in results if item["ok"])
    return {"count": len(results), "successful": ok, "failed": len(results) - ok, "results": results}


@app.get("/api/v1/daily-forecast/{locality}")
def daily_forecast(locality: str):
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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Locality forecast failed: {str(exc)}")


@app.get("/api/v1/rainfall-forecast")
def rainfall_forecast(months: int = 6):
    if months < 1 or months > 36:
        raise HTTPException(status_code=422, detail="months must be between 1 and 36")
    if not RAINFALL_MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Rainfall model file not found on server")
    try:
        model = SARIMAXResults.load(str(RAINFALL_MODEL_PATH))
        future = model.get_forecast(steps=months)
        mean = future.predicted_mean.round(1)
        ci = future.conf_int(alpha=0.05).round(1)
        predictions = []
        for i, (date, value) in enumerate(mean.items(), start=1):
            predictions.append(
                {
                    "month": i,
                    "period": str(date.date()),
                    "forecast_rainfall_mm": float(value),
                    "lower_95_mm": float(ci.iloc[i - 1, 0]),
                    "upper_95_mm": float(ci.iloc[i - 1, 1]),
                }
            )
        return {"scope": "city_wide", "months_requested": months, "predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(exc)}")

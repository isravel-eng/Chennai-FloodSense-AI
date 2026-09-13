"""Chennai FloodSense AI — FastAPI application.

Endpoints
---------
Public (no authentication required):
  GET  /api/v1/health
  GET  /api/v1/localities
  GET  /api/v1/flood-risk/{locality}
  GET  /api/v1/flood-risk-all
  GET  /api/v1/daily-forecast/{locality}
  GET  /api/v1/rainfall-forecast/locality/{locality}
  GET  /api/v1/rainfall-forecast

Authentication (uses Supabase Auth via anon client):
  POST /api/v1/auth/register   — create Supabase Auth account + profile row
  POST /api/v1/auth/login      — sign in, returns session tokens
  POST /api/v1/auth/logout     — invalidate session

User profile (requires valid JWT bearer token):
  GET  /api/v1/users/me        — return authenticated user's profile

Rainfall log (authenticated read; service-role write handled internally):
  GET  /api/v1/rainfall-logs/{locality}  — last N daily observations

Design decisions:
  - Flood-risk / forecast endpoints remain public so the map loads without login.
  - User-specific and write endpoints require auth.
  - CORS origin is read from FRONTEND_ORIGIN env var; defaults to "*" so local
    dev works without configuration.
  - Supabase credentials are never returned to the client.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.live_prediction import predict_live_flood
from model_1_rainfall.locality_forecast import forecast_locality

LOOKUP_PATH = ROOT / "data" / "processed" / "locality_lookup.csv"
RAINFALL_MODEL_PATH = ROOT / "models" / "rainfall_model.pkl"

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Chennai FloodSense AI API")

# CORS: tighten in production by setting FRONTEND_ORIGIN env var.
_allowed_origins_raw = os.environ.get("FRONTEND_ORIGIN", "*")
_allowed_origins = (
    ["*"]
    if _allowed_origins_raw.strip() == "*"
    else [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Supabase helpers (backend service-role client)
# ---------------------------------------------------------------------------

def _supabase_configured() -> bool:
    try:
        from backend.database import is_configured
        return is_configured()
    except Exception:
        return False


def _get_db():
    from backend.database import get_client
    return get_client()


def _verify_jwt(authorization: str) -> dict:
    """Verify a bearer token with Supabase and return the user dict.

    Raises HTTPException(401) on any failure.
    """
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    try:
        db = _get_db()
        response = db.auth.get_user(token)
        if response is None or response.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": response.user.id, "email": response.user.email}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}")


# ---------------------------------------------------------------------------
# Locality helpers
# ---------------------------------------------------------------------------

def _known_localities() -> list[str]:
    if not LOOKUP_PATH.exists():
        return []
    return pd.read_csv(LOOKUP_PATH)["locality"].dropna().astype(str).tolist()


def _known_locality_names_lower() -> set[str]:
    return {n.lower() for n in _known_localities()}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Localities
# ---------------------------------------------------------------------------

@app.get("/api/v1/localities")
def get_localities():
    if not LOOKUP_PATH.exists():
        raise HTTPException(status_code=500, detail="Locality lookup not found on server")
    try:
        df = pd.read_csv(LOOKUP_PATH)
        required = {"locality", "latitude", "longitude", "elevation_m_approx"}
        missing = required.difference(df.columns)
        if missing:
            raise HTTPException(
                status_code=500, detail=f"Locality lookup missing columns: {sorted(missing)}"
            )
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


# ---------------------------------------------------------------------------
# Flood risk (public — map loads without login)
# ---------------------------------------------------------------------------

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
        raise HTTPException(
            status_code=503,
            detail="Weather service unavailable. Check internet access and try again.",
        )
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
    return {
        "count": len(results),
        "successful": ok,
        "failed": len(results) - ok,
        "results": results,
    }


@app.get("/api/v1/daily-forecast/{locality}")
def daily_forecast(locality: str):
    result = flood_risk(locality)
    return {
        "locality": result["locality"],
        "updated_at": result["updated_at"],
        "forecast_source": "Open-Meteo daily forecast",
        "days": result.get("next_7_days", []),
    }


# ---------------------------------------------------------------------------
# Rainfall forecasts (public)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    native_locality: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/v1/auth/register", status_code=201)
def register(body: RegisterRequest):
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")

    # Validate locality against known list
    if body.native_locality.lower() not in _known_locality_names_lower():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown locality '{body.native_locality}'. "
            "Use GET /api/v1/localities to see supported options.",
        )

    db = _get_db()

    # 1. Create Supabase Auth account
    try:
        auth_resp = db.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
    except Exception as exc:
        error_msg = str(exc).lower()
        if "already registered" in error_msg or "already exists" in error_msg:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        raise HTTPException(status_code=422, detail=f"Registration failed: {exc}")

    if auth_resp is None or auth_resp.user is None:
        raise HTTPException(status_code=422, detail="Registration failed — Supabase returned no user.")

    user_id = auth_resp.user.id

    # 2. Insert application profile row
    try:
        db.table("users").insert(
            {
                "id": user_id,
                "name": body.name.strip(),
                "email": body.email.lower().strip(),
                "native_locality": body.native_locality,
            }
        ).execute()
    except Exception as exc:
        # Profile insert failed — attempt to clean up the auth account
        try:
            db.auth.admin.delete_user(user_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Account created but profile could not be saved: {exc}",
        )

    return {
        "id": user_id,
        "name": body.name.strip(),
        "email": body.email.lower().strip(),
        "native_locality": body.native_locality,
        "session": {
            "access_token": auth_resp.session.access_token if auth_resp.session else None,
            "refresh_token": auth_resp.session.refresh_token if auth_resp.session else None,
        },
    }


@app.post("/api/v1/auth/login")
def login(body: LoginRequest):
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")
    db = _get_db()
    try:
        resp = db.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if resp is None or resp.session is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Fetch application profile
    profile_resp = (
        db.table("users")
        .select("id, name, email, native_locality, created_at")
        .eq("id", resp.user.id)
        .single()
        .execute()
    )
    profile = profile_resp.data or {}

    return {
        "session": {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
            "expires_in": resp.session.expires_in,
        },
        "user": {
            "id": resp.user.id,
            "email": resp.user.email,
            "name": profile.get("name", ""),
            "native_locality": profile.get("native_locality", ""),
            "created_at": profile.get("created_at", ""),
        },
    }


@app.post("/api/v1/auth/logout")
def logout(authorization: str = Header(default="")):
    if not _supabase_configured():
        return {"status": "ok"}
    jwt = _verify_jwt(authorization)
    try:
        token = authorization.split(" ", 1)[1].strip()
        db = _get_db()
        db.auth.sign_out(token)
    except Exception:
        pass  # best-effort
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

@app.get("/api/v1/users/me")
def get_me(authorization: str = Header(default="")):
    jwt_user = _verify_jwt(authorization)
    db = _get_db()
    resp = (
        db.table("users")
        .select("id, name, email, native_locality, created_at")
        .eq("id", jwt_user["id"])
        .single()
        .execute()
    )
    if resp is None or resp.data is None:
        raise HTTPException(status_code=404, detail="User profile not found")
    return resp.data


# ---------------------------------------------------------------------------
# Rainfall log (authenticated read)
# ---------------------------------------------------------------------------

@app.get("/api/v1/rainfall-logs/{locality}")
def get_rainfall_log(locality: str, days: int = 30, authorization: str = Header(default="")):
    _verify_jwt(authorization)
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    db = _get_db()
    resp = (
        db.table("rainfall_logs")
        .select("observed_at, locality, rainfall_mm, source")
        .ilike("locality", locality)
        .order("observed_at", desc=True)
        .limit(days)
        .execute()
    )
    rows = resp.data or []
    rows_sorted = sorted(rows, key=lambda r: r["observed_at"])
    return {
        "locality": locality,
        "days_requested": days,
        "count": len(rows_sorted),
        "observations": rows_sorted,
    }

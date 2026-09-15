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

Authentication (uses Supabase Admin Auth via service-role client):
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
  - Registration uses auth.admin.create_user() (service-role path) so the
    backend has full control over email confirmation and the user UUID is
    available immediately.
  - Email confirmation behaviour is controlled by REQUIRE_EMAIL_CONFIRMATION
    env var (default "false" for demo/staging; set "true" for production).
  - CORS origin is read from FRONTEND_ORIGIN env var; defaults to "*" so local
    dev works without configuration.
  - Supabase credentials are never returned to the client.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Chennai FloodSense AI API")

# CORS: tighten in production by setting FRONTEND_ORIGIN env var.
# In local dev, leaving FRONTEND_ORIGIN unset (or "*") allows any origin.
# In production on Render: FRONTEND_ORIGIN=https://chennai-floodsense-ai-1.onrender.com
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
# Email confirmation configuration
# ---------------------------------------------------------------------------

def _email_confirm_enabled() -> bool:
    """Return True when production email-verification mode is active.

    Set REQUIRE_EMAIL_CONFIRMATION=true in the Render backend environment to
    require users to verify their email before they can log in.

    Default is False (demo/staging mode) — users are auto-confirmed and can
    log in immediately after registration.
    """
    return os.environ.get("REQUIRE_EMAIL_CONFIRMATION", "false").lower() == "true"


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

    Uses the service-role client's auth.get_user() which validates the JWT
    server-side without trusting any user-supplied ID.

    Raises HTTPException(401) on any failure.
    """
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")
    try:
        db = _get_db()
        response = db.auth.get_user(token)
        if response is None or response.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": str(response.user.id), "email": response.user.email}
    except HTTPException:
        raise
    except Exception as exc:
        # Log the real error server-side; return a safe message to the client.
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Token verification failed")


# ---------------------------------------------------------------------------
# Locality helpers
# ---------------------------------------------------------------------------

def _known_localities() -> list[str]:
    if not LOOKUP_PATH.exists():
        return []
    return pd.read_csv(LOOKUP_PATH)["locality"].dropna().astype(str).tolist()


def _known_locality_names_lower() -> set[str]:
    return {n.lower() for n in _known_localities()}


def _canonical_locality(name: str) -> str | None:
    """Return the canonical locality name (preserving original casing) or None."""
    lookup = {n.lower(): n for n in _known_localities()}
    return lookup.get(name.lower())


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
        if exc.response is not None and exc.response.status_code == 429:
            raise HTTPException(status_code=503, detail="Weather service rate limit exceeded. Please try again later.")
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
    with ThreadPoolExecutor(max_workers=min(4, len(names))) as executor:
        import time
        
        def _predict_with_delay(name: str):
            # Stagger startup slightly to avoid hitting rate limits instantly
            time.sleep(0.1 * names.index(name) % 4)
            return _predict(name)
            
        results = list(executor.map(_predict_with_delay, names))
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


@app.get("/api/v1/weather/{locality}")
def get_weather(locality: str):
    from live.weather_api import get_weather_for_locality
    try:
        return get_weather_for_locality(locality)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {exc}")


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


def _is_duplicate_email_error(exc: Exception) -> bool:
    """Detect Supabase duplicate-email errors from the admin API response."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in (
        "already registered",
        "already exists",
        "user already registered",
        "email address already registered",
        "duplicate",
    ))


@app.post("/api/v1/auth/register", status_code=201)
def register(body: RegisterRequest):
    """Create a Supabase Auth account and a public.users profile row.

    Registration flow:
      1. Validate inputs (locality, non-empty fields).
      2. Create the auth.users row via auth.admin.create_user() — the admin API
         bypasses email rate-limits and gives us full control over confirmation.
      3. Insert the public.users profile using the exact UUID returned by Auth.
      4. If profile insertion fails, delete the auth user to avoid orphans.

    Email confirmation behaviour (controlled by REQUIRE_EMAIL_CONFIRMATION env var):
      - false (default/demo): user is auto-confirmed → session returned immediately.
      - true  (production):   user must verify email → requires_verification=true,
                               no session in response.
    """
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")

    # --- Input validation ---
    name = body.name.strip()
    email = body.email.lower().strip()
    native_locality = body.native_locality.strip()

    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    if not email:
        raise HTTPException(status_code=422, detail="Email is required.")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")

    # --- Locality validation ---
    canonical = _canonical_locality(native_locality)
    if canonical is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown locality '{native_locality}'. "
                "Use GET /api/v1/localities to see supported options."
            ),
        )

    db = _get_db()
    confirm_immediately = not _email_confirm_enabled()

    # --- Step 1: Create Supabase Auth user via Admin API ---
    # Using admin.create_user() instead of auth.sign_up() because:
    #   - The admin API uses the service-role key path — no anon rate limits.
    #   - email_confirm parameter gives explicit control over confirmation state.
    #   - The returned user.id is immediately usable for the FK insert.
    auth_user = None
    try:
        resp = db.auth.admin.create_user(
            {
                "email": email,
                "password": body.password,
                "email_confirm": confirm_immediately,
            }
        )
        auth_user = resp.user
    except Exception as exc:
        if _is_duplicate_email_error(exc):
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists.",
            )
        # Log full error server-side, return safe message to client.
        logger.error("Auth user creation failed for %s: %s", email, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Registration failed. Please check your details and try again. Error: {str(exc)}",
        )

    if auth_user is None:
        raise HTTPException(
            status_code=500,
            detail="Registration failed — Auth service returned no user.",
        )

    user_id = str(auth_user.id)

    # --- Step 2: Insert public.users profile ---
    # The service-role client bypasses RLS, so the insert succeeds even before
    # the user has a session. The FK references auth.users(id) which is now
    # committed because admin.create_user() returned successfully.
    try:
        db.table("users").insert(
            {
                "id": user_id,
                "name": name,
                "email": email,
                "native_locality": canonical,
            }
        ).execute()
    except Exception as exc:
        # Profile insert failed — clean up the auth user to prevent orphans.
        try:
            db.auth.admin.delete_user(user_id)
            logger.info("Cleaned up orphaned auth user %s after profile insert failure", user_id)
        except Exception as cleanup_exc:
            logger.error("Failed to clean up auth user %s: %s", user_id, cleanup_exc)

        logger.error("Profile insert failed for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Account setup failed. Please try registering again.",
        )

    # --- Step 3: Return response ---
    if confirm_immediately:
        # Auto-confirmed: sign the user in to get a session token.
        # We sign in here because admin.create_user() doesn't return a session.
        try:
            sign_in_resp = db.auth.sign_in_with_password(
                {"email": email, "password": body.password}
            )
            session_data = {
                "access_token": sign_in_resp.session.access_token,
                "refresh_token": sign_in_resp.session.refresh_token,
                "expires_in": sign_in_resp.session.expires_in,
            } if sign_in_resp and sign_in_resp.session else None
        except Exception as exc:
            logger.warning("Post-registration sign-in failed for %s: %s", email, exc)
            session_data = None

        return {
            "id": user_id,
            "name": name,
            "email": email,
            "native_locality": canonical,
            "requires_verification": False,
            "session": session_data,
        }
    else:
        # Email confirmation required — no session yet.
        return {
            "id": user_id,
            "name": name,
            "email": email,
            "native_locality": canonical,
            "requires_verification": True,
            "session": None,
        }


@app.post("/api/v1/auth/login")
def login(body: LoginRequest):
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Authentication service not configured")
    db = _get_db()
    try:
        resp = db.auth.sign_in_with_password(
            {"email": body.email.lower().strip(), "password": body.password}
        )
    except Exception as exc:
        logger.info("Login failed for %s: %s", body.email, exc)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if resp is None or resp.session is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Fetch application profile using service-role client (bypasses RLS)
    try:
        service_db = _get_db()
        profile_resp = (
            service_db.table("users")
            .select("id, name, email, native_locality, created_at")
            .eq("id", str(resp.user.id))
            .single()
            .execute()
        )
        profile = profile_resp.data or {}
    except Exception as exc:
        logger.warning("Could not fetch profile for %s after login: %s", resp.user.id, exc)
        profile = {}

    return {
        "session": {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
            "expires_in": resp.session.expires_in,
        },
        "user": {
            "id": str(resp.user.id),
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
    # Verify the token before signing out — best effort.
    try:
        _verify_jwt(authorization)
    except HTTPException:
        return {"status": "ok"}
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
    """Return the authenticated user's profile.

    Verifies the bearer token server-side — never trusts a user-supplied ID.
    """
    jwt_user = _verify_jwt(authorization)
    db = _get_db()
    try:
        resp = (
            db.table("users")
            .select("id, name, email, native_locality, created_at")
            .eq("id", jwt_user["id"])
            .single()
            .execute()
        )
    except Exception as exc:
        logger.error("Profile fetch failed for %s: %s", jwt_user["id"], exc)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profile: {str(exc)}")

    if resp is None or resp.data is None:
        raise HTTPException(status_code=404, detail="User profile not found")
    return resp.data


# ---------------------------------------------------------------------------
# Rainfall log (authenticated read)
# ---------------------------------------------------------------------------

@app.get("/api/v1/rainfall-logs/{locality}")
def get_rainfall_log(locality: str, days: int = 30, authorization: str = Header(default="")):
    """Return the last N daily rainfall observations for a locality.

    Requires a valid bearer token — rainfall history is considered user data.
    The service-role client is used for the read so RLS is bypassed, allowing
    the backend to return data for any locality (not just the authenticated
    user's locality).
    """
    _verify_jwt(authorization)
    if not _supabase_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    db = _get_db()
    try:
        resp = (
            db.table("rainfall_logs")
            .select("observed_at, locality, rainfall_mm, source")
            .ilike("locality", locality)
            .order("observed_at", desc=True)
            .limit(days)
            .execute()
        )
    except Exception as exc:
        logger.error("Rainfall log fetch failed for %s: %s", locality, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve rainfall logs")

    rows = resp.data or []
    rows_sorted = sorted(rows, key=lambda r: r["observed_at"])
    return {
        "locality": locality,
        "days_requested": days,
        "count": len(rows_sorted),
        "observations": rows_sorted,
    }

"""Rainfall history module for the Chennai FloodSense AI pipeline.

Provides:
  - get_recent_rainfall()  — retrieve rolling rainfall stats for a locality
  - get_locality_climatology() — fallback using static master dataset

Storage backend selection
-------------------------
When SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set, observations are
read from (and written to) PostgreSQL via backend.repositories.rainfall_repository.

When those environment variables are absent (e.g., local development without
Supabase), the legacy CSV file is used exactly as before.  This means the
existing behaviour is preserved without any code changes to callers.

LOG_COLUMNS is kept for backward compatibility with locality_forecast.py.
LIVE_LOG_PATH is kept so that code that checks path.exists() does not break.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
LIVE_LOG_PATH = ROOT / "data" / "processed" / "live_rainfall_log.csv"
LOG_COLUMNS = ["date", "locality", "rainfall_mm"]


# ---------------------------------------------------------------------------
# Internal: decide which storage backend to use
# ---------------------------------------------------------------------------

def _use_postgres() -> bool:
    """Return True when the Supabase environment is configured."""
    try:
        from backend.database import is_configured
        return is_configured()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CSV-based log (legacy / local-dev fallback)
# ---------------------------------------------------------------------------

def _ensure_log_exists():
    if not LIVE_LOG_PATH.exists():
        LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LIVE_LOG_PATH, index=False)


class RainfallLog:
    """CSV-backed rainfall log — used only when PostgreSQL is not configured."""

    def __init__(self, path: Path = LIVE_LOG_PATH):
        self.path = path
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=LOG_COLUMNS).to_csv(self.path, index=False)

    def append(self, locality: str, rainfall_mm: float, day: date = None):
        day = day or date.today()
        pd.DataFrame(
            [{"date": day.isoformat(), "locality": locality, "rainfall_mm": rainfall_mm}]
        ).to_csv(self.path, mode="a", header=False, index=False)

    def read(self, locality: str) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["date"])
        return df[df["locality"].str.lower() == locality.lower()].sort_values("date")


# ---------------------------------------------------------------------------
# Public API: upsert an observation
# ---------------------------------------------------------------------------

def log_observation(locality: str, rainfall_mm: float, day: date = None) -> None:
    """Persist one daily rainfall observation.

    Writes to PostgreSQL when configured; appends to the CSV otherwise.
    Safe to call with the same (locality, day) more than once — both backends
    handle duplicates gracefully (PostgreSQL: UPSERT; CSV: append, and the
    reading code de-duplicates by keeping the last value).
    """
    day = day or date.today()
    if _use_postgres():
        try:
            from backend.repositories import rainfall_repository
            rainfall_repository.upsert(locality, day, rainfall_mm, source="open_meteo")
        except Exception as exc:
            # Non-fatal: log the error but do not interrupt the prediction.
            import traceback
            traceback.print_exc()
    else:
        RainfallLog().append(locality, rainfall_mm, day)


# ---------------------------------------------------------------------------
# Public API: read history
# ---------------------------------------------------------------------------

def _read_postgres(locality: str) -> pd.DataFrame:
    from backend.repositories import rainfall_repository
    return rainfall_repository.read(locality)


def _read_csv(locality: str) -> pd.DataFrame:
    return RainfallLog().read(locality)


# ---------------------------------------------------------------------------
# Climatology fallback
# ---------------------------------------------------------------------------

def get_locality_climatology(locality: str, month: int) -> dict:
    """Return median rainfall stats from the static training dataset.

    Used when there are insufficient live observations for a locality.
    Preserved exactly from the original implementation.
    """
    df = pd.read_csv(MASTER_PATH)
    subset = df[(df["locality"].str.lower() == locality.lower()) & (df["month"] == month)]
    if subset.empty:
        subset = df[df["month"] == month]
    median = float(subset["rainfall_mm"].median())
    return {
        "rainfall_mm": median,
        "rainfall_3d_mm": float(subset["rainfall_3d_mm"].median()),
        "rainfall_7d_mm": float(subset["rainfall_7d_mm"].median()),
        "rainfall_30d_mm": float(subset["rainfall_30d_mm"].median()),
        "rainfall_lag_1": median,
        "rainfall_lag_2": median,
        "rainfall_lag_3": median,
        "rainfall_lag_7": median,
        "source": "climatology_fallback",
    }


# ---------------------------------------------------------------------------
# Main entry point used by live_prediction.py
# ---------------------------------------------------------------------------

def get_recent_rainfall(locality: str, month: int = None, min_log_days: int = 30) -> dict:
    """Return rolling rainfall statistics for a locality.

    Reads from PostgreSQL when configured, CSV otherwise.  Falls back to
    climatology from the static master dataset when fewer than min_log_days
    observations are available.

    The returned dict has the same keys as the original implementation so
    live_features.py requires no changes.
    """
    month = month or datetime.now().month
    try:
        if _use_postgres():
            entries = _read_postgres(locality)
        else:
            entries = _read_csv(locality)
    except Exception:
        return get_locality_climatology(locality, month)

    if len(entries) >= min_log_days:
        return {
            "rainfall_mm": float(entries.iloc[-1]["rainfall_mm"]),
            "rainfall_3d_mm": float(entries.tail(3)["rainfall_mm"].sum()),
            "rainfall_7d_mm": float(entries.tail(7)["rainfall_mm"].sum()),
            "rainfall_30d_mm": float(entries.tail(30)["rainfall_mm"].sum()),
            "rainfall_lag_1": float(entries.iloc[-2]["rainfall_mm"]),
            "rainfall_lag_2": float(entries.iloc[-3]["rainfall_mm"]),
            "rainfall_lag_3": float(entries.iloc[-4]["rainfall_mm"]),
            "rainfall_lag_7": float(entries.iloc[-8]["rainfall_mm"]),
            "source": "live_log",
        }
    return get_locality_climatology(locality, month)

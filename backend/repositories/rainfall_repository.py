"""Rainfall data-access layer — PostgreSQL (Supabase) implementation.

This module provides the same logical interface as the original RainfallLog
class in live/rainfall_history.py but persists data in PostgreSQL rather than
a local CSV file.

Column mapping (CSV → PostgreSQL):
    date        → observed_at  (type: date — daily granularity, matching model)
    locality    → locality
    rainfall_mm → rainfall_mm

The returned DataFrames always have columns [date, locality, rainfall_mm] with
the same dtypes/sort order as the CSV reader so the rest of the ML pipeline
requires zero changes.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from backend.database import get_client


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert(
    locality: str,
    observed_at: date,
    rainfall_mm: float,
    source: str = "open_meteo",
) -> None:
    """Insert or update a daily rainfall observation.

    Uses ON CONFLICT DO UPDATE so re-running the same day's prediction does
    not create duplicate rows.  The service-role client bypasses RLS.
    """
    client = get_client()
    client.table("rainfall_logs").upsert(
        {
            "locality": locality,
            "observed_at": observed_at.isoformat(),
            "rainfall_mm": float(rainfall_mm),
            "source": source,
        },
        on_conflict="locality,observed_at",
    ).execute()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read(locality: str, limit: int = 365) -> pd.DataFrame:
    """Fetch rainfall observations for a locality, newest-first then reversed.

    Returns a DataFrame with columns [date, locality, rainfall_mm] sorted
    ascending by date — the same shape that RainfallLog.read() returns from
    the CSV.

    Parameters
    ----------
    locality:
        Exact locality name as stored (case-insensitive query).
    limit:
        Maximum number of rows to fetch.  365 days is more than enough for
        the ML pipeline which requires at most 30-day rolling windows plus
        lag features up to 7 days back (38 rows minimum for live_log path).
    """
    client = get_client()
    response = (
        client.table("rainfall_logs")
        .select("observed_at, locality, rainfall_mm")
        .ilike("locality", locality)
        .order("observed_at", desc=False)
        .limit(limit)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=["date", "locality", "rainfall_mm"])

    df = pd.DataFrame(rows)
    # Normalise column names to match the CSV interface expected by the rest
    # of the pipeline.
    df = df.rename(columns={"observed_at": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "locality", "rainfall_mm"]]


# ---------------------------------------------------------------------------
# Read for locality_forecast (month-start aggregated series)
# ---------------------------------------------------------------------------

def read_as_monthly_series(locality: str) -> pd.DataFrame:
    """Return daily observations as a DataFrame suitable for SARIMA fitting.

    locality_forecast.py expects a DataFrame with columns [date, locality,
    rainfall_mm] where date can be either daily or month-start — it resamples
    internally.  This function returns the same shape as read() so the
    existing _read_history_file() helper works without changes.
    """
    return read(locality, limit=3650)  # up to 10 years of daily data

"""
rainfall_history.py
--------------------
Answers: "What rainfall has this locality actually received recently?"

Model 2 needs rainfall_3d_mm / rainfall_7d_mm / rainfall_30d_mm and
rainfall_lag_1/2/3/7 - all of which describe ACTUAL PAST rainfall, not
forecast. In a live deployment these should ideally come from a small
local database that is updated once per day (e.g. a cron job appending
today's observed rainfall). Until that pipeline exists, this module
provides two things:

1. `get_locality_climatology(locality, month)` - a seasonal fallback drawn
   from the 1993-2023 master_dataset.csv: the historical median rainfall
   for that locality in that calendar month. Used when no live daily log
   is available yet, so the live app still returns a sane answer instead
   of crashing.

2. `RainfallLog` - a tiny append-only CSV-backed log
   (data/processed/live_rainfall_log.csv) that a scheduled job can append
   to daily with real observed rainfall (from the weather API's
   `current_precipitation_mm`, summed over the day, or a rain-gauge feed).
   Once at least 30 days of real entries exist for a locality, computations
   automatically switch from climatology to the real rolling log.

Run standalone: python live/rainfall_history.py "Sholinganallur"
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
LIVE_LOG_PATH = ROOT / "data" / "processed" / "live_rainfall_log.csv"

LOG_COLUMNS = ["date", "locality", "rainfall_mm"]


def _ensure_log_exists():
    if not LIVE_LOG_PATH.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LIVE_LOG_PATH, index=False)


class RainfallLog:
    """Append-only per-day, per-locality observed rainfall log."""

    def __init__(self, path: Path = LIVE_LOG_PATH):
        self.path = path
        if not self.path.exists():
            pd.DataFrame(columns=LOG_COLUMNS).to_csv(self.path, index=False)

    def append(self, locality: str, rainfall_mm: float, day: date = None):
        day = day or date.today()
        row = pd.DataFrame([{
            "date": day.isoformat(),
            "locality": locality,
            "rainfall_mm": rainfall_mm,
        }])
        row.to_csv(self.path, mode="a", header=False, index=False)

    def read(self, locality: str) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["date"])
        return df[df["locality"].str.lower() == locality.lower()].sort_values("date")


def get_locality_climatology(locality: str, month: int) -> dict:
    """
    Historical median daily rainfall, and the rolled-up 3d/7d/30d medians,
    for `locality` in calendar `month`, computed from master_dataset.csv.
    Used as a fallback until enough live log data accumulates.
    """
    df = pd.read_csv(MASTER_PATH)
    subset = df[
        (df["locality"].str.lower() == locality.lower()) & (df["month"] == month)
    ]
    if subset.empty:
        subset = df[df["month"] == month]  # citywide fallback

    return {
        "rainfall_mm": float(subset["rainfall_mm"].median()),
        "rainfall_3d_mm": float(subset["rainfall_3d_mm"].median()),
        "rainfall_7d_mm": float(subset["rainfall_7d_mm"].median()),
        "rainfall_30d_mm": float(subset["rainfall_30d_mm"].median()),
        "rainfall_lag_1": float(subset["rainfall_mm"].median()),
        "rainfall_lag_2": float(subset["rainfall_mm"].median()),
        "rainfall_lag_3": float(subset["rainfall_mm"].median()),
        "rainfall_lag_7": float(subset["rainfall_mm"].median()),
        "source": "climatology_fallback",
    }


def get_recent_rainfall(locality: str, month: int = None, min_log_days: int = 30) -> dict:
    """
    Preferred entry point: uses the live log if it has enough history for
    this locality, otherwise falls back to seasonal climatology.
    """
    month = month or datetime.now().month
    log = RainfallLog()
    entries = log.read(locality)

    if len(entries) >= min_log_days:
        entries = entries.sort_values("date")
        last_30 = entries.tail(30)
        today_idx = len(entries) - 1
        result = {
            "rainfall_mm": float(entries.iloc[today_idx]["rainfall_mm"]),
            "rainfall_3d_mm": float(entries.tail(3)["rainfall_mm"].sum()),
            "rainfall_7d_mm": float(entries.tail(7)["rainfall_mm"].sum()),
            "rainfall_30d_mm": float(last_30["rainfall_mm"].sum()),
            "rainfall_lag_1": float(entries.iloc[-2]["rainfall_mm"]) if len(entries) >= 2 else 0.0,
            "rainfall_lag_2": float(entries.iloc[-3]["rainfall_mm"]) if len(entries) >= 3 else 0.0,
            "rainfall_lag_3": float(entries.iloc[-4]["rainfall_mm"]) if len(entries) >= 4 else 0.0,
            "rainfall_lag_7": float(entries.iloc[-8]["rainfall_mm"]) if len(entries) >= 8 else 0.0,
            "source": "live_log",
        }
        return result

    return get_locality_climatology(locality, month)


if __name__ == "__main__":
    locality = sys.argv[1] if len(sys.argv) > 1 else "Sholinganallur"
    result = get_recent_rainfall(locality)
    for k, v in result.items():
        print(f"{k}: {v}")

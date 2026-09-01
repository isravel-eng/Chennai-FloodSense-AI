import sys
from datetime import date, datetime
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
    def __init__(self, path: Path = LIVE_LOG_PATH):
        self.path = path
        if not self.path.exists(): pd.DataFrame(columns=LOG_COLUMNS).to_csv(self.path, index=False)
    def append(self, locality: str, rainfall_mm: float, day: date = None):
        day = day or date.today()
        pd.DataFrame([{"date": day.isoformat(), "locality": locality, "rainfall_mm": rainfall_mm}]).to_csv(self.path, mode="a", header=False, index=False)
    def read(self, locality: str) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["date"])
        return df[df["locality"].str.lower() == locality.lower()].sort_values("date")

def get_locality_climatology(locality: str, month: int) -> dict:
    df = pd.read_csv(MASTER_PATH)
    subset = df[(df["locality"].str.lower() == locality.lower()) & (df["month"] == month)]
    if subset.empty: subset = df[df["month"] == month]
    median = float(subset["rainfall_mm"].median())
    return {"rainfall_mm": median, "rainfall_3d_mm": float(subset["rainfall_3d_mm"].median()), "rainfall_7d_mm": float(subset["rainfall_7d_mm"].median()), "rainfall_30d_mm": float(subset["rainfall_30d_mm"].median()), "rainfall_lag_1": median, "rainfall_lag_2": median, "rainfall_lag_3": median, "rainfall_lag_7": median, "source": "climatology_fallback"}

def get_recent_rainfall(locality: str, month: int = None, min_log_days: int = 30) -> dict:
    month = month or datetime.now().month
    entries = RainfallLog().read(locality)
    if len(entries) >= min_log_days:
        return {"rainfall_mm": float(entries.iloc[-1]["rainfall_mm"]), "rainfall_3d_mm": float(entries.tail(3)["rainfall_mm"].sum()), "rainfall_7d_mm": float(entries.tail(7)["rainfall_mm"].sum()), "rainfall_30d_mm": float(entries.tail(30)["rainfall_mm"].sum()), "rainfall_lag_1": float(entries.iloc[-2]["rainfall_mm"]), "rainfall_lag_2": float(entries.iloc[-3]["rainfall_mm"]), "rainfall_lag_3": float(entries.iloc[-4]["rainfall_mm"]), "rainfall_lag_7": float(entries.iloc[-8]["rainfall_mm"]), "source": "live_log"}
    return get_locality_climatology(locality, month)

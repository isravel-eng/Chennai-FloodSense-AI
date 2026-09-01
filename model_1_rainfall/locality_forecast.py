"""Locality-wise long-term monthly SARIMA forecasting (12/24/36 months)."""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
CANDIDATE_ORDERS = [(1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 0), (2, 1, 1)]
CANDIDATE_SEASONAL = [(1, 1, 1, 12), (0, 1, 1, 12), (1, 1, 0, 12)]
MIN_MONTHS = 24


def load_locality_monthly(locality: str, data_path: Path = DATA_PATH) -> pd.Series:
    df = pd.read_csv(data_path)
    if not {"locality", "rainfall_mm"}.issubset(df.columns):
        raise ValueError("Dataset must contain locality and rainfall_mm")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif {"year", "month"}.issubset(df.columns):
        dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    else:
        raise ValueError("Dataset needs date or year/month columns")
    mask = df["locality"].astype(str).str.lower().eq(locality.lower())
    subset = df.loc[mask].copy()
    subset["date"] = dates.loc[subset.index]
    subset["rainfall_mm"] = pd.to_numeric(subset["rainfall_mm"], errors="coerce")
    subset = subset.dropna(subset=["date", "rainfall_mm"])
    if subset.empty:
        raise ValueError(f"Unknown locality or no rainfall history: {locality}")
    return subset.set_index("date")["rainfall_mm"].resample("MS").sum(min_count=1)


def _select_model(series: pd.Series):
    train = series.dropna()
    if len(train) < MIN_MONTHS:
        return None
    best = None
    for order in CANDIDATE_ORDERS:
        for seasonal in CANDIDATE_SEASONAL:
            try:
                fit = SARIMAX(train, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                candidate = (float(fit.aic), order, seasonal, fit)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            except Exception:
                continue
    return best


def forecast_locality(locality: str, horizon_months: int = 12, data_path: Path = DATA_PATH) -> dict:
    if horizon_months not in (12, 24, 36):
        raise ValueError("horizon_months must be 12, 24, or 36")
    monthly = load_locality_monthly(locality, data_path)
    observed = monthly.dropna()
    if len(observed) < MIN_MONTHS:
        return {"locality": locality, "status": "insufficient_historical_data", "message": "Insufficient historical data for locality-specific forecasting", "observed_months": int(len(observed)), "horizon_months": horizon_months, "forecast": []}
    selected = _select_model(monthly)
    if selected is None:
        return {"locality": locality, "status": "model_fit_failed", "message": "No supported SARIMA specification could be fitted", "observed_months": int(len(observed)), "horizon_months": horizon_months, "forecast": []}
    aic, order, seasonal, fit = selected
    future = fit.get_forecast(steps=horizon_months)
    ci = future.conf_int(alpha=0.05)
    mean = np.maximum(np.asarray(future.predicted_mean, dtype=float), 0.0)
    lower = np.maximum(np.asarray(ci.iloc[:, 0], dtype=float), 0.0)
    upper = np.maximum(np.asarray(ci.iloc[:, 1], dtype=float), 0.0)
    rows = [{"month": ts.strftime("%Y-%m"), "forecast_mm": round(float(mean[i]), 2), "lower_95_mm": round(float(lower[i]), 2), "upper_95_mm": round(float(upper[i]), 2)} for i, ts in enumerate(future.predicted_mean.index)]
    return {"locality": locality, "status": "ok", "message": "Locality-specific SARIMA forecast", "observed_months": int(len(observed)), "first_observation": str(observed.index.min().date()), "last_observation": str(observed.index.max().date()), "model": {"name": "SARIMA", "order": list(order), "seasonal_order": list(seasonal), "aic": round(aic, 2)}, "horizon_months": horizon_months, "forecast": rows, "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2)}

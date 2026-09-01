"""Locality-wise monthly rainfall forecasting.

Uses SARIMA when enough history is available and a deterministic linear-trend
fallback when a locality has a short history or SARIMA cannot be fitted.
"""

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
    monthly = subset.set_index("date")["rainfall_mm"].resample("MS").sum(min_count=1)
    monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
    return monthly


def _select_model(series: pd.Series):
    train = series.dropna().copy()
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


def _fallback_forecast(observed: pd.Series, horizon_months: int) -> list[dict]:
    """Forecast with a simple non-negative linear trend for short/failed histories."""
    y = observed.dropna().astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    if len(y) == 1:
        slope, intercept = 0.0, y[0]
    else:
        slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual_std = float(np.std(y - fitted, ddof=1)) if len(y) > 2 else max(float(np.mean(y)) * 0.25, 1.0)
    future_x = np.arange(len(y), len(y) + horizon_months, dtype=float)
    values = np.maximum(intercept + slope * future_x, 0.0)
    margin = max(1.96 * residual_std, 1.0)
    start = observed.index.max() + pd.offsets.MonthBegin(1)
    dates = pd.date_range(start=start, periods=horizon_months, freq="MS")
    return [{"month": ts.strftime("%Y-%m"), "forecast_mm": round(float(values[i]), 2), "lower_95_mm": round(float(max(values[i] - margin, 0.0)), 2), "upper_95_mm": round(float(values[i] + margin), 2)} for i, ts in enumerate(dates)]


def forecast_locality(locality: str, horizon_months: int = 12, data_path: Path = DATA_PATH) -> dict:
    if horizon_months not in (12, 24, 36):
        raise ValueError("horizon_months must be 12, 24, or 36")
    monthly = load_locality_monthly(locality, data_path)
    observed = monthly.dropna()

    # Short histories cannot support a reliable 12-month seasonal SARIMA fit.
    # Still return a useful forecast graph instead of failing the locality page.
    if len(observed) < MIN_MONTHS:
        rows = _fallback_forecast(observed, horizon_months)
        return {"locality": locality, "status": "fallback_forecast", "message": "Limited historical data; showing a simple trend forecast", "observed_months": int(len(observed)), "horizon_months": horizon_months, "model": {"name": "Trend fallback"}, "forecast": rows, "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2)}

    selected = _select_model(monthly)
    if selected is None:
        rows = _fallback_forecast(observed, horizon_months)
        return {"locality": locality, "status": "fallback_forecast", "message": "SARIMA could not be fitted; showing a simple trend forecast", "observed_months": int(len(observed)), "horizon_months": horizon_months, "model": {"name": "Trend fallback"}, "forecast": rows, "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2)}

    aic, order, seasonal, fit = selected
    try:
        future = fit.get_forecast(steps=horizon_months)
        ci = future.conf_int(alpha=0.05)
        mean = np.maximum(np.asarray(future.predicted_mean, dtype=float), 0.0)
        lower = np.maximum(np.asarray(ci.iloc[:, 0], dtype=float), 0.0)
        upper = np.maximum(np.asarray(ci.iloc[:, 1], dtype=float), 0.0)
        dates = pd.date_range(start=observed.index.max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
        rows = [{"month": dates[i].strftime("%Y-%m"), "forecast_mm": round(float(mean[i]), 2), "lower_95_mm": round(float(lower[i]), 2), "upper_95_mm": round(float(upper[i]), 2)} for i in range(horizon_months)]
    except Exception:
        rows = _fallback_forecast(observed, horizon_months)
        return {"locality": locality, "status": "fallback_forecast", "message": "SARIMA forecast index was unavailable; showing a simple trend forecast", "observed_months": int(len(observed)), "horizon_months": horizon_months, "model": {"name": "Trend fallback"}, "forecast": rows, "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2)}

    return {"locality": locality, "status": "ok", "message": "Locality-specific SARIMA forecast", "observed_months": int(len(observed)), "first_observation": str(observed.index.min().date()), "last_observation": str(observed.index.max().date()), "model": {"name": "SARIMA", "order": list(order), "seasonal_order": list(seasonal), "aic": round(aic, 2)}, "horizon_months": horizon_months, "forecast": rows, "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2)}

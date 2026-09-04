"""Locality-wise monthly rainfall forecasting.

Uses SARIMA when enough history is available and a deterministic linear-trend
fallback when a locality has a short history or SARIMA cannot be fitted.

The forecast timeline is anchored to the current calendar month. Historical
observations are still used for model fitting, but the returned forecast months
always begin in the month in which the request is made.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "master_dataset.csv"
LIVE_LOG_PATH = ROOT / "data" / "processed" / "live_rainfall_log.csv"
CANDIDATE_ORDERS = [(1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 0), (2, 1, 1)]
CANDIDATE_SEASONAL = [(1, 1, 1, 12), (0, 1, 1, 12), (1, 1, 0, 12)]
MIN_MONTHS = 24


def _read_history_file(path: Path, *, required: bool = False) -> pd.DataFrame:
    """Read a rainfall history file without exposing server filesystem paths."""
    if not path.exists():
        if required:
            raise RuntimeError("Required rainfall history dataset is missing")
        return pd.DataFrame(columns=["date", "locality", "rainfall_mm"])

    df = pd.read_csv(path)
    if not {"locality", "rainfall_mm"}.issubset(df.columns):
        raise RuntimeError("Rainfall history dataset has invalid columns")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif {"year", "month"}.issubset(df.columns):
        dates = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=1), errors="coerce"
        )
    else:
        raise RuntimeError("Rainfall history dataset needs date or year/month columns")

    df = df.copy()
    df["date"] = dates
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    return df.dropna(subset=["date", "rainfall_mm"])


def load_locality_monthly(locality: str, data_path: Path = DATA_PATH) -> pd.Series:
    """Load historical rainfall and append any newer live observations.

    The static dataset is required. Live observations are optional. They are
    combined for model fitting, while the forecast output timeline is anchored
    independently to the current calendar month.
    """
    base = _read_history_file(data_path, required=True)
    live = _read_history_file(LIVE_LOG_PATH)
    frames = [base]
    if not live.empty:
        frames.append(live)
    df = pd.concat(frames, ignore_index=True)

    mask = df["locality"].astype(str).str.lower().eq(locality.lower())
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"Unknown locality or no rainfall history: {locality}")

    # Preserve concat order (base first, live second) so that when the same
    # locality/date exists in both sources, the live value wins deterministically.
    subset = subset.sort_values("date", kind="mergesort").drop_duplicates(
        subset=["date"], keep="last"
    )
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
                fit = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                candidate = (float(fit.aic), order, seasonal, fit)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            except Exception:
                continue
    return best


def _current_month_start() -> pd.Timestamp:
    """Return the first day of the current calendar month."""
    return pd.Timestamp.today().to_period("M").to_timestamp()


def _forecast_dates(horizon_months: int) -> pd.DatetimeIndex:
    """Build forecast labels from the current month for the requested horizon."""
    return pd.date_range(
        start=_current_month_start(),
        periods=horizon_months,
        freq="MS",
    )


def _fallback_forecast(observed: pd.Series, horizon_months: int) -> list[dict]:
    """Forecast with a simple non-negative linear trend for short/failed histories."""
    y = observed.dropna().astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    if len(y) == 1:
        slope, intercept = 0.0, y[0]
    else:
        slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual_std = (
        float(np.std(y - fitted, ddof=1))
        if len(y) > 2
        else max(float(np.mean(y)) * 0.25, 1.0)
    )
    future_x = np.arange(len(y), len(y) + horizon_months, dtype=float)
    values = np.maximum(intercept + slope * future_x, 0.0)
    margin = max(1.96 * residual_std, 1.0)
    dates = _forecast_dates(horizon_months)
    return [
        {
            "month": ts.strftime("%Y-%m"),
            "forecast_mm": round(float(values[i]), 2),
            "lower_95_mm": round(float(max(values[i] - margin, 0.0)), 2),
            "upper_95_mm": round(float(values[i] + margin), 2),
        }
        for i, ts in enumerate(dates)
    ]


def forecast_locality(locality: str, horizon_months: int = 12, data_path: Path = DATA_PATH) -> dict:
    if horizon_months not in (12, 24, 36):
        raise ValueError("horizon_months must be 12, 24, or 36")
    monthly = load_locality_monthly(locality, data_path)
    observed = monthly.dropna()

    if observed.empty:
        raise ValueError(f"Unknown locality or no rainfall history: {locality}")

    # IMPORTANT: forecast labels always begin in the current calendar month.
    # This is independent of the last historical observation used for fitting.
    forecast_start = _current_month_start()

    # Short histories cannot support a reliable seasonal SARIMA fit.
    if len(observed) < MIN_MONTHS:
        rows = _fallback_forecast(observed, horizon_months)
        return {
            "locality": locality,
            "status": "fallback_forecast",
            "message": "Limited historical data; showing a simple trend forecast",
            "observed_months": int(len(observed)),
            "first_observation": str(observed.index.min().date()),
            "last_observation": str(observed.index.max().date()),
            "forecast_start": forecast_start.strftime("%Y-%m"),
            "horizon_months": horizon_months,
            "model": {"name": "Trend fallback"},
            "forecast": rows,
            "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2),
        }

    selected = _select_model(monthly)
    if selected is None:
        rows = _fallback_forecast(observed, horizon_months)
        return {
            "locality": locality,
            "status": "fallback_forecast",
            "message": "SARIMA could not be fitted; showing a simple trend forecast",
            "observed_months": int(len(observed)),
            "first_observation": str(observed.index.min().date()),
            "last_observation": str(observed.index.max().date()),
            "forecast_start": forecast_start.strftime("%Y-%m"),
            "horizon_months": horizon_months,
            "model": {"name": "Trend fallback"},
            "forecast": rows,
            "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2),
        }

    aic, order, seasonal, fit = selected
    try:
        future = fit.get_forecast(steps=horizon_months)
        ci = future.conf_int(alpha=0.05)
        mean = np.maximum(np.asarray(future.predicted_mean, dtype=float), 0.0)
        lower = np.maximum(np.asarray(ci.iloc[:, 0], dtype=float), 0.0)
        upper = np.maximum(np.asarray(ci.iloc[:, 1], dtype=float), 0.0)

        # Deliberately label the requested prediction window from the current
        # month rather than from the final timestamp in the historical dataset.
        dates = _forecast_dates(horizon_months)

        rows = [
            {
                "month": dates[i].strftime("%Y-%m"),
                "forecast_mm": round(float(mean[i]), 2),
                "lower_95_mm": round(float(lower[i]), 2),
                "upper_95_mm": round(float(upper[i]), 2),
            }
            for i in range(horizon_months)
        ]
    except Exception:
        rows = _fallback_forecast(observed, horizon_months)
        return {
            "locality": locality,
            "status": "fallback_forecast",
            "message": "SARIMA forecast failed; showing a simple trend forecast",
            "observed_months": int(len(observed)),
            "first_observation": str(observed.index.min().date()),
            "last_observation": str(observed.index.max().date()),
            "forecast_start": forecast_start.strftime("%Y-%m"),
            "horizon_months": horizon_months,
            "model": {"name": "Trend fallback"},
            "forecast": rows,
            "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2),
        }

    return {
        "locality": locality,
        "status": "ok",
        "message": "Locality-specific SARIMA forecast",
        "observed_months": int(len(observed)),
        "first_observation": str(observed.index.min().date()),
        "last_observation": str(observed.index.max().date()),
        "forecast_start": forecast_start.strftime("%Y-%m"),
        "model": {"name": "SARIMA", "order": list(order), "seasonal_order": list(seasonal), "aic": round(aic, 2)},
        "horizon_months": horizon_months,
        "forecast": rows,
        "annual_total_forecast_mm": round(sum(x["forecast_mm"] for x in rows), 2),
    }

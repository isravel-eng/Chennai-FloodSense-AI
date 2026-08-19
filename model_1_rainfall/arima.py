"""
arima.py
--------
Baseline non-seasonal ARIMA model for city-wide monthly rainfall.
Kept for comparison against SARIMA in sarima.py - a plain ARIMA cannot
capture the Oct-Dec northeast monsoon seasonality, so it is expected to
under-perform and is not the model actually shipped in models/.

Run standalone: python model_1_rainfall/arima.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"

TEST_MONTHS = 24


def main():
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["month_start"])
    series = df.set_index("month_start")["avg_rainfall_mm"].asfreq("MS").interpolate(method="time").bfill().ffill()

    train, test = series[:-TEST_MONTHS], series[-TEST_MONTHS:]

    best_aic, best_order, best_fit = np.inf, None, None
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    fit = ARIMA(train, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
                except Exception:
                    continue

    print(f"Best non-seasonal ARIMA order: {best_order} (AIC={best_aic:.2f})")

    forecast = best_fit.forecast(steps=TEST_MONTHS)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"ARIMA{best_order} holdout MAE : {mae:.2f} mm")
    print(f"ARIMA{best_order} holdout RMSE: {rmse:.2f} mm")
    print(
        "\nAs expected, plain ARIMA smooths toward the mean and misses the "
        "monsoon spike - see sarima.py for the seasonal model actually used."
    )


if __name__ == "__main__":
    main()

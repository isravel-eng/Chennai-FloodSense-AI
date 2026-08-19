"""
holt_winters.py
----------------
Baseline Holt-Winters (triple exponential smoothing) model for city-wide
monthly rainfall. This was the Model 1 originally shipped in an earlier
build of FloodSense AI. It has since been replaced by SARIMA (see
sarima.py), which produces both point forecasts and prediction intervals
and integrates more naturally with statsmodels' SARIMAX diagnostics.
Kept here for comparison / regression testing only.

Run standalone: python model_1_rainfall/holt_winters.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"

TEST_MONTHS = 24


def main():
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["month_start"])
    series = df.set_index("month_start")["avg_rainfall_mm"].asfreq("MS").interpolate(method="time").bfill().ffill()
    series = series.clip(lower=0.01)  # multiplicative seasonality needs > 0

    train, test = series[:-TEST_MONTHS], series[-TEST_MONTHS:]

    fit = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        damped_trend=True,
    ).fit()

    forecast = fit.forecast(TEST_MONTHS)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"Holt-Winters holdout MAE : {mae:.2f} mm")
    print(f"Holt-Winters holdout RMSE: {rmse:.2f} mm")
    print(
        "\nHolt-Winters has no native confidence-interval support in "
        "statsmodels and was superseded by SARIMA for that reason - "
        "see sarima.py."
    )


if __name__ == "__main__":
    main()

"""
sarima.py
---------
Trains the final Model 1: a SARIMA model forecasting city-wide average
MONTHLY rainfall (mm) for Chennai.

Inputs : data/processed/monthly_rainfall_citywide.csv
Outputs: models/rainfall_model.pkl          (fitted SARIMAXResults, pickled
                                              via statsmodels' own .save())
         models/rainfall_preprocessing.pkl  (dict of metadata needed to use
                                              the model - order, last date,
                                              training series stats)

IMPORTANT SCOPE NOTE (read this before wiring into the live app):
Model 1 forecasts MONTHLY rainfall trained on 1993-2023 history. It is a
research / trend-forecasting component, not a near-term (hours/next-24h)
predictor. The live/ layer does NOT feed Model 1's output into Model 2 -
Model 2 consumes real live weather-API rainfall instead. See
live/live_prediction.py and README.md for the full explanation.

Run standalone: python model_1_rainfall/sarima.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly_rainfall_citywide.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_MONTHS = 24
CANDIDATE_ORDERS = [
    # (p, d, q)
    (1, 0, 1), (1, 1, 1), (2, 1, 1), (0, 1, 1), (1, 0, 0),
]
CANDIDATE_SEASONAL_ORDERS = [
    # (P, D, Q, s)
    (1, 1, 1, 12), (0, 1, 1, 12), (1, 1, 0, 12),
]


def grid_search(train: pd.Series):
    best_aic, best_spec, best_fit = np.inf, None, None
    for order in CANDIDATE_ORDERS:
        for seasonal_order in CANDIDATE_SEASONAL_ORDERS:
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_spec = (order, seasonal_order)
                    best_fit = fit
            except Exception:
                continue
    return best_spec, best_aic, best_fit


def main():
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["month_start"])
    series = df.set_index("month_start")["avg_rainfall_mm"].asfreq("MS")

    n_missing = int(series.isna().sum())
    if n_missing:
        print(f"Note: {n_missing} months have no station readings in this "
              f"period - interpolating (time-based) before fitting.")
        series = series.interpolate(method="time").bfill().ffill()

    train, test = series[:-TEST_MONTHS], series[-TEST_MONTHS:]

    print(f"Training on {len(train)} months, holding out last {len(test)} months")
    print("Grid searching (p,d,q)(P,D,Q,12) by AIC ...")
    (order, seasonal_order), aic, holdout_fit = grid_search(train)
    print(f"Best spec: order={order} seasonal_order={seasonal_order} AIC={aic:.2f}")

    forecast = holdout_fit.get_forecast(steps=TEST_MONTHS)
    pred_mean = forecast.predicted_mean
    mae = mean_absolute_error(test, pred_mean)
    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    print(f"Holdout MAE : {mae:.2f} mm")
    print(f"Holdout RMSE: {rmse:.2f} mm")

    # Refit on the FULL series with the winning spec so the shipped model
    # has the most recent data available for forecasting going forward.
    print("Refitting on full history with winning spec ...")
    final_model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_fit = final_model.fit(disp=False)

    model_path = MODELS_DIR / "rainfall_model.pkl"
    final_fit.save(str(model_path))
    print(f"Saved {model_path}")

    preprocessing = {
        "order": order,
        "seasonal_order": seasonal_order,
        "seasonal_period": 12,
        "trained_on_last_date": str(series.index.max().date()),
        "trained_on_first_date": str(series.index.min().date()),
        "target": "avg_rainfall_mm (city-wide average across all localities)",
        "frequency": "MS (month start)",
        "holdout_mae_mm": round(float(mae), 2),
        "holdout_rmse_mm": round(float(rmse), 2),
        "aic": round(float(aic), 2),
        "notes": (
            "Forecasts MONTHLY rainfall. Not used directly as a live/daily "
            "input to Model 2 - see live/live_prediction.py."
        ),
    }
    preprocessing_path = MODELS_DIR / "rainfall_preprocessing.pkl"
    with open(preprocessing_path, "wb") as f:
        pickle.dump(preprocessing, f)
    print(f"Saved {preprocessing_path}")

    print("\nNext 6 months forecast (research output):")
    future = final_fit.get_forecast(steps=6)
    future_df = pd.DataFrame({
        "month": future.predicted_mean.index.strftime("%Y-%m"),
        "forecast_mm": future.predicted_mean.round(1).values,
        "lower_95": future.conf_int(alpha=0.05).iloc[:, 0].round(1).values,
        "upper_95": future.conf_int(alpha=0.05).iloc[:, 1].round(1).values,
    })
    print(future_df.to_string(index=False))


if __name__ == "__main__":
    main()

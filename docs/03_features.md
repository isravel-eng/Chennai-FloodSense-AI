# Feature Documentation

## Model 2 Feature Contract

The live pipeline is designed to reproduce the exact feature order expected by `models/flood_model.pkl`.

Current feature order:

```text
rainfall_mm
rainfall_3d_mm
rainfall_7d_mm
rainfall_30d_mm
latitude
longitude
elevation_m_approx
month
month_sin
month_cos
is_northeast_monsoon
rainfall_lag_1
rainfall_lag_2
rainfall_lag_3
rainfall_lag_7
```

## Feature Groups

### Rainfall intensity / accumulation

- `rainfall_mm`
- `rainfall_3d_mm`
- `rainfall_7d_mm`
- `rainfall_30d_mm`

These represent current or accumulated rainfall information at different time windows.

### Location

- `latitude`
- `longitude`
- `elevation_m_approx`

These provide locality context.

### Seasonality

- `month`
- `month_sin`
- `month_cos`
- `is_northeast_monsoon`

The cyclic month features represent seasonal position without treating December and January as distant points.

### Rainfall lags

- `rainfall_lag_1`
- `rainfall_lag_2`
- `rainfall_lag_3`
- `rainfall_lag_7`

These provide recent temporal context.

## Live Feature Pipeline

```text
Locality
  -> locality lookup
  -> Open-Meteo current / forecast rainfall
  -> recent rainfall history or climatology fallback
  -> feature construction
  -> exact 15-feature order
  -> XGBoost
  -> probability + LOW/MEDIUM/HIGH band
```

## Important Contract Rule

Do not add temperature, humidity, wind, or other new variables to the existing XGBoost model without retraining it. The current model was not trained on those variables.

A future Version 2 model may use expanded weather features after an appropriate training-data update and validation cycle.

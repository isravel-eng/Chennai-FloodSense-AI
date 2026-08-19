# ML Architecture

## System Overview

```text
                         CHENNAI FLOODSENSE AI
                                  |
                    +-------------+-------------+
                    |                           |
             MODEL 1: SARIMA             LIVE FLOOD PATH
                    |                           |
       Monthly city-wide rainfall              |
             research forecast                  |
                                                v
                                      User selects locality
                                                |
                                                v
                                      Locality lookup
                                      lat/lon/elevation
                                                |
                                                v
                                       Open-Meteo API
                                      current + forecast
                                                |
                                                v
                                  Recent rainfall history
                                  or climatology fallback
                                                |
                                                v
                                      Feature builder
                                      exact 15 features
                                                |
                                                v
                                      XGBoost classifier
                                                |
                                                v
                                  Flood probability / risk band
                                      LOW / MEDIUM / HIGH
```

## Model 1

Input: historical city-wide monthly rainfall.

Output: monthly rainfall forecast with uncertainty interval.

Interface:

```bash
python predict_end_to_end.py --forecast-rainfall --months 12
```

Model 1 is intentionally not connected directly to the live flood-risk prediction path.

## Model 2

Input: daily rainfall, accumulated rainfall windows, lag features, location, elevation, and seasonality.

Output: flood probability and risk band.

The live layer obtains current and next-24-hour weather information and constructs the exact feature contract required by the saved XGBoost model.

## Data Flow

```text
Historical data -> feature engineering -> model training -> saved model

Live locality -> weather API -> rainfall history -> feature builder
             -> saved XGBoost -> risk prediction
```

## Version 1 Boundary

Temperature, humidity, and wind are available from the weather API but are not fed to the current XGBoost model because they were not part of its training features. A future model version requires retraining and evaluation before using them.

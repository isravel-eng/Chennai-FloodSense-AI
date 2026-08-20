# Chennai FloodSense AI

A two-model rainfall/flood-risk system for Chennai, built on 1993–2023
historical rainfall data (30 localities), now extended with a **live
weather layer** so predictions reflect real, current 2026 conditions
instead of only historical data.

```
                    CHENNAI FLOODSENSE AI
                           │
                           ▼
                 ┌─────────────────────┐
                 │   USER SELECTS      │
                 │     LOCALITY        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   LIVE WEATHER API  │   (Open-Meteo)
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │     LIVE FEATURE BUILDER    │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ MODEL 2        │
                    │ XGBoost        │
                    └───────┬────────┘
                            │
                            ▼
                    Flood Probability
                            │
                            ▼
                  LOW / MEDIUM / HIGH
```

## Why two separate models

| | Model 1 — `model_1_rainfall/` | Model 2 — `model_2_flood/` |
|---|---|---|
| Algorithm | SARIMA (statsmodels) | Selected tree classifier (XGBoost/RF candidates) |
| Predicts | **Monthly** city-wide rainfall (mm) | Flood probability for a given locality/day |
| Trained on | 1993–2023 monthly rainfall | 1993–2023 daily rainfall + locality/season features |
| Role | Research / long-range trend forecasting | The model driving live risk predictions |
| Evaluation | 24-month research holdout | Chronological validation + untouched 2022–2023 test |

**Model 1's forecast is monthly**. A live flood-risk tool needs a near-term
(hours/next-24h) number. Model 1 therefore remains a standalone research
component, while Model 2 receives its near-term rainfall input from the real
Open-Meteo weather API.

## Live architecture (`live/`)

```
Sholinganallur
       │
       ▼
locality_lookup.csv ──► latitude / longitude / elevation
       │
       ▼
Open-Meteo weather API ──► current precipitation, next-24h forecast
       │
       ▼
Recent rainfall (3d / 7d / 30d + lags) ──► live log if available, else
                                            seasonal climatology fallback
       │
       ▼
live_features.py ──► exact V2 feature vector, in training order
       │
       ▼
Model 2 (`flood_model.pkl`)
       │
       ▼
Two predictions: CURRENT risk + NEXT-24H risk
```

| File | Responsibility |
|---|---|
| `live/weather_api.py` | Calls Open-Meteo for current + hourly forecast precipitation/temp/humidity/wind for a locality's coordinates. |
| `live/rainfall_history.py` | Provides recent locality rainfall from the append-only live log, with a historical seasonal fallback. |
| `live/live_features.py` | Converts weather + history + location into the **exact V2 feature vector** expected by the model. |
| `live/live_prediction.py` | Orchestrates the live current-risk and next-24h-risk predictions. |

### Why temperature/humidity/wind are not model inputs

Open-Meteo provides these values, but the historical training dataset does not
contain them. Passing them into the model without retraining would create a
train/inference mismatch. They remain available from the API for future use.

## Repository layout

```
Chennai-FloodSense-AI/
├── data/
│   ├── raw/master_dataset.csv              1993-2023 source data (30 localities)
│   └── processed/
│       ├── model2_features.csv             V2 engineered features for Model 2
│       ├── monthly_rainfall_citywide.csv   monthly series for Model 1
│       ├── locality_lookup.csv             lat/lon/elevation per locality
│       └── live_rainfall_log.csv           created at runtime
├── models/
│   ├── rainfall_model.pkl                  fitted SARIMAXResults
│   ├── rainfall_preprocessing.pkl          SARIMA metadata
│   ├── flood_model.pkl                     fitted selected Model 2 classifier
│   ├── flood_model_selection.pkl           V2 validation/test metrics + threshold
│   └── flood_preprocessing.pkl             V2 feature order + locality stats
├── model_1_rainfall/
├── model_2_flood/
│   ├── feature_engineering.py              V2 feature generation
│   ├── preprocessing.py                    canonical feature schema
│   ├── 02_evaluation.py                    temporal split + model comparison/tuning + final test
│   ├── evaluation.py                       untouched chronological test report
│   └── predict_flood.py                    stable inference interface
├── live/                                   live Open-Meteo prediction layer
├── tests/                                  offline-safe tests
├── docs/MODEL_V2.md                        detailed V2 documentation
├── .github/workflows/retrain-model.yml    free automatic retraining
├── predict_end_to_end.py                   CLI entry point
├── requirements.txt
└── README.md
```

## Model 2 V2 upgrade

Model 2 now uses a chronological:

- **Train:** 1993–2017
- **Validation:** 2018–2021
- **Final test:** 2022–2023

It compares Random Forest, XGBoost baseline and a tuned XGBoost configuration
using **validation PR-AUC**. The final decision threshold is selected on the
validation period only.

V2 adds:

- day-of-year cyclic seasonality
- rainfall change from the previous observation
- normalized 7-day rainfall
- normalized 30-day rainfall
- 7-day/30-day rainfall concentration ratio
- class-imbalance handling
- stable LOW/MEDIUM/HIGH risk bands

The live feature builder uses the same schema, so training and inference do
not silently use different feature sets.

Full model notes: `docs/MODEL_V2.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Rebuilding Model 2 from scratch

```bash
python model_2_flood/feature_engineering.py
python model_2_flood/preprocessing.py
python model_2_flood/02_evaluation.py
python model_2_flood/evaluation.py
```

## Using it

```bash
python predict_end_to_end.py --live Sholinganallur
python predict_end_to_end.py --historical Alandur --date 2017-11-13
python predict_end_to_end.py --forecast-rainfall --months 12
```

## Running tests

```bash
python -m pytest tests/ -v
```

The live weather tests are offline-safe by default. Set
`RUN_LIVE_NETWORK_TESTS=1` to additionally call the real Open-Meteo API.

## Automatic free retraining

`.github/workflows/retrain-model.yml` rebuilds the Model 2 features, trains the
candidate models, runs the offline tests, and commits the generated Model 2
artifacts whenever the historical dataset or Model 2 source code changes.
No paid ML service is required.

## Known limitations

- The live rainfall log starts empty and falls back to historical seasonal
  climatology until enough observed entries accumulate.
- Next-24h flood risk uses forecast precipitation in the rainfall input slot
  as a documented near-term proxy. A separately trained forecast model would
  require historical forecast-versus-observed training data.
- Model 1 (SARIMA) is monthly and remains a separate research/trend component.
- This is a research/prototype flood-risk classifier, not an official flood
  warning system.

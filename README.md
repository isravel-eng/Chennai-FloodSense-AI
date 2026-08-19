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
| Algorithm | SARIMA (statsmodels) | XGBoost (selected over RandomForest by PR-AUC) |
| Predicts | **Monthly** city-wide rainfall (mm) | Flood probability for a given locality/day |
| Trained on | 1993–2023 monthly rainfall | 1993–2023 daily rainfall + locality/season features |
| Role | Research / long-range trend forecasting | The model actually driving live risk predictions |
| Holdout metric | MAE 4.73mm, RMSE 6.03mm (24-month holdout) | PR-AUC 0.727 (rare-event holdout, 1.1% positive rate) |

**Model 1's forecast is monthly** (e.g. "November 2026 will average ~21mm/day
citywide"). A live flood-risk tool needs a near-term (hours/next-24h)
number. Multiplying the SARIMA monthly output by some constant to fake a
daily value was an earlier prototype shortcut and is **not** what this
version does. Instead:

- Model 1 stays a standalone research component (`predict_end_to_end.py --forecast-rainfall`).
- Model 2's live rainfall input comes from a **real weather API** (Open-Meteo), not from Model 1.

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
live_features.py ──► exact 15-feature vector, in training order
       │
       ▼
Existing XGBoost (flood_model.pkl)
       │
       ▼
Two predictions: CURRENT risk + NEXT-24H risk
```

| File | Responsibility |
|---|---|
| `live/weather_api.py` | Calls Open-Meteo for current + hourly forecast precipitation/temp/humidity/wind for a locality's coordinates. |
| `live/rainfall_history.py` | Answers "what rainfall has this locality actually received recently?" — via an append-only live log (`data/processed/live_rainfall_log.csv`), falling back to 1993–2023 seasonal climatology until 30+ days of real log data exist. |
| `live/live_features.py` | Converts weather + history + location into the **exact** 15-feature vector `flood_model.pkl` expects, in the exact trained order. Two builders: `build_current_features()` and `build_forecast_24h_features()`. |
| `live/live_prediction.py` | Orchestrates the above into `predict_live_flood(locality)`, returning both a current-risk and next-24h-risk prediction. |

### Why the model isn't fed temperature/humidity/wind (yet)

`flood_model.pkl` was trained only on rainfall + location + season features.
Feeding it temperature/humidity/wind now — even though Open-Meteo provides
them — would be invalid (the model was never trained on those inputs).
This is **Version 1** of the live layer. **Version 2** (future work) would
retrain Model 2 on an expanded feature set including those live weather
variables once enough live-labeled data exists to justify it.

## Repository layout

```
Chennai-FloodSense-AI/
├── data/
│   ├── raw/master_dataset.csv              1993-2023 source data (30 localities)
│   └── processed/
│       ├── model2_features.csv             + lag/cyclical features for Model 2
│       ├── monthly_rainfall_citywide.csv   monthly series for Model 1
│       ├── locality_lookup.csv             lat/lon/elevation per locality
│       └── live_rainfall_log.csv           created at runtime by rainfall_history.py
├── models/
│   ├── rainfall_model.pkl                  fitted SARIMAXResults
│   ├── rainfall_preprocessing.pkl          SARIMA order + metadata
│   ├── flood_model.pkl                     fitted XGBClassifier
│   └── flood_preprocessing.pkl             feature order + locality stats
├── model_1_rainfall/
│   ├── stationarity.py    ADF tests, informs SARIMA differencing
│   ├── arima.py           non-seasonal baseline (comparison only)
│   ├── holt_winters.py    exponential-smoothing baseline (comparison only)
│   └── sarima.py          trains & saves the final Model 1
├── model_2_flood/
│   ├── feature_engineering.py   builds processed/ CSVs from raw data
│   ├── preprocessing.py         builds flood_preprocessing.pkl (feature order)
│   ├── 02_evaluation.py         trains RandomForest + XGBoost, selects winner by PR-AUC, saves flood_model.pkl
│   ├── evaluation.py            loads the saved model, prints a full evaluation report
│   └── predict_flood.py         stable predict_flood(features_dict) interface used by everything else
├── live/                        the new live-data layer (see above)
├── tests/
│   ├── test_model1.py           SARIMA sanity checks
│   ├── test_model2.py           XGBoost sanity + risk-band checks
│   ├── test_live_weather.py     offline-safe weather-API unit tests
│   └── test_live_prediction.py  offline-safe full live-pipeline test (mocked weather)
├── predict_end_to_end.py        CLI: --live, --historical, --forecast-rainfall
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Rebuilding everything from scratch

```bash
# 1. Feature engineering (raw CSV -> processed CSVs)
python model_2_flood/feature_engineering.py

# 2. Train Model 1 (SARIMA)
python model_1_rainfall/stationarity.py     # optional diagnostics
python model_1_rainfall/sarima.py           # saves models/rainfall_model.pkl

# 3. Train Model 2 (XGBoost)
python model_2_flood/preprocessing.py       # saves models/flood_preprocessing.pkl
python model_2_flood/02_evaluation.py       # trains, compares, saves models/flood_model.pkl
python model_2_flood/evaluation.py          # prints a full evaluation report
```

## Using it

```bash
# Live prediction for a locality (requires internet access to Open-Meteo)
python predict_end_to_end.py --live Sholinganallur

# Historical backtest (no internet required, uses the 1993-2023 dataset)
python predict_end_to_end.py --historical Alandur --date 2017-11-13

# Model 1 standalone research forecast
python predict_end_to_end.py --forecast-rainfall --months 12
```

Or programmatically:

```python
from live.live_prediction import predict_live_flood
result = predict_live_flood("Sholinganallur")
print(result["current"]["risk_band"], result["next_24h"]["risk_band"])
```

## Running tests

```bash
python -m pytest tests/ -v
```

`test_live_weather.py` and `test_live_prediction.py` are written to run
fully offline (they use a canned weather fixture / a mocked weather dict)
so the suite passes even without internet access. Set
`RUN_LIVE_NETWORK_TESTS=1` to additionally hit the real Open-Meteo API.

## Known limitations / next steps

- **Live rainfall log starts empty.** Until `live/rainfall_history.py`'s
  log has 30+ real daily entries per locality, recent-rainfall features
  fall back to 1993–2023 seasonal climatology medians, not this week's
  actual rainfall. Wire up a daily cron job that appends observed
  rainfall (from the weather API or a rain-gauge feed) via `RainfallLog.append()`.
- **Version 1 live model ignores temperature/humidity/wind** even though
  they're available from Open-Meteo, because `flood_model.pkl` was never
  trained on them. Retraining with those features is future work (Version 2).
- **Forecast-rainfall risk uses next-24h precipitation as a proxy** for
  the `rainfall_mm` feature slot the model was trained on with *actual*
  daily rainfall — a documented approximation, not a perfect substitute.
- **Model 1 (SARIMA) is monthly**, kept as a separate research/trend
  component; it is not wired into the live flood-risk path.

<<<<<<<
=======
>>>>>>>
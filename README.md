# Chennai FloodSense AI — Machine Learning

This branch is the **single source of truth for the ML implementation**.

## Simple ML structure

```text
machine-learning/
├── data/
├── model_1_rainfall/
│   └── sarima.py
├── model_2_flood/
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict_flood.py
├── live/
├── models/
├── tests/
└── docs/
```

## Model 1 — Rainfall Forecasting

**Algorithm:** SARIMA

```text
Historical monthly rainfall
        ↓
      SARIMA
        ↓
Rainfall forecast
```

The model uses a simple seasonal configuration with a 12-month period. It forecasts city-wide monthly rainfall and is a trend/research component.

Run:

```bash
python model_1_rainfall/sarima.py
```

## Model 2 — Flood Risk

**Algorithm:** Random Forest Classifier

The model uses six easy-to-explain inputs:

```text
rainfall_mm
rainfall_3d_mm
rainfall_7d_mm
rainfall_30d_mm
latitude
longitude
```

```text
Rainfall + Location
        ↓
   Random Forest
        ↓
 Flood probability
        ↓
   LOW / MEDIUM / HIGH
```

Build features and train:

```bash
python model_2_flood/feature_engineering.py
python model_2_flood/train.py
```

The prediction interface is:

```python
from model_2_flood.predict_flood import predict_flood

result = predict_flood({
    "rainfall_mm": 80,
    "rainfall_3d_mm": 140,
    "rainfall_7d_mm": 220,
    "rainfall_30d_mm": 300,
    "latitude": 13.05,
    "longitude": 80.25,
})
```

## Important

The old XGBoost comparison, tuned-model selection, 20-feature preprocessing contract and validation-threshold pipeline have been removed from the canonical implementation because they made the project unnecessarily difficult to explain.

The current ML implementation is intentionally simple and presentation-friendly. The previous implementation is preserved in `machine-learning-backup-2026-09-01`.

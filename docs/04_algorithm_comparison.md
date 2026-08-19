# Algorithm Comparison

## Model 1 — Rainfall Forecasting

The Model 1 directory contains three forecasting approaches:

| Approach | File | Role |
|---|---|---|
| ARIMA | `model_1_rainfall/arima.py` | Non-seasonal baseline |
| Holt-Winters | `model_1_rainfall/holt_winters.py` | Exponential-smoothing baseline |
| SARIMA | `model_1_rainfall/sarima.py` | Final Model 1 |

The current repository uses SARIMA as the final Model 1. The README records a 24-month holdout result of **MAE 4.73 mm** and **RMSE 6.03 mm**.

## Model 2 — Flood Classification

The Model 2 evaluation pipeline compares:

- Random Forest
- XGBoost

The current README states that XGBoost was selected over Random Forest using **PR-AUC**, which is appropriate for the rare-event setting. The recorded rare-event holdout has a **1.1% positive rate**, and the selected XGBoost model has **PR-AUC 0.727**.

## Why PR-AUC Matters

With a highly imbalanced flood label, accuracy can be misleading because a classifier can obtain high accuracy by predicting the majority class. Precision-recall performance is therefore a more useful evaluation view for the flood-event class.

## Decision

```text
Model 1
ARIMA / Holt-Winters / SARIMA
             |
             +--> SARIMA retained as final rainfall model

Model 2
Random Forest / XGBoost
             |
             +--> XGBoost retained by PR-AUC
```

## Reproducibility

Model 2 comparison and selection are implemented in:

`model_2_flood/02_evaluation.py`

The saved winner is:

`models/flood_model.pkl`

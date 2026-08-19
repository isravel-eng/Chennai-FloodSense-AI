# Model Evaluation

## Model 1 — SARIMA

The current documented evaluation is a **24-month holdout**:

| Metric | Result |
|---|---:|
| MAE | 4.73 mm |
| RMSE | 6.03 mm |

Model 1 is a monthly city-wide research forecast and is not directly used by the live flood-risk path.

## Runtime Verification

The current saved SARIMA model was verified through:

```bash
python predict_end_to_end.py --forecast-rainfall --months 12
```

The command successfully returned 12 monthly forecast rows with point forecasts and 95% interval columns (`forecast_mm`, `lower_95`, `upper_95`).

The repository's pytest test could not yet be executed in the local environment because `pytest` is not installed in the active virtual environment. Install the project test dependency before treating the full test suite as verified.

## Model 2 — XGBoost

The current documented rare-event holdout results are:

| Metric | Result |
|---|---:|
| PR-AUC | 0.727 |
| Positive rate | 1.1% |

Model 2 is the model used by the live flood-risk pipeline.

## Evaluation Principle

Evaluation should be based on time-aware or otherwise leakage-safe holdouts. Future observations must not be used to construct features for earlier prediction dates.

## Current Status

- Model 1 saved model: runtime forecast verified
- Model 1 documented holdout: MAE 4.73 mm, RMSE 6.03 mm
- Model 2 saved model: used by historical and live prediction paths
- Model 2 documented PR-AUC: 0.727
- Full automated test suite: pending `pytest` installation and execution

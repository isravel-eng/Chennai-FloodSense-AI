# Model 1 Validation

## Current Model

Model 1 is the saved SARIMA rainfall-forecasting component in `model_1_rainfall/sarima.py` with the fitted artifact stored as `models/rainfall_model.pkl`.

## Verification Performed

Command:

```bash
python predict_end_to_end.py --forecast-rainfall --months 12
```

Result: the saved model successfully produced 12 monthly forecast rows from January through December 2024, including point forecasts and 95% lower/upper interval columns.

Example output structure:

```text
            forecast_mm  lower_95  upper_95
2024-01-01          8.7      -8.6      26.0
...
2024-12-01         17.2      -0.9      35.3
```

This confirms that the saved SARIMA artifact can be loaded and used for inference.

## Existing Holdout Evaluation

The project README records a 24-month holdout evaluation:

- MAE: 4.73 mm
- RMSE: 6.03 mm

## Interpretation

The forecast is a **monthly city-wide research forecast**. It should not be interpreted as a direct daily flood-risk forecast. The live flood-risk path uses weather API rainfall inputs and the XGBoost Model 2 classifier.

## Test Status

The intended sanity test is:

```bash
python -m pytest tests/test_model1.py -v
```

At the current development environment, this command was not executed successfully because `pytest` is not installed in the active virtual environment. Install the test dependency and rerun it before marking automated Model 1 testing complete.

## Next Validation Step

When 2024–2025 verified rainfall data are added, run a new time-based holdout evaluation and compare the updated SARIMA model against the current baseline rather than replacing the model solely because newer data are available.

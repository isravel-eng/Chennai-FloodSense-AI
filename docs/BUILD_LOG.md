# FloodSense AI — V2 Build Log

## 2026-08-20 — Model 2 upgrade

### Completed

1. Replaced random train/test splitting with chronological validation:
   - Train: 1993–2017
   - Validation: 2018–2021
   - Test: 2022–2023
2. Added rainfall/seasonality features:
   - day-of-year sine/cosine
   - one-step rainfall change
   - normalized 7-day rainfall
   - normalized 30-day rainfall
   - 7-day/30-day rainfall concentration ratio
3. Added XGBoost tuning candidate while retaining Random Forest and XGBoost baseline comparison.
4. Kept PR-AUC as the primary selection metric because the flood target is highly imbalanced.
5. Selected the alert threshold using validation data only.
6. Updated the live feature builder to use the same V2 feature schema as training.
7. Updated Model 2 tests for the new feature schema.
8. Added `docs/MODEL_V2.md`.
9. Added a free GitHub Actions workflow to retrain Model 2 when the source dataset or Model 2 code changes.

### Local V2 experiment

Using the current 16,554-row historical dataset:

- Random Forest validation PR-AUC: **0.1061**
- XGBoost baseline validation PR-AUC: **0.1722**
- tuned XGBoost validation PR-AUC: **0.1861**
- selected model: **XGBoost tuned**
- untouched 2022–2023 test PR-AUC: **0.0505**
- test recall at the validation-selected threshold: **0.443**
- test F1 at the validation-selected threshold: **0.116**

These numbers are intentionally reported from the chronological evaluation. They are not replaced by a random-split score.

### Validation of the implementation

The V2 Model 2/live test subset passes locally:

```text
16 passed
```

The pre-existing Model 1 artifact has a separate pickle compatibility problem in the current Python 3.13 environment; Model 1 source code was not changed by this V2 upgrade. The GitHub retraining workflow uses Python 3.11 for Model 2.

## Next project step

Integrate the V2 model artifacts into the FastAPI/React application and keep the UI output aligned with the presentation:

`locality → current/next-24h risk → LOW / MEDIUM / HIGH → map/alert`

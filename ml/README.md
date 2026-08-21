# Chennai FloodSense AI — ML Branch

This branch isolates the Machine Learning/Data Processing work from `main` so Backend, Frontend, and Deployment can be developed independently without changing the stable main branch.

## Scope

Phase I from the project presentation requires:

- Literature survey
- Dataset collection
- Data preprocessing
- EDA
- Feature engineering
- Initial ML model
- Algorithm comparison
- Model evaluation
- GitHub repository
- UI wireframe support

Phase II requires the optimized ML model to feed the FastAPI flood-prediction module.

## Current baseline

The repository currently uses two models:

- Model 1: SARIMA for monthly city-wide rainfall research forecasting.
- Model 2: XGBoost for locality/day-level flood probability.

The existing historical dataset covers 1993–2023. Model 2 currently uses rainfall, location, elevation, seasonality and rainfall-lag features, with a rare-event target. The current pipeline compares Random Forest and XGBoost using PR-AUC and selects the better classifier.

## 2024–2026 data upgrade

The next ML dataset extension should add observations for 2024, 2025 and 2026 using authoritative rainfall/weather sources while preserving the existing schema. Do **not** fabricate 2024–2026 locality flood labels. New rows without verified flood labels should be kept as unlabeled/live-weather data and excluded from supervised target training until labels are available.

Recommended source hierarchy:

1. India Meteorological Department (IMD) rainfall observations and historical datasets.
2. Open-Meteo for current/forecast weather features already used by the live layer.
3. Verified municipal/government flood-event records for locality-level labels.

## ML upgrade plan

1. Extend the raw weather dataset beyond 2023.
2. Normalize dates, locality names and units.
3. Preserve a source column and observation-quality metadata.
4. Rebuild 3/7/30-day rainfall aggregates and lag features chronologically.
5. Prevent temporal leakage: train on older dates and test on later dates.
6. Compare Random Forest and XGBoost with PR-AUC, precision, recall, F1 and calibration-oriented diagnostics.
7. Use a final time-based holdout for honest evaluation.
8. Produce feature-importance / SHAP-style explainability artifacts when dependencies permit.
9. Version datasets and preprocessing metadata separately from trained model binaries.
10. Export the final model contract for the backend: exact feature names/order, expected units, probability output and risk-band thresholds.

## Branch ownership model

- `main` — stable integration branch.
- `ml` — ML/data processing experiments and validated model releases.
- `upgrade/model-2-v2` — existing Model 2 V2 work.
- `backend` — FastAPI work (friend/team member).
- `frontend` — React/Leaflet work (friend/team member).
- `deployment` — Docker/cloud deployment work (friend/team member).

Backend, frontend and deployment branches can be created from `main` and merged independently. ML changes should reach integration through pull requests from `ml` after evaluation.

## Important limitation

The project presentation promises locality-wise flood risk, but the existing historical supervised labels are documented as rare-event labels. For a defensible 2024–2026 retraining run, only rows with verified flood-event labels should become positive/negative supervised examples. Weather-only records can improve the live feature history but cannot, by themselves, create a trustworthy flood label.

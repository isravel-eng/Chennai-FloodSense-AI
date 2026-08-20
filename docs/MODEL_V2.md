# Chennai FloodSense AI — Model 2 V2

## Scope

This upgrade stays aligned with the project presentation: locality-wise flood-risk prediction using historical rainfall/flood patterns and near-term weather forecasts, with `LOW / MEDIUM / HIGH` output.

The upgrade intentionally does **not** add deep learning, paid cloud ML services, or a complicated online-learning system. Model 1 (SARIMA monthly rainfall research forecast) is unchanged.

## Changes

### 1. Time-aware validation

Model 2 now uses a chronological split:

- **Train:** 1993–2017
- **Validation:** 2018–2021
- **Final test:** 2022–2023

The validation period chooses the model configuration and alert threshold. The final test period is kept untouched until the final evaluation.

### 2. Better rainfall features

The model now adds:

- `day_of_year_sin`, `day_of_year_cos`
- `rainfall_change_1d`
- `rainfall_7d_per_day`
- `rainfall_30d_per_day`
- `rainfall_7d_ratio_30d`

Existing rainfall accumulation and locality features remain.

### 3. Algorithm comparison + tuning

Three free local candidates are compared:

- Random Forest baseline
- XGBoost baseline
- tuned XGBoost

The winner is selected using **validation PR-AUC** because flood events are rare.

### 4. Imbalance handling

Random Forest uses balanced class weights. XGBoost uses `scale_pos_weight` calculated from the training partition only.

### 5. Threshold selection

The best-F1 operating threshold is selected on the validation period only and stored in `models/flood_model_selection.pkl`. The application also keeps stable `LOW / MEDIUM / HIGH` bands for the presentation/UI.

### 6. Live inference alignment

`live/live_features.py` now creates the same V2 feature schema as training. Open-Meteo temperature, humidity and wind are **not** passed into the model because the historical training dataset does not contain those fields.

The next-24h path continues to use Open-Meteo forecast precipitation as a documented near-term proxy for the rainfall input. A separately trained forecast model would require historical forecast-versus-observed training data, which is outside this focused upgrade.

## Rebuild locally

```bash
python model_2_flood/feature_engineering.py
python model_2_flood/preprocessing.py
python model_2_flood/02_evaluation.py
python model_2_flood/evaluation.py
python -m pytest tests/test_model2.py tests/test_live_prediction.py tests/test_live_weather.py -v
```

## Current local validation result

Using the 1993–2023 dataset:

- Validation PR-AUC improved from the XGBoost baseline to the tuned XGBoost candidate in the V2 experiment.
- The untouched 2022–2023 test result is reported by the training script and is **not** replaced by a random-split score.

The model remains a prototype/research classifier, not an official flood-warning system.

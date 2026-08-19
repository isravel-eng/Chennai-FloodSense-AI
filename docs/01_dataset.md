# Dataset Documentation

## Overview

Chennai FloodSense AI currently uses historical rainfall and flood-related data covering **1993–2023** across **30 Chennai localities**. The repository also contains a live weather layer for current 2026 prediction inputs.

## Dataset Components

### Raw dataset

`data/raw/master_dataset.csv`

Main historical source dataset used to build the Model 2 processed features.

### Model 2 processed features

`data/processed/model2_features.csv`

Engineered daily features used by the flood-risk model, including rainfall windows, lag features, cyclical month features, locality information, and season indicators. The live pipeline reproduces the same 15-feature order expected by the saved XGBoost model.

### Model 1 monthly series

`data/processed/monthly_rainfall_citywide.csv`

City-wide monthly rainfall series used by the SARIMA rainfall-forecasting component.

### Locality lookup

`data/processed/locality_lookup.csv`

Provides locality-level latitude, longitude, and approximate elevation used by the live feature builder.

### Live rainfall log

`data/processed/live_rainfall_log.csv`

Created at runtime by the live rainfall-history component. Until sufficient real observations are available, the pipeline can fall back to 1993–2023 seasonal climatology.

## Current Data Timeline

| Period | Role |
|---|---|
| 1993–2023 | Historical training / research data |
| 2024–2025 | Planned extension for model re-evaluation and retraining |
| 2026 | Current-year live weather inputs and validation period; not yet a complete training year |

## Important Limitation

The live rainfall-history mechanism can use seasonal climatology until enough real locality-level observations are accumulated. This is a Version 1 limitation and should be addressed before treating the system as a fully operational flood-monitoring solution.

## Planned Data Update

Obtain verified 2024–2025 rainfall observations, evaluate their impact on Model 1, and improve the live rainfall-history layer. Treat 2026 observations initially as a current/live validation period rather than as a completed annual training year.

# Dataset Documentation

## Overview

Chennai FloodSense AI currently uses historical rainfall and flood-related data covering **1993–2023** across **30 Chennai localities**. The repository also contains a live weather layer for current 2026 prediction inputs. fileciteturn3file0

## Dataset Components

### Raw dataset

`data/raw/master_dataset.csv`

This is the main historical source dataset used to build the Model 2 processed features.

### Model 2 processed features

`data/processed/model2_features.csv`

This contains engineered daily features used by the flood-risk model, including rainfall windows, lag features, cyclical month features, locality information, and season indicators. The live pipeline reproduces the same 15-feature order expected by the saved XGBoost model. fileciteturn3file0

### Model 1 monthly series

`data/processed/monthly_rainfall_citywide.csv`

This is the city-wide monthly rainfall series used by the SARIMA rainfall-forecasting component.

### Locality lookup

`data/processed/locality_lookup.csv`

Provides locality-level latitude, longitude, and approximate elevation used by the live feature builder.

### Live rainfall log

`data/processed/live_rainfall_log.csv`

This file is created at runtime by the live rainfall-history component. Until sufficient real observations are collected, the pipeline can fall back to 1993–2023 seasonal climatology. fileciteturn3file0

## Current Data Timeline

| Period | Role |
|---|---|
| 1993–2023 | Historical training / research data |
| 2024–2025 | Planned extension for model re-evaluation and retraining |
| 2026 | Current-year live weather inputs and validation period; not yet a complete training year |

## Important Data Limitation

The current live rainfall-history mechanism may use seasonal climatology until enough real locality-level observations are accumulated. This is an acknowledged limitation of Version 1 and should be addressed before treating the system as a fully reliable operational flood-monitoring solution. fileciteturn3file0

## Planned Data Update

The next data-engineering stage is to obtain verified 2024–2025 rainfall observations, evaluate their impact on Model 1, and improve the live rainfall-history layer. 2026 observations should initially be treated as a current/live validation period rather than as a completed annual training set.

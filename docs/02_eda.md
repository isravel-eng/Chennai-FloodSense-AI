# EDA Documentation

## Objective

EDA is used to understand rainfall behavior, locality variation, seasonality, missingness, and the rare-event nature of flood labels before model training.

## Data Views

The project contains:

- Historical raw data: `data/raw/master_dataset.csv`
- Model 2 engineered data: `data/processed/model2_features.csv`
- City-wide monthly series for Model 1: `data/processed/monthly_rainfall_citywide.csv`

## Required EDA Checks

### 1. Temporal coverage

Confirm the historical range and inspect observations by year and month.

### 2. Rainfall distribution

Inspect daily rainfall distributions, zero-rainfall frequency, high-rainfall extremes, and summary statistics.

### 3. Seasonality

Compare rainfall by month and identify the Northeast Monsoon period used by the feature pipeline.

### 4. Locality variation

Compare rainfall behavior across the 30 localities and verify locality-level coordinates/elevation metadata.

### 5. Flood-event imbalance

Measure the positive flood-event rate. The current README reports a 1.1% positive rate for the rare-event holdout used for Model 2 evaluation.

### 6. Feature relationships

Inspect relationships between rainfall windows/lags, seasonality, location, and the flood label. Avoid leakage from future observations.

## EDA Deliverables

The final EDA report should contain:

1. Dataset shape and date range
2. Missing-value report
3. Rainfall summary statistics
4. Monthly/seasonal rainfall plots
5. Locality comparison
6. Extreme-rainfall analysis
7. Flood-class distribution
8. Feature correlation or association analysis
9. Key observations and limitations

## Current Status

The repository contains the datasets and feature-engineering pipeline, but this document deliberately does not invent numerical EDA findings that have not been recorded as project results. The plots and measured findings should be added when the EDA notebook/report is finalized.

# Machine Learning

The machine-learning branch contains the complete Machine Learning work for Chennai FloodSense AI.

## ML Branch

**Branch:** `machine-learning`

All ML development should be done in this branch only. The ML branch is kept separate from the application code on `main` so model development can continue without interfering with backend, frontend, or deployment work.

## ML Scope

The machine-learning branch contains:

- Model 1 — rainfall forecasting
- Model 2 — flood-risk classification
- Dataset collection and preprocessing
- Feature engineering
- Model training and evaluation
- Model comparison and selection
- Trained model artifacts
- Live-weather-to-ML feature processing
- ML tests
- ML documentation

## Models

### Model 1 — Rainfall Forecasting

- Purpose: city-wide rainfall forecasting
- Current approach: SARIMA
- Output: monthly rainfall forecast

### Model 2 — Flood Risk Prediction

- Purpose: locality/day flood-risk prediction
- Current candidate models: Random Forest and XGBoost
- Evaluation focus: PR-AUC because the flood-event target is highly imbalanced
- Output: flood probability and risk band

## Current ML Data

The original training data covers historical Chennai rainfall observations through 2023. Future ML work includes adding verified newer observations and retraining the models without fabricating locality-level data.

## Integration With Main

The ML branch is **not intended to be merged wholesale into `main`**.

The application teams work on `main`:

```text
main/
├── backend/
├── frontend/
└── deployment/
```

The ML team works on:

```text
machine-learning/
├── data/
├── model_1_rainfall/
├── model_2_flood/
├── live/
├── models/
├── tests/
└── docs/
```

When the ML implementation is ready, the backend should integrate with the ML prediction interface/API rather than importing unrelated ML development files into `main`.

## Team Branch Responsibilities

| Branch | Responsibility |
|---|---|
| `main` | Overall application integration |
| `machine-learning` | All ML work |

Inside `main`, the application work is organized into:

- `backend/`
- `frontend/`
- `deployment/`

## ML Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Comparison
   ↓
Evaluation
   ↓
Model Selection
   ↓
Prediction Interface
   ↓
Backend Integration
```

## Important Rule

Do not create separate ML branches for Model 1, Model 2, V2, datasets, or experiments unless the team later adopts a dedicated experiment workflow. For the current project structure, **`machine-learning` is the single canonical ML development branch.**

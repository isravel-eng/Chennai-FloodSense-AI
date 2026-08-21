# Chennai FloodSense AI — Machine Learning

This branch is the **single source of truth for all Machine Learning work** in Chennai FloodSense AI.

## Branch scope

```text
machine-learning/
├── data/                 # ML datasets and processed features
├── model_1_rainfall/     # Model 1: rainfall forecasting
├── model_2_flood/        # Model 2: locality flood-risk classifier
├── live/                 # ML live-weather feature/prediction layer
├── models/               # trained ML artifacts
├── tests/                # ML tests
├── docs/                 # ML documentation, evaluation and build records
├── .github/              # ML retraining automation
├── predict_end_to_end.py # ML-only CLI
└── requirements.txt      # ML dependencies
```

## Model 1

SARIMA-based city-wide monthly rainfall forecasting. This is a research/trend component and is kept separate from the near-term live flood-risk path.

## Model 2

XGBoost/RandomForest flood-risk classification workflow with feature engineering, preprocessing, evaluation and live inference support. Model selection uses PR-AUC because documented flood events are a rare class.

## Live ML layer

The `live/` package converts locality coordinates and live weather/rainfall history into the feature contract expected by the flood-risk model and produces current and next-24-hour risk predictions.

## Data

The branch contains the existing historical dataset and processed ML datasets. Future 2024–2026 data additions must preserve source provenance, locality consistency, temporal ordering and flood-label validity. Do not fabricate locality-level observations by copying city/district totals across localities.

## ML-only rule

Do not add FastAPI application code, React components, PostgreSQL application code, UI implementation or deployment infrastructure here. The integrated application lives on `main`.

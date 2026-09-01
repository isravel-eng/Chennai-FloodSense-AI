# Backend

FastAPI service for Chennai FloodSense AI.

## Flow

```text
Request
  ↓
FastAPI endpoint
  ↓
Weather / locality data
  ↓
Prediction function
  ↓
JSON response
```

## Endpoints

```text
GET /api/v1/health
GET /api/v1/localities
GET /api/v1/flood-risk/{locality}
GET /api/v1/daily-forecast/{locality}
GET /api/v1/rainfall-forecast/locality/{locality}
GET /api/v1/rainfall-forecast
```

## Folder roles

```text
api/              → HTTP endpoints
live/             → live weather + prediction helpers
model_1_rainfall/ → rainfall forecasting
model_2_flood/    → flood-risk prediction
data/             → runtime datasets
models/           → saved model files
```

The backend is the bridge between the frontend and the ML prediction stack. Keep UI code out of this branch and keep experimental ML work on `machine-learning`.
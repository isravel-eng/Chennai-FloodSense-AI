# Backend Handover Specification

## Objective

Expose the existing ML pipeline through a backend API without changing the trained model or feature contract.

## Backend Input

Primary request input:

```json
{
  "locality": "Sholinganallur"
}
```

## Backend Processing

```text
HTTP request
    |
    v
Validate locality
    |
    v
Load / call live prediction pipeline
    |
    +--> locality lookup
    +--> weather API
    +--> rainfall history
    +--> live feature builder
    +--> XGBoost model
    |
    v
Return prediction JSON
```

## Suggested Endpoints

### Current + next 24-hour prediction

```text
GET /api/v1/flood-risk/{locality}
```

Example:

```text
GET /api/v1/flood-risk/Sholinganallur
```

### Health check

```text
GET /api/v1/health
```

## Suggested Response

```json
{
  "locality": "Sholinganallur",
  "updated": "2026-08-19T14:15",
  "current": {
    "rainfall_mm": 0.1,
    "probability": 0.0001,
    "risk_band": "LOW"
  },
  "next_24h": {
    "forecast_rainfall_mm": 28.8,
    "probability": 0.0,
    "risk_band": "LOW"
  }
}
```

The numeric values above are an example of the structure, not a permanent API response.

## ML Integration Boundary

The backend should treat the ML layer as a service/module with a stable interface. It should not recreate feature engineering in Java/Node/etc. The Python ML pipeline remains the source of truth for model inputs.

## Model Artifacts

Required artifacts currently include:

```text
models/rainfall_model.pkl
models/rainfall_preprocessing.pkl
models/flood_model.pkl
models/flood_preprocessing.pkl
```

Model 1 is a research forecast component. Model 2 is the model driving live flood-risk predictions.

## Operational Requirements

- Handle unknown locality names with a clear 4xx response.
- Handle weather API timeout/failure explicitly.
- Do not silently substitute arbitrary weather values.
- Log model/version and prediction timestamp.
- Keep model artifacts versioned.
- Do not expose internal stack traces to API clients.
- Add request/response validation.

## Handover Checklist

- [ ] Backend API created
- [ ] Locality validation implemented
- [ ] ML prediction module integrated
- [ ] Weather API failure handling implemented
- [ ] API response schema finalized
- [ ] CORS/authentication requirements decided
- [ ] Docker/deployment strategy decided
- [ ] Integration test completed

# Backend Handover Specification

## 1. Objective

Expose the existing ML pipeline through a backend API without changing the trained model or feature contract.

The backend is responsible for HTTP/API concerns, locality validation, persistence, error handling, and integration with the frontend. The Python ML pipeline remains the source of truth for feature engineering and prediction.

---

## 2. Supported Localities

The backend **must validate locality names against `data/processed/locality_lookup.csv`**. Do not invent or silently normalize an unknown locality to another locality.

There are currently **30 locality entries**:

| # | Locality | Latitude | Longitude | Elevation (m approx.) |
|---:|---|---:|---:|---:|
| 1 | Alandur | 13.0067 | 80.2000 | 9 |
| 2 | Ambathur | 13.1143 | 80.1548 | 15 |
| 3 | Anna university | 13.0108 | 80.2339 | 8 |
| 4 | Ayanavaram taluk office | 13.1080 | 80.2320 | 9 |
| 5 | CD Hospital Tondiarpet | 13.1250 | 80.2900 | 3 |
| 6 | Chennai AP | 12.9941 | 80.1709 | 8 |
| 7 | Chennai collectorate building | 13.0940 | 80.2900 | 4 |
| 8 | Chennai nungambakkam | 13.0604 | 80.2496 | 8 |
| 9 | Chennai port trust | 13.0989 | 80.2963 | 4 |
| 10 | DGP Office | 13.0524 | 80.2824 | 5 |
| 11 | Gov hr sec school MGR Nagar | 13.0350 | 80.2100 | 10 |
| 12 | Govt. arts college | 13.0350 | 80.2400 | 7 |
| 13 | MYLAPORE-TRIPLICANE TALUK | 13.0368 | 80.2676 | 4 |
| 14 | Pachaiyappa college | 13.0850 | 80.2707 | 7 |
| 15 | Perambur Corporation park | 13.1150 | 80.2350 | 9 |
| 16 | Purasawalkam - Perambur | 13.0900 | 80.2500 | 8 |
| 17 | Sholinganallur | 12.9010 | 80.2279 | 5 |
| 18 | Zone 02 Manali | 13.1700 | 80.2600 | 3 |
| 19 | Zone 03 Puzhal | 13.1600 | 80.1900 | 10 |
| 20 | Zone 06 D65 Kolathur | 13.1200 | 80.2100 | 12 |
| 21 | Zone 06 T.V.K Nagar | 13.1150 | 80.2200 | 10 |
| 22 | Zone 07 U18 D81 Vanagaram | 13.0650 | 80.1450 | 15 |
| 23 | Zone 08 Anna Nagar | 13.0850 | 80.2101 | 10 |
| 24 | Zone 08 Malar colony | 13.0700 | 80.2250 | 9 |
| 25 | Zone 11 Valasaravakkam | 13.0450 | 80.1750 | 12 |
| 26 | Zone 12 Meenambakkam | 12.9941 | 80.1709 | 8 |
| 27 | Zone 13 Adyar | 13.0012 | 80.2565 | 4 |
| 28 | Zone 13 Adyar Eco Park | 12.9950 | 80.2500 | 4 |
| 29 | Zone 14 U41 Perungudi | 12.9650 | 80.2420 | 3 |
| 30 | Zone 15 Sholinganallur | 12.9010 | 80.2279 | 5 |

These names and coordinates come directly from `data/processed/locality_lookup.csv`. fileciteturn8file0L2-L2

### Locality handling requirements

- Treat the locality value as an identifier, not free-form text.
- Return a clear `404` or `400` response for an unsupported locality.
- Preserve the canonical locality name in API responses.
- The backend may expose a dropdown/list endpoint using the canonical names above.
- Note that `Sholinganallur` and `Zone 15 Sholinganallur` are separate locality entries even though they currently have the same coordinates. Do not merge them unless the ML/data layer is deliberately changed.

---

## 3. Recommended API Endpoints

### Get supported localities

```text
GET /api/v1/localities
```

Recommended response:

```json
{
  "count": 30,
  "localities": [
    {
      "name": "Alandur",
      "latitude": 13.0067,
      "longitude": 80.2,
      "elevation_m_approx": 9
    }
  ]
}
```

The frontend should use this endpoint rather than hard-coding the locality list.

### Current + next 24-hour flood-risk prediction

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

### Optional later endpoints

```text
GET /api/v1/forecast/{locality}
GET /api/v1/map/risk
GET /api/v1/alerts
```

These should be added only when their data contracts are finalized.

---

## 4. Backend Input

Primary prediction request input:

```json
{
  "locality": "Sholinganallur"
}
```

For the initial API, the locality is the only user-provided ML input. The backend/ML layer obtains the remaining features from the locality lookup, weather API, rainfall history, and feature builder.

---

## 5. Backend Processing Flow

```text
HTTP request
    |
    v
Validate locality against locality_lookup.csv
    |
    v
Get locality coordinates/elevation
    |
    v
Call live prediction pipeline
    |
    +--> Open-Meteo weather API
    +--> rainfall history / climatology fallback
    +--> live feature builder
    +--> XGBoost model
    |
    v
Return current + next-24h risk
```

The current live layer uses Open-Meteo and builds the exact 15-feature vector expected by the existing XGBoost model. fileciteturn3file0

---

## 6. Model 2 Feature Contract

The backend must **not recreate these calculations itself**. It should call the Python ML layer.

The current XGBoost model expects these 15 features in this exact order:

```text
1.  rainfall_mm
2.  rainfall_3d_mm
3.  rainfall_7d_mm
4.  rainfall_30d_mm
5.  latitude
6.  longitude
7.  elevation_m_approx
8.  month
9.  month_sin
10. month_cos
11. is_northeast_monsoon
12. rainfall_lag_1
13. rainfall_lag_2
14. rainfall_lag_3
15. rainfall_lag_7
```

The live feature builder is responsible for producing this vector in training order. fileciteturn3file0

### Important

Do **not** add temperature, humidity, or wind to the current XGBoost input. The saved flood model was not trained on those variables. They are available from the weather API but are not valid model inputs for Version 1. fileciteturn3file0

---

## 7. Suggested Response Contract

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

The numeric values above are an example of the response structure, not a permanent response.

Recommended risk-band values:

```text
LOW
MEDIUM
HIGH
```

---

## 8. Model 1 — SARIMA Handover

Model 1 is a separate monthly, city-wide rainfall research forecast. It is **not currently part of the live flood-risk prediction path**. The current CLI command is:

```bash
python predict_end_to_end.py --forecast-rainfall --months 12
```

The existing repository documents the current Model 1 holdout metrics as MAE 4.73 mm and RMSE 6.03 mm on a 24-month holdout. fileciteturn3file0

The backend should not use Model 1 output as a direct substitute for near-term daily/hourly rainfall unless the ML contract is deliberately redesigned.

---

## 9. Model Artifacts

Required artifacts currently include:

```text
models/rainfall_model.pkl
models/rainfall_preprocessing.pkl
models/flood_model.pkl
models/flood_preprocessing.pkl
```

Model 1 is the research forecast component. Model 2 is the model driving live flood-risk predictions. fileciteturn3file0

---

## 10. Live Data Limitation

The live rainfall-history component currently uses an append-only rainfall log when sufficient real observations are available. Until enough real locality-level observations are accumulated, it can fall back to 1993–2023 seasonal climatology. fileciteturn3file0

This limitation must be visible to the backend/API design. A future response may need a field such as:

```json
{
  "history_source": "climatology_fallback"
}
```

so the frontend can distinguish observed-history predictions from fallback-history predictions.

---

## 11. Backend Operational Requirements

- Validate locality before calling the ML pipeline.
- Use the canonical locality names from `locality_lookup.csv`.
- Handle unknown locality names with a clear 4xx response.
- Handle weather API timeout/failure explicitly.
- Do not silently substitute arbitrary weather values.
- Log model/version and prediction timestamp.
- Keep model artifacts versioned.
- Do not expose internal stack traces to API clients.
- Add request/response validation.
- Keep ML feature engineering in Python rather than duplicating it in the backend language.
- Do not merge locality records with identical coordinates unless the ML/data layer is updated accordingly.

---

## 12. Backend Handover Checklist

### ML integration

- [ ] Python ML prediction module integrated
- [ ] `flood_model.pkl` loaded successfully
- [ ] `flood_preprocessing.pkl` loaded successfully
- [ ] Exact 15-feature order preserved
- [ ] Model probability returned
- [ ] Risk band returned

### Locality integration

- [ ] `locality_lookup.csv` integrated
- [ ] All 30 canonical locality names supported
- [ ] Latitude/longitude/elevation available to prediction layer
- [ ] Unknown locality returns 4xx
- [ ] `/api/v1/localities` implemented
- [ ] Frontend dropdown/map uses API locality list

### Live weather

- [ ] Open-Meteo integration connected
- [ ] Current precipitation handled
- [ ] Next-24h precipitation handled
- [ ] Weather API timeout/error handling implemented
- [ ] Rainfall-history source exposed when required

### API

- [ ] `/api/v1/health`
- [ ] `/api/v1/localities`
- [ ] `/api/v1/flood-risk/{locality}`
- [ ] Response schema finalized
- [ ] Error schema finalized
- [ ] CORS requirements decided
- [ ] Authentication requirements decided

### Testing and deployment

- [ ] Integration test completed
- [ ] Unknown-locality test completed
- [ ] Weather API failure test completed
- [ ] ML prediction response test completed
- [ ] Docker/deployment strategy decided

# Chennai FloodSense AI — API Contract

Base URL while developing locally: `http://127.0.0.1:8000`

## Core endpoints

### Health
`GET /api/v1/health`

### Localities
`GET /api/v1/localities`

Returns locality name, latitude, longitude and elevation.

### Current + next 24h + next 7 days flood risk
`GET /api/v1/flood-risk/{locality}`

Response contains:

- `current`: observed/current rainfall, probability and risk band
- `next_24h`: next-24-hour rainfall, probability and risk band
- `next_7_days`: one row per forecast day containing date, rainfall, flood probability and risk band
- `context`: recent rainfall and season information

### Daily forecast only
`GET /api/v1/daily-forecast/{locality}`

Returns the next seven calendar days. The daily rainfall source is the live Open-Meteo forecast; Model 2 converts each forecast day into a flood probability/risk band.

### All-locality live risk
`GET /api/v1/flood-risk-all`

Runs the live prediction for every configured locality.

## Long-term rainfall

### City-wide monthly forecast
`GET /api/v1/rainfall-forecast?months=12`

Supported horizons: 1–36 months. This is the existing city-wide Model 1 SARIMA forecast.

### Locality-wise long-term forecast
`GET /api/v1/rainfall-forecast/locality/{locality}?months=12`

Supported horizons: 12, 24, or 36 months.

This endpoint fits a locality-specific seasonal SARIMA model from that locality's historical daily rainfall after monthly aggregation. It does **not** copy the city-wide forecast to the locality. If the locality has fewer than 24 observed monthly points, the API returns `insufficient_historical_data` instead of fabricating a forecast.

## Important modelling boundary

Daily 1–7 day rainfall is supplied by the live weather forecast because the historical locality series is irregular and does not support a defensible fitted daily ARIMA/SARIMA model. The academic time-series component remains ARIMA/SARIMA/Holt-Winters and the locality-specific long-term forecast.

The Open-Meteo forecast API supports current conditions, hourly variables and daily aggregations including `precipitation_sum` and `rain_sum`, with timezone-aware daily output. See the provider documentation before deployment.

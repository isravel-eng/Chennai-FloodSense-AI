# Backend

FastAPI application for the Chennai FloodSense AI integration branch.

## API contract

The current backend Swagger export demonstrates these endpoints:

```text
GET /api/v1/health
GET /api/v1/localities
GET /api/v1/flood-risk/{locality}
```

The demonstrated server runs on `127.0.0.1:8000`.

### Health

```json
{
  "status": "ok"
}
```

### Localities

Returns the available Chennai localities with their coordinates and approximate elevation. The demonstrated API currently returns 30 localities.

### Flood risk

Example path:

```text
GET /api/v1/flood-risk/Ambathur
```

The demonstrated response contains:

- `locality`
- `updated_at`
- `current.rainfall_input_mm`
- `current.probability`
- `current.risk_band`
- `next_24h.forecast_rainfall_mm`
- `next_24h.probability`
- `next_24h.risk_band`
- `context.rainfall_last_7d_mm`
- `context.rainfall_last_30d_mm`
- `context.rainfall_history_source`
- `context.is_northeast_monsoon`

## Ownership

Backend developers should add the actual FastAPI source code here. ML implementation belongs only to the `machine-learning` branch.

# Chennai FloodSense AI — API Contract (Backend ↔ Frontend)

Base URL while developing locally:http://127.0.0.1:8000


CORS is open for local development, so this can be called directly from
a React app running on a different port (e.g. localhost:3000).

---

## 1. Health check

GET /api/v1/health

Response `200`:
```json
{ "status": "ok" }
```

Use this to check if the backend is reachable before showing predictions.

---

## 2. Get all supported localities
GET /api/v1/localities

Response `200`:
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

**Use this to build the locality dropdown/search box.**
Do NOT hard-code the 30 locality names in the frontend — always fetch
them from here, so if the backend's data changes, the frontend stays
in sync automatically.

---

## 3. Get flood risk for a locality (main prediction endpoint)
GET /api/v1/flood-risk/{locality}

`{locality}` must be one of the exact names returned by `/api/v1/localities`
(case-insensitive, but spelling must match — e.g. "Sholinganallur", not "sholinganalur").
If it has spaces, URL-encode it (e.g. "MYLAPORE-TRIPLICANE TALUK" → `MYLAPORE-TRIPLICANE%20TALUK`).

Example:GET /api/v1/flood-risk/Sholinganallur


### Success response `200`
```json
{
  "locality": "Sholinganallur",
  "updated_at": "2026-08-21T17:00",
  "current": {
    "rainfall_input_mm": 0.5,
    "probability": 0.0001,
    "risk_band": "LOW"
  },
  "next_24h": {
    "forecast_rainfall_mm": 3.1,
    "probability": 0.0001,
    "risk_band": "LOW"
  },
  "context": {
    "rainfall_last_7d_mm": 18.2,
    "rainfall_last_30d_mm": 60.5,
    "rainfall_history_source": "climatology_fallback",
    "is_northeast_monsoon": false
  }
}
```

**Fields the frontend should display:**
- `current.risk_band` and `next_24h.risk_band` → the main thing to show,
  values are always one of: `"LOW"`, `"MEDIUM"`, `"HIGH"`
  (suggested colors: green / yellow / red)
- `current.probability` / `next_24h.probability` → a number between 0 and 1,
  can optionally show as a percentage
- `updated_at` → timestamp, show as "last updated" text
- `context.rainfall_history_source` → if this says `"climatology_fallback"`,
  consider showing a small note like "based on seasonal averages" since
  it means real recent rainfall data isn't available yet for that locality

### Error responses

| Status | When it happens | Response body |
|---|---|---|
| `404` | Locality name not recognized | `{"detail": "Unknown locality '...'. Known localities: ..."}` |
| `503` | Weather service (Open-Meteo) unreachable/timeout | `{"detail": "Weather service unavailable, try again shortly."}` |
| `500` | Unexpected server-side error | `{"detail": "Prediction failed: ..."}` |

**Frontend should handle all three** — e.g. show a friendly message like
"Couldn't get a prediction right now, please try again" rather than a
blank screen or raw JSON.

---

## 4. Known limitations (good to show in the UI)

- Rainfall history may fall back to 1993–2023 seasonal averages instead of
  real recent data for a locality (`context.rainfall_history_source`).
  Not a bug — just a data-maturity limitation on Version 1.
- The model only uses rainfall + location + season — NOT temperature,
  humidity, or wind, even though those are visually interesting. Don't
  imply the app is using them.

---

## 5. Status
Last tested and confirmed working: 2026-08-21, against `Sholinganallur`,
returned real live data successfully.
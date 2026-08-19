# UI Wireframe

## Purpose

The UI should expose the existing prediction pipeline without changing the ML feature contract.

## Main Screen

```text
+-------------------------------------------------------+
|              CHENNAI FLOODSENSE AI                   |
|          Local Flood Risk Prediction                  |
+-------------------------------------------------------+
|                                                       |
|  Select locality                                      |
|  [ Sholinganallur                         v ]         |
|                                                       |
|  [ GET CURRENT RISK ]   [ NEXT 24H ]                 |
|                                                       |
+-------------------------------------------------------+
| CURRENT RISK                                          |
|                                                       |
|  Probability: 0.0001                                  |
|  Risk: LOW                                            |
|  Rainfall today: 0.1 mm                               |
|                                                       |
+-------------------------------------------------------+
| NEXT 24 HOURS                                         |
|                                                       |
|  Forecast rainfall: 28.8 mm                           |
|  Probability: 0.0                                     |
|  Risk: LOW                                            |
|                                                       |
+-------------------------------------------------------+
| DATA CONTEXT                                          |
|  Last 7 days: 20.9 mm                                 |
|  Last 30 days: 120.7 mm                               |
|  Weather source: Open-Meteo                           |
+-------------------------------------------------------+
```

## Backend Contract

The frontend should send a locality identifier/name to the backend. The backend should call the live prediction service and return structured JSON rather than exposing model internals directly.

Suggested response shape:

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

The exact field names should be finalized with the backend implementation before integration.

# Frontend

React + Leaflet + Recharts UI for Chennai FloodSense AI.

## Pages

- **Map** — Chennai locality map with current, next-24h and next-7-day flood risk.
- **Localities** — searchable locality list.
- **Rainfall Prediction** — locality-wise 12/24/36-month SARIMA rainfall forecast with confidence intervals.

## Backend

Set `VITE_API_BASE_URL` when the API is not running at `http://127.0.0.1:8000/api/v1`.

The frontend uses the backend boundary only; Python ML code is not imported into React.

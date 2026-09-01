# Chennai FloodSense AI — Frontend

React + Leaflet dashboard for Chennai flood-risk monitoring.

## Flow

```text
User selects locality
        ↓
Frontend requests API
        ↓
Backend returns prediction
        ↓
Map + risk panel + rainfall chart
        ↓
User sees result
```

## Structure

```text
src/
├── App.jsx
├── main.jsx
└── styles.css
index.html
package.json
README.md
```

## Screens

```text
Map → locality markers + flood risk
Localities → searchable locality list
Rainfall Prediction → SARIMA forecast
```

## Run

```bash
npm install
npm run dev
```

Backend URL:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

# Chennai FloodSense AI — Backend

FastAPI service that connects the frontend to the flood-prediction and rainfall-forecasting logic.

## Flow

```text
Frontend request
      ↓
FastAPI endpoint
      ↓
Read locality / weather data
      ↓
Run prediction logic
      ↓
Return JSON response
      ↓
Frontend displays result
```

## API flow

```text
GET /health
     ↓
Server status

GET /localities
     ↓
Locality coordinates

GET /flood-risk/{locality}
     ↓
Live weather + ML prediction
     ↓
Current + 24h + 7-day risk

GET /rainfall-forecast/locality/{locality}
     ↓
SARIMA forecast
     ↓
12 / 24 / 36 month rainfall prediction
```

## Structure

```text
backend/
├── api/                 # FastAPI endpoints
├── live/                # Live weather + prediction helpers
├── model_1_rainfall/    # Rainfall forecasting code
├── model_2_flood/       # Flood-risk model code
├── data/                # Runtime data
├── models/              # Trained model files
└── README.md
```

## Run

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`

## Responsibility

Keep API and runtime prediction integration here. Frontend UI stays on `frontend`. Model-development experiments stay on `machine-learning`.
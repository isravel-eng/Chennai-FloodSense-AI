# Chennai FloodSense AI

Application for locality-level flood-risk monitoring and rainfall forecasting in Chennai.

## Branches

| Branch | Owns |
|---|---|
| `main` | Final integrated application |
| `frontend` | React + Leaflet dashboard |
| `backend` | FastAPI + runtime prediction integration |
| `machine-learning` | Model development, training, evaluation and ML experiments |

## Architecture

```text
                 ┌───────────────┐
                 │    Frontend   │
                 │ React + Map   │
                 └───────┬───────┘
                         │ HTTP
                         ↓
                 ┌───────────────┐
                 │    Backend    │
                 │    FastAPI    │
                 └───────┬───────┘
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
        Live Weather           ML Models
              │                     │
              └──────────┬──────────┘
                         ↓
                  Flood / Rainfall
                     Prediction
```

## Development rule

Work in the branch that owns the code. Merge into `main` only for integration.

```text
frontend  → UI
backend   → API + runtime integration
ML        → models + experiments
main      → integration
```

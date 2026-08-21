# Chennai FloodSense AI

Overall integration repository for the Chennai FloodSense AI project.

The repository is intentionally split so each team member can work without mixing responsibilities:

```text
Chennai-FloodSense-AI/
├── backend/              # FastAPI backend
├── frontend/             # React + map UI
├── deployment/           # Docker / cloud deployment
├── .gitignore
└── README.md
```

## Branch ownership

| Branch | Purpose |
|---|---|
| `main` | Overall application: backend, frontend, deployment and integration documentation |
| `machine-learning` | All ML work only: datasets, Model 1, Model 2, live ML inference, training, evaluation, tests and ML artifacts |

The previous `upgrade/model-2-v2` branch is being retired. Its required Model 2 V2 work is preserved in `machine-learning`.

## Team integration

### Machine Learning

The ML branch owns the complete prediction stack and exposes the contract required by the backend. It contains the rainfall model, flood-risk model, feature engineering, evaluation, live weather/rainfall processing and model artifacts.

### Backend

The backend consumes the ML prediction contract and exposes HTTP endpoints to the frontend. The current API contract demonstrated in the backend Swagger export includes:

```text
GET /api/v1/health
GET /api/v1/localities
GET /api/v1/flood-risk/{locality}
```

The demonstrated flood-risk response contains locality, update time, current rainfall/probability/risk, next-24-hour forecast/probability/risk, and rainfall context.

See `backend/README.md` for the integration contract.

### Frontend

The frontend owns locality selection, interactive Chennai map display, flood-risk visualization and user-facing warnings.

See `frontend/README.md`.

### Deployment

Deployment owns Docker/container configuration, environment variables, service orchestration and cloud deployment configuration.

See `deployment/README.md`.

## Development rule

Do not place ML training code, datasets or trained model artifacts on `main`. Do not place backend/frontend/deployment implementation inside `machine-learning`.

Changes should be made in the appropriate branch and then integrated into `main` only when the complete application is ready.

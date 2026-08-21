# Deployment

Deployment configuration for the integrated Chennai FloodSense AI application.

## Responsibilities

- Dockerfiles and container configuration
- Environment-variable management
- Backend/frontend service orchestration
- Production configuration
- Cloud deployment configuration
- Health checks and deployment documentation

## Planned services

```text
frontend
   │
   ▼
backend (FastAPI)
   │
   ▼
machine-learning prediction layer
```

The exact deployment topology should be finalized after the backend and frontend implementations are merged into `main`.

## Rule

Do not copy the ML training dataset or development notebooks into deployment images. Deployment should contain only the runtime artifacts required by the integrated application.

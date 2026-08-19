# Project Documentation

This folder contains the project documents prepared for the ML-to-backend handover.

| Document | Purpose |
|---|---|
| [01_dataset.md](01_dataset.md) | Dataset sources, timeline, and limitations |
| [02_eda.md](02_eda.md) | EDA plan and required findings |
| [03_features.md](03_features.md) | Model 2 feature contract and live feature construction |
| [04_algorithm_comparison.md](04_algorithm_comparison.md) | Model 1 and Model 2 algorithm choices |
| [05_model_evaluation.md](05_model_evaluation.md) | Recorded evaluation metrics and test status |
| [06_ml_architecture.md](06_ml_architecture.md) | End-to-end ML architecture |
| [07_ui_wireframe.md](07_ui_wireframe.md) | Initial frontend/UI wireframe and response shape |
| [08_model1_validation.md](08_model1_validation.md) | SARIMA runtime verification and validation record |
| [09_backend_handover.md](09_backend_handover.md) | Backend API and ML integration contract |
| [10_handover_checklist.md](10_handover_checklist.md) | Completion and final handover checklist |

## Current Phase

The ML pipeline is functional enough for backend integration testing. Model 1 SARIMA inference has been manually verified through the CLI. Automated tests still need `pytest` installed and executed.

The next major ML/data task is to evaluate verified 2024–2025 data before deciding whether to retrain the models. The backend can begin against the Version 1 API contract while that evaluation is performed separately.

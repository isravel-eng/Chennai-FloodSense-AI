# Project Handover Checklist

## ML Work Completed

- [x] GitHub repository created
- [x] Core ML pipeline committed
- [x] Saved Model 1 artifact available
- [x] Saved Model 2 artifact available
- [x] Live weather layer implemented
- [x] Historical prediction path available
- [x] Model 1 runtime forecast verified
- [x] Dataset documentation added
- [x] EDA documentation structure added
- [x] Feature contract documented
- [x] Algorithm comparison documented
- [x] Evaluation metrics documented
- [x] ML architecture documented
- [x] UI wireframe documented
- [x] Backend handover specification added

## Still Required Before Final Delivery

- [ ] Install `pytest` and run the complete test suite
- [ ] Resolve the XGBoost serialized-model version warning by exporting/re-saving the model using the recommended compatible model format/version
- [ ] Add verified 2024–2025 rainfall data
- [ ] Re-evaluate Model 1 with a time-based holdout
- [ ] Decide whether the newer data materially improves the model
- [ ] Improve the live rainfall-history log so recent local observations replace climatology fallback
- [ ] Finalize backend API contract
- [ ] Connect backend to the ML prediction layer
- [ ] Connect frontend to backend
- [ ] Add deployment configuration
- [ ] Add final project screenshots/results

## Handover Boundary

The ML team hands over:

1. Trained model artifacts
2. Feature preprocessing artifacts
3. Prediction functions / CLI
4. Live weather integration
5. Dataset and feature documentation
6. Model evaluation results
7. API integration contract

The backend team owns API exposure, service orchestration, error handling, authentication/CORS if required, deployment, and frontend integration.

## Current Version Boundary

The current live model is Version 1. It uses rainfall, location, elevation, lag, and seasonality features. Temperature, humidity, and wind are not currently model inputs and must not be added at inference time without retraining.

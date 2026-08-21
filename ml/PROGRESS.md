# ML branch progress

## Done

- Created isolated `ml` branch from `main`.
- Documented Phase I/II ML scope from the project presentation.
- Documented a 2024–2026 data acquisition contract.
- Recorded verified IMD 2024 and 2026 data leads without contaminating the training dataset.

## Not yet safe to claim as completed

- Full locality-level 2024–2026 daily dataset for all 30 localities.
- Verified 2024–2026 locality-level flood labels.
- Full retraining and honest time-based evaluation on the extended labelled dataset.
- Final optimized model release for Backend integration.

The reason is methodological: the available newer IMD material found so far is station/district aggregate rainfall, while Model 2 expects locality/day rows and a documented flood target. The ML branch therefore preserves provenance instead of manufacturing labels or duplicating aggregate observations across localities.

## Next ML implementation step

Acquire a consistent daily source at station/grid resolution, map it to the existing locality coordinates, obtain verified flood-event labels, then run a chronological retraining/evaluation pipeline. Only after that should the new model artifact replace the current Model 2 artifact for backend integration.

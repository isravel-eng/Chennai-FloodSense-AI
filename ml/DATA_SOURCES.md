# Data acquisition plan — 2024–2026

## Required additions

The current project data stops at 2023. The ML branch needs newer observations for:

- Model 1 rainfall forecasting: monthly city-wide rainfall.
- Model 2 flood-risk features: daily/locality rainfall, rolling rainfall and lags.
- Live layer validation: observed weather versus API-derived features.

## Source strategy

### IMD

Use India Meteorological Department rainfall observations as the primary authoritative source. IMD exposes daily, weekly, monthly and cumulative rainfall information and provides a historical-data route/contact for deeper historical queries.

Reference: https://mausam.imd.gov.in/responsive/rainfallinformation_state.php

### Open-Meteo

Use the same weather provider already wired into `live/` for current and forecast weather. Do not treat forecast values as historical ground truth when an observed dataset is available.

### Flood labels

Acquire locality-level event labels only from verifiable government/municipal/event records. Do not infer a flood label solely from a rainfall threshold unless the project explicitly documents that proxy target as a separate experiment.

## 2024–2026 dataset contract

Recommended columns:

`date, locality, latitude, longitude, elevation_m_approx, rainfall_mm, source, observation_type, flood_occurred_documented`

Where `flood_occurred_documented` is unknown, store it as null and exclude that row from supervised Model 2 training/evaluation.

## Quality rules

- Deduplicate by `(locality, date, source)`.
- Convert rainfall to millimetres.
- Parse all dates to ISO `YYYY-MM-DD`.
- Keep source provenance for every imported row.
- Validate locality names against `data/processed/locality_lookup.csv`.
- Do chronological splits for model evaluation.
- Never fill an unknown flood label with 0.

## Why this matters

The current Model 2 uses a rare-event documented-flood target. Extending rainfall data is straightforward, but extending the **labelled** flood target requires evidence. The project should distinguish observed weather data from verified flood-event labels so that the new model does not gain apparent accuracy from fabricated negatives.

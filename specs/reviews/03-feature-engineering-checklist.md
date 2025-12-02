# Stage 03 — Feature Engineering Review Checklist

Pre‑conditions
- Inputs available: `indices_final.csv`, `aligned_indices.parquet`, `aligned_environment.parquet`, `community_metrics.parquet`
- Config keys present: `temporal.resolution_hours`, `temporal.cyclic_terms`, `stations.include`, `covariates.scale`, `covariates.to_scale`

Inputs/Methods
- UTC alignment and inner joins on `station, datetime`
- Temporal features computed; grouping IDs and AR1 sequence validated
- Environment merged from aligned artifact; optional scaling per config

Outputs
- `analysis_ready.parquet` with required columns
- `feature_engineering_schema.csv`, `temporal_feature_checks.png`, `feature_engineering_summary.json`

Acceptance Criteria
- Cyclic terms within [-1,1]; contiguous `time_within_day`
- Merge completeness ≥ 95%; imputation ≤ 5% and logged
- Deterministic checksum and row counts per station/year

Sign‑off
- Reviewer, Date
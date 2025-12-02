# Stage 00 — Data Prep & Alignment Review Checklist

Pre‑conditions
- Config keys present: `temporal.resolution_hours`, `stations.include`, `community_metrics.season_definition`
- Raw inputs available and listed in manifest

Inputs/Methods
- UTC normalization applied; 2‑hour alignment confirmed
- Station codes harmonized to `9M, 14M, 37M`
- Environmental units converted to °C and m

Outputs
- `aligned_detections.parquet` exists with `datetime, station`
- `aligned_environment.parquet` exists with `datetime, station, temperature, depth`
- `aligned_indices.parquet` exists with `datetime, station, indices`
- `aligned_spl.parquet` exists when SPL used
- `alignment_schema.csv` and `alignment_summary.json` written

Acceptance Criteria
- Completeness ≥ 95% for detections+environment per station/year
- Environmental imputation ≤ 5%; all imputation logged
- Deterministic checksum recorded; rerun stable
- Unit conversions and dropped rows summarized

Sign‑off
- Reviewer, Date
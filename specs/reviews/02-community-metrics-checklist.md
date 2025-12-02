# Stage 02 — Community Metrics Review Checklist

Pre‑conditions
- Aligned detections present: `aligned_detections.parquet`
- Config keys present: `community_metrics.assume_zero_when_missing`, `community_metrics.season_definition`

Inputs/Methods
- Metrics derived by grouping `station, datetime`
- Column mappings from `det_column_names.csv` applied
- No implicit zero unless explicit absence; rules documented

Outputs
- `community_metrics.parquet` with keys and all metrics
- `community_metrics_schema.csv` and `community_metrics_summary.json`

Acceptance Criteria
- Deterministic derivation; non‑negative counts; binary ∈ {0,1}
- Summary includes station/year totals and presence fraction

Sign‑off
- Reviewer, Date
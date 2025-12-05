# 03 Feature Engineering — Stage Spec

Title
- Temporal and grouping features; merge environmental covariates; analysis‑ready dataset

Purpose
- Create cyclic temporal terms, grouping IDs for random effects, and within‑day sequence needed for AR1; merge environmental variables; output a clean, deterministic dataset for modeling.

Inputs
- Final indices list: `data/processed/indices_final.csv` (from Stage 01)
- Aligned indices: `data/interim/aligned_indices.parquet` (from Stage 00)
- Aligned environment: `data/interim/aligned_environment.parquet` (from Stage 00)
- Community metrics: `data/processed/community_metrics.parquet` (from Stage 03)

Outputs
- `data/processed/analysis_ready.parquet`
  - Columns:
    - Keys: `datetime` (UTC ISO), `datetime_local` (EST, UTC-5 fixed), `date`, `station`
    - Temporal: `hour`, `sin_hour`, `cos_hour`, `day_of_year`, `hour_of_day` (all derived from `datetime_local` for biological interpretation)
    - Grouping: `day_id`, `month_id`
    - Sequence: `time_within_day` (0‑based within each `day_id`)
    - Predictors: final indices from Stage 01
    - Covariates: `temperature`, `depth` (per station/time)
    - Optional scaled covariates: `temperature_z`, `depth_z` (enabled via config)
    - Responses: from Stage 03 community metrics
- `results/tables/feature_engineering_schema.csv`
- `results/figures/temporal_feature_checks.png`
- `results/logs/feature_engineering_summary.json`

Methods
- Time normalization:
  - Parse timestamps to UTC; standardize to 2‑hour resolution; align via inner join on `datetime`+`station` using aligned inputs.
- Temporal features (derived from `datetime_local` for biological interpretation):
  - `hour_of_day = hour(datetime_local)` — local hour (0-23) for day/night patterns
  - `sin_hour = sin(2π * hour_of_day / 24)` — cyclic encoding of daily pattern
  - `cos_hour = cos(2π * hour_of_day / 24)` — cyclic encoding of daily pattern
  - `day_of_year = yday(datetime_local)` — seasonal patterns
- Grouping factors:
  - `day_id = paste(date, station, sep="_")`
  - `month_id = format(date, "%Y-%m")`
- Sequence for AR1:
  - `time_within_day = rank(order(datetime))` within each `day_id` starting at 0.
- Covariate merge:
  - Use `aligned_environment.parquet` for `temperature` and `depth` per `station`+`datetime`.
- Responses:
  - Join `community_metrics.parquet` on `station, datetime`; no derivation in this stage.

Parameters
- `time_resolution_hours`: default `2`.
- `merge_strategy`: `inner` (default), optional `left` for indices; imputation window `max_gap_hours=2`.
- `stations`: `9M, 14M, 37M`.
- `timezone`: `UTC`.

Acceptance Criteria
- All required columns exist with correct types; schema file generated.
- Deterministic transforms: same inputs → same outputs; checksum recorded.
- Temporal cyclic terms in range: `sin_hour, cos_hour ∈ [-1, 1]`.
- Grouping IDs unique per station/date; `time_within_day` sequences contiguous starting at 0.
- Responses non‑negative for counts; binary flags ∈ {0,1}.
- Merge completeness ≥ 95% of timestamps; imputation events logged and ≤ 5%.
- Summary JSON includes row counts per station/year and missingness report.

Edge Cases
- Daylight saving or timezone inconsistencies → normalize to UTC (ignore DST shifts).
- SPL at 1‑hour resolution → aggregate to 2‑hour blocks if used; document method.
- Missing environmental rows → bounded forward‑fill; exceedance flagged.

Performance
- Target runtime: < 15 minutes full; < 2 minutes sample.
- Memory: streaming reads for Excel; prefer column subsets.

- Upstream: Stage 01 indices list; Stage 00 aligned indices/environment; Stage 03 community metrics.
- Downstream: GLMM/GAMM stages consume `analysis_ready.parquet`.

Change Record
- 2025‑12‑04: Changed `datetime_local` from America/New_York (DST-aware) to fixed EST (UTC-5). DST caused alternating hour gaps in downstream heatmaps; fixed offset ensures consistent 2-hour bins year-round.
- 2025‑12‑03: **IMPLEMENTED** - Created analysis-ready dataset with 13,102 observations (2021 only, 3 stations). Features: temporal (hour_of_day, sin_hour, cos_hour, day_of_year from datetime_local), grouping (day_id, month_id), AR1 sequence (time_within_day), 20 acoustic indices, 2 environmental covariates, 9 community metrics. All validation passed.
- 2025‑12‑03: Updated to use `datetime_local` (America/New_York) for all temporal feature extraction. Rationale: biological patterns follow local day/night cycles, not UTC. `datetime` (UTC) retained for merging/alignment only.
- 2025‑11‑21: Renumbered to Stage 03; inputs switched to aligned indices/environment and Stage 02 community metrics; added optional covariate scaling controlled via config; acceptance criteria retained.
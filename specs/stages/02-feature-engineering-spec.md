# 02 Feature Engineering — Stage Spec

Title
- Temporal and grouping features; merge environmental covariates; analysis‑ready dataset

Purpose
- Create cyclic temporal terms, grouping IDs for random effects, and within‑day sequence needed for AR1; merge environmental variables; output a clean, deterministic dataset for modeling.

Inputs
- Index time series:
  - `data/processed/indices_final.csv` (list of kept indices)
  - Source indices files from `data/raw/indices/` (same as Stage 01)
- Detections (manual annotations):
  - `data/raw/2018/detections/Master_Manual_<station>_2h_2018.xlsx`
  - `data/raw/2021/detections/Master_Manual_<station>_2h_2021.xlsx`
- Environmental:
  - `data/raw/2018/environmental/Master_<station>_Temp_2018.xlsx`, `...Depth_2018.xlsx`
  - `data/raw/2021/environmental/Master_<station>_Temp_2021.xlsx`, `...Depth_2021.xlsx`
- SPL (optional):
  - `data/raw/2018/rms_spl/Master_rmsSPL_<station>_1h_2018.xlsx`
  - `data/raw/2021/rms_spl/Master_rmsSPL_<station>_1h_2021.xlsx`

Outputs
- `data/processed/analysis_ready.parquet`
  - Columns:
    - Keys: `datetime` (UTC ISO), `date`, `station`
    - Temporal: `hour`, `sin_hour`, `cos_hour`, `day_of_year`, `hour_of_day`
    - Grouping: `day_id`, `month_id`
    - Sequence: `time_within_day` (0‑based within each `day_id`)
    - Predictors: final indices from Stage 01
    - Covariates: `temperature`, `depth` (per station/time)
    - Responses: per overview (fish activity/richness, dolphin call counts, presence flags, vessel presence)
- `results/tables/feature_engineering_schema.csv`
- `results/figures/temporal_feature_checks.png`
- `results/logs/feature_engineering_summary.json`

Methods
- Time normalization:
  - Parse timestamps to UTC; standardize to 2‑hour resolution; align across sources via inner join on `datetime`+`station`.
- Temporal features:
  - `sin_hour = sin(2π * hour / 24)`
  - `cos_hour = cos(2π * hour / 24)`
  - `day_of_year = yday(date)`
  - `hour_of_day = hour(datetime)`
- Grouping factors:
  - `day_id = paste(date, station, sep="_")`
  - `month_id = format(date, "%Y-%m")`
- Sequence for AR1:
  - `time_within_day = rank(order(datetime))` within each `day_id` starting at 0.
- Covariate merge:
  - Map `temperature` and `depth` per `station`+`datetime` with forward‑fill allowance ≤ 2 hours for occasional gaps; record imputation.
- Response derivation:
  - Fish metrics: activity (intensity sum), richness (species count), presence (binary)
  - Dolphin metrics: burst pulse, click, whistle counts; activity (sum); presence (binary)
  - Vessel presence: binary
  - Derive directly from detections files; document column mappings.

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

Dependencies
- Upstream: Stage 01 outputs; raw detections/environmental files.
- Downstream: GLMM/GAMM stages consume `analysis_ready.parquet`.

Change Record
- YYYY‑MM‑DD: Draft created; acceptance gates aligned to overview.
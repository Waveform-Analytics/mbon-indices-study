# 00 Data Preparation & Temporal Alignment — Stage Spec

Title
- Canonical data cleaning, station harmonization, and temporal alignment to 2‑hour bins

Purpose
- Produce a clean, aligned, and documented dataset that standardizes keys, timestamps, station codes, and units across detections, indices, environmental, and SPL sources. Provide deterministic and auditable transforms for downstream stages.

Inputs
- Detections (manual annotations):
  - `data/raw/2018/detections/Master_Manual_<station>_2h_2018.xlsx`
  - `data/raw/2021/detections/Master_Manual_<station>_2h_2021.xlsx`
- Environmental:
  - `data/raw/<year>/environmental/Master_<station>_Temp_<year>.xlsx`
  - `data/raw/<year>/environmental/Master_<station>_Depth_<year>.xlsx`
- Acoustic indices:
  - `data/raw/indices/*` (per station/year CSVs)
- SPL (optional):
  - `data/raw/<year>/rms_spl/Master_rmsSPL_<station>_1h_<year>.xlsx`
- Metadata:
  - `data/raw/metadata/det_column_names.csv` (column name mapping)
  - `data/raw/metadata/Updated_Index_Categories_v2.csv` (index metadata)

Outputs
- `data/interim/aligned_base.parquet`
  - Keys: `datetime` (UTC, ISO), `datetime_local` (America/New_York), `date`, `hour`, `station`
  - Columns: canonical detection fields (fish/dolphin/vessel), environmental (`temperature`, `depth`), optional SPL fields
- `data/interim/aligned_indices.parquet`
  - Keys: `datetime` (UTC, ISO), `datetime_local` (America/New_York), `station`
  - Columns: acoustic indices aligned/aggregated to 2‑hour bins per station
- `data/interim/aligned_detections.parquet`
  - Keys: `datetime` (UTC, ISO), `datetime_local` (America/New_York), `station`
  - Columns: canonical detection fields aligned to 2‑hour bins
- `data/interim/aligned_environment.parquet`
  - Keys: `datetime` (UTC, ISO), `datetime_local` (America/New_York), `station`
  - Columns: environmental covariates aligned to 2‑hour bins (`temperature`, `depth`)
- `data/interim/aligned_spl.parquet`
  - Keys: `datetime` (UTC, ISO), `datetime_local` (America/New_York), `station`
  - Columns: SPL features aggregated to 2‑hour bins
- `results/tables/alignment_schema.csv`
- `results/figures/alignment_completeness.png`
- `results/logs/alignment_summary.json` (row counts, missingness, imputation events, unit conversions)

Methods
- Timestamp normalization
  - Parse all timestamps to timezone‑aware; convert to `UTC` for alignment and merging.
  - Create `datetime_local` by converting UTC to `America/New_York` timezone for biological interpretation (day/night cycles, fish activity patterns). Local time accounts for EST/EDT transitions.
  - Normalize to 2‑hour resolution; for sources with 1‑hour resolution (SPL), aggregate to 2‑hour bins via mean unless otherwise specified.
- Station harmonization
  - Map station names to canonical set `{9M, 14M, 37M}`; drop/flag any unknown codes.
- Column normalization
  - Apply `det_column_names.csv` to standardize detection columns; coerce types (ints for counts, bools for presence).
- Temporal alignment
  - Perform inner join on `datetime`+`station` across detections and environmental.
  - Align acoustic indices to 2‑hour bins per `station` by aggregating hourly values (mean unless specified) and write `aligned_indices.parquet`.
  - Write separate aligned artifacts for detections (`aligned_detections.parquet`), environmental (`aligned_environment.parquet`), and SPL (`aligned_spl.parquet`) to support downstream contracts.
- Unit handling
  - Ensure `temperature` in °C and `depth` in meters; record any conversions.
- Missing data policy
  - Environmental gaps: forward‑fill up to `max_gap_hours=2`; record imputation events.
  - Detections gaps: do not impute; keep as missing; record completeness per station/year.
  - SPL aggregation gaps: align and report; no imputation beyond aggregation.
- Determinism
  - Sort by `station, datetime`; write outputs with fixed column order; record input file hashes in summary.

Parameters
- `timezone`: `UTC`
- `time_resolution_hours`: `2`
- `env_max_gap_hours`: `2`
- `stations`: `9M, 14M, 37M`
- `spl_aggregation`: `mean`

Acceptance Criteria
- All outputs have canonical keys and types as per schema; schema CSV generated.
- Alignment completeness ≥ 95% across detections+environmental per station/year; completeness reported.
- Environmental imputation events ≤ 5% of aligned rows; all imputation logged.
- Deterministic checksum: reruns on same inputs produce identical `aligned_base.parquet` and summary.
- Unit conversions, station mappings, and dropped rows are summarized and saved to JSON.

Edge Cases
- Overlapping timestamps or duplicates → de‑duplicate by `station, datetime` with defined rule (keep last, report count).
- DST anomalies → neutralized by UTC conversion.
- Unknown or new station codes → flagged and excluded; summary records occurrences.

Performance
- Target runtime: < 10 minutes full; < 1 minute sample.
- Memory: streaming reads for Excel via chunking where needed.

Dependencies
- Upstream: raw files present; metadata mapping for detections and indices.
- Downstream: Stage 01 Index Reduction consumes `data/interim/aligned_indices.parquet`.
- Downstream: Stage 03 Community Metrics consumes `data/interim/aligned_detections.parquet`.

Change Record
- 2025‑12‑03: Added `datetime_local` column (America/New_York timezone) to all aligned outputs. Rationale: biological patterns (fish/dolphin activity) follow local day/night cycles, not UTC. UTC retained for alignment/merging; local time used for temporal feature extraction in downstream stages.
- 2025‑11‑28: **IMPLEMENTED** - All outputs produced and verified against acceptance criteria. Implementation complete.
  - Added config-driven loader architecture: `config/analysis.yml:sources.*` specifies datetime columns, sheet names, timezones per data type.
  - Added unified datetime parser (`src/python/mbon_indices/utils/datetime.py`) with support for `datetime_column` and `compose_datetime` (handles Excel Date+Time combination for SPL).
  - Added environment metadata mapping: `data/raw/metadata/env_column_names.yml` for temperature/depth raw→canonical column names.
  - Critical fix: SPL requires combining Date column (date part) + Time column (time part) via `compose_datetime` to avoid microsecond offsets from Excel datetime serialization.
  - Scripts: stage00-a_verify_loaders.py (config check), stage00-b_align.py (alignment orchestration), stage00-c_generate_qa.py (QA artifacts).
  - Acceptance criteria met: 100% completeness for detections/environment 2021; <5% imputation; all artifacts produced.
- 2025‑11‑21: Added explicit aligned indices output and indices alignment steps; clarified downstream dependency for Stage 01.
- 2025‑11‑21: Split aligned outputs into separate files for detections, environmental, and SPL; updated downstream dependencies for Stage 03.
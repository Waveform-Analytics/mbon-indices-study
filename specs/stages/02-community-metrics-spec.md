# 02 Community Metrics — Stage Spec

Title
- Derive biological community metrics from detections aligned to 2‑hour bins

Purpose
- Compute standardized response variables per station/time for modeling and exploratory analyses.

Inputs
- Aligned detections: `data/interim/aligned_detections.parquet` (from Stage 00)
- Metadata: `data/raw/metadata/det_column_names.csv` for column mappings

Outputs
- `data/processed/community_metrics.parquet`
  - Keys: `datetime`, `station`
  - Metrics:
    - Fish activity (sum of intensity values across fish detections)
    - Fish richness (count of unique fish species present)
    - Fish presence (binary)
    - Dolphin echolocation count
    - Dolphin burst pulse count
    - Dolphin whistle count
    - Dolphin activity (sum across call types)
    - Dolphin presence (binary)
    - Vessel presence (binary)
- `results/tables/community_metrics_schema.csv`
- `results/logs/community_metrics_summary.json` (counts per station/year, missingness)

Methods
- Use `aligned_detections.parquet` as the canonical timeline; derive metrics by grouping on `station, datetime`.
- Column mapping: apply `det_column_names.csv` to identify fish/dolphin/vessel fields using `keep_species=1` filter; excludes interruption columns and non-target species.
- Fish metrics:
  - Species selection: filter `group=fish` AND `keep_species=1` (8 species: Silver perch, Oyster toadfish boat whistle/grunt, Black drum, Spotted seatrout, Red drum, Atlantic croaker, Weakfish).
  - Activity: sum intensity columns per bin; missing treated as zero only when absence is explicit; otherwise leave missing and report.
  - Richness: count unique species detected (>0) per bin.
  - Presence: binary indicator if any fish species present.
- Dolphin metrics:
  - Columns: Bottlenose dolphin echolocation, burst pulses, whistles (convert first two from string to numeric).
  - Counts: extract the 3 call type counts from detections; presence if any call type > 0.
  - Activity: sum of the three counts.
- Vessel metric:
  - Presence: binary from Vessel column.
- Validation:
  - Ensure non‑negative counts; presence flags ∈ {0,1}.

Parameters
- `assume_zero_when_missing`: see `config/analysis.yml -> community_metrics.assume_zero_when_missing`.
- `season_definition`: see `config/analysis.yml -> community_metrics.season_definition`.
- `species_column`: from detection metadata mapping.
- `intensity_columns`: from detection metadata mapping.

Acceptance Criteria
- Schema and types as per `community_metrics_schema.csv`.
- Deterministic derivation: reruns produce identical metrics given identical inputs.
- Non‑negative counts; presence flags valid.
- Summary JSON includes per station/year totals and fraction of bins with any presence.

Edge Cases
- Incomplete species labeling → treat richness conservatively and log.
- Multiple detections per bin → aggregate by sum/count; deduplicate species before counting richness.

Performance
- Target runtime: < 5 minutes full; < 1 minute sample.

- Upstream: Stage 00 aligned detections.
- Downstream: Stage 04 Exploratory Visualization; Stage 05 GLMM; Stage 06 GAMM.

Change Record
- 2025‑12‑03: **IMPLEMENTED** - Computed community metrics from aligned detections. Outputs: 26,250 observations (3 stations, 2018-2021), 11 columns (keys + 9 metrics). Fish presence ~51%, dolphin presence 7-50% by station, vessel presence 5-38%. All validation passed: non-negative counts, binary flags, deterministic derivation.
- 2025‑11‑21: Draft created; metrics defined per overview and aligned detections; parameters now reference config; renumbered to Stage 02.
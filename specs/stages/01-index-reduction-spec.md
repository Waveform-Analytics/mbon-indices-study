# 01 Index Reduction — Stage Spec

## Purpose
- Reduce ~60 acoustic indices to a distinct, low-collinearity subset suitable for GLMM/GAMM modeling while preserving coverage of spectral, temporal, and complexity aspects of the soundscape.

## Inputs
- Aligned indices: `data/interim/aligned_indices.parquet`
  - Columns: `datetime`, `station`, acoustic index columns aligned to 2‑hour bins
- Metadata: `data/raw/metadata/Updated_Index_Categories_v2.csv`
  - Columns: `index_name`, `category`, `frequency_band`, `description`.
- Key columns expected in indices files:
  - `datetime` (ISO), `station` (`9M|14M|37M`), index columns (multiple), optional `date`, `hour`.

## Outputs
- `data/processed/indices_final.csv`
  - Columns: `index_name`, `kept` (bool), `reason`, `category`, `band`.
- `results/figures/index_correlation_heatmap.png`
- `results/figures/index_correlation_sensitivity_0_8.png`
- `results/tables/index_reduction_report.csv`
  - Pairwise correlation summary and VIF table.
- `results/indices/index_final_list.json`
  - Ordered list of ~10 indices with rationale and categories.
- `results/indices/index_final_list_8kHz.json`
- `results/indices/index_final_list_FullBW.json`
- `results/logs/stage01_index_reduction_YYYYMMDD_HHMMSS.txt`
  - Timestamped execution log with all steps, decisions, and outputs.
- `results/logs/archive/`
  - Previous run logs.

## Methods
- Correlation pruning:
  - Standardize each index (z‑score) within station‑year.
  - Compute pairwise Pearson correlations across all data.
  - Primary threshold: drop one index from each pair with `|r| > 0.6`.
  - Greedy selection: process pairs sorted by correlation strength (highest first); for each pair where both indices remain, keep one using decision rules.
  - Selection rule within correlated pairs (priority): coverage (fewer missing values) → alphabetical tiebreaker (deterministic).
- VIF analysis:
  - Compute VIF on remaining set; iteratively remove indices with `VIF > 2`.
  - Fallback policy: if achieving `VIF <= 2` would violate category coverage or reduce the final list below 5, allow `VIF <= 5` for specific indices with explicit justification recorded in the report.
- Domain coverage check:
  - Ensure representation of categories: spectral energy, temporal modulation, complexity/entropy.
  - If pruning removes a whole category, reintroduce the best candidate with lowest correlation/VIF.

## Parameters
- `correlation_threshold`: see `config/analysis.yml -> thresholds.correlation_r`.
- `vif_threshold`: see `config/analysis.yml -> thresholds.vif`.
- `vif_threshold_fallback`: see `config/analysis.yml -> thresholds.vif_fallback`.
- `min_coverage_fraction`: see `config/analysis.yml -> thresholds.min_coverage_fraction`.
- `bands_policy`: see `config/analysis.yml -> predictors.band_policy`.
- `analysis_band`: see `config/analysis.yml -> predictors.analysis_band`.

## Acceptance Criteria
- Final list size is between approximately 10-15 indices.
- No pair among final indices has `|r| > correlation_threshold`.
- All final indices have `VIF <= vif_threshold`; if not achievable without violating coverage or list-size constraints, allow up to `vif_threshold_fallback` with per-index justification captured in the report.
- Each major category (spectral/temporal/complexity) is represented by ≥1 index.
- Indices chosen are present for ≥`min_coverage_fraction` of records across stations and years.
- Heatmap and report generated; rationale documented for each dropped/kept index.

## Edge Cases
- Missing `datetime` or station mismatches → exclude affected rows; report fraction excluded.
- Station/year coverage imbalance → weight correlations by station/year to avoid dominance; document approach.
- Highly similar indices across bands → choose single band unless justified (document).

## Performance
- Target runtime: < 10 minutes on full dataset; < 1 minute on sample.
- Memory: fit in standard laptop RAM; chunked reading if necessary.

## Dependencies
- Upstream: raw indices and metadata availability.
- Upstream: Stage 00 aligned indices (`data/interim/aligned_indices.parquet`) and metadata.
- Downstream: Stage 02 Feature Engineering expects `indices_final.csv` list and metadata categories.

## Change Record
- 2025‑12‑08: Tightened thresholds to |r| > 0.6 and VIF ≤ 2 per ecological best practices (Zuur et al. 2010, Graham 2003). Stricter VIF recommended for GLMM stability. Updated acceptance criteria to 10-15 indices. See `results/logs/RUN_HISTORY.md` for run-specific outcomes.
- 2025‑12‑02: **IMPLEMENTED** - Completed VIF analysis and output generation. Note: `FrequencyResolution` removed from indices loader (constant metadata field, not an index). Note: `aROI` and `nROI` indices present in raw data but missing from metadata file `Updated_Index_Categories_v2.csv`; retained as legitimate indices pending documentation update.
- 2025‑12‑02: Added correlation pruning with greedy algorithm. Simplified decision rules to: (1) coverage (fewer missing values), (2) alphabetical tiebreaker. Rationale: interpretability is subjective and hard to operationalize; using VIF in pairwise decisions creates circular dependency with subsequent VIF analysis step; alphabetical provides deterministic, reproducible tiebreaker. Manual review of dropped indices remains available if domain knowledge suggests reconsideration. Added timestamped logging with archiving: `results/logs/stage01_index_reduction_YYYYMMDD_HHMMSS.txt` captures all steps, decisions, and outputs for audit trail and debugging.
- 2025‑11‑21: Adopted per station‑year Pearson aggregation by median |r|; added 0.8 sensitivity artifact; set final target to 5–10 indices; thresholds remain 0.7 and VIF 5 (fallback 10).
- 2025‑11‑21: Clarified VIF fallback policy and switched inputs to aligned indices from Stage 00; updated dependencies accordingly.
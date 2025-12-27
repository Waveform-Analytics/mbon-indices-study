# 01 Index Reduction — Stage Spec

## Purpose
- Reduce ~60 acoustic indices to a distinct, low-collinearity subset suitable for GAMM modeling while preserving coverage of spectral, temporal, and complexity aspects of the soundscape.

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
- `results/figures/index_vif_progression.png`
  - Shows VIF reduction across iterations.
- `results/tables/index_reduction_report.csv`
  - VIF history and final values.
- `results/indices/index_final_list.json`
  - Ordered list of final indices with categories.
- `results/logs/stage01_index_reduction_YYYYMMDD_HHMMSS.txt`
  - Timestamped execution log with all steps, decisions, and outputs.
- `results/logs/archive/`
  - Previous run logs.

## Methods

### VIF-Based Reduction

Index reduction uses Variance Inflation Factor (VIF) as the sole criterion. VIF measures multicollinearity in a multivariate sense — how well each index can be predicted by all other indices combined. This approach is fully deterministic with no arbitrary tiebreaker decisions.

**Procedure:**
1. Standardize each index (z‑score) within station‑year (for numerical stability)
2. Compute VIF for all indices
3. Iteratively remove the index with highest VIF until all remaining indices have `VIF <= 2`
4. Fallback policy: if achieving `VIF <= 2` would reduce the final list below 5 indices, allow `VIF <= 5` for specific indices with explicit justification

**Why VIF-only (no pairwise correlation pruning):**
- VIF captures multicollinearity more holistically than pairwise correlations
- VIF pruning is fully deterministic — always remove the highest VIF index
- No arbitrary tiebreaker decisions required
- High pairwise correlations inherently inflate VIF, so correlated pairs are addressed

**Domain coverage check:**
- After VIF pruning, verify representation across categories (spectral, temporal, complexity, diversity, amplitude)
- If a category is unrepresented, this is noted in the report for manual review

### Historical: Sensitivity Analysis Script

A sensitivity analysis script (`scripts/sensitivity_analysis_index_reduction.py`) exists from when the pipeline used correlation-based pruning with arbitrary tiebreakers. This script demonstrated that tiebreaker choices materially affected model outcomes, which motivated the switch to VIF-only reduction. The script is retained for reference but is no longer needed since VIF-only reduction is deterministic.

## Parameters
- `vif_threshold`: see `config/analysis.yml -> thresholds.vif` (default: 2)
- `vif_threshold_fallback`: see `config/analysis.yml -> thresholds.vif_fallback` (default: 5)
- `min_coverage_fraction`: see `config/analysis.yml -> thresholds.min_coverage_fraction`

## Acceptance Criteria
- Final list size is approximately 15-20 indices (VIF-only typically retains more than correlation+VIF)
- All final indices have `VIF <= vif_threshold`; if not achievable without reducing below 5 indices, allow up to `vif_threshold_fallback` with justification
- Each major category (spectral, temporal, complexity, diversity, amplitude) is represented by ≥1 index
- Indices chosen are present for ≥`min_coverage_fraction` of records across stations and years
- VIF report generated; rationale documented for each removed index

## Edge Cases
- Missing `datetime` or station mismatches → exclude affected rows; report fraction excluded
- Near-perfect collinearity (VIF = infinity) → remove first, then continue iteration
- Minimum list size → stop at 5 indices even if VIF threshold not met

## Performance
- Target runtime: < 10 minutes on full dataset; < 1 minute on sample.
- Memory: fit in standard laptop RAM; chunked reading if necessary.

## Dependencies
- Upstream: raw indices and metadata availability.
- Upstream: Stage 00 aligned indices (`data/interim/aligned_indices.parquet`) and metadata.
- Downstream: Stage 02 Feature Engineering expects `indices_final.csv` list and metadata categories.

## Implementation Notes

**Code structure:**
- Core reduction functions are in `src/python/mbon_indices/reduction.py` (reusable module)
- Main pipeline script: `scripts/stage01_index_reduction.py` (imports from reduction module)
- Historical sensitivity analysis: `scripts/sensitivity_analysis_index_reduction.py` (retained for reference)

**Standardization note:** The `standardize_indices` function in the reduction module applies z-score standardization within station-year groups for VIF calculations. This is internal to Stage 01 only — the standardized values are NOT saved. Downstream stages receive raw index values and apply their own standardization for model fitting.

## Change Record
- 2025-12-16: **Switched to VIF-only reduction** — removed correlation-based pruning step. Sensitivity analysis revealed that arbitrary tiebreaker choices during correlation pruning materially affected model outcomes (different indices selected, AIC differences >200 for some responses). VIF-only approach is fully deterministic and captures multicollinearity holistically. Expected final count ~17 indices (vs ~14 with correlation+VIF). See `scripts/sensitivity_analysis_index_reduction.py` for the analysis that motivated this change.
- 2025-12-16: Refactored implementation — extracted core reduction functions into `src/python/mbon_indices/reduction.py` module for reuse. Updated sensitivity analysis to be a comprehensive post-pipeline script that includes model comparison (not just index list comparison). Added implementation notes section.
- 2025-12-12: Added sensitivity analysis for pair selection robustness check per statistical consultation.
- 2025‑12‑08: Tightened thresholds to |r| > 0.6 and VIF ≤ 2 per ecological best practices (Zuur et al. 2010, Graham 2003). Stricter VIF recommended for model stability. Updated acceptance criteria to 10-15 indices. See `results/logs/RUN_HISTORY.md` for run-specific outcomes.
- 2025‑12‑02: **IMPLEMENTED** - Completed VIF analysis and output generation. Note: `FrequencyResolution` removed from indices loader (constant metadata field, not an index). Note: `aROI` and `nROI` indices present in raw data but missing from metadata file `Updated_Index_Categories_v2.csv`; retained as legitimate indices pending documentation update.
- 2025‑12‑02: Added correlation pruning with greedy algorithm. Simplified decision rules to: (1) coverage (fewer missing values), (2) alphabetical tiebreaker. Rationale: interpretability is subjective and hard to operationalize; using VIF in pairwise decisions creates circular dependency with subsequent VIF analysis step; alphabetical provides deterministic, reproducible tiebreaker. Manual review of dropped indices remains available if domain knowledge suggests reconsideration. Added timestamped logging with archiving: `results/logs/stage01_index_reduction_YYYYMMDD_HHMMSS.txt` captures all steps, decisions, and outputs for audit trail and debugging.
- 2025‑11‑21: Adopted per station‑year Pearson aggregation by median |r|; added 0.8 sensitivity artifact; set final target to 5–10 indices; thresholds remain 0.7 and VIF 5 (fallback 10).
- 2025‑11‑21: Clarified VIF fallback policy and switched inputs to aligned indices from Stage 00; updated dependencies accordingly.
# 01 Index Reduction — Stage Spec

Title
- Index Reduction: Correlation + VIF pruning to ~10 indices

Purpose
- Reduce ~60 acoustic indices to a distinct, low-collinearity subset suitable for GLMM/GAMM modeling while preserving coverage of spectral, temporal, and complexity aspects of the soundscape.

Inputs
- Source indices: `data/raw/indices/`
  - Per-station/year CSVs (e.g., `Acoustic_Indices_9M_2021_8kHz_v2_Final.csv`, `..._FullBW_v2_Final.csv` and `indices/culled/*Final.csv`).
- Metadata: `data/raw/metadata/Updated_Index_Categories_v2.csv`
  - Columns: `index_name`, `category`, `frequency_band`, `description`.
- Key columns expected in indices files:
  - `datetime` (ISO), `station` (`9M|14M|37M`), index columns (multiple), optional `date`, `hour`.

Outputs
- `data/processed/indices_final.csv`
  - Columns: `index_name`, `kept` (bool), `reason`, `category`, `band`.
- `results/figures/index_correlation_heatmap.png`
- `results/figures/index_correlation_sensitivity_0_8.png`
- `results/tables/index_reduction_report.csv`
  - Pairwise correlation summary and VIF table.
- `results/indices/index_final_list.json`
  - Ordered list of ~10 indices with rationale and categories.

Methods
- Correlation pruning:
  - Standardize each index (z‑score) within station‑year.
  - Compute pairwise Pearson correlations per station‑year, then aggregate by median absolute correlation across station‑years.
  - Primary threshold: drop one index from each pair with aggregated `|r| > 0.7`.
  - Sensitivity analysis: report pairs with aggregated `|r| > 0.8` and note differences in final selection.
  - Selection rule within correlated pairs (priority): coverage → interpretability → preliminary VIF → category balance.
- VIF analysis:
  - Compute VIF on remaining set; iteratively remove indices with `VIF > 5` (fallback `10`) until all below threshold.
- Domain coverage check:
  - Ensure representation of categories: spectral energy, temporal modulation, complexity/entropy.
  - If pruning removes a whole category, reintroduce the best candidate with lowest correlation/VIF.

Parameters
- `correlation_threshold`: default `0.7` (absolute).
- `vif_threshold`: default `5` (allow `10` if necessary).
- `min_coverage_fraction`: default `0.95` (fraction of timestamps present across stations/years to be considered robust).
- `bands_to_include`: `8kHz`, `FullBW` (can be set per analysis).

Acceptance Criteria
- Final list size is between approximately 5-10 indices.
- No pair among final indices has `|r| > correlation_threshold`.
- All final indices have `VIF <= vif_threshold`.
- Each major category (spectral/temporal/complexity) is represented by ≥1 index.
- Indices chosen are present for ≥`min_coverage_fraction` of records across stations and years.
- Heatmap and report generated; rationale documented for each dropped/kept index.

Edge Cases
- Missing `datetime` or station mismatches → exclude affected rows; report fraction excluded.
- Station/year coverage imbalance → weight correlations by station/year to avoid dominance; document approach.
- Highly similar indices across bands → choose single band unless justified (document).

Performance
- Target runtime: < 10 minutes on full dataset; < 1 minute on sample.
- Memory: fit in standard laptop RAM; chunked reading if necessary.

Dependencies
- Upstream: raw indices and metadata availability.
- Downstream: Feature Engineering stage expects `indices_final.csv` list and metadata categories.

Change Record
- 2025‑11‑21: Adopted per station‑year Pearson aggregation by median |r|; added 0.8 sensitivity artifact; set final target to 5–10 indices; thresholds remain 0.7 and VIF 5 (fallback 10).
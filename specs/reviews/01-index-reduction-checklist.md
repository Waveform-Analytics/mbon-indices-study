# Stage 01 — Index Reduction Review Checklist

Pre‑conditions
- Aligned indices present: `data/interim/aligned_indices.parquet`
- Config keys present: `thresholds.correlation_r`, `thresholds.vif`, `thresholds.vif_fallback`, `predictors.band_policy`, `predictors.analysis_band`

Inputs/Methods
- Pearson correlations per station‑year aggregated by median |r|
- Threshold 0.7 applied; sensitivity at 0.8 reported
- VIF ≤ 5; fallback ≤ 10 only with justification and category coverage preserved
- Bands treated separately; per‑band artifacts produced

Outputs
- `index_correlation_heatmap.png` and `index_correlation_sensitivity_0_8.png`
- `index_reduction_report.csv` with correlations and VIF table
- `index_final_list_8kHz.json`, `index_final_list_FullBW.json`
- `indices_final.csv` for selected analysis band

Acceptance Criteria
- Final indices count between 5–10
- No pair among final indices exceeds `correlation_r`
- Category coverage present across spectral/temporal/complexity
- Coverage ≥ `min_coverage_fraction`

Sign‑off
- Reviewer, Date
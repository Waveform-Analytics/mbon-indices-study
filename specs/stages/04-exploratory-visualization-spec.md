# 04 Exploratory Visualization — Stage Spec

## Purpose
- Provide simple, interpretable visuals to inspect variability and co‑movement between biological community metrics and acoustic indices before modeling.

## Inputs
- `data/processed/analysis_ready.parquet` (community metrics, indices, covariates)
- `data/processed/indices_final.csv` (column name list identifying which columns are acoustic indices)
- Column identification via config:
  - Responses: `config/analysis.yml -> responses` (9 community metrics)
  - Covariates: `config/analysis.yml -> covariates` (temperature, depth)
  - Indices: `config/analysis.yml -> predictors.final_list_path` (20 acoustic indices)

## Outputs
- `results/tables/descriptive_stats.csv` — mean, SD, min, max by station/season/hour
- `results/figures/exploratory/community_metrics_distributions.png` — 9-panel histograms
- `results/figures/exploratory/scatter/indices_vs_<response>.png` — all indices vs each response (13 files: 9 linear + 4 log-scale for skewed responses)
- `results/figures/exploratory/heatmaps/heatmap_<variable>.png` — date × hour heatmaps with all 3 stations vertically stacked (31 files: 9 responses + 20 indices + 2 covariates)
- Appends summary to `results/logs/RUN_HISTORY.md`

## Methods
- Descriptive stats:
  - Compute mean, SD, min, max for each metric by `station`, `season`, and `hour_of_day`.
- Distributions:
  - 9-panel faceted histograms for community metrics.
  - Binary variables use discrete bins; continuous variables use 30 bins.
- Scatter/violin overlays:
  - One figure per response, with all 20 indices as subplots.
  - Violin plots for binary responses; scatter + regression line for continuous.
  - Annotate Pearson r with significance stars.
  - Auto-detect skewed responses and generate additional log-scale versions.
- Heatmaps:
  - One figure per variable with 3 station panels stacked vertically.
  - X-axis: date (full year), Y-axis: hour of day (local time, EST).
  - Midnight-centered: hours reordered so midnight appears in vertical center.
  - Horizontal colorbar at bottom.
- Timezone alignment:
  - Use `datetime_local` (fixed EST, UTC-5) for biological interpretability.
  - Fixed offset ensures consistent 2-hour bins year-round (no DST gaps).

## Parameters
- `heatmap_color_scheme`: see `config/analysis.yml -> exploratory.heatmap_color_scheme`
- `heatmap_midnight_center`: see `config/analysis.yml -> exploratory.heatmap_midnight_center`

## Acceptance Criteria
- Coverage: all 3 stations included in every visualization.
- Axes labeled with units; color scheme consistent across heatmaps.
- Binary presence metrics show only {0, 1} in distributions.
- Deterministic outputs: identical figures given identical inputs.
- Heatmaps show no white-stripe artifacts from timezone issues.

## Edge Cases
- Sparse bins: annotate with counts.
- Skewed data: auto-generate log-scale scatter plots.

## Performance
- Target runtime: < 2 minutes.

## Dependencies
- Upstream: Stage 03 feature engineering.
- Downstream: none; figures for sanity checks before modeling.

## Change Record
- 2025-12-05: Simplified spec to match implementation. Removed unused config params. Updated output paths to reflect actual structure (flat heatmaps with multi-station panels, scatter folder).
- 2025-12-04: Updated timezone from America/New_York (DST-aware) to fixed EST (UTC-5). DST caused alternating hour gaps in heatmaps; fixed offset ensures consistent 2-hour bins.
- 2025-12-03: Clarified inputs section—`indices_final.csv` is a column list, not data. Updated timezone references to local time for biological interpretability.
# 04 Exploratory Visualization — Stage Spec

Title
- Descriptive summaries and visuals across site, season, and time of day

Purpose
- Provide simple, interpretable visuals to inspect variability and co‑movement between biological community metrics and acoustic indices before modeling.

Inputs
- `data/processed/analysis_ready.parquet` (community metrics, indices, covariates)
- `data/processed/indices_final.csv` (column name list identifying which columns are acoustic indices; not data to merge)
- Column identification via config:
  - Responses: `config/analysis.yml -> responses` (9 community metrics)
  - Covariates: `config/analysis.yml -> covariates` (temperature, depth)
  - Indices: `config/analysis.yml -> predictors.final_list_path` (20 acoustic indices)

Outputs
- `results/figures/exploratory/`
  - Site/season/diel line plots and bar graphs for community metrics
  - Overlays of biological metrics vs selected acoustic indices over seasons
  - Faceted plots by station and month
- `results/tables/descriptive_stats.csv` (mean, range, SD by site/season/hour bins)
- `results/figures/exploratory/distributions/community_metrics_facets.png`
- `results/figures/exploratory/scatter_overlays/<station>/<metric>_vs_<index>.png`
- `results/figures/exploratory/heatmaps/metrics/<station>/<metric>.png`
- `results/figures/exploratory/heatmaps/indices/<station>/<index>.png`
- `results/figures/exploratory/heatmaps/env/<station>/<variable>.png`

Methods
- Descriptive stats:
  - Compute mean, SD, min, max for each metric by `station`, `season`, and `hour_of_day` bins.
- Visualizations:
  - Line plots of metrics over time per station; monthly faceting.
  - Bar graphs of aggregated metrics by season and station.
  - Overlay scatter/line plots: community metrics vs selected indices (per station); annotate Pearson r; no fitted model beyond linear reference.
  - Distributions: 9‑panel faceted histograms/KDEs for community metrics; validate binary variables show only {0,1}.
  - Heatmaps: per station, for each community metric, each final index, and each environmental variable; x = local date, y = hour of day (local).
- Index selection for overlays:
  - Use the final indices list; select indices from config representing spectral/temporal/complexity.
- Reproducibility:
  - Fixed seeds for any sampling; write plots with deterministic filenames.
- Timezone alignment:
  - Use `datetime_local` and `hour_of_day` (both derived from America/New_York local time) for biological interpretability.
  - Rationale: diel patterns (day/night activity) follow local sunrise/sunset, not UTC.
 - Midnight centering:
   - To center midnight vertically, shift timestamps by `exploratory.heatmap_shift_hours` (default 12h) before plotting.
   - Compute display hour as `(hour_of_day + heatmap_shift_hours) % 24`.
   - Define heatmap column day as the date of the shifted timestamp ("shifted_date"), ensuring continuity across midnight without simple half‑day swapping.

Parameters
- `season_definition`: see `config/analysis.yml -> community_metrics.season_definition`.
- `overlay_indices`: see `config/analysis.yml -> exploratory.overlay_indices`.
- `diel_bins`: see `config/analysis.yml -> exploratory.diel_bins`.
- `figure_size`: see `config/analysis.yml -> exploratory.figure_size`.
- `font_size`: see `config/analysis.yml -> exploratory.font_size`.
- `heatmap_color_scheme`: see `config/analysis.yml -> exploratory.heatmap_color_scheme`.
 - `heatmap_midnight_center`: see `config/analysis.yml -> exploratory.heatmap_midnight_center`.
 - `heatmap_shift_hours`: see `config/analysis.yml -> exploratory.heatmap_shift_hours`.

Acceptance Criteria
- Coverage: visuals include all stations; summaries produced for each station and month.
- Clarity: axes labeled, units noted, captions include data ranges and binning.
- No inferential claims: visuals are descriptive; results not used for formal model selection.
- Deterministic outputs: identical figures given identical inputs.
- Time alignment: all heatmaps use local date on x and `hour_of_day` (local) on y; consistent across stations.
- Binary validation: presence metrics distributions show only {0,1}; deviations are logged.
- Publication quality: minimum font size and DPI per config; color scheme consistent.
 - Midnight centering: when enabled, heatmaps use shifted timestamps for x‑axis day (shifted_date) and rotated hour values; no vertical swapping artifacts.

Edge Cases
- Sparse bins → annotate with counts and suppress misleading bars.
- Outlier values → show ranges explicitly; avoid truncation without note.

Performance
- Target runtime: < 10 minutes full; < 2 minutes sample.

Dependencies
- Upstream: Stage 03 feature engineering.
- Downstream: none strictly; figures for collaborator sharing and sanity checks.

Change Record
- 2025‑12‑03: Clarified inputs section—`indices_final.csv` is a column list, not data; added config references for identifying response/covariate/index columns. Updated all timezone references from UTC to local time (`datetime_local`, America/New_York) for biological interpretability of diel patterns.
- 2025‑11‑21: Updated to consume `analysis_ready.parquet` only; added distributions, scatter overlays with Pearson r, and per‑station heatmaps for metrics/indices/environment; parameters reference config; timezone alignment emphasized.
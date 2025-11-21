# 04 Exploratory Visualization — Stage Spec

Title
- Descriptive summaries and visuals across site, season, and time of day

Purpose
- Provide simple, interpretable visuals to inspect variability and co‑movement between biological community metrics and acoustic indices before modeling.

Inputs
- `data/processed/community_metrics.parquet`
- `data/processed/analysis_ready.parquet` (for indices and covariates)
- `data/processed/indices_final.csv` (final indices list)

Outputs
- `results/figures/exploratory/`
  - Site/season/diel line plots and bar graphs for community metrics
  - Overlays of biological metrics vs selected acoustic indices over seasons
  - Faceted plots by station and month
- `results/tables/descriptive_stats.csv` (mean, range, SD by site/season/hour bins)

Methods
- Descriptive stats:
  - Compute mean, SD, min, max for each metric by `station`, `season` (e.g., meteorological or month), and `hour_of_day` bins.
- Visualizations:
  - Line plots of metrics over time per station; monthly faceting.
  - Bar graphs of aggregated metrics by season and station.
  - Overlay plots tracking variability in fish activity vs selected indices; similarly for dolphin presence/counts.
- Index selection for overlays:
  - Use the final indices list; start with a small set representing spectral/temporal/complexity.
- Reproducibility:
  - Fixed seeds for any sampling; write plots with deterministic filenames.

Parameters
- `season_definition`: `month` (default) or custom seasons
- `overlay_indices`: list of indices to include in overlays
- `diel_bins`: number of bins for hour‑of‑day summaries (default 12)

Acceptance Criteria
- Coverage: visuals include all stations; summaries produced for each station and month.
- Clarity: axes labeled, units noted, captions include data ranges and binning.
- No inferential claims: visuals are descriptive; results not used for formal model selection.
- Deterministic outputs: identical figures given identical inputs.

Edge Cases
- Sparse bins → annotate with counts and suppress misleading bars.
- Outlier values → show ranges explicitly; avoid truncation without note.

Performance
- Target runtime: < 10 minutes full; < 2 minutes sample.

Dependencies
- Upstream: Stage 03 community metrics; Stage 02 feature engineering.
- Downstream: none strictly; figures for collaborator sharing and sanity checks.

Change Record
- 2025‑11‑21: Draft created based on collaborator feedback; descriptive stats and overlay figures defined.
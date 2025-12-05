# Stage 04 — Exploratory Visualization Review Checklist

Pre‑conditions
- Inputs available: `analysis_ready.parquet`, `indices_final.csv`
- Config keys present: `exploratory.figure_size`, `exploratory.font_size`, `exploratory.heatmap_color_scheme`, `exploratory.dpi`, `exploratory.heatmap_midnight_center`, `exploratory.heatmap_shift_hours`

Methods
- Descriptive stats by station, season, hour_of_day
- Scatter overlays with Pearson r annotation; no fitted models beyond linear reference
- 9‑panel distributions for community metrics with binary validation
- Per‑station heatmaps with UTC x and midnight‑centered y via shifted timestamps

Outputs
- `descriptive_stats.csv`
- Distributions facet: `community_metrics_facets.png`
- Scatter overlays per station
- Heatmaps for metrics/indices/environment per station

Acceptance Criteria
- Coverage across stations and months
- Labels, units, font sizes, color scheme per config; DPI ≥ config value
- Deterministic outputs; timezone alignment verified; no vertical swapping artifacts

Sign‑off
- Reviewer, Date
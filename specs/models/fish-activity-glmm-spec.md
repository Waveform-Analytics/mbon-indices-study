# Fish Activity — GLMM Spec

## Purpose
- Fit GLMM for `fish_activity` using final indices and covariates; evaluate effects and diagnostics.

## Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`

## Formula/Structure
- Response: `fish_activity`
- Family: see `config/analysis.yml -> responses.fish_activity.family`
- Fixed effects: all final indices + `temperature`, `depth`, `sin_hour`, `cos_hour`
- Random effects: `(1|station) + (1|month_id) + (1|day_id)`
- Autocorrelation: `ar1(time_within_day + 0 | day_id)` when enabled in config

## Diagnostics
- DHARMa residuals (uniformity, dispersion, temporal autocorrelation)
- Fixed effects CIs; random effects variance summary
- AIC recorded

## Acceptance Criteria
- Convergence without warnings
- Dispersion acceptable or justified
- Residuals and autocorrelation diagnostics pass or are addressed
- Effects table produced and interpretable

## Change Record
- 2025‑11‑21: Stub created; references config families and final indices; diagnostics listed.
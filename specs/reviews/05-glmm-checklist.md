# Stage 05 — GLMM Modeling Review Checklist

Pre‑conditions
- Inputs present: `data/processed/analysis_ready.parquet`, `data/processed/indices_final.csv`
- Config keys present: `responses.*.family`, `temporal.cyclic_terms`, `random_effects`, `autocorrelation.glmm_ar1`, `autocorrelation.sequence_column`

Scope
- One checklist per response metric; run GLMMs using `glmmTMB`

Methods
- Formula: response ~ final indices + temperature + depth + sin_hour + cos_hour + (1|station) + (1|month_id) + (1|day_id)
- Family/link from `config/analysis.yml -> responses.<metric>.family`
- Temporal autocorrelation: `ar1(sequence_column + 0 | day_id)` when enabled

Diagnostics
- Residuals: DHARMa simulate/plot; uniformity tests
- Dispersion: `testDispersion`
- Autocorrelation: `testTemporalAutocorrelation` using `time_within_day`
- Random effects: summary of station/month/day variances
- Fixed effects: confidence intervals; note indices with CIs not overlapping zero

Outputs
- `results/models/glmm/<metric>.rds`
- `results/diagnostics/glmm/<metric>/residuals.png`
- `results/diagnostics/glmm/<metric>/dispersion.txt`
- `results/diagnostics/glmm/<metric>/autocorrelation.txt`
- `results/tables/glmm/<metric>_effects.csv`
- `results/tables/glmm/<metric>_random_effects.csv`
- `results/tables/glmm/<metric>_aic.csv`

Acceptance Criteria
- Convergence achieved; no optimizer warnings
- Dispersion acceptable or justified in notes
- Residual diagnostics do not show severe violations
- Autocorrelation assessed; AR1 specified when needed
- AIC recorded; effects table generated; rationale documented for non‑significant indices when retained

Sign‑off
- Reviewer, Date
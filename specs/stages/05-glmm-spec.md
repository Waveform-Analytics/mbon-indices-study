# 05 GLMM Modeling — Stage Spec

Title
- GLMMs for community metrics with nested random effects and AR1

Purpose
- Fit GLMMs per response metric using the final indices and covariates, with proper temporal structure and diagnostics.

Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`

Outputs
- `results/models/glmm/<metric>.rds`
- `results/tables/glmm/<metric>_effects.csv`
- `results/tables/glmm/<metric>_random_effects.csv`
- `results/tables/glmm/<metric>_aic.csv`
- `results/diagnostics/glmm/<metric>/residuals.png`
- `results/diagnostics/glmm/<metric>/dispersion.txt`
- `results/diagnostics/glmm/<metric>/autocorrelation.txt`

Methods
- Software: `glmmTMB`
- Formula per metric:
  - response ~ predictors from config (`predictors.final_list_path`) + covariates from config (`covariates.*`) + cyclic terms from config (`temporal.cyclic_terms`) + random effects from config (`random_effects`)
- Family: see `config/analysis.yml -> responses.<metric>.family`
- Temporal autocorrelation: when `autocorrelation.glmm_ar1` is true, include `ar1(time_within_day + 0 | day_id)`
- Model fitting notes:
  - Use standardized indices as provided in `analysis_ready.parquet`
  - Drop predictors that cause singular fits; record justification

Parameters
- Families per response: see `config/analysis.yml -> responses.*`
- Random effects: see `config/analysis.yml -> random_effects`
- Cyclic terms: see `config/analysis.yml -> temporal.cyclic_terms`
- AR1: see `config/analysis.yml -> autocorrelation.glmm_ar1` and `autocorrelation.sequence_column`
- Predictors list: see `config/analysis.yml -> predictors.final_list_path`
- Covariates enabled: see `config/analysis.yml -> covariates.*`

Diagnostics
- Residuals via DHARMa: simulate, plot, uniformity
- Dispersion: `testDispersion`
- Temporal autocorrelation: `testTemporalAutocorrelation` using `time_within_day`
- Random effects: variance components by station/month/day
- Fixed effects: confidence intervals; mark indices with CIs not overlapping zero

Acceptance Criteria
- Convergence without warnings; optimizer stable
- Dispersion acceptable or justified; residual diagnostics pass or are addressed
- Autocorrelation assessed; AR1 included when needed
- Effects table and random effects table produced; AIC recorded

Edge Cases
- Overdispersion in count models → prefer `nbinom2` as configured
- Non-convergence → adjust optimizer or drop problematic predictors; record changes

Performance
- Target runtime: per metric < 10 minutes; smoke tests < 1 minute

Dependencies
- Upstream: Stage 03 Feature Engineering; Stage 01 final indices
- Downstream: Stage 07 Cross-Validation; Stage 08 Model Selection

Change Record
- 2025‑11‑21: Draft created; config‑referenced families and AR1; diagnostics defined.
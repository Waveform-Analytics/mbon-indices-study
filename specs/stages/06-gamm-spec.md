# 06 GAMM (bam) Modeling — Stage Spec

Title
- GAMMs with smooth terms for indices, covariates, and cyclic temporal patterns

Purpose
- Fit GAMMs per response metric to capture non-linear relationships and cyclic temporal structure; compare to GLMM via AIC and CV.

Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`

Outputs
- `results/models/gamm/<metric>.rds`
- `results/diagnostics/gamm/<metric>/smooths.png`
- `results/tables/gamm/<metric>_edf.csv`
- `results/tables/gamm/<metric>_aic.csv`

Methods
- Software: `mgcv::bam`
- Formula per metric:
  - response ~ smooths over predictors from config (`predictors.final_list_path`, each with `k=gamm.smooth_k`) + smooths over covariates from config (`covariates.*`) + cyclic smooths over `hour_of_day` and `day_of_year` with `k=gamm.cyclic_k` + random effects from config (`random_effects`) via `s(<re>, bs="re")`
- Family: counts use `nb()`; binary use `binomial` (link logit)
 - Fitting: use `gamm.method` and `gamm.discrete` from config; enable shrinkage selection `select=TRUE` when `gamm.select` is true in config

Parameters
- `smooth_k`: see `config/analysis.yml -> gamm.smooth_k` (default 5)
- `cyclic_k`: see `config/analysis.yml -> gamm.cyclic_k` (default 12)
- `method`: see `config/analysis.yml -> gamm.method` (default fREML)
- `discrete`: see `config/analysis.yml -> gamm.discrete` (default true)
 - `select`: see `config/analysis.yml -> gamm.select` (default true)
- Predictors list: see `config/analysis.yml -> predictors.final_list_path`
- Covariates enabled: see `config/analysis.yml -> covariates.*`
- Random effects: see `config/analysis.yml -> random_effects`
- Cyclic terms: see `config/analysis.yml -> temporal.cyclic_terms`

Diagnostics
- EDF summary: interpret linearity vs non-linearity
- Smooth plots for indices and covariates
- AIC recorded; comparison with GLMM

Acceptance Criteria
- Model fits without errors; EDFs interpretable
- Cyclic smooths exhibit reasonable diel/seasonal structure
- AIC computed and recorded; ΔAIC used in selection logic

Edge Cases
- Overfitting (high edf) → adjust `k` or penalization; record change
- Non-convergence → simplify smooths or reduce `k`

Performance
- Target runtime: per metric < 15 minutes; smoke tests < 2 minutes

Dependencies
- Upstream: Stage 03 Feature Engineering; Stage 01 final indices
- Downstream: Stage 07 Cross-Validation; Stage 08 Model Selection

Change Record
- 2025‑11‑21: Draft created; config‑referenced GAMM parameters; diagnostics defined.
# Stage 06 — GAMM (bam) Modeling Review Checklist

Pre‑conditions
- Inputs present: `data/processed/analysis_ready.parquet`, `data/processed/indices_final.csv`
- Config keys present: `responses.*.family`, `temporal.cyclic_terms`

Scope
- One checklist per response metric; run GAMMs using `mgcv::bam`

Methods
- Formula: response ~ s(each final index, k=5) + s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) + s(station, bs="re") + s(month_id, bs="re") + s(day_id, bs="re`)
- Family: `nb()` for counts, `binomial` for binary
- Method: `fREML`, `discrete=TRUE`

Diagnostics
- EDF summary: check linearity vs non‑linearity
- Smooth plots for each predictor
- AIC recorded; compare against GLMM

Outputs
- `results/models/gamm/<metric>.rds`
- `results/diagnostics/gamm/<metric>/smooths.png`
- `results/tables/gamm/<metric>_edf.csv`
- `results/tables/gamm/<metric>_aic.csv`

Acceptance Criteria
- Model fits without errors; EDFs interpretable
- Cyclic smooths show expected diel/seasonal structure
- AIC recorded and compared to GLMM; note ΔAIC decisions

Sign‑off
- Reviewer, Date
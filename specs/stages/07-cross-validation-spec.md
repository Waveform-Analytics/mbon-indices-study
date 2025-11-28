# 07 Cross-Validation — Stage Spec

Title
- Out-of-sample evaluation for GLMM and GAMM models

Purpose
- Evaluate predictive performance of GLMM and GAMM per response metric using reproducible splits; avoid leakage and ensure fair comparison.

Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`
- Models produced in Stages 05 and 06 (per metric)
- `config/cv.yml` for strategies, metrics, seeds

Outputs
- `results/cv/glmm/<metric>_cv.csv`
- `results/cv/gamm/<metric>_cv.csv`
- `results/cv/summary/<metric>_summary.csv`
- Optional: `results/figures/cv/<metric>_performance.png`

Methods
- Use strategies defined in `config/cv.yml -> primary_strategy` and `secondary_strategy`.
- For each split and metric:
  - Fit models on training data only; predict on test data.
  - Compute metrics as defined in `config/cv.yml -> metrics`.
- Set seeds from `config/cv.yml -> seeds.cv`.

Parameters
- Strategies, metrics, and seeds: see `config/cv.yml` keys

Acceptance Criteria
- CV executed for all metrics under primary strategy; secondary strategy executed where feasible
- Metrics computed and saved; no evidence of data leakage
- Results reproducible with seeds; summary CSV aggregates per metric across splits

Dependencies
- Upstream: Stage 05 and 06 models
- Downstream: Stage 08 Model Selection consumes CV results

Change Record
- 2025‑11‑21: Thin scaffold created; references config strategies/metrics; artifacts defined.
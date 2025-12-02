# 08 Model Selection — Stage Spec

Title
- Choose GLMM vs GAMM per response using AIC and CV performance

Purpose
- Combine in-sample fit (AIC) and out-of-sample CV metrics to select the preferred model per metric, with transparent rationale.

Inputs
- `results/tables/glmm/<metric>_aic.csv`
- `results/tables/gamm/<metric>_aic.csv`
- `results/cv/glmm/<metric>_cv.csv`
- `results/cv/gamm/<metric>_cv.csv`
- `config/analysis.yml -> selection.*` thresholds and preferences

Outputs
- `results/selection/final_selection.json`
- `results/tables/selection_summary.csv`

Methods
- Apply AIC thresholds and preferences from `config/analysis.yml -> selection.*`.
- Use CV primary metrics from `config/analysis.yml -> selection.cv_primary_metric_counts` and `selection.cv_primary_metric_binary`.
- Combine AIC and CV outcomes per config to select model; record metrics and rationale.

Parameters
- Selection thresholds and metric preferences: see `config/analysis.yml -> selection.*`

Acceptance Criteria
- Per metric, final selection recorded with AIC, CV metrics, decision rationale
- JSON and summary CSV saved; reproducible decisions from config keys

Dependencies
- Upstream: Stage 07 CV; Stages 05/06 AIC
- Downstream: Stage 09 Visualization; Stage 10 Reporting

Change Record
- 2025‑11‑21: Thin scaffold created; references config thresholds and CV preferences.
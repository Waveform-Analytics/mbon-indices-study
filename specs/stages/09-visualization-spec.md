# 09 Final Visualization — Stage Spec

Title
- Publishable figures summarizing model effects and performance

Purpose
- Create clear, publication-quality figures for selected models per metric.

Inputs
- `results/selection/final_selection.json`
- GLMM/GAMM outputs and diagnostics from Stages 05/06
- `data/processed/analysis_ready.parquet`
- Figure parameters: reuse `config/analysis.yml -> exploratory.*`

Outputs
- `results/figures/final/glmm/<metric>_forest.png`
- `results/figures/final/gamm/<metric>_smooths.png`
- `results/figures/final/<metric>_predicted_vs_observed.png`
- Optional: `results/figures/final/<metric>_timeseries.png`

Methods
- Generate figures guided by model selection and figure parameters in `config/analysis.yml -> exploratory.*`.
- For GLMM, create effects and random-effects summaries; for GAMM, create smooth summaries and edf annotations.
- Include predicted vs observed visuals per metric as configured.

Parameters
- Figure parameters: see `config/analysis.yml -> exploratory.*`

Acceptance Criteria
- Figures include all selected metrics; readable axes/labels/units; deterministic outputs
- Effects and performance visuals present for each chosen model type

Dependencies
- Upstream: Stage 08 selection; Stages 05/06 diagnostics
- Downstream: Stage 10 Reporting

Change Record
- 2025‑11‑21: Thin scaffold created; uses exploratory figure config.
# MBON Acoustic Indices — Spec Index

## Status Board
- Project: In Progress
- Stages:
  - 00 Data Prep & Alignment: ✅ Implemented (2025-11-28)
  - 01 Index Reduction: ✅ Implemented (2025-12-16) — VIF-only, ~17 final indices
  - 02 Community Metrics: ✅ Implemented (2025-12-03) — 9 response variables
  - 03 Feature Engineering: ✅ Implemented (2025-12-03) — Analysis-ready dataset
  - 04 Exploratory Visualization: ✅ Implemented (2025-12-05) — Heatmaps, scatter plots, distributions
  - 05a Modeling (GAMM): 🔄 In Progress — GAMM fitting with data-driven AR1 rho
  - 05b Validation: 📋 Spec Ready — AR1 validation, week-based k-fold CV
  - ~~06-08~~: Merged — Model selection in 05; validation in 05b
  - 09 Results Presentation: Draft — Quarto slides for interpretation
  - 10 Reporting: Draft — Manuscript preparation

## Glossary
- GAMM: Generalized Additive Mixed Model
- AIC: Akaike Information Criterion
- CV: Cross-Validation
- VIF: Variance Inflation Factor
- EDF: Effective Degrees of Freedom (smoothness measure in GAMMs)

## Links
- Stage Specs: `specs/stages/`
- Templates: `specs/templates/`
- ADRs: `specs/risks/`
- Format Guide: `specs/SPEC_FORMAT.md`

## Folder Structure
- config/: analysis.yml, cv.yml, stations.yml
- data/:
  - raw/, metadata/, external/
  - interim/: aligned_detections.parquet, aligned_environment.parquet, aligned_spl.parquet, aligned_indices.parquet
  - processed/: indices_final.csv, community_metrics.parquet, analysis_ready.parquet
  - manifests/, sample/
- envs/: pyproject.toml, uv.lock (Python); renv/ (R)
- specs/: stages/, templates/, risks/, SPEC_FORMAT.md, _index.md
- src/python/mbon_indices/: data/, transform/, metrics/, viz/, utils/
- src/r/: gamm/, common/
- pipelines/: Snakefile; rules/
- results/:
  - models/: <metric>/gamm.rds
  - tables/: <metric>/gamm_summary.csv, model_summary.csv
  - figures/: exploratory/... (distributions, overlays, heatmaps)
- notebooks/: marimo/ (optional; reads from processed/ only)
- reports/: rendered Quarto site
- tests/: python/ unit tests; r/ smoke tests
- scripts/: thin CLI wrappers
- .gitignore
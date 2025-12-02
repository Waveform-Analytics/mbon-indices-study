# MBON Acoustic Indices — Spec Index

Status Board
- Project: In Progress
- Stages:
  - 00 Data Prep & Alignment: ✅ Implemented (2025-11-28)
  - 01 Index Reduction: ✅ Implemented (2025-12-02) — 20 final indices
  - 02 Community Metrics: Approved → Ready for Implementation
  - 03 Feature Engineering: Approved
  - 04 Exploratory Visualization: Approved
  - 05 GLMM Modeling: Approved
  - 06 GAMM Modeling: Approved
  - 07 Cross-Validation: Draft
  - 08 Model Selection: Draft
  - 09 Visualization: Draft
  - 10 Reporting: Draft

Glossary
- GLMM: Generalized Linear Mixed Model
- GAMM: Generalized Additive Mixed Model
- AIC: Akaike Information Criterion

Links
- Stage Specs: specs/stages/
- Templates: specs/templates/
- Per-Metric Specs: specs/models/
- Review Checklists: specs/reviews/
- ADRs: specs/risks/

Folder Structure
- config/: analysis.yml, cv.yml, stations.yml
- data/:
  - raw/, metadata/, external/
  - interim/: aligned_detections.parquet, aligned_environment.parquet, aligned_spl.parquet, aligned_indices.parquet
  - processed/: indices_final.csv, community_metrics.parquet, analysis_ready.parquet
  - manifests/, sample/
- envs/: pyproject.toml, uv.lock (Python); renv/ (R)
- specs/: stages/, models/, reviews/, templates/, risks/, _index.md
- src/python/mbon_indices/: data/, transform/, metrics/, viz/, utils/
- src/r/: glmm/, gamm/, common/
- pipelines/: Snakefile; rules/
- results/:
  - models/: glmm/<metric>.rds, gamm/<metric>.rds
  - diagnostics/: glmm/<metric>/..., gamm/<metric>/...
  - tables/: glmm/..., gamm/...
  - figures/: exploratory/... (distributions, overlays, heatmaps)
- notebooks/: marimo/ (optional; reads from processed/ only)
- reports/: rendered Quarto site
- tests/: python/ unit tests; r/ smoke tests
- scripts/: thin CLI wrappers
- .gitignore
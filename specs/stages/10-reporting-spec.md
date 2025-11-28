# 10 Reporting — Stage Spec

Title
- Public-facing report/site documenting methods, results, and reproducibility

Purpose
- Assemble a structured report integrating specs, figures, tables, and reproducibility information suitable for sharing.

Inputs
- Specs and checklists (Stages 00–09)
- Final figures and tables
- Environment info (uv lock, renv) and seeds

Outputs
- `reports/site/index.html` and pages:
  - Methods, Results, Figures, CV performance, Reproducibility
- Optional: PDF export for manuscript-ready sharing

Methods
- Render a report/site that links to artifacts and specs, using figure/reporting parameters from config.
- Include environment versions, seeds, and config snapshots.
- Document decisions (selection rationale) and limitations.

Parameters
- Reporting parameters: reference existing figure config (`config/analysis.yml -> exploratory.*`); extend as needed.

Acceptance Criteria
- All pages render; links to artifacts/specs functional; reproducibility details included

Dependencies
- Upstream: Stage 09 figures and Stage 08 selection

Change Record
- 2025‑11‑21: Thin scaffold created; high-level reporting structure documented.
## Spec Philosophy & Workflow
- Treat specs as living contracts: each stage has its own spec with clear inputs/outputs, methods, acceptance criteria, and change record.
- Gate implementation behind spec approval: no code starts until the spec is reviewed and accepted.
- Keep specs small and composable: per-stage and per-metric specs, plus a master index that links them.
- Traceability: every code artifact maps back to a spec section and acceptance test.

## Planning Files & Templates
- `specs/_index.md`: master index outlining stages, links, status, and glossary.
- `specs/stages/<stage>-spec.md`: one per stage (index reduction, feature engineering, GLMM, GAMM, CV, selection, visualization, reports).
- `specs/models/<metric>-<model>-spec.md`: one per metric/model (e.g., `fish-activity-glmm-spec.md`).
- `specs/data/manifest-spec.md`: data manifest schema and rules (hashes, sizes, provenance, units).
- `specs/config/config-spec.md`: config keys, types, defaults, validation rules.
- `specs/risks/adr-<nn>.md`: Architecture Decision Records for major choices (e.g., Snakemake vs targets; Python vs R split).
- `specs/reviews/<stage>-checklist.md`: review checklist used at approval gate.

### Stage Spec Template (each `*-spec.md`)
- Purpose: concise statement of what this stage does and why.
- Inputs: file paths, schemas, required columns, expected sizes.
- Outputs: file paths, schemas, figures/tables, and naming conventions.
- Methods: formulae/algorithms, parameter bounds, model families; cite the overview plan.
- Parameters: configurable keys with types and defaults (reference `config/*.yml`).
- Acceptance Criteria: objective checks to pass before merging (e.g., VIF < 5, DHARMa dispersion p > 0.05, CV metrics computed for all splits).
- Edge Cases: missing data handling, station anomalies, seasonality gaps.
- Performance: expected runtime and resource use on sample and full datasets.
- Dependencies: upstream/downstream stages and contracts.
- Change Record: date, change summary, rationale, impact.

## Sub‑Task Breakdown (gated by specs)
1) Project Skeleton & Environments
- Artifacts: env specs (`envs/python-env.yml`, `renv` lock), `pre-commit`, Quarto skeleton.
- Spec: `specs/adr-01-environments.md` describing environment choices & reproducibility.
- Gate: CI runs, pre-commit hooks pass.

2) Data Manifest & Access Contract
- Artifacts: `data/manifests/manifest.csv`, `.gitignore` baseline, `data/sample/` examples.
- Spec: `specs/data/manifest-spec.md`.
- Gate: manifest validator passes; sample artifacts reproducible.

3) Index Reduction
- Artifacts: `processed/indices_final.csv`, correlation heatmap, reduction report.
- Spec: `specs/stages/index-reduction-spec.md`.
- Gate: correlation pruning applied, all VIF < threshold; doc includes rationale.

4) Feature Engineering
- Artifacts: `processed/analysis_ready.parquet` with temporal and grouping variables.
- Spec: `specs/stages/feature-engineering-spec.md`.
- Gate: schema validation, deterministic transforms, unit tests green.

5) GLMM Modeling
- Artifacts: per-metric `model.rds`, diagnostics (DHARMa plots), effects tables.
- Spec: `specs/stages/glmm-spec.md` + per-metric specs in `specs/models/`.
- Gate: convergence, dispersion acceptable, autocorrelation assessed, interpretable fixed/random effects.

6) GAMM (bam) Modeling
- Artifacts: per-metric `model.rds`, smooth plots, edf summaries, AIC.
- Spec: `specs/stages/gamm-spec.md` + per-metric specs.
- Gate: edf interpretation documented; cyclic smooths validated; AIC computed.

7) Cross‑Validation
- Artifacts: split files, CV results for GLMM/GAMM.
- Spec: `specs/stages/cv-spec.md`.
- Gate: station‑wise and within‑month splits executed; metrics computed; seeds fixed.

8) Model Selection
- Artifacts: `results/final_selection.json` + summary table.
- Spec: `specs/stages/selection-spec.md`.
- Gate: decision criteria applied consistently; ties resolved with interpretability rules.

9) Visualization & Reporting
- Artifacts: forest plots, smooth plots, predicted vs observed; Quarto site pages.
- Spec: `specs/stages/visualization-spec.md`, `specs/stages/reports-spec.md`.
- Gate: figures pass review checklist (readability, captions, reproducibility); site renders.

## Review & Approval Flow
- Create/Update spec → Internal review using the stage checklist → Approve spec (record decision in ADR) → Implement code → Produce artifacts → Verify acceptance criteria → Merge.
- Each PR includes: spec link, affected artifacts, acceptance checklist, and before/after examples.

## Change Control & Traceability
- ADRs for material changes (methods, tools, architecture); Minor parameter changes logged in the stage spec change record.
- Versioned configs: copy `config/analysis.yml` to `config/archive/analysis-YYYYMMDD.yml` on breaking changes.
- Tag results with spec version in filenames or metadata (`results/…/meta.json`).

## Separate Specs vs Single Doc
- Separate specs: best for modularity, clarity, and parallel work; easier to gate and review. Recommended.
- Single master doc: maintain `specs/_index.md` to provide big‑picture, link to stage and per‑metric specs; include glossary and status table.

## Naming & Locations (examples)
- Stage specs: `specs/stages/03-glmm-spec.md`, `specs/stages/04-gamm-spec.md` (prefix numbers match pipeline order).
- Per‑metric specs: `specs/models/fish-activity-glmm-spec.md`, `specs/models/dolphin-presence-glmm-spec.md`.
- Checklists: `specs/reviews/glmm-checklist.md`, `specs/reviews/gamm-checklist.md`.
- ADRs: `specs/risks/adr-02-snakemake-vs-targets.md`.

## Best Practices for Success
- Write acceptance criteria that are testable and automated (validators, smoke tests, diagnostics thresholds).
- Provide example I/O artifacts in each spec so reviewers can reason concretely.
- Freeze scope per stage; if mid‑stream changes arise, update spec first, capture the rationale, then modify code.
- Keep specs short (1–3 pages) but precise; use tables for schemas and parameters.
- Maintain a status board (in `_index.md`) with stage state: Draft → In Review → Approved → Implemented → Verified.

## Next Planning Steps
- Draft `specs/_index.md` (status board + glossary).
- Author templates for: Stage Spec, Per‑Metric Spec, Review Checklist, ADR.
- Start with two specs in detail: Index Reduction and Feature Engineering; submit for review before coding.
- Decide env tooling in ADR (conda+renv vs Docker) and record rationale.
# Tasks: Acoustic Indices Only Model Comparison

**Input**: Design documents from `/specs/extra-analysis/003-acoustic-indices-only/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Tests**: Not requested for this feature (data analysis script with manual verification)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **R scripts**: `scripts/` at repository root
- **Outputs**: `results/tables/`, `results/models/`

---

## Phase 1: Setup

**Purpose**: Create script structure and reuse helper functions from stage05d_baseline.R

- [X] T001 Create `scripts/stage05e_acoustic_only.R` with header, package loading, and argument parsing (support `--metric` flag)
- [X] T002 Add configuration loading from `config/analysis.yml` (response metrics, families)
- [X] T003 [P] Copy or source helper functions from `scripts/stage05d_baseline.R`: `load_full_model()`, `extract_rho()`, `get_gam_family()`, `extract_model_metrics()`, `fit_model_safe()`, `get_all_index_columns()`, `ensure_output_dirs()`

**Checkpoint**: Script skeleton ready with model loading capability

---

## Phase 2: Foundational (Acoustic-Only Specific)

**Purpose**: Add the acoustic-only formula builder (key difference from baseline)

- [X] T004 Add helper function `build_acoustic_only_formula(response, index_cols)` in `scripts/stage05e_acoustic_only.R` - builds formula with 61 index smooth terms + random effects (NO temperature, depth, hour_of_day, day_of_year)
- [X] T005 Add data loading: merge `data/interim/aligned_indices.parquet` with response variables and grouping factors from `data/processed/analysis_ready.parquet`

**Checkpoint**: Foundation ready - acoustic-only formula builder available

---

## Phase 3: User Story 1 - Compare Acoustic-Only vs Full Model Performance (Priority: P1) 🎯 MVP

**Goal**: Fit acoustic-only models and produce comparison table showing ΔAIC for all 9 response variables

**Independent Test**: Run `Rscript scripts/stage05e_acoustic_only.R`, verify `results/tables/acoustic_only_comparison.csv` contains ΔAIC values for all 9 metrics

### Implementation for User Story 1

- [X] T006 [US1] Implement acoustic-only fitting loop: for each metric, load full model, extract rho, fit acoustic-only model with same rho in `scripts/stage05e_acoustic_only.R`
- [X] T007 [US1] Add acoustic-only model saving to `results/models/<metric>/gamm_acoustic_only.rds`
- [X] T008 [US1] Implement comparison metrics calculation: full_aic, acoustic_only_aic, delta_aic, full_dev_explained, acoustic_only_dev_explained, acoustic_only_converged
- [X] T009 [US1] Implement CSV export to `results/tables/acoustic_only_comparison.csv` with schema from plan.md
- [X] T010 [US1] Add console summary output showing ΔAIC and deviance explained for each metric

**Checkpoint**: Acoustic-only comparison complete - can answer "How much explanatory power do indices have alone?" for each response variable

---

## Phase 4: User Story 2 - Access Individual Acoustic-Only Model Objects (Priority: P2)

**Goal**: Ensure saved acoustic-only models are valid and inspectable

**Independent Test**: Load `results/models/fish_presence/gamm_acoustic_only.rds`, run `summary()` and `plot()` without errors

### Implementation for User Story 2

- [X] T011 [US2] Add model validation check after fitting: verify `summary()` works before saving in `scripts/stage05e_acoustic_only.R`
- [X] T012 [US2] Update console output to confirm each model saved successfully with path

**Checkpoint**: All 9 acoustic-only model objects saved and inspectable

---

## Phase 5: User Story 3 - Interpret Results in Context of Baseline Comparison (Priority: P2)

**Goal**: Enable synthesis of acoustic-only results with baseline comparison results

**Independent Test**: Review both `results/tables/acoustic_only_comparison.csv` and `results/tables/baseline_comparison.csv`, verify comparable format

### Implementation for User Story 3

- [X] T013 [US3] Add interpretation column to output based on ΔAIC thresholds (>10 = full model much better, 4-10 = moderate, <4 = similar)
- [X] T014 [US3] Add console summary comparing deviance explained: acoustic-only vs baseline vs full model (reads baseline_comparison.csv if exists)
- [X] T015 [US3] Add summary statistics showing which model component (indices vs env/temporal) contributes more for each response

**Checkpoint**: Results enable synthesis with baseline comparison for complete picture

---

## Phase 6: Polish & Integration

**Purpose**: Final validation and documentation

- [X] T016 Add `--metric` argument support for single-model testing in `scripts/stage05e_acoustic_only.R`
- [X] T017 Add summary log output showing: models processed, convergence status, files created
- [X] T018 Run full script and verify all outputs generated correctly
- [X] T019 [P] Update `rebuild-docs.sh` to copy `results/tables/acoustic_only_comparison.csv` if adding to results viewer

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - US1 (P1): Can proceed immediately after Foundational
  - US2 (P2): Depends on US1 (needs acoustic-only models to exist)
  - US3 (P2): Depends on US1 (needs comparison table)
- **Polish (Phase 6)**: Depends on US1 completion

### User Story Dependencies

- **User Story 1 (P1)**: Independent - start after Foundational
- **User Story 2 (P2)**: Depends on US1 model saving (T007)
- **User Story 3 (P2)**: Depends on US1 comparison table (T009)

### Parallel Opportunities

Within Phase 1 (Setup):
- T003 can run in parallel with T001-T002

Within Phase 6 (Polish):
- T019 can run in parallel with other polish tasks

---

## Parallel Example: User Story 1

```bash
# After foundational is complete, run the main fitting loop:
Rscript scripts/stage05e_acoustic_only.R --metric fish_presence  # Test single metric first
Rscript scripts/stage05e_acoustic_only.R                          # Then run all 9 metrics
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T005)
3. Complete Phase 3: User Story 1 - Acoustic-Only Comparison (T006-T010)
4. **STOP and VALIDATE**: Run `Rscript scripts/stage05e_acoustic_only.R --metric fish_presence`
5. Verify `results/tables/acoustic_only_comparison.csv` is generated correctly
6. Run full comparison for all 9 metrics

**MVP delivers**: The inverse of baseline comparison - "How well do indices predict alone?"

### Incremental Delivery

1. Complete Setup + Foundational → Script skeleton ready
2. Add User Story 1 (Acoustic-Only Comparison) → Core result (MVP!)
3. Add User Story 2 (Model Objects) → Models inspectable
4. Add User Story 3 (Synthesis with Baseline) → Complete picture for publication
5. Polish → Integration with results viewer

### Code Reuse from stage05d_baseline.R

This script can reuse most helper functions from the baseline comparison script:
- `load_full_model()` - identical
- `extract_rho()` - identical
- `get_gam_family()` - identical
- `extract_model_metrics()` - identical
- `fit_model_safe()` - identical
- `get_all_index_columns()` - identical
- `ensure_output_dirs()` - identical

The ONLY new function needed is `build_acoustic_only_formula()` which is the inverse of `build_baseline_formula()`.

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Setup | T001-T003 | Script skeleton, helper functions |
| Foundational | T004-T005 | Acoustic-only formula, data loading |
| US1: Acoustic-Only Comparison | T006-T010 | Core ΔAIC analysis (MVP) |
| US2: Model Objects | T011-T012 | Model validation and saving |
| US3: Synthesis | T013-T015 | Compare with baseline results |
| Polish | T016-T019 | Integration, validation |

**Total Tasks**: 19
**MVP Scope**: T001-T010 (10 tasks)
**Parallel Opportunities**: 2 tasks marked [P]
**Key Reuse**: 7 helper functions from stage05d_baseline.R

# Tasks: Baseline Model Comparison & VIF Validation

**Input**: Design documents from `/specs/002-baseline-comparison/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Tests**: Not requested for this feature (data analysis script with manual verification)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US4)
- Include exact file paths in descriptions

## Path Conventions

- **R scripts**: `scripts/` at repository root
- **Outputs**: `results/tables/`, `results/models/`, `results/figures/`

---

## Phase 1: Setup

**Purpose**: Create script structure and helper functions

- [x] T001 Create `scripts/stage05d_baseline.R` with header, package loading, and argument parsing (support `--metric` and `--baseline-only` flags)
- [x] T002 Add configuration loading from `config/analysis.yml` (response metrics, families)
- [x] T003 [P] Add helper function `load_full_model(metric)` to load GAMM from `results/models/<metric>/gamm.rds`
- [x] T004 [P] Add helper function `extract_rho(model)` to get AR1 correlation parameter from full model
- [x] T005 [P] Add helper function `get_gam_family(family_name)` (reuse pattern from stage05_modeling.R)

**Checkpoint**: Script skeleton ready with model loading capability

---

## Phase 2: Foundational (Shared Utilities)

**Purpose**: Core functions used across multiple user stories

- [x] T006 Add helper function `extract_model_metrics(model)` returning list: aic, deviance_explained, converged
- [x] T007 [P] Add helper function `build_baseline_formula(response)` per research.md specification
- [x] T008 [P] Add helper function `fit_model_safe(formula, data, family, rho)` with error handling (returns model or NULL)
- [x] T009 [P] Add output directory creation logic ensuring `results/models/<metric>/` exists

**Checkpoint**: Foundation ready - all helper functions available for user story implementation

---

## Phase 3: User Story 1 - Compare Model Performance Metrics (Priority: P1) 🎯 MVP

**Goal**: Fit baseline models and produce comparison table showing ΔAIC for all 9 response variables

**Independent Test**: Run `Rscript scripts/stage05d_baseline.R --baseline-only`, verify `results/tables/baseline_comparison.csv` contains ΔAIC values for all 9 metrics

### Implementation for User Story 1

- [x] T010 [US1] Implement baseline fitting loop: for each metric, load full model, extract rho, fit baseline with same rho in `scripts/stage05d_baseline.R`
- [x] T011 [US1] Add baseline model saving to `results/models/<metric>/gamm_baseline.rds`
- [x] T012 [US1] Implement comparison metrics calculation: full_aic, baseline_aic, delta_aic, full_dev_explained, baseline_dev_explained
- [x] T013 [US1] Add interpretation column based on ΔAIC thresholds (>10 strong, 4-10 moderate, <4 weak)
- [x] T014 [US1] Implement CSV export to `results/tables/baseline_comparison.csv` with schema from plan.md
- [x] T015 [US1] Add console summary output showing which metrics benefit from indices

**Checkpoint**: Baseline comparison complete - can answer "Do indices add explanatory power?" for each response variable

---

## Phase 4: User Story 2 - Access Individual Baseline Model Objects (Priority: P2)

**Goal**: Ensure saved baseline models are valid and inspectable

**Independent Test**: Load `results/models/fish_presence/gamm_baseline.rds`, run `summary()` and `plot()` without errors

### Implementation for User Story 2

- [ ] T016 [US2] Add model validation check after fitting: verify `summary()` works before saving in `scripts/stage05d_baseline.R`
- [ ] T017 [US2] Add convergence status to saved model metadata (or separate JSON log)
- [ ] T018 [US2] Update console output to confirm each model saved successfully with path

**Checkpoint**: All 9 baseline model objects saved and inspectable

---

## Phase 5: User Story 3 - Interpret Results for Publication (Priority: P3)

**Goal**: Add interpretation guidance to outputs

**Independent Test**: Review `results/tables/baseline_comparison.csv` and verify interpretation column is populated

### Implementation for User Story 3

- [ ] T019 [US3] Ensure interpretation column uses standard language from Burnham & Anderson in `scripts/stage05d_baseline.R`
- [ ] T020 [US3] Add summary statistics to console: count of metrics with strong/moderate/weak evidence

**Checkpoint**: Results are publication-ready with clear interpretation

---

## Phase 6: User Story 4 - VIF Validation Analysis (Priority: P2)

**Goal**: Compare 17-index vs 60-index models to validate VIF filtering approach

**Independent Test**: Run `Rscript scripts/stage05d_baseline.R` (full), verify `results/tables/vif_validation.csv` and `results/tables/vif_significance_comparison.csv` exist

### Implementation for User Story 4

- [x] T021 [US4] Add data loading for 60 indices from `data/interim/aligned_indices.parquet` and join with analysis data
- [x] T022 [US4] Add helper function `get_all_index_columns(data)` to extract ~60 index column names
- [x] T023 [US4] Add helper function `build_60index_formula(response, index_cols)` with k=3 for convergence
- [x] T024 [US4] Implement 60-index model fitting loop with enhanced error handling for memory/convergence issues
- [x] T025 [US4] Add 60-index model saving to `results/models/<metric>/gamm_60index.rds`
- [x] T026 [US4] Implement VIF validation comparison: full17_aic vs full60_aic, delta_aic, converged_60index
- [x] T027 [US4] Export VIF validation table to `results/tables/vif_validation.csv`
- [x] T028 [P] [US4] Add helper function `extract_significant_terms(model, threshold=0.05)` returning data.frame
- [x] T029 [US4] Implement significance comparison between 17-index and 60-index models
- [x] T030 [US4] Export significance comparison to `results/tables/vif_significance_comparison.csv`

**Checkpoint**: VIF validation complete - can assess whether 60→17 filtering lost useful information

---

## Phase 7: Polish & Integration

**Purpose**: Final integration and visualization

- [ ] T031 Add main execution block that runs baseline comparison (always) and VIF validation (unless --baseline-only)
- [ ] T032 Add `--metric` argument support for single-model testing
- [ ] T033 Add summary log output showing: models processed, convergence status, files created
- [ ] T034 [P] Create visualization `results/figures/baseline_comparison.png` showing ΔAIC by metric
- [ ] T035 Run full script and verify all outputs generated correctly
- [ ] T036 Update `rebuild-docs.sh` to copy new outputs if adding to results viewer

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 (P1): Can proceed immediately after Foundational
  - US2 (P2): Depends on US1 (needs baseline models to exist)
  - US3 (P3): Depends on US1 (needs comparison table)
  - US4 (P2): Can run in parallel with US2/US3 after Foundational
- **Polish (Phase 7)**: Depends on US1 completion; US4 completion optional

### User Story Dependencies

- **User Story 1 (P1)**: Independent - start after Foundational
- **User Story 2 (P2)**: Depends on US1 model saving (T011)
- **User Story 3 (P3)**: Depends on US1 comparison table (T014)
- **User Story 4 (P2)**: Independent of US1-3, can run in parallel

### Parallel Opportunities

Within Phase 1 (Setup):
- T003, T004, T005 can run in parallel

Within Phase 2 (Foundational):
- T007, T008, T009 can run in parallel

Within Phase 6 (User Story 4):
- T028 can run in parallel with T024-T027

---

## Implementation Strategy

### MVP First (User Story 1 Only) - FOR TOMORROW'S MEETING

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T009)
3. Complete Phase 3: User Story 1 - Baseline Comparison (T010-T015)
4. **STOP and VALIDATE**: Run `Rscript scripts/stage05d_baseline.R --baseline-only --metric fish_presence`
5. Verify `results/tables/baseline_comparison.csv` is generated correctly
6. Run full baseline comparison for all 9 metrics

**MVP delivers**: The core research question answer - "Do indices add value?"

### Incremental Delivery

1. Complete Setup + Foundational → Script skeleton ready
2. Add User Story 1 (Baseline Comparison) → Core result for meeting (MVP!)
3. Add User Story 2 (Model Objects) → Models inspectable
4. Add User Story 3 (Interpretation) → Publication-ready output
5. Add User Story 4 (VIF Validation) → Methodology validation complete
6. Polish → Visualization and integration

### Time-Constrained Path (Meeting Tomorrow)

If short on time, prioritize:
1. ✅ T001-T009: Setup + Foundational
2. ✅ T010-T015: User Story 1 (Baseline Comparison)
3. ⏸️ Skip US2, US3, US4 for now
4. ⏸️ Skip visualization (T034)

This gives you the key ΔAIC comparison table for the meeting.

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Setup | T001-T005 | Script skeleton, model loading |
| Foundational | T006-T009 | Helper utilities |
| US1: Baseline Comparison | T010-T015 | Core ΔAIC analysis (MVP) |
| US2: Model Objects | T016-T018 | Model validation and saving |
| US3: Interpretation | T019-T020 | Publication guidance |
| US4: VIF Validation | T021-T030 | 17-index vs 60-index comparison |
| Polish | T031-T036 | Integration, visualization |

**Total Tasks**: 36
**MVP Scope**: T001-T015 (15 tasks)
**Parallel Opportunities**: 8 tasks marked [P]

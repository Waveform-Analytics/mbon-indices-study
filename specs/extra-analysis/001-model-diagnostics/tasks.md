# Tasks: Model Diagnostics & Effect Interpretation

**Input**: Design documents from `/specs/extra-analysis/001-model-diagnostics/`
**Prerequisites**: plan.md (required), spec.md (required), research.md

**Tests**: Not requested for this feature (data analysis script with manual verification)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **R scripts**: `scripts/` at repository root
- **Outputs**: `results/tables/` and `results/figures/`

---

## Phase 1: Setup

**Purpose**: Create script structure and helper functions

- [x] T001 Create `scripts/stage05c_diagnostics.R` with header, package loading, and argument parsing
- [x] T002 Add configuration loading from `config/analysis.yml` (response metrics, index columns)
- [x] T003 [P] Add helper function `load_model(metric)` to load GAMM from `results/models/<metric>/gamm.rds`
- [x] T004 [P] Add helper function `check_convergence(model)` to verify model converged before processing

**Checkpoint**: Script skeleton ready with model loading capability

---

## Phase 2: Foundational (Shared Utilities)

**Purpose**: Core functions used across multiple user stories

- [x] T005 Add helper function `get_index_columns()` to extract acoustic index column names from config
- [x] T006 [P] Add helper function `get_model_type(metric)` to classify as "presence" or "count"
- [x] T007 [P] Add output directory creation logic for `results/tables/<metric>/` and `results/figures/<metric>/`

**Checkpoint**: Foundation ready - all helper functions available for user story implementation

---

## Phase 3: User Story 1 - Effect Sizes (Priority: P1) 🎯 MVP

**Goal**: Calculate and export effect sizes showing predicted response change across index ranges

**Independent Test**: Run script with `--metric fish_presence`, verify `results/tables/effect_sizes.csv` contains probability changes for all 17 indices

### Implementation for User Story 1

- [x] T008 [US1] Implement `calculate_effect_sizes(model, data, index_cols, model_type)` function in `scripts/stage05c_diagnostics.R`
  - For each index: create newdata at 10th and 90th percentile, other vars at median
  - Predict response at both points
  - Calculate effect: probability_change for presence, fold_change for counts
- [x] T009 [US1] Add effect size loop over all 9 metrics in main script body
- [x] T010 [US1] Implement CSV export to `results/tables/effect_sizes.csv` with columns: metric, index, low_value, high_value, low_pred, high_pred, effect_size, effect_type
- [x] T011 [US1] Add console output summarizing largest effect sizes per metric

**Checkpoint**: Effect sizes table complete - can communicate practical significance to stakeholders

---

## Phase 4: User Story 2 - Model Assumption Validation (Priority: P2)

**Goal**: Validate model assumptions via concurvity, zero-inflation, and random effects diagnostics

**Independent Test**: Run script, verify `results/tables/<metric>/concurvity.csv` exists for all 9 metrics and flags any terms > 0.8

### Implementation for User Story 2

- [x] T012 [P] [US2] Implement `check_concurvity(model, threshold = 0.8)` function in `scripts/stage05c_diagnostics.R`
  - Run `mgcv::concurvity(model, full=TRUE)`
  - Extract worst-case values for each smooth term
  - Flag terms exceeding threshold
- [x] T013 [US2] Add concurvity loop and export to `results/tables/<metric>/concurvity.csv`
- [x] T014 [P] [US2] Implement `check_zero_inflation(model, data, response_col)` function
  - Calculate observed zero proportion from data
  - Calculate predicted zero proportion from model
  - Flag if gap > 10%
- [x] T015 [US2] Add zero-inflation loop (count models only) and export to `results/tables/zero_inflation_check.csv`
- [x] T016 [P] [US2] Implement `plot_random_effects_qq(model, metric)` function
  - Extract station and month_id random effect estimates
  - Generate QQ plots for each
  - Combine into 2-panel figure
- [x] T017 [US2] Add random effects QQ loop and save to `results/figures/<metric>/random_effects_qq.png`

**Checkpoint**: All assumption diagnostics complete - model validity confirmed or issues flagged

---

## Phase 5: User Story 3 - Station-Specific Diagnostics (Priority: P3)

**Goal**: Generate residual plots faceted by station to identify location-specific model issues

**Independent Test**: Run script, verify `results/figures/<metric>/residuals_by_station.png` shows 3-panel plot (9M, 14M, 37M) for each metric

### Implementation for User Story 3

- [x] T018 [US3] Implement `plot_residuals_by_station(model, data, metric)` function in `scripts/stage05c_diagnostics.R`
  - Extract deviance residuals
  - Join with station from data
  - Create ggplot with residuals vs fitted, faceted by station
- [x] T019 [US3] Add residuals-by-station loop and save to `results/figures/<metric>/residuals_by_station.png`
- [x] T020 [US3] Add summary statistics per station (mean residual, SD) to console output

**Checkpoint**: Station-level diagnostics complete - any systematic issues by location are visible

---

## Phase 6: Polish & Integration

**Purpose**: Final integration and documentation

- [x] T021 Add main execution block that runs all diagnostics in sequence
- [x] T022 Add `--metric` argument support for single-model testing
- [x] T023 Add summary log output showing counts: models processed, flags raised, files created
- [x] T024 Update `results/logs/` with diagnostics run metadata (timestamp, metrics processed)
- [x] T025 Run full script and verify all outputs generated correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories CAN proceed in parallel (different functions)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (Effect Sizes)**: Independent - can start after Foundational
- **User Story 2 (Assumption Validation)**: Independent - can start after Foundational
- **User Story 3 (Station Diagnostics)**: Independent - can start after Foundational

### Parallel Opportunities

Within Phase 2 (Foundational):
- T005, T006, T007 can run in parallel

Within User Story 2:
- T012, T014, T016 can run in parallel (different functions)

---

## Parallel Example: User Story 2

```bash
# Launch all diagnostic functions in parallel:
Task: "Implement check_concurvity() function"
Task: "Implement check_zero_inflation() function"
Task: "Implement plot_random_effects_qq() function"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T007)
3. Complete Phase 3: User Story 1 - Effect Sizes (T008-T011)
4. **STOP and VALIDATE**: Run `Rscript scripts/stage05c_diagnostics.R --metric fish_presence`
5. Verify `results/tables/effect_sizes.csv` is generated correctly

### Incremental Delivery

1. Complete Setup + Foundational → Script skeleton ready
2. Add User Story 1 (Effect Sizes) → Can present to stakeholders (MVP!)
3. Add User Story 2 (Assumption Validation) → Model validity confirmed
4. Add User Story 3 (Station Diagnostics) → Full diagnostic suite complete
5. Polish → Production-ready script

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Setup | T001-T004 | Script skeleton, model loading |
| Foundational | T005-T007 | Helper utilities |
| US1: Effect Sizes | T008-T011 | Practical significance (MVP) |
| US2: Assumptions | T012-T017 | Concurvity, zero-inflation, RE diagnostics |
| US3: Station | T018-T020 | Residuals by location |
| Polish | T021-T025 | Integration, logging |

**Total Tasks**: 25
**MVP Scope**: T001-T011 (11 tasks)
**Parallel Opportunities**: 8 tasks marked [P]

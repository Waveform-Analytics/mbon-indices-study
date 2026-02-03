# Feature Specification: Baseline Model Comparison & VIF Validation

**Feature Branch**: `002-baseline-comparison`
**Created**: 2026-02-02
**Status**: Draft
**Input**: User description: "Compare GAMM performance with and without acoustic indices to directly answer: Do indices add explanatory power beyond environmental and temporal variables alone? Also validate VIF filtering by comparing 17-index vs 60-index models."

## Clarifications

### Session 2026-02-02
- Q: Should VIF validation analysis (17-index vs 60-index comparison) be included in this spec or kept separate? → A: Include in this spec (Option A) - both are model comparison analyses with similar structure.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare Model Performance Metrics (Priority: P1)

A researcher wants to determine whether acoustic indices provide meaningful explanatory power beyond what environmental and temporal variables already capture. They need a direct comparison showing how much the indices contribute to model fit.

**Why this priority**: This is the core deliverable that directly answers the research question. Without this comparison, there's no way to quantify the added value of acoustic indices.

**Independent Test**: Can be fully tested by fitting baseline models for all 9 response variables and comparing AIC/deviance explained against existing full models.

**Acceptance Scenarios**:

1. **Given** fitted full GAMM models exist for all 9 response variables, **When** the analyst runs the baseline comparison script, **Then** baseline models are fitted using only environmental + temporal + random effects (no indices).
2. **Given** both full and baseline models exist for a response variable, **When** the comparison is computed, **Then** ΔAIC and Δdeviance_explained are calculated and saved to a summary table.
3. **Given** the comparison table is complete, **When** the analyst reviews results, **Then** they can identify which response variables benefit most from acoustic indices (ΔAIC > 10 threshold).

---

### User Story 2 - Access Individual Baseline Model Objects (Priority: P2)

A researcher wants to inspect the baseline model details (coefficient estimates, smooth term visualizations, summary statistics) to understand how well environmental and temporal variables alone explain biological responses.

**Why this priority**: Model objects enable detailed inspection beyond summary metrics, supporting supplementary analyses or troubleshooting. Secondary to the comparison table because the comparison table answers the primary research question.

**Independent Test**: Can be fully tested by loading a saved baseline model object and calling standard model inspection functions (summary, plot).

**Acceptance Scenarios**:

1. **Given** baseline model fitting is complete for fish_presence, **When** the analyst loads `results/models/fish_presence/gamm_baseline.rds`, **Then** the model object is a valid fitted GAMM that can be inspected with `summary()` and `plot()`.
2. **Given** all 9 baseline models are saved, **When** the analyst lists the model directory contents, **Then** each metric folder contains both `gamm.rds` (full) and `gamm_baseline.rds` (baseline).

---

### User Story 3 - Interpret Results for Publication (Priority: P3)

A researcher needs to communicate findings to stakeholders and reviewers. They want clear interpretation of what the ΔAIC values mean in practical terms.

**Why this priority**: Interpretation guidance aids communication but doesn't change the core analysis output. Tertiary because it builds on the comparison table from P1.

**Independent Test**: Can be tested by reviewing the comparison table output format and verifying interpretation guidance is included in documentation or output notes.

**Acceptance Scenarios**:

1. **Given** the comparison table shows ΔAIC values, **When** the analyst reviews the output, **Then** standard interpretation guidelines are provided (e.g., ΔAIC > 10 = strong evidence indices add value; ΔAIC 4-10 = moderate evidence; ΔAIC < 4 = weak/no evidence).

---

### User Story 4 - VIF Validation Analysis (Priority: P2)

A researcher wants to validate that the VIF filtering approach (reducing 60 indices to 17) was appropriate. They need to compare models with all 60 indices against the 17-index filtered models to see if performance differs and whether similar indices emerge as significant.

**Why this priority**: This validates a key methodological decision. If 60-index models perform similarly and select similar indices, it confirms VIF filtering removed redundant (not useful) information. Secondary to baseline comparison because baseline directly answers the primary research question.

**Independent Test**: Can be tested by fitting 60-index models and comparing AIC/significance patterns against existing 17-index models.

**Acceptance Scenarios**:

1. **Given** fitted 17-index (full) GAMM models exist, **When** the analyst runs the VIF validation script, **Then** 60-index models are fitted using all available acoustic indices.
2. **Given** both 17-index and 60-index models exist for a response variable, **When** the comparison is computed, **Then** ΔAIC between them is calculated and saved.
3. **Given** both models are fitted, **When** the analyst reviews which indices are significant, **Then** they can compare whether the 17-index model's significant indices appear significant in the 60-index model as well.

---

### Edge Cases

- What happens if a baseline model fails to converge? Script should log a warning and continue with other models, marking that metric as failed in the output table.
- What happens if the full model file is missing for a metric? Script should skip that metric with a clear error message rather than crashing.
- How does the system handle metrics where baseline deviance explained exceeds full model? This shouldn't happen mathematically (nested models), but if it does due to numerical issues, flag it in the output for manual review.
- What if AR1 correlation estimation differs between full and baseline models? Use the full model's rho value for both to ensure fair comparison.
- What if 60-index models fail to converge? This is more likely with 60 smooth terms. Log warning, record as "failed" in output, and continue. Do not block baseline comparison results.
- What if 60-index models hit memory limits? Log error with memory usage estimate and suggest reducing k values or running on a machine with more RAM.
- How to handle significance comparison between 17-index and 60-index models? Extract p-values for all index smooth terms from both models; flag indices significant in one but not the other.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST fit baseline GAMM models for all 9 response variables using the formula: `response ~ s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) + s(station, bs="re") + s(month_id, bs="re")`.
- **FR-002**: System MUST use the same distribution family for each baseline model as its corresponding full model (binomial for presence metrics, negative binomial for count metrics).
- **FR-003**: System MUST use the AR1 correlation parameter (rho) from the full model when fitting the baseline model to ensure comparable autocorrelation structure.
- **FR-004**: System MUST save each baseline model object to `results/models/<metric>/gamm_baseline.rds`.
- **FR-005**: System MUST compute AIC and deviance explained for both full and baseline models.
- **FR-006**: System MUST produce a comparison table at `results/tables/baseline_comparison.csv` with columns: metric, full_aic, baseline_aic, delta_aic, full_dev_explained, baseline_dev_explained.
- **FR-007**: System MUST check model convergence status and record it in the output; non-converged models should be flagged but not halt execution.
- **FR-008**: System MUST load existing full model objects from `results/models/<metric>/gamm.rds` rather than refitting them.
- **FR-009**: System MUST fit 60-index GAMM models for all 9 response variables using all available acoustic indices (no VIF filtering) plus environmental + temporal + random effects.
- **FR-010**: System MUST save each 60-index model object to `results/models/<metric>/gamm_60index.rds`.
- **FR-011**: System MUST produce a VIF validation table at `results/tables/vif_validation.csv` with columns: metric, full_aic (17-index), full60_aic (60-index), delta_aic, full_dev_explained, full60_dev_explained, converged_60index.
- **FR-012**: System MUST extract significant index terms (p < 0.05) from both 17-index and 60-index models and output to `results/tables/vif_significance_comparison.csv` for comparison.

### Key Entities

- **Full Model (17-index)**: Existing GAMM with 17 VIF-filtered acoustic indices + environmental + temporal + random effects. Stored at `results/models/<metric>/gamm.rds`.
- **Baseline Model**: New GAMM with only environmental + temporal + random effects (no indices). To be stored at `results/models/<metric>/gamm_baseline.rds`.
- **60-Index Model**: New GAMM with all 60 acoustic indices (no VIF filtering) + environmental + temporal + random effects. To be stored at `results/models/<metric>/gamm_60index.rds`.
- **Response Variables**: 9 metrics - 3 presence (fish_presence, dolphin_presence, vessel_presence) and 6 count (fish_activity, fish_richness, dolphin_burst_pulse, dolphin_echolocation, dolphin_whistle, dolphin_activity).
- **Comparison Metrics**: AIC (Akaike Information Criterion) and deviance explained (proportion of null deviance accounted for by the model).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Baseline models fitted and saved for all 9 response variables with convergence status recorded.
- **SC-002**: Comparison table produced with complete data for all converged model pairs (full vs baseline).
- **SC-003**: ΔAIC calculated correctly as (baseline_AIC - full_AIC) so positive values indicate indices improve fit.
- **SC-004**: Results enable clear yes/no answer to research question: "Do acoustic indices add explanatory power?" for each response variable, using ΔAIC > 10 as the threshold for meaningful contribution.
- **SC-005**: Analysis is reproducible - running the script again produces identical results given the same input data and full models.
- **SC-006**: 60-index models attempted for all 9 response variables with convergence status recorded.
- **SC-007**: VIF validation table produced comparing 17-index vs 60-index model performance.
- **SC-008**: Significance comparison table produced showing which indices are significant in each model type, enabling assessment of whether VIF filtering removed useful vs redundant indices.

## Assumptions

- All 9 full GAMM models have already converged successfully (confirmed in existing model_summary.csv).
- The existing full models use the formula structure documented in stage05_modeling.R with 17 acoustic indices.
- Using the full model's rho value for baseline models is appropriate for fair comparison (alternative: re-estimate rho for baseline, but this changes the correlation structure).
- ΔAIC > 10 is an appropriate threshold for "meaningful" contribution based on Burnham & Anderson guidelines.
- The data file `data/processed/analysis_ready.parquet` contains all variables needed for baseline model fitting.
- Deviance explained is extracted from the GAM summary using standard mgcv conventions.
- All 60 acoustic indices are available in `data/processed/analysis_ready.parquet` (the VIF filtering was done during analysis, not during data preparation).
- 60-index models may have convergence issues due to high collinearity; this is expected and part of what the validation demonstrates.

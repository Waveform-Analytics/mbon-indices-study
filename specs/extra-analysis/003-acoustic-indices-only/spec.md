# Feature Specification: Acoustic Indices Only Model Comparison

**Feature Branch**: `003-acoustic-indices-only`
**Created**: 2026-02-05
**Status**: Draft
**Input**: User description: "Run GAMM models with ONLY acoustic indices (no environmental or temporal variables) to compare against full models. Use all 60 indices consistent with current approach."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare Acoustic-Only vs Full Model Performance (Priority: P1)

A researcher wants to understand how much explanatory power comes from acoustic indices alone, compared to the full model that includes environmental and temporal covariates. This helps assess whether indices can stand alone as predictors of biological activity.

**Why this priority**: This is the core deliverable that directly answers the research question. It's the inverse of the baseline comparison (which tested env/temporal-only), completing the picture of which components drive model performance.

**Independent Test**: Can be fully tested by fitting acoustic-indices-only models for all 9 response variables and comparing AIC/deviance explained against existing full models.

**Acceptance Scenarios**:

1. **Given** fitted full GAMM models exist for all 9 response variables, **When** the analyst runs the acoustic-only comparison script, **Then** models are fitted using only the 60 acoustic indices + random effects (no temperature, depth, hour_of_day, day_of_year).
2. **Given** both full and acoustic-only models exist for a response variable, **When** the comparison is computed, **Then** ΔAIC and Δdeviance_explained are calculated and saved to a summary table.
3. **Given** the comparison table is complete, **When** the analyst reviews results, **Then** they can identify which response variables are well-predicted by acoustic indices alone vs require environmental/temporal context.

---

### User Story 2 - Access Individual Acoustic-Only Model Objects (Priority: P2)

A researcher wants to inspect the acoustic-only model details (coefficient estimates, smooth term significance, summary statistics) to understand which indices drive predictions when environmental/temporal confounders are removed.

**Why this priority**: Model objects enable detailed inspection beyond summary metrics. This reveals which specific indices are predictive of each response variable in isolation, supporting targeted interpretation. Secondary to the comparison table because the comparison answers the primary research question.

**Independent Test**: Can be fully tested by loading a saved acoustic-only model object and calling standard model inspection functions (summary, plot).

**Acceptance Scenarios**:

1. **Given** acoustic-only model fitting is complete for fish_presence, **When** the analyst loads `results/models/fish_presence/gamm_acoustic_only.rds`, **Then** the model object is a valid fitted GAMM that can be inspected with `summary()` and `plot()`.
2. **Given** all 9 acoustic-only models are saved, **When** the analyst lists the model directory contents, **Then** each metric folder contains `gamm_acoustic_only.rds` alongside the existing model files.

---

### User Story 3 - Interpret Results in Context of Baseline Comparison (Priority: P2)

A researcher wants to synthesize findings from both the baseline comparison (env/temporal-only) and this acoustic-only comparison to understand the relative contribution of each component type. They need a combined view showing how much explanatory power comes from indices vs environmental/temporal factors.

**Why this priority**: This synthesis provides the complete picture for publication. Combined with baseline results, it answers: "What proportion of model performance comes from acoustic indices vs environmental/temporal covariates?" Same priority as P2 because it builds on P1.

**Independent Test**: Can be tested by reviewing both comparison tables (baseline_comparison.csv and acoustic_only_comparison.csv) and verifying interpretable summary is produced.

**Acceptance Scenarios**:

1. **Given** both baseline_comparison.csv and acoustic_only_comparison.csv exist, **When** the analyst reviews the outputs, **Then** they can compare deviance explained by indices-only vs env/temporal-only for each response variable.
2. **Given** the comparison tables are complete, **When** the analyst interprets results, **Then** they can determine whether indices or env/temporal variables are more predictive for each response.

---

### Edge Cases

- What happens if an acoustic-only model fails to converge? Script should log a warning and continue with other models, marking that metric as failed in the output table.
- What happens if the full model file is missing for a metric? Script should skip that metric with a clear error message rather than crashing.
- How does the system handle metrics where acoustic-only deviance explained exceeds full model? This could happen if env/temporal terms introduce multicollinearity or overfitting. Flag it in output for review but it's not necessarily an error.
- What if AR1 correlation estimation differs between full and acoustic-only models? Use the full model's rho value for both to ensure fair comparison, consistent with the baseline approach.
- What if acoustic-only models fit poorly (very low deviance explained)? This is a valid scientific finding - indices may genuinely require environmental/temporal context to be predictive. Record the result without treating it as an error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST fit acoustic-only GAMM models for all 9 response variables using only the 60 acoustic indices + random effects, with formula: `response ~ s(index1, k=3) + s(index2, k=3) + ... + s(index60, k=3) + s(station, bs="re") + s(month_id, bs="re")`.
- **FR-002**: System MUST use the same distribution family for each acoustic-only model as its corresponding full model (binomial for presence metrics, negative binomial for count metrics).
- **FR-003**: System MUST use the AR1 correlation parameter (rho) from the full model when fitting the acoustic-only model to ensure comparable autocorrelation structure.
- **FR-004**: System MUST save each acoustic-only model object to `results/models/<metric>/gamm_acoustic_only.rds`.
- **FR-005**: System MUST compute AIC and deviance explained for both full and acoustic-only models.
- **FR-006**: System MUST produce a comparison table at `results/tables/acoustic_only_comparison.csv` with columns: metric, full_aic, acoustic_only_aic, delta_aic, full_dev_explained, acoustic_only_dev_explained, acoustic_only_converged.
- **FR-007**: System MUST check model convergence status and record it in the output; non-converged models should be flagged but not halt execution.
- **FR-008**: System MUST load existing full model objects from `results/models/<metric>/gamm.rds` rather than refitting them.
- **FR-009**: System MUST use all 60 acoustic indices from `data/interim/aligned_indices.parquet`, consistent with the current full model approach (vif_enabled: false in config/analysis.yml).
- **FR-010**: System MUST use k=3 for index smooth terms, matching the current full model configuration (gamm.smooth_k: 3).

### Key Entities

- **Full Model (60-index)**: Existing GAMM with 60 acoustic indices + environmental + temporal + random effects. Stored at `results/models/<metric>/gamm.rds`.
- **Acoustic-Only Model**: New GAMM with only 60 acoustic indices + random effects (no temperature, depth, hour_of_day, day_of_year). To be stored at `results/models/<metric>/gamm_acoustic_only.rds`.
- **Baseline Model**: Existing GAMM with only environmental + temporal + random effects (no indices). Stored at `results/models/<metric>/gamm_baseline.rds`.
- **Response Variables**: 9 metrics - 3 presence (fish_presence, dolphin_presence, vessel_presence) and 6 count (fish_activity, fish_richness, dolphin_burst_pulse, dolphin_echolocation, dolphin_whistle, dolphin_activity).
- **Comparison Metrics**: AIC (Akaike Information Criterion) and deviance explained (proportion of null deviance accounted for by the model).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Acoustic-only models fitted and saved for all 9 response variables with convergence status recorded.
- **SC-002**: Comparison table produced with complete data for all converged model pairs (full vs acoustic-only).
- **SC-003**: ΔAIC calculated correctly as (acoustic_only_AIC - full_AIC) so positive values indicate the full model with env/temporal variables fits better.
- **SC-004**: Results enable assessment of acoustic indices' standalone predictive power for each response variable, compared to the full model.
- **SC-005**: Analysis is reproducible - running the script again produces identical results given the same input data and full models.
- **SC-006**: Results can be combined with baseline_comparison.csv to show the relative contribution of indices vs environmental/temporal variables to overall model performance.

## Assumptions

- All 9 full GAMM models have already converged successfully and use 60 acoustic indices (vif_enabled: false).
- The existing full models use the formula structure documented in stage05_modeling.R with 60 indices at k=3.
- Using the full model's rho value for acoustic-only models is appropriate for fair comparison.
- The data file `data/interim/aligned_indices.parquet` contains all 60 acoustic indices.
- Deviance explained is extracted from the GAM summary using standard mgcv conventions.
- The baseline comparison (002-baseline-comparison) has been run, so baseline_comparison.csv exists for synthesis.
- Random effects (station, month_id) are still appropriate even without temporal smooth terms (day_of_year, hour_of_day), since they capture site-specific and monthly variation.

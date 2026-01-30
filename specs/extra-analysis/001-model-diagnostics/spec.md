# Feature Specification: Model Diagnostics & Effect Interpretation

**Feature Branch**: `001-model-diagnostics`
**Created**: 2026-01-28
**Status**: Draft
**Input**: User description: "Model Diagnostics & Effect Interpretation - Validate model assumptions and quantify practical significance of index effects"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review Effect Sizes for Publication (Priority: P1)

A researcher needs to communicate the practical significance of acoustic indices to stakeholders. Rather than just saying "ACI is statistically significant," they want to say "a change in ACI from low to high values corresponds to a 15% increase in fish presence probability."

**Why this priority**: Effect sizes are the most important output for stakeholder communication and publication. Statistical significance without practical magnitude is incomplete.

**Independent Test**: Can be fully tested by running effect size calculations on the 9 existing GAMM models and producing a summary table showing predicted response changes across index ranges.

**Acceptance Scenarios**:

1. **Given** a fitted GAMM model for fish_presence, **When** the analyst runs the effect size calculation, **Then** the output shows predicted probability change from the 10th to 90th percentile of each acoustic index.
2. **Given** a fitted GAMM model for fish_activity (count), **When** the analyst runs the effect size calculation, **Then** the output shows predicted count change (or fold-change) from the 10th to 90th percentile of each acoustic index.

---

### User Story 2 - Validate Model Assumptions Before Publication (Priority: P2)

A researcher wants to ensure model assumptions are reasonably met before submitting results for publication or presentation. This includes checking that smooth terms aren't confounded (concurvity), that count models adequately capture zeros, and that random effects are approximately normal.

**Why this priority**: Assumption validation is standard practice and a reviewer would expect these checks. However, it's secondary to effect sizes because the models have already been validated via cross-validation.

**Independent Test**: Can be fully tested by running concurvity checks on the 9 models and producing a summary table flagging any problematic terms (concurvity > 0.8).

**Acceptance Scenarios**:

1. **Given** a fitted GAMM with 17 index smooth terms, **When** the analyst runs the concurvity check, **Then** the output reports worst-case concurvity for each smooth term.
2. **Given** a count model (e.g., fish_activity), **When** the analyst runs the zero-inflation check, **Then** the output compares observed vs predicted zero proportions.
3. **Given** a model with station and month_id random effects, **When** the analyst runs random effects diagnostics, **Then** QQ plots are generated to assess normality.

---

### User Story 3 - Identify Station-Specific Model Issues (Priority: P3)

A researcher wants to check whether the model performs consistently across all three stations (9M, 14M, 37M) or whether one station is systematically over/under-predicted.

**Why this priority**: Station-level diagnostics are useful for identifying microhabitat issues but are less critical than overall effect interpretation.

**Independent Test**: Can be fully tested by generating residual plots faceted by station for any single response variable.

**Acceptance Scenarios**:

1. **Given** a fitted GAMM and residuals, **When** the analyst generates station-faceted residual plots, **Then** the output shows residuals vs fitted values separately for each station.

---

### Edge Cases

- What happens when a model has no significant index terms? Effect sizes should still be calculable (they'll just be small).
- How does the system handle models that failed to converge? The script should check convergence status before processing and skip non-converged models with a warning.
- What happens if concurvity is extremely high (>0.95)? Flag with a warning but continue processing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST calculate effect sizes for all 9 response variables showing predicted change across the 10th to 90th percentile range of each acoustic index.
- **FR-002**: System MUST differentiate effect size presentation: probability change for presence models (fish_presence, dolphin_presence, vessel_presence) and count/fold-change for count models.
- **FR-003**: System MUST run concurvity analysis on all fitted models and flag any smooth terms with concurvity > 0.8.
- **FR-004**: System MUST compare observed vs predicted zero proportions for all 6 count models (fish_activity, fish_richness, dolphin_burst_pulse, dolphin_echolocation, dolphin_whistle, dolphin_activity).
- **FR-005**: System MUST generate residual plots faceted by station (9M, 14M, 37M) for all 9 response variables.
- **FR-006**: System MUST generate QQ plots for station and month_id random effects for all 9 response variables.
- **FR-007**: System MUST produce a consolidated effect sizes table across all models and indices.
- **FR-008**: System MUST check model convergence status before processing and skip non-converged models with a warning.

### Key Entities

- **Response Variables**: 9 metrics (3 presence, 6 count) with fitted GAMM objects stored in `results/models/<metric>/gamm.rds`
- **Acoustic Indices**: 17 indices after VIF reduction, used as smooth predictors in all models
- **Stations**: 3 locations (9M, 14M, 37M) used for spatial diagnostics
- **Random Effects**: station and month_id, included in all models

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Effect size table produced for all 9 response variables, with interpretable magnitude values (probability change for presence, fold-change for counts).
- **SC-002**: Concurvity values computed for all smooth terms across all models, with clear flagging of terms exceeding 0.8 threshold.
- **SC-003**: Zero-inflation comparison completed for all 6 count models, with gaps > 10% clearly flagged.
- **SC-004**: Residual-by-station plots generated for all 9 models, enabling visual inspection of station-specific patterns.
- **SC-005**: Random effects QQ plots generated for all 9 models, enabling assessment of normality assumption.
- **SC-006**: All diagnostic outputs organized in existing results directory structure for easy integration with reporting.

## Assumptions

- All 9 GAMM models have already converged successfully (confirmed in model_summary.csv).
- The 10th to 90th percentile range is appropriate for effect size calculation (avoids extreme values that may be extrapolation).
- Concurvity threshold of 0.8 is standard for flagging potential issues (not a hard failure criterion).
- Zero-inflation gap of 10% is a reasonable threshold for flagging concern (domain-specific judgment may override).
- Existing mgcv functions (concurvity(), predict()) are sufficient for all calculations.
# Research: Model Diagnostics & Effect Interpretation

**Date**: 2026-01-28
**Feature**: 001-model-diagnostics

## Summary

This feature uses well-established R/mgcv functions. Minimal research required - all approaches are standard biostatistical practice.

## Decisions

### 1. Concurvity Calculation

**Decision**: Use `mgcv::concurvity(model, full=TRUE)`

**Rationale**:
- Standard mgcv function specifically designed for this purpose
- Returns both "worst" (conservative) and "observed" (actual) concurvity values
- Threshold of 0.8 is commonly used in literature

**Alternatives considered**:
- Manual correlation of smooth basis functions - unnecessary, mgcv handles this
- VIF on smooth predictions - less appropriate for GAMs than concurvity

### 2. Effect Size Calculation

**Decision**: Use `predict()` with newdata varying one index at a time (10th to 90th percentile), other predictors at median values

**Rationale**:
- Standard "marginal effects" approach
- 10th-90th percentile avoids extrapolation at extremes
- Median for other predictors is neutral reference point

**Alternatives considered**:
- Average marginal effects (AME) - more complex, not necessary for this use case
- Full range (min to max) - risks extrapolation beyond data support

### 3. Effect Size Presentation

**Decision**:
- Presence models: Probability change (absolute difference)
- Count models: Fold-change (ratio of high/low predictions)

**Rationale**:
- Probability change is intuitive for binary outcomes ("15% more likely")
- Fold-change is standard for count data ("1.5x more fish activity")

**Alternatives considered**:
- Log-odds ratio for presence - less intuitive for stakeholders
- Absolute count change - scale-dependent, harder to compare across metrics

### 4. Zero-Inflation Check

**Decision**: Compare observed zero proportion to predicted zero proportion from fitted model

**Rationale**:
- Simple, interpretable comparison
- Gap > 10% flags potential need for zero-inflated model
- Does not require fitting additional models

**Alternatives considered**:
- Vuong test comparing NB vs ZINB - requires fitting zero-inflated model
- DHARMa residual diagnostics - adds dependency, more complex

### 5. Random Effects Diagnostics

**Decision**: QQ plots of extracted random effect estimates (BLUPs)

**Rationale**:
- Visual check of normality assumption
- Standard diagnostic for mixed models
- Easy to generate from mgcv model object

**Alternatives considered**:
- Formal normality tests (Shapiro-Wilk) - often too sensitive with large samples
- Caterpillar plots - useful but QQ plots more directly assess normality

## No Additional Research Needed

All required functionality is available in the current R environment:
- `mgcv::concurvity()` - concurvity calculation
- `predict.gam()` - predictions for effect sizes
- `residuals.gam()` - deviance residuals
- `ggplot2::stat_qq()` - QQ plots
- Base R quantile functions - percentile calculations

No new packages or complex integrations required.

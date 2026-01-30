# Analysis Roadmap

## Context

Following stakeholder discussions (Jan 2026), we identified additional analyses to strengthen our understanding of whether acoustic indices add explanatory power for biological community metrics. These complement the existing GAMM models (Stage 05).

**Research Question**: Can acoustic indices quickly characterize soundscapes as indicators of biological community metrics?

---

# Feature Specs

Each section below is scoped as a separate feature specification, listed in implementation order.

**Location**: `specs/extra-analysis/` (separate from main pipeline stages)

**Naming convention**: `XXX-short-name/spec.md` where XXX is a 3-digit number matching the branch name.

---

## Feature 1: Model Diagnostics & Effect Interpretation

**Spec location**: `specs/extra-analysis/001-model-diagnostics/spec.md`
**Branch**: `001-model-diagnostics`
**Status**: Spec draft complete

### Purpose
Validate model assumptions and quantify practical significance of index effects. These are standard checks a biostatistician would expect before publication.

### Checks to Implement

**1. Concurvity Check**
- Run `mgcv::concurvity()` on each fitted model
- Flag any smooth terms with concurvity > 0.8 (nearly confounded)
- Output: `results/tables/<metric>/concurvity.csv`

**2. Effect Size Summary**
- For each significant index, calculate predicted response change across the index range (10th to 90th percentile)
- For presence models: change in probability
- For count models: change in expected count (or fold-change)
- Output: `results/tables/effect_sizes.csv`

**3. Zero-Inflation Check (counts only)**
- Compare observed zero proportion to model-predicted zero proportion
- Flag if gap > 10% (may need zero-inflated model)
- Output: `results/tables/zero_inflation_check.csv`

**4. Residuals by Station**
- Generate residual plots faceted by station
- Check for systematic over/under-prediction or heteroscedasticity
- Output: `results/figures/<metric>/residuals_by_station.png`

**5. Random Effects Diagnostics**
- QQ plots of station and month_id random effects
- Check approximate normality assumption
- Output: `results/figures/<metric>/random_effects_qq.png`

### Why This Matters
- Concurvity: Validates that VIF reduction was sufficient for smooth terms
- Effect sizes: Answers "how much does it matter?" not just "is it significant?"
- Zero-inflation: Ensures count model family is appropriate
- Residual checks: Catches systematic model failures by location
- Random effects: Validates model assumptions

---

## Feature 2: Baseline Model Comparison

**Spec location**: `specs/extra-analysis/002-baseline-comparison/spec.md`
**Branch**: `002-baseline-comparison`
**Status**: Not started

### Purpose
Compare GAMM performance with and without acoustic indices to directly answer: "Do indices add explanatory power beyond environmental and temporal variables alone?"

### Approach
- Fit "baseline" GAMMs for each response variable using only:
  - Environmental: `s(temperature, k=5) + s(depth, k=5)`
  - Temporal: `s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12)`
  - Random effects: `s(station, bs="re") + s(month_id, bs="re")`
- Use identical AR1 correlation structure as full models
- Compare AIC and deviance explained between full model (with 17 indices) and baseline model (no indices)

### Expected Outputs
- Baseline model objects: `results/models/<metric>/gamm_baseline.rds`
- Comparison table: `results/tables/baseline_comparison.csv` with columns:
  - metric, full_AIC, baseline_AIC, delta_AIC, full_dev_explained, baseline_dev_explained
- Interpretation: ΔAIC > 10 suggests indices contribute meaningfully

### Why This Matters
If baseline models perform nearly as well as full models (small ΔAIC), indices aren't adding value. If ΔAIC is large, indices capture information beyond what temperature, depth, and time provide.

---

## Feature 3: Spatial Cross-Validation

**Spec location**: `specs/extra-analysis/003-spatial-cv/spec.md`
**Branch**: `003-spatial-cv`
**Status**: Not started

### Purpose
Test spatial generalization: "Can the model predict biological responses at a new, unseen location?"

### Approach
- 3-fold CV where each fold holds out one station entirely:
  - Fold 1: Train on 9M + 14M, test on 37M
  - Fold 2: Train on 9M + 37M, test on 14M
  - Fold 3: Train on 14M + 37M, test on 9M
- Calculate same performance metrics as existing CV (AUC for presence, RMSE/R² for counts)
- Compare to existing leave-week-out CV results

### Expected Outputs
- CV performance table: `results/tables/spatial_cv_performance.csv`
- Comparison table: `results/tables/cv_comparison.csv` (temporal vs spatial CV)
- Figure: `results/figures/cv_comparison.png` (leave-week-out vs leave-station-out)

### Why This Matters
The current leave-week-out CV tests temporal generalization (can we predict new time periods?). Leave-station-out tests whether the model works at new deployment locations - critical for the practical application of using indices to characterize soundscapes at new sites.

---

# Approaches Considered But Not Prioritized

### All 60 Indices Without VIF Filtering
**Decision**: Not recommended at this time.

**Rationale**:
- VIF reduction removed redundant indices (highly correlated predictors don't add independent information)
- 60 smooth terms in a GAMM risks convergence issues and overfitting
- Multicollinearity inflates standard errors and makes coefficient estimates unstable
- "Let the model decide" sounds appealing but often produces hard-to-interpret results

**Alternative if needed**: Could compare 17-index vs 60-index AIC to validate VIF approach, but this is low priority.

### Separate Models Per Station (Microhabitat Effects)
**Decision**: Not the optimal approach for this question.

**Rationale**:
- The current random effect `s(station, bs="re")` already captures station-level baseline differences
- Fitting 27 separate models (9 responses × 3 stations) reduces sample size and statistical power
- Doesn't provide a direct statistical test of whether effects differ by station
- Interpretation becomes complex with 27 sets of results

**Better alternative if needed**: Add station × index interaction terms for 2-3 key indices of interest, rather than separate models.

---

# Implementation Notes

## Folder Structure

```
specs/
├── stages/           # Main pipeline (00-05, 10)
└── extra-analysis/   # Additional analyses (this roadmap)
    ├── 001-model-diagnostics/spec.md   ← Feature 1 (in progress)
    ├── 002-baseline-comparison/spec.md ← Feature 2 (planned)
    └── 003-spatial-cv/spec.md          ← Feature 3 (planned)
```

## Code Reuse
- `stage05_modeling.R` is modular enough to adapt for baseline models
- `stage05b_validation.R` can be adapted for leave-station-out CV
- Diagnostics mostly use built-in `mgcv` functions (`concurvity()`, `predict()`, etc.)

## Dependencies
- Features 2 and 3 are independent (can be done in parallel after Feature 1)
- Feature 1 can run on existing models immediately
- All features depend on Stage 05 being complete (it is)

---

**Created**: 2026-01-28 | **Last Updated**: 2026-01-28 | **Status**: Planning

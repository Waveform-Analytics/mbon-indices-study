# Implementation Plan: Baseline Model Comparison & VIF Validation

**Branch**: `002-baseline-comparison` | **Date**: 2026-02-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-baseline-comparison/spec.md`

## Summary

Fit baseline GAMM models (environmental + temporal only, no indices) for all 9 response variables and compare against full models to quantify index contribution. Secondary analysis: validate VIF filtering by comparing 17-index vs 60-index models.

## Technical Context

**Language/Version**: R 4.x (matching existing stage05 scripts)
**Primary Dependencies**: mgcv (already used), dplyr, arrow, ggplot2
**Storage**: RDS (models), CSV (tables), PNG (figures)
**Testing**: Manual verification via script execution and output review
**Target Platform**: Local R environment (macOS)
**Project Type**: Data analysis scripts (extending existing pipeline)
**Performance Goals**: Process all 9 models in under 10 minutes per comparison type
**Constraints**: Must integrate with existing `results/` directory structure; 60-index models may hit memory/convergence limits
**Scale/Scope**: 9 GAMM models × 3 variants (full, baseline, 60-index)

## Constitution Check

*GATE: Checking against `.specify/memory/constitution.md`*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Research-First | ✅ Pass | Manual verification, step-by-step review |
| II. Script Naming | ✅ Pass | Will use `stage05d_baseline.R` |
| III. Output Structure | ✅ Pass | Outputs to `results/models/`, `results/tables/` |
| IV. Configuration-Driven | ✅ Pass | Uses existing `config/analysis.yml` |
| V. Data Flow | ✅ Pass | Inputs from existing parquet/rds files |

**Quality Gates:**
- Convergence check: Will verify convergence before computing metrics
- Output verification: User reviews before proceeding
- Documentation sync: Will update `rebuild-docs.sh` if needed

No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/002-baseline-comparison/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── checklists/
│   └── requirements.md  # Spec validation checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
scripts/
├── stage05_modeling.R           # Existing - fits full GAMMs (17 indices)
├── stage05b_validation.R        # Existing - AR1 and CV validation
├── stage05c_diagnostics.R       # Existing - diagnostics and effect sizes
└── stage05d_baseline.R          # NEW - baseline and VIF validation

results/
├── models/<metric>/
│   ├── gamm.rds                 # Existing - full model (17 indices)
│   ├── gamm_baseline.rds        # NEW - baseline model (no indices)
│   └── gamm_60index.rds         # NEW - 60-index model (VIF validation)
├── tables/
│   ├── baseline_comparison.csv  # NEW - baseline vs full comparison
│   ├── vif_validation.csv       # NEW - 17-index vs 60-index comparison
│   └── vif_significance_comparison.csv  # NEW - significant terms comparison
└── figures/
    └── baseline_comparison.png  # NEW - visualization of ΔAIC
```

**Structure Decision**: Single new R script (`stage05d_baseline.R`) following existing naming convention.

---

## Phase 0: Research

### Technical Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Baseline formula | Remove all `s(index, k=5)` terms, keep environmental + temporal + random effects | Matches spec FR-001 |
| AR1 rho handling | Use rho from existing full model | Spec specifies this for fair comparison (same autocorrelation structure) |
| 60-index data source | Join from `data/interim/aligned_indices.parquet` | Contains all indices before VIF filtering |
| 60-index k value | Use k=3 instead of k=5 | Reduce parameters to help convergence with 60 smooth terms |
| Deviance explained | Extract from `summary(model)$dev.expl` | Standard mgcv convention |
| AIC extraction | Use `AIC(model)` | Standard R function |
| ΔAIC interpretation | baseline_AIC - full_AIC | Positive = indices improve fit |

### Data Availability

**Confirmed:**
- Full models: `results/models/<metric>/gamm.rds` (9 models, all converged)
- 17 indices: in `data/processed/analysis_ready.parquet`
- 60 indices: in `data/interim/aligned_indices.parquet` (62 indices available)
- Environmental/temporal: temperature, depth, hour_of_day, day_of_year, station, month_id

### Dependencies

All required packages already installed:
- `mgcv` - model fitting
- `arrow` - read parquet
- `dplyr` - data manipulation
- `ggplot2` - visualization

No new dependencies required.

---

## Phase 1: Design

### Data Flow

```
Input:
  results/models/<metric>/gamm.rds          # Full models (17-index)
  data/processed/analysis_ready.parquet     # Analysis data (17 indices)
  data/interim/aligned_indices.parquet      # All ~60 indices

Processing:
  Part A - Baseline Comparison (P1):
    1. Load each full model
    2. Extract rho parameter
    3. Fit baseline model (no indices) with same rho
    4. Compute AIC and deviance explained for both
    5. Calculate ΔAIC

  Part B - VIF Validation (P2):
    1. Merge 60 indices with analysis data
    2. Fit 60-index models (may fail for some metrics)
    3. Compare AIC/significance with 17-index models

Output:
  results/models/<metric>/gamm_baseline.rds
  results/models/<metric>/gamm_60index.rds
  results/tables/baseline_comparison.csv
  results/tables/vif_validation.csv
  results/tables/vif_significance_comparison.csv
  results/figures/baseline_comparison.png
```

### Key Functions

**1. `build_baseline_formula(response)`**
- Returns formula: `response ~ s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) + s(station, bs='re') + s(month_id, bs='re')`

**2. `fit_baseline_model(formula, data, family, rho)`**
- Fit GAMM using bam() with provided rho
- Return model or NULL on failure

**3. `extract_model_metrics(model)`**
- Return list: aic, deviance_explained, converged

**4. `build_60index_formula(response, index_cols)`**
- Like full model but with all 60 indices, k=3

**5. `extract_significant_terms(model, threshold=0.05)`**
- Return data.frame of term, edf, p_value, significant

### Output Schemas

**baseline_comparison.csv**
```
metric,full_aic,baseline_aic,delta_aic,full_dev_explained,baseline_dev_explained,interpretation
fish_presence,12345.6,12567.8,222.2,0.45,0.32,"Strong evidence indices add value"
```

**vif_validation.csv**
```
metric,full17_aic,full60_aic,delta_aic,full17_dev_explained,full60_dev_explained,converged_60index
fish_presence,12345.6,12340.2,-5.4,0.45,0.46,TRUE
```

**vif_significance_comparison.csv**
```
metric,index,significant_17,significant_60,pval_17,pval_60
fish_presence,ACI,TRUE,TRUE,0.001,0.003
```

### Interpretation Guidelines

ΔAIC thresholds (Burnham & Anderson):
- ΔAIC > 10: Strong evidence indices add value
- ΔAIC 4-10: Moderate evidence
- ΔAIC < 4: Weak/no evidence

---

## Quickstart

```r
# Run baseline comparison only (MVP for meeting)
Rscript scripts/stage05d_baseline.R --baseline-only

# Run full analysis (baseline + VIF validation)
Rscript scripts/stage05d_baseline.R

# Run on single metric for testing
Rscript scripts/stage05d_baseline.R --metric fish_presence

# Check outputs
cat results/tables/baseline_comparison.csv
```

---

## Implementation Priority

**For tomorrow's meeting - MVP scope:**
1. ✅ Baseline comparison (User Story 1, P1) - answers the core research question
2. ⏸️ VIF validation (User Story 4, P2) - can defer if time-constrained

The baseline comparison alone provides the key result: "Do indices add explanatory power?"

---

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Implement `stage05d_baseline.R` (baseline comparison first)
3. Run and verify outputs
4. Add to results viewer if time permits

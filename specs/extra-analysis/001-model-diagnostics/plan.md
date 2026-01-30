# Implementation Plan: Model Diagnostics & Effect Interpretation

**Branch**: `001-model-diagnostics` | **Date**: 2026-01-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/extra-analysis/001-model-diagnostics/spec.md`

## Summary

Add diagnostic checks and effect size calculations to the existing GAMM modeling pipeline. This extends Stage 05 with standard biostatistical validation (concurvity, zero-inflation, residual diagnostics) and practical effect size quantification for stakeholder communication.

## Technical Context

**Language/Version**: R 4.x (matching existing stage05 scripts)
**Primary Dependencies**: mgcv (already used), ggplot2, dplyr, tidyr, arrow
**Storage**: RDS (models), CSV (tables), PNG (figures) - matches existing output structure
**Testing**: Manual verification via script execution and output review
**Target Platform**: Local R environment (macOS/Linux)
**Project Type**: Data analysis scripts (extending existing pipeline)
**Performance Goals**: Process all 9 models in under 5 minutes
**Constraints**: Must integrate with existing `results/` directory structure
**Scale/Scope**: 9 GAMM models, 17 indices, 3 stations

## Constitution Check

*GATE: No constitution defined for this project - no gates to enforce.*

The project uses a flexible research-oriented workflow without formal architectural constraints.

## Project Structure

### Documentation (this feature)

```text
specs/extra-analysis/001-model-diagnostics/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output (minimal - standard R functions)
├── checklists/
│   └── requirements.md  # Spec validation checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
scripts/
├── stage05_modeling.R           # Existing - fits GAMMs
├── stage05b_validation.R        # Existing - AR1 and CV validation
├── stage05c_diagnostics.R       # NEW - this feature
├── generate_diagnostics.R       # Existing - basic diagnostics
└── generate_smooth_plots.R      # Existing - smooth visualizations

results/
├── models/<metric>/gamm.rds     # Existing - fitted models (input)
├── tables/
│   ├── effect_sizes.csv         # NEW - consolidated effect sizes
│   ├── zero_inflation_check.csv # NEW - zero-inflation comparison
│   └── <metric>/
│       └── concurvity.csv       # NEW - per-model concurvity
└── figures/<metric>/
    ├── residuals_by_station.png # NEW - station-faceted residuals
    └── random_effects_qq.png    # NEW - RE normality check
```

**Structure Decision**: Single new R script (`stage05c_diagnostics.R`) that loads existing model objects and generates all diagnostic outputs. Follows existing naming convention (stage05a, 05b, 05c).

## Complexity Tracking

No constitution violations to justify - this is a straightforward extension of existing analysis code.

---

## Phase 0: Research

### Technical Decisions

This feature uses well-established R/mgcv functions. No significant research required.

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Concurvity calculation | Use `mgcv::concurvity(model, full=TRUE)` | Standard mgcv function, returns worst-case values |
| Effect size calculation | Use `predict()` with newdata at 10th/90th percentiles | Standard approach, hold other predictors at median |
| Zero-inflation check | Compare `mean(y == 0)` vs `mean(predict(model, type="response") < 0.5)` | Simple proportion comparison |
| Residual extraction | Use `residuals(model, type="deviance")` | Standard for GAMMs |
| Random effects | Extract via `gam.vcomp()` or model coefficients | mgcv provides access to RE estimates |

### Dependencies

All required packages are already installed for stage05:
- `mgcv` - model fitting and diagnostics
- `ggplot2` - plotting
- `dplyr` / `tidyr` - data manipulation
- `arrow` - read parquet data

No new dependencies required.

---

## Phase 1: Design

### Data Flow

```
Input:
  results/models/<metric>/gamm.rds (9 files)
  data/processed/analysis_ready.parquet

Processing:
  1. Load each model
  2. Check convergence
  3. Calculate concurvity
  4. Calculate effect sizes (varies by model type)
  5. Check zero-inflation (counts only)
  6. Generate residual plots by station
  7. Generate random effects QQ plots

Output:
  results/tables/effect_sizes.csv
  results/tables/zero_inflation_check.csv
  results/tables/<metric>/concurvity.csv
  results/figures/<metric>/residuals_by_station.png
  results/figures/<metric>/random_effects_qq.png
```

### Key Functions

**1. `calculate_effect_sizes(model, data, index_cols, model_type)`**
- For each index: predict at 10th and 90th percentile, other vars at median
- Return: data.frame with index, low_pred, high_pred, effect_size, effect_type

**2. `check_concurvity(model, threshold = 0.8)`**
- Run `concurvity(model, full=TRUE)`
- Flag terms exceeding threshold
- Return: data.frame with term, worst_concurvity, flagged

**3. `check_zero_inflation(model, data, response_col)`**
- Compare observed vs predicted zero proportions
- Return: data.frame with observed_zeros, predicted_zeros, gap, flagged

**4. `plot_residuals_by_station(model, data)`**
- Extract deviance residuals
- Facet by station
- Return: ggplot object

**5. `plot_random_effects_qq(model)`**
- Extract station and month_id random effects
- Generate QQ plots
- Return: ggplot object (2-panel)

### Output Schemas

**effect_sizes.csv**
```
metric,index,low_value,high_value,low_pred,high_pred,effect_size,effect_type,significant
fish_presence,ACI,0.12,0.89,0.31,0.46,0.15,probability_change,TRUE
fish_activity,ACI,0.12,0.89,12.3,18.7,1.52,fold_change,TRUE
```

**concurvity.csv** (per metric)
```
term,worst,observed,flagged
s(ACI),0.72,0.65,FALSE
s(temperature),0.85,0.81,TRUE
```

**zero_inflation_check.csv**
```
metric,observed_zero_prop,predicted_zero_prop,gap,flagged
fish_activity,0.42,0.38,0.04,FALSE
dolphin_whistle,0.78,0.65,0.13,TRUE
```

---

## Quickstart

```r
# Run diagnostics on all fitted models
Rscript scripts/stage05c_diagnostics.R

# Run on single model for testing
Rscript scripts/stage05c_diagnostics.R --metric fish_presence

# Check outputs
ls results/tables/effect_sizes.csv
ls results/tables/*/concurvity.csv
ls results/figures/*/residuals_by_station.png
```

---

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Implement `stage05c_diagnostics.R`
3. Run and verify outputs
4. Integrate into reporting (Stage 10)

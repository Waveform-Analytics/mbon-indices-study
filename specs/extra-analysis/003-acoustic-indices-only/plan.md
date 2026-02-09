# Implementation Plan: Acoustic Indices Only Model Comparison

**Branch**: `003-acoustic-indices-only` | **Date**: 2026-02-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/extra-analysis/003-acoustic-indices-only/spec.md`

## Summary

Fit GAMM models using ONLY acoustic indices (no environmental or temporal covariates) for all 9 response variables and compare against full models. This completes the decomposition of model performance alongside the baseline comparison (which tested env/temporal-only).

## Technical Context

**Language/Version**: R 4.x (matching existing stage05 scripts)
**Primary Dependencies**: mgcv (already used), dplyr, arrow
**Storage**: RDS (models), CSV (tables)
**Testing**: Manual verification via script execution and output review
**Target Platform**: Local R environment (macOS)
**Project Type**: Data analysis scripts (extending existing pipeline)
**Performance Goals**: Process all 9 models in under 10 minutes
**Constraints**: Must integrate with existing `results/` directory structure
**Scale/Scope**: 9 GAMM models × 1 new variant (acoustic-only)

## Constitution Check

*GATE: Checking against `.specify/memory/constitution.md`*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Research-First | ✅ Pass | Manual verification, step-by-step review |
| II. Script Naming | ✅ Pass | Will use `stage05e_acoustic_only.R` |
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
specs/extra-analysis/003-acoustic-indices-only/
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
├── stage05_modeling.R           # Existing - fits full GAMMs (60 indices)
├── stage05b_validation.R        # Existing - AR1 and CV validation
├── stage05c_diagnostics.R       # Existing - diagnostics and effect sizes
├── stage05d_baseline.R          # Existing - baseline comparison
└── stage05e_acoustic_only.R     # NEW - acoustic-only comparison

results/
├── models/<metric>/
│   ├── gamm.rds                 # Existing - full model (60 indices)
│   ├── gamm_baseline.rds        # Existing - baseline (no indices)
│   └── gamm_acoustic_only.rds   # NEW - acoustic-only (no env/temporal)
└── tables/
    ├── baseline_comparison.csv      # Existing - baseline vs full
    └── acoustic_only_comparison.csv # NEW - acoustic-only vs full
```

**Structure Decision**: Single new R script (`stage05e_acoustic_only.R`) following existing naming convention. Pattern mirrors `stage05d_baseline.R` closely.

---

## Phase 0: Research

### Technical Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Acoustic-only formula | 61 index smooth terms + random effects only | Inverse of baseline: tests indices alone |
| Terms to EXCLUDE | temperature, depth, hour_of_day, day_of_year | These are the env/temporal terms tested in baseline |
| Random effects | Keep station + month_id | These capture site/temporal grouping, not confounders |
| AR1 rho handling | Use rho from existing full model | Same approach as baseline for fair comparison |
| Index k value | k=3 | Matches current full model (config: gamm.smooth_k: 3) |
| Deviance explained | Extract from `summary(model)$dev.expl` | Standard mgcv convention |
| ΔAIC calculation | acoustic_only_AIC - full_AIC | Positive = full model better (env/temporal help) |

### Data Availability

**Confirmed:**
- Full models: `results/models/<metric>/gamm.rds` (9 models, all converged, 60 indices)
- 61 indices: in `data/interim/aligned_indices.parquet`
- Response variables: fish_activity, fish_richness, fish_presence, dolphin_burst_pulse, dolphin_echolocation, dolphin_whistle, dolphin_activity, dolphin_presence, vessel_presence

### Dependencies

All required packages already installed (same as stage05d_baseline.R):
- `mgcv` - model fitting
- `arrow` - read parquet
- `dplyr` - data manipulation

No new dependencies required.

---

## Phase 1: Design

### Data Flow

```
Input:
  results/models/<metric>/gamm.rds          # Full models (60-index)
  data/interim/aligned_indices.parquet      # All 61 indices
  data/processed/analysis_ready.parquet     # Response variables + grouping

Processing:
  1. Load each full model
  2. Extract rho parameter
  3. Merge 61 indices with response/grouping data
  4. Fit acoustic-only model (indices + random effects, no env/temporal)
  5. Compute AIC and deviance explained for both
  6. Calculate ΔAIC

Output:
  results/models/<metric>/gamm_acoustic_only.rds
  results/tables/acoustic_only_comparison.csv
```

### Key Functions

**1. `build_acoustic_only_formula(response, index_cols)`**
- Returns formula: `response ~ s(index1, k=3) + s(index2, k=3) + ... + s(station, bs='re') + s(month_id, bs='re')`
- NO temperature, depth, hour_of_day, day_of_year

**2. `fit_model_safe(formula, data, family, rho)`**
- Fit GAMM using bam() with provided rho
- Return model or NULL on failure
- (Reuse from stage05d_baseline.R)

**3. `extract_model_metrics(model)`**
- Return list: aic, deviance_explained, converged
- (Reuse from stage05d_baseline.R)

**4. `get_all_index_columns(data)`**
- Identify all acoustic index columns (exclude response, temporal, environmental)
- (Reuse from stage05d_baseline.R)

### Output Schema

**acoustic_only_comparison.csv**
```
metric,full_aic,acoustic_only_aic,delta_aic,full_dev_explained,acoustic_only_dev_explained,acoustic_only_converged
fish_presence,12345.6,12456.7,111.1,0.45,0.40,TRUE
```

### Interpretation Guidelines

ΔAIC interpretation (acoustic_only_AIC - full_AIC):
- ΔAIC > 10: Full model substantially better → env/temporal add significant value
- ΔAIC 4-10: Moderate difference
- ΔAIC < 4: Similar performance → indices capture most information
- ΔAIC < 0: Acoustic-only better (unexpected, flag for review)

### Synthesis with Baseline Comparison

Combined analysis can show:
- **Full model deviance explained** (reference)
- **Baseline deviance explained** (env/temporal contribution)
- **Acoustic-only deviance explained** (index contribution)

This reveals whether indices or env/temporal terms are more predictive for each response.

---

## Quickstart

```r
# Run acoustic-only comparison
Rscript scripts/stage05e_acoustic_only.R

# Run on single metric for testing
Rscript scripts/stage05e_acoustic_only.R --metric fish_presence

# Check outputs
cat results/tables/acoustic_only_comparison.csv

# Compare with baseline
head results/tables/baseline_comparison.csv
head results/tables/acoustic_only_comparison.csv
```

---

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Implement `stage05e_acoustic_only.R` (can largely reuse stage05d_baseline.R structure)
3. Run and verify outputs
4. Optionally add to results viewer

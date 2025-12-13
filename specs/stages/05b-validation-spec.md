# 05b Model Validation — Stage Spec

## Purpose
- Validate model assumptions (AR1 autocorrelation structure)
- Assess predictive performance via cross-validation
- Ensure models generalize beyond training data

This spec covers validation steps that follow model fitting (Stage 05a).

---

## Inputs
- Fitted models from Stage 05a: `results/models/<metric>/glmm.rds`, `results/models/<metric>/gamm.rds`
- Analysis-ready data: `data/processed/analysis_ready.parquet`
- Model summaries: `results/tables/<metric>/glmm_summary.csv`

---

## Outputs

### AR1 Validation
- `results/diagnostics/acf_by_response.png` — ACF plots per response metric
- `results/tables/ar1_rho_estimates.csv` — estimated rho values per response

### Cross-Validation
- `results/tables/cv_performance_summary.csv` — per-response CV metrics (RMSE/AUC)
- `results/figures/cv_performance_by_week.png` — performance variation across folds
- `results/logs/cv_run_log.json` — run metadata, fold details

---

## Methods

### AR1 Autocorrelation Validation

The GLMM uses AR1 correlation structure to handle temporal autocorrelation within days. To validate this structure:

1. **Compute empirical ACF**: Calculate autocorrelation function on model residuals at lags 1, 2, 3... up to ~12 (24 hours at 2-hour resolution)
2. **Plot ACF**: Visualize decay pattern — expect exponential decay if AR1 is appropriate
3. **Extract estimated rho**: glmmTMB estimates rho from data; extract and compare across responses

**Interpretation:**
- If lag-1 autocorrelation is near zero, AR1 may be unnecessary
- If ACF shows non-exponential decay (e.g., oscillation), AR1 may be misspecified
- Consistent rho estimates across responses (~0.4–0.7 typical for ecological time series) suggests appropriate structure

**Implementation:**
- Use `acf()` on DHARMa residuals or Pearson residuals
- Extract rho from fitted glmmTMB object via `glmmTMB::VarCorr()`
- Plot ACF with confidence bands

---

### Cross-Validation Strategy

**Primary approach: Week-based k-fold**

Hold out one week at a time, train on remaining weeks, cycle through all weeks. This respects temporal structure better than random k-fold.

**Procedure:**
1. Define folds by ISO week number
2. For each fold:
   - Exclude that week's observations from training data
   - Refit model on remaining data
   - Predict held-out week
   - Compute performance metrics
3. Aggregate metrics across folds

**Fold requirements:**
- Minimum observations per fold: 20 (skip sparse weeks)
- Record which weeks were skipped and why

**Performance metrics:**
| Response type | Primary metric | Secondary metrics |
|---------------|----------------|-------------------|
| Count (fish_activity, dolphin_*, etc.) | RMSE | MAE, R² |
| Binary (fish_presence, dolphin_presence, vessel_presence) | AUC | Accuracy, Sensitivity, Specificity |

**Rationale:** Week-at-a-time holdout tests whether the model generalizes to unseen time periods while maintaining enough training data for stable estimates. This is more realistic than random CV for time series data.

---

### Optional: Hour-of-Day Performance Stratification

*Status: Future work*

Stratify CV performance metrics by hour-of-day to assess whether prediction accuracy varies with diel period. This could reveal ecologically interesting patterns (e.g., model performs better during crepuscular periods when fish calling is most predictable).

**If implemented:**
- Compute per-hour RMSE/AUC from CV predictions
- Plot performance by hour with confidence intervals
- Output: `results/figures/cv_performance_by_hour.png`

---

## Parameters
- `cv.primary_strategy.fold_by`: `iso_week` (from `config/cv.yml`)
- `cv.primary_strategy.min_fold_rows`: 20
- `cv.metrics.counts`: `["rmse", "mae", "r2"]`
- `cv.metrics.binary`: `["auc", "accuracy", "sensitivity", "specificity"]`

---

## Acceptance Criteria
- ACF plots generated for all responses
- AR1 rho estimates extracted and documented
- CV completed for all 9 responses
- CV performance summary table produced
- No systematic issues flagged (e.g., all weeks failing, extreme performance variation)

---

## Edge Cases
- **Sparse weeks**: Skip folds with < 20 observations; document in log
- **Model non-convergence during CV**: Use simpler model specification for that fold; flag in output
- **Extreme performance variation**: If CV RMSE varies > 3× across folds, investigate temporal patterns

---

## Performance
- Target: < 30 minutes for full CV across all responses (parallel fitting recommended)

---

## Dependencies
- Upstream: Stage 05a (fitted models)
- Downstream: Results interpretation, manuscript

---

## Change Record
- 2025-12-12: Created spec. AR1 validation via ACF and data-driven rho estimation. Week-based k-fold CV per statistical consultation (Tiago). Hour-of-day stratification marked as future work.
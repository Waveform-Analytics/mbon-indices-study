# 05b Model Validation — Stage Spec

## Purpose
- Validate model assumptions (AR1 autocorrelation structure)
- Assess predictive performance via cross-validation
- Ensure models generalize beyond training data

This spec covers validation steps that follow model fitting (Stage 05a).

---

## Inputs
- Fitted GAMM models from Stage 05a: `results/models/<metric>/gamm.rds`
- Analysis-ready data: `data/processed/analysis_ready.parquet`
- Model summaries: `results/tables/<metric>/gamm_summary.csv`
- Model summary with rho values: `results/tables/model_summary.csv`

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

The GAMM uses an AR1 correlation structure via the `rho` parameter in `bam()`. In Stage 05a, rho is estimated using preliminary model residuals. This validation step confirms the AR1 structure is adequate.

**Validation procedure:**

1. **Review estimated rho values**: Check `results/tables/model_summary.csv` for per-response rho values
2. **Compute residual ACF**: Calculate autocorrelation function on final model residuals at lags 1, 2, 3... up to ~12 (24 hours at 2-hour resolution)
3. **Compare pre/post AR1**: If residual ACF at lag-1 is near zero, the AR1 correction is working

**Interpretation:**
- Lag-1 ACF of final model residuals should be near zero (AR1 absorbed the autocorrelation)
- If residual ACF still shows strong autocorrelation, rho may be underestimated
- Rho values typically range 0.3–0.8 for ecological time series; values outside this range warrant investigation

**Implementation:**
- Extract rho values from `model_summary.csv` (logged during Stage 05a)
- Use `acf()` on deviance residuals from fitted GAMM
- Plot ACF with confidence bands to visualize residual autocorrelation

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
- 2025-12-16: Updated for GAMM-only approach. Removed GLMM/glmmTMB references. AR1 validation now confirms rho estimates from Stage 05a are adequate.
- 2025-12-12: Created spec. AR1 validation via ACF and data-driven rho estimation. Week-based k-fold CV per statistical consultation (Tiago). Hour-of-day stratification marked as future work.
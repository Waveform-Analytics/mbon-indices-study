# 05a Modeling — Stage Spec

## Purpose
- Fit Generalized Additive Mixed Models (GAMMs) for each response metric to predict biological community metrics from acoustic indices.

### Analysis Framing: Prediction with Interpretability

**Decision (2025-12-12):** This analysis is **prediction-focused but with interpretability**.

- **Primary goal**: Demonstrate that acoustic indices can predict biological community metrics — a proof-of-concept for potential monitoring applications at other sites.
- **Secondary goal**: Maintain ecological interpretability so results can tie back to meaningful relationships (e.g., "ACI relates to fish activity").
- **Not the goal**: Rigorous causal inference about which specific indices drive community dynamics.

**Why GAMM (not GLMM)?**

Initial modeling compared GLMM and GAMM approaches. Results showed non-linear relationships between acoustic indices and community metrics — relationships that GLMM (which assumes linearity on the link scale) cannot capture. GAMM's smooth functions can represent these non-linear patterns while still providing interpretable output through smooth plots and effective degrees of freedom (EDF).

**Implications:**
- Index reduction uses VIF pruning (not PCA) to preserve interpretability
- Cross-validation assesses predictive generalization, not just in-sample fit
- Smooth terms are interpreted via plots and EDF values
- Multicollinearity among predictors is less concerning than for pure inference

## Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`

## Outputs
Per response metric:
- `results/models/<metric>/gamm.rds` — fitted GAMM object
- `results/tables/<metric>/gamm_summary.csv` — smooth terms with EDFs and p-values
- `results/tables/<metric>/scaling_params.csv` — mean/SD for back-transformation
- `results/figures/<metric>/gamm_diagnostics.png` — residual diagnostics
- `results/figures/<metric>/gamm_smooths.png` — overview grid of all smooth terms
- `results/figures/<metric>/smooth_<term>.png` — individual smooth plots per term

Summary outputs:
- `results/tables/model_summary.csv` — all responses, convergence status, key metrics
- `results/logs/modeling_summary.json` — run metadata

Results interpretation:
- `results/results_summary.html` — Quarto-generated reveal.js slides for reviewing results

---

## Methods

### The Research Question

We want to understand: **Do acoustic indices predict biological community metrics?**

For example: "Does the Acoustic Complexity Index (ACI) relate to fish calling activity, after accounting for environmental conditions and the structure of our sampling?"

### Predictor Scaling

**Why scale predictors?**

Acoustic indices and environmental covariates are on vastly different scales:
- `KURTt` has SD > 10,000
- `VARt` has SD ~ 0.0004
- `temperature` ranges ~10–33

This causes numerical instability — optimizers struggle when parameters span many orders of magnitude.

**Solution: Z-score standardization**

Before fitting, all continuous predictors (indices + temperature + depth) are standardized:

```
x_scaled = (x - mean(x)) / sd(x)
```

This transforms each predictor to mean=0, SD=1.

**Implementation:**
- Scaling is performed at model-fitting time (not stored in parquet)
- Scaling parameters (mean, SD) are saved to `results/tables/<metric>/scaling_params.csv`
- Response variables are NOT scaled (they use appropriate link functions)
- Cyclic terms (hour_of_day, day_of_year) are NOT scaled (handled by cyclic splines)

---

### GAMM (Generalized Additive Mixed Model)

**What it does:** Models potentially NON-LINEAR relationships between acoustic indices and community metrics. Instead of assuming "more ACI = proportionally more fish activity," it can capture patterns like "fish activity increases with ACI up to a point, then levels off."

GAMMs also properly handle:
- Non-normal response distributions (counts, presence/absence)
- Repeated measurements at the same stations
- Temporal patterns (diel and seasonal)

**The Formula Explained:**

```
response ~ s(index1, k=5) + s(index2, k=5) + ... +
           s(temperature, k=5) + s(depth, k=5) +
           s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) +
           s(station, bs="re") + s(month_id, bs="re")
```

| Component | What it means |
|-----------|---------------|
| `s(ACI, k=5)` | Smooth (potentially non-linear) function of ACI with up to ~4 degrees of wiggliness. If the relationship is actually linear, the smooth will estimate a straight line. |
| `bs="cc"` | Cyclic cubic spline — the curve wraps around (hour 23 connects smoothly to hour 0) |
| `k=12` | For cyclic terms, allows more flexibility to capture complex diel/seasonal patterns |
| `bs="re"` | Random effect smooth — equivalent to random intercept, accounts for station/month baseline differences |

**Why these Distribution Families:**

| Response type | Family | Why |
|---------------|--------|-----|
| Count data (fish_activity, dolphin_echolocation, etc.) | `nb` (negative binomial) | Counts are non-negative integers. Ecological count data is almost always **overdispersed** (variance > mean). Negative binomial handles this. |
| Binary data (fish_presence, dolphin_presence, vessel_presence) | `binomial` | These are 0/1 outcomes. Binomial with logit link models the probability of presence. |

**Understanding smooth terms:**

Unlike GLMM coefficients, GAMM smooth terms don't have a single "effect size." Instead:
- **EDF (effective degrees of freedom)**: Higher = more non-linear. EDF ≈ 1 means essentially linear.
- **Smooth plots**: Show the estimated shape of the relationship
- **p-value**: Tests whether the smooth term significantly improves the model

**Interpreting results:**
- Look at smooth plots to understand relationship shapes
- EDF indicates complexity: EDF=1 is linear, EDF=3-4 is moderately wiggly
- Confidence bands on smooth plots show uncertainty

**Key Assumptions:**
1. Smooth functions can adequately represent the true relationships
2. The chosen basis dimension (k) is sufficient but not excessive
3. `select=TRUE` shrinks unnecessary wiggles toward zero (built-in regularization)
4. Observations are independent after accounting for random effects and temporal correlation

---

### Temporal Autocorrelation (AR1)

**The problem:** Observations close in time are not independent. Fish activity at 10:00 is correlated with fish activity at 12:00 — environmental conditions, animal behavior, and acoustic conditions persist over hours. Ignoring this violates the independence assumption and can inflate significance of predictors.

**Solution:** Include an AR1 (first-order autoregressive) correlation structure via the `rho` parameter in `bam()`. This models correlation between consecutive observations within the same temporal sequence.

**Data-driven rho estimation:**

Rather than using an arbitrary fixed value, we estimate `rho` from the data:

1. **Fit preliminary model**: Fit the GAMM without AR1 correlation (rho = 0)
2. **Extract residuals**: Get the deviance residuals from this model
3. **Compute lag-1 ACF**: Calculate the autocorrelation at lag 1 — this is the empirical rho
4. **Refit with estimated rho**: Fit the final model using this data-driven rho value

```r
# Pseudocode for data-driven rho estimation
preliminary_fit <- bam(formula, data, family, method = "fREML", rho = 0)
residuals <- residuals(preliminary_fit, type = "deviance")
rho_estimated <- acf(residuals, lag.max = 1, plot = FALSE)$acf[2]
final_fit <- bam(formula, data, family, method = "fREML", rho = rho_estimated)
```

**Implementation notes:**
- Rho is estimated separately for each response metric (different responses may have different temporal dynamics)
- The estimated rho value is logged in the model summary for transparency
- Typical ecological time series have rho in the range 0.3–0.8
- If estimated rho is negative or near zero, AR1 may not be needed for that response

---

### Diagnostics

**GAMM Diagnostics (via gam.check and gratia):** 
- **Residual QQ plot**: Points should fall on diagonal line
- **Residuals vs fitted**: Should show no pattern
- **Response vs fitted**: Should show reasonable prediction
- **Basis dimension check**: k-index should not be too low (would suggest k is insufficient)

**Smooth-specific diagnostics:**
- **EDF relative to k**: If EDF ≈ k-1, may need to increase k
- **Smooth plots**: Check for unrealistic wiggliness or edge effects
- **Concurvity**: Check for confounding between smooth terms

---

## Workflow

### Iterative Review (Quarto Slides)  

To support iterative review of results, we produce a reveal.js slide deck via Quarto.

**What the slides show:**
- **Overview**: Which responses have been modeled, convergence status
- **Per-response slides**:
  - Significant smooth terms with EDF values
  - Smooth plots for key predictors
  - Diagnostic plots
  - Model fit statistics

**Steps:**
1. Run modeling script → generates CSVs and PNGs
2. Run `quarto render results/results_summary.qmd` → regenerates slides
3. Review slides in browser → identify issues or interesting findings
4. Iterate on model specification if needed

---

## Parameters
- `responses.<metric>.family` — distribution family per response
- `random_effects` — grouping variables for random effects (station, month_id)
- `gamm.smooth_k` — basis dimension for index/covariate smooths (default 5)
- `gamm.cyclic_k` — basis dimension for cyclic smooths (default 12)
- `gamm.method` — fitting method (default: "fREML")
- `gamm.select` — whether to use built-in selection (default: TRUE)
- `gamm.rho` — AR1 correlation parameter (default: data-driven, estimated from preliminary model residuals)
- `scaling.enabled` — whether to z-score standardize predictors (default: true)
- `scaling.exclude` — predictors to exclude from scaling (default: cyclic terms)

## Acceptance Criteria
- All models converge without errors
- Diagnostics reviewed; major issues noted in output
- Smooth plots generated for all predictor terms
- EDF values reported and interpreted

## Implementation Notes
- **Pilot first**: Start with `fish_activity` to test full pipeline
- **Expand**: Once pilot works, run remaining 8 responses
- **Language**: R (mgcv, gratia packages); Quarto for slides
- **Comments**: Code will be heavily commented for learning purposes
- **Output management**: Clean slate approach — delete per-metric output directories at start of each run

## Edge Cases
- **Non-convergence**: Reduce smooth complexity (lower k) or simplify random effects; document changes
- **High concurvity**: May indicate confounded predictors; consider removing one
- **EDF at boundary (≈ k-1)**: Increase k for that term

## Performance
- Target: < 5 minutes per response; < 45 minutes full run

## Dependencies
- Upstream: Stage 03 (analysis_ready.parquet), Stage 01 (indices_final.csv)
- Downstream: Stage 05b (validation), results interpretation, manuscript

**Note:** Model validation (cross-validation) is specified separately in `05b-validation-spec.md`.

## Change Record
- 2025-12-16: Added **data-driven rho estimation** for AR1 temporal autocorrelation. Rather than using an arbitrary fixed value, rho is estimated from lag-1 ACF of preliminary model residuals, separately for each response metric.
- 2025-12-16: **Switched to GAMM-only approach** — removed GLMM fitting and AIC comparison. Initial modeling showed non-linear relationships between indices and responses that GLMM cannot capture. GAMM's smooth functions provide better fit while maintaining interpretability through smooth plots and EDF values. Simplified spec, outputs, and workflow accordingly.
- 2025-12-12: Resolved inference vs prediction framing: prediction-primary with interpretability. Split validation (AR1, CV) to separate spec 05b-validation-spec.md per statistical consultation.
- 2025-12-09: Added predictor scaling requirement. Acoustic indices and covariates must be z-score standardized before fitting to ensure numerical stability.
- 2025-12-08: Updated indices reference to be generic ("final acoustic indices from Stage 01") rather than hardcoded count.
- 2025-12-06: Reorganized for clarity. Added Quarto slides and output management.
- 2025-12-05: Created merged spec from stages 05-06. Added detailed explanations of formulas, families, and AIC. Simplified to single stage with AIC comparison.
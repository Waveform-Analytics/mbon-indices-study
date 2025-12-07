# 05 Modeling — Stage Spec

Title
- GLMM and GAMM modeling with AIC-based model selection

Implementation Note
- Since this is a learning experience, the R code will include more explanatory comments than typical production code. Comments will explain what each step does and why.

Purpose
- Fit both GLMM and GAMM for each response metric, compare via AIC, and select the better-fitting model. Goal is inference (understanding relationships between acoustic indices and community metrics), not prediction. (this is open for discussion)

Inputs
- `data/processed/analysis_ready.parquet`
- `data/processed/indices_final.csv`

Outputs
Per response metric:
- `results/models/<metric>/glmm.rds` — fitted GLMM object
- `results/models/<metric>/gamm.rds` — fitted GAMM object
- `results/tables/<metric>/glmm_summary.csv` — fixed effects with CIs
- `results/tables/<metric>/gamm_summary.csv` — smooth terms with EDFs
- `results/tables/<metric>/model_comparison.csv` — AIC comparison, selected model
- `results/figures/<metric>/glmm_diagnostics.png` — residual diagnostics
- `results/figures/<metric>/gamm_smooths.png` — smooth term plots

Summary outputs:
- `results/tables/model_selection_summary.csv` — all responses, which model won, ΔAIC
- `results/logs/modeling_summary.json` — run metadata

Results interpretation:
- `results/results_summary.html` — Quarto-generated reveal.js slides for reviewing results

---

## Methods

### The Research Question

We want to understand: **Do acoustic indices predict biological community metrics?**

For example: "Does the Acoustic Complexity Index (ACI) relate to fish calling activity, after accounting for environmental conditions and the structure of our sampling?"

### GLMM (Generalized Linear Mixed Model)

**What it does:** Tests whether acoustic indices have LINEAR relationships with community metrics, while properly handling:
- Non-normal response distributions (counts, presence/absence)
- Repeated measurements at the same stations
- Temporal autocorrelation within days

**The Formula Explained:**

```
response ~ indices + covariates + sin_hour + cos_hour + (1|station) + (1|month_id) + ar1(time_within_day + 0 | day_id)
```

Breaking this down:

| Component | What it means | Why we include it |
|-----------|---------------|-------------------|
| `response` | The thing we're trying to explain (e.g., fish_activity) | Our dependent variable |
| `indices` | The 20 acoustic indices (ACI, BI, etc.) | These are our predictors of interest - we want to know if they relate to community metrics |
| `covariates` | Temperature and depth | Environmental controls - we want to isolate the effect of acoustic indices from confounding environmental variation |
| `sin_hour + cos_hour` | Cyclic encoding of time of day | Accounts for diel (day/night) patterns in a smooth, continuous way |
| `(1\|station)` | Random intercept for station | Different stations may have systematically different baseline levels - this accounts for that without "using up" degrees of freedom |
| `(1\|month_id)` | Random intercept for month | Seasonal baseline differences |
| `ar1(time_within_day + 0 \| day_id)` | First-order autoregressive correlation within each day | Observations close together in time are more similar than distant ones - this handles that non-independence |

**Why AR1 doesn't need continuity across midnight:**

The AR1 structure treats each day independently - correlation "resets" at midnight. This is intentional and appropriate because:

1. **AR1 handles short-term environmental continuity**: Adjacent 2-hour bins within the same day are correlated because conditions persist (the fish calling at 14:00 are likely still there at 16:00).

2. **Cyclic terms handle the wrap-around**: `sin_hour + cos_hour` encode time of day as a smooth circle where 23:00 is mathematically "close to" 01:00. This captures the diel pattern without AR1 needing to connect across midnight.

3. **Random effects handle longer patterns**: `(1|month_id)` captures seasonal baselines that persist across days.

The key insight: AR1 models "nearby observations are similar due to unmeasured short-term environmental continuity," not "the diel cycle wraps around." The cyclic terms handle the biological expectation that midnight should look like midnight.

**Why these Distribution Families:**

| Response type | Family | Why |
|---------------|--------|-----|
| Count data (fish_activity, dolphin_echolocation, etc.) | `nbinom2` (negative binomial) | Counts are non-negative integers. Poisson assumes mean=variance, but ecological count data is almost always **overdispersed** (variance > mean). Negative binomial adds a parameter to handle this overdispersion. |
| Binary data (fish_presence, dolphin_presence, vessel_presence) | `binomial` | These are 0/1 outcomes. Binomial with logit link models the log-odds of presence as a linear function of predictors. |

**Key Assumptions:**
1. Relationships between indices and response are approximately LINEAR (on the link scale)
2. Random effects are normally distributed
3. AR1 structure adequately captures temporal autocorrelation
4. Observations are independent AFTER accounting for random effects and AR1

**What is the "link scale"?**

GLMs don't model the response directly - they model a *transformed* version of the expected response. This transformation is the "link function":

| Family | Link | What we model | Interpretation |
|--------|------|---------------|----------------|
| Negative binomial | log | log(expected count) | A coefficient of 0.1 means a 1-unit increase in the predictor *multiplies* expected count by exp(0.1) ≈ 1.11, i.e., ~11% increase |
| Binomial | logit | log(p / (1-p)) | A coefficient of 0.5 means a 1-unit increase in the predictor adds 0.5 to the log-odds of presence |

So the "linear on link scale" assumption means: we assume `log(expected fish calls) = β₀ + β₁×ACI + ...` — a straight line relationship on the log scale. On the original count scale, this corresponds to multiplicative (exponential) effects, not additive ones.

---

### GAMM (Generalized Additive Mixed Model)

**What it does:** Like GLMM, but allows NON-LINEAR relationships. Instead of assuming "more ACI = proportionally more fish activity," it can capture patterns like "fish activity increases with ACI up to a point, then levels off."

**The Formula Explained:**

```
response ~ s(ACI, k=5) + s(BI, k=5) + ... + s(temperature, k=5) + s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) + s(station, bs="re") + s(month_id, bs="re")
```

| Component | What it means |
|-----------|---------------|
| `s(ACI, k=5)` | Smooth (potentially non-linear) function of ACI with up to ~4 degrees of wiggliness. If the relationship is actually linear, the smooth will estimate a straight line. |
| `bs="cc"` | Cyclic cubic spline - the curve wraps around (hour 23 connects smoothly to hour 0) |
| `k=12` | For cyclic terms, allows more flexibility to capture complex diel/seasonal patterns |
| `bs="re"` | Random effect smooth - equivalent to random intercept in GLMM |

**Why use GAMM in addition to GLMM?**
- If relationships are truly linear, GLMM is simpler and preferred
- If relationships are non-linear, GAMM will fit better and reveal the shape
- We fit both and let AIC tell us which is more appropriate

**Key Assumptions:**
1. Smooth functions can adequately represent the true relationships
2. The chosen basis dimension (k) is sufficient but not excessive
3. `select=TRUE` helps by shrinking unnecessary wiggles toward zero

---

### Model Comparison via AIC

**What is AIC?**
Akaike Information Criterion balances model fit against complexity. Lower AIC = better balance of fit and parsimony.

**Is AIC appropriate here?**
Yes, with caveats:
- AIC is valid for comparing models fit to THE SAME DATA with THE SAME RESPONSE
- We compare GLMM vs GAMM for each response separately (not across responses)
- Both models use the same likelihood family, so comparison is valid

**Interpretation:**
| ΔAIC | Interpretation |
|------|----------------|
| < 2 | Models essentially equivalent |
| 2-4 | Weak preference for lower-AIC model |
| 4-10 | Moderate preference |
| > 10 | Strong preference |

**When models are equivalent (ΔAIC < 4):** We prefer GLMM because:
1. Simpler to interpret (linear coefficients)
2. More familiar to reviewers/readers
3. Parsimony principle

---

### Diagnostics

**GLMM Diagnostics (via DHARMa):**
- Simulated residuals: Should look uniform if model is correct
- QQ plot: Points should fall on diagonal line
- Residuals vs predicted: Should show no pattern
- Dispersion test: Checks if variance assumption is met

**GAMM Diagnostics:**
- EDF (effective degrees of freedom): Higher = more non-linear. EDF ≈ 1 means relationship is essentially linear.
- Smooth plots: Visualize the estimated non-linear relationships
- gam.check: Residual diagnostics similar to GLMM

---

### Results Interpretation (Quarto Slides)

To support iterative review of results as they're generated, we produce a reveal.js slide deck via Quarto.

**Why Quarto?**
- Native reveal.js output with clean markdown source
- Can read CSV outputs (language-agnostic interface to model results)
- One command to regenerate: `quarto render results/results_summary.qmd`
- Slides update as new responses are modeled

**What the slides show:**
- **Overview**: Which responses have been modeled, which model won for each
- **Per-response slides**:
  - Model comparison (ΔAIC, selected model)
  - Significant predictors with effect sizes
  - Diagnostic plots (embedded PNGs)
  - Interpretation templates (fill-in-the-blank)

**Interpretation templates** help translate statistical output to plain English:
- Count models: "A 1-unit increase in [index] is associated with a [exp(β)]× change in expected [response] (95% CI: [lower]–[upper])."
- Binary models: "A 1-unit increase in [index] multiplies the odds of [response] by [exp(β)] (95% CI: [lower]–[upper])."

**Workflow:**
1. Run modeling script → generates CSVs and PNGs
2. Run `quarto render` → regenerates slides from current outputs
3. Review slides in browser → identify issues or interesting findings
4. Iterate on model specification if needed

---

## Parameters (from config)
- `responses.<metric>.family` — distribution family per response
- `random_effects` — grouping variables for random effects
- `gamm.smooth_k` — basis dimension for index/covariate smooths (default 5)
- `gamm.cyclic_k` — basis dimension for cyclic smooths (default 12)
- `autocorrelation.glmm_ar1` — whether to include AR1 in GLMM

## Acceptance Criteria
- All models converge without errors
- Diagnostics reviewed; major issues noted in output
- AIC comparison completed for all responses
- Selected model justified per ΔAIC guidelines above

## Implementation Notes
- **Pilot first**: Start with `fish_activity` to test full pipeline
- **Expand**: Once pilot works, run remaining 8 responses
- **Language**: R (glmmTMB, mgcv, DHARMa packages); Quarto for slides
- **Comments**: Code will be heavily commented for learning purposes
- **Output management**: Clean slate approach — delete `results/models/`, `results/tables/*/` (per-metric), and `results/figures/*/` (per-metric) at start of each run. This keeps outputs tidy and avoids confusion from stale results. Summary files are regenerated fresh each run.

## Edge Cases
- **Non-convergence**: Simplify random effects or reduce smooth complexity; document changes
- **Singular fits**: Drop problematic terms; record justification
- **Equivalent models (ΔAIC < 4)**: Prefer simpler (GLMM) per parsimony

## Performance
- Target: < 5 minutes per response (pilot); < 45 minutes full run

## Dependencies
- Upstream: Stage 03 (analysis_ready.parquet), Stage 01 (indices_final.csv)
- Downstream: Results interpretation, manuscript

## Change Record
- 2025-12-06: Added Quarto-based results slides for iterative review during modeling. Added output management (clean slate). Added explanations for AR1 midnight discontinuity and link scale concept.
- 2025-12-05: Created merged spec from stages 05-06. Added detailed explanations of formulas, families, and AIC. Simplified to single stage with AIC comparison. Deferred cross-validation.
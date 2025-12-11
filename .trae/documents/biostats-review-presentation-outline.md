# Biostats Expert Review — Presentation Outline

**Purpose:** Curated document for biostats advisor review (Dec 12, 2025)
**Format:** Reveal.js slides → PDF
**Repository:** [link to repo for reviewer to access specs/code]

---

## Proposed Structure

### Section 1: Context (1-2 slides)

**Slide 1: Research Question**
- Can acoustic indices predict biological community metrics in estuarine environments?
- 3 stations, May River SC, 2021 data
- 9 response variables (fish activity/richness/presence, dolphin metrics, vessel presence)
- ~60 acoustic indices as candidate predictors

**Slide 2: Data Overview**
- 13,102 observations (2-hour bins)
- 4 data sources aligned to common temporal resolution:
  - Detections (manual annotations of fish/dolphin/vessel)
  - Environment (temperature, depth from sensors)
  - Acoustic indices (~60 indices across 5 categories: Amplitude, Complexity, Diversity, Spectral, Temporal)
  - SPL (sound pressure levels)
- Temporal structure: station, month, day, hour

---

### Section 2: Methods Summary (2-3 slides)

**Slide 3: Pipeline Overview**
- Stage 00: Data alignment (4 sources → 2-hour bins)
- Stage 01: Index reduction (60 → 14 via correlation/VIF)
- Stage 02-03: Response variables + feature engineering
- Stage 05: GLMM vs GAMM modeling

*Why both models?* GLMM assumes linear predictor-response relationships; GAMM allows non-linear (smooth) relationships. We fit both and compare — if relationships are truly linear, GLMM preferred (simpler, more interpretable); if non-linear, GAMM will fit better.

**Slide 4: Model Specifications**
- GLMM: linear fixed effects + AR1 autocorrelation + random intercepts (station, month)
- GAMM: smooth terms (k=5) + cyclic splines for hour/day + random effects + AR1 (rho)
- Both: negative binomial for counts, binomial for presence/absence
- Comparison: AIC (prefer simpler model if ΔAIC < 4)

---

### Section 3: Question 1 — Index Reduction (2-3 slides)

**Slide 5: What We Did**
- Correlation pruning: removed one of each pair with |r| > 0.6
- VIF screening: iteratively removed indices with VIF > 2
- Result: 60 → 17 (correlation) → 14 (VIF)
- Categories preserved: Amplitude, Complexity, Diversity, Spectral, Temporal

**Slide 6: Concern**
- Is 14 indices too many?
- Is correlation + VIF the right approach, or should we use something else (e.g., PCA, LASSO)?
- Model shrinkage (GAMM select=TRUE) effectively removed 4 more — should we formalize this?

**Slide 7: Specific Questions for Advisor**
- Q1: Is correlation + VIF standard practice, or would you recommend a different approach?
- Q2: Given that GAMM shrinkage removed 4 indices, should we adopt a two-stage approach (VIF → model-based selection)?
- Q3: 14 predictors for 13K observations — is this ratio acceptable?

MW:

---

### Section 4: Question 2 — Modeling Results (4-5 slides)

**Slide 8: Model Comparison Summary**
- fish_activity pilot results:
  - GLMM AIC: 34,903
  - GAMM AIC: 28,833
  - ΔAIC: 6,071 (strong preference for GAMM)
- GLMM diagnostics show systematic misfit (DHARMa KS test fails)
- GAMM selected for interpretation

**Slide 9: GAMM Results — Significant Predictors**
- Show table of significant terms (10 of 14 indices significant)
- Key findings:
  - hour_of_day highly significant (EDF=8.2) — strong diel pattern
  - BI significant but NEGATIVE relationship (higher BI → less fish activity)
  - ACI, EAS, VARt among strongest positive predictors
  - ADI, BioEnergy, EPS_KURT, MEANt shrunk away

**Slide 10: GAMM Results — Smooth Plots (Overview)**
- Show full gamm_smooths.png as overview
- Note: "Let's zoom in on a few key relationships..."

**Slide 11: Smooth Plot Zoom — hour_of_day**
- Show hour_of_day panel with annotation
- Commentary: Strong diel pattern (EDF=8.2), peaks ~8 PM, lowest ~10 AM
- Matches known fish calling behavior — validates model is capturing real biology

**Slide 12: Smooth Plot Zoom — BI (negative relationship)**
- Show BI panel with annotation
- Commentary: Higher BI → less fish activity (counterintuitive?)
- Possible explanation: BI elevated when other sound sources dominate (snapping shrimp, etc.)
- Question for advisor: Is this ecologically interpretable or a modeling artifact?

**Slide 13: Smooth Plot Zoom — VARt (non-linear)**
- Show VARt panel with annotation
- Commentary: "Goldilocks" relationship — fish activity peaks at intermediate variance
- Demonstrates why GAMM outperforms GLMM (can't capture this with a line)

*(Note: Model saved as .rds — can regenerate individual panels if needed)*

**Slide 14: Unexpected Results**
- Temperature NOT significant in GAMM (p=0.12) but highly significant in GLMM
- day_of_year NOT significant in GAMM (p=0.18) despite visible seasonality in data
- Hypothesis: indices absorb seasonal/temperature signal?
- Show fish_activity heatmap alongside to illustrate the visible seasonality

**Slide 15: Methodological Concerns**
- GLMM uses AR1 autocorrelation; GAMM uses rho parameter — are these comparable?
- AIC comparison: GLMM uses ML, GAMM uses fREML — is this technically valid?
- 10 of 14 indices significant — overfitting concern or genuine signal?
- Depth coefficient flips sign between GLMM (+) and GAMM (-) — what does this mean?

**Slide 16: GLMM Diagnostics**
- Show glmm_diagnostics.png
- Commentary: KS test fails (p=0), outliers detected — systematic misfit
- This supports GAMM selection: linear terms can't capture the true relationships

---

### Section 5: Open Questions (1 slide)

**Slide 17: Summary of Questions for Discussion**

1. **Index reduction:** Is correlation + VIF appropriate? Should we reduce further, or is model shrinkage sufficient?

2. **Model choice:** GAMM strongly preferred by AIC — any concerns? Is fREML vs ML comparison valid?

3. **Autocorrelation:** Is our AR1 handling adequate in both models?

4. **Interpretation:** Temperature/seasonality not significant in GAMM — absorbed by indices, or a problem?

5. **Next steps:** Expand to all 9 responses? Additional diagnostics? Anything we're missing?

---

## Supporting Materials to Include

- [ ] Model comparison table (model_comparison.csv)
- [ ] GAMM summary table (gamm_summary.csv) — formatted nicely
- [ ] GAMM smooth plots (gamm_smooths.png + individual zooms)
- [ ] GLMM diagnostics (glmm_diagnostics.png)
- [ ] Heatmaps: fish_activity (to show the pattern we're predicting)
- [ ] Index reduction summary (from Stage 01 — correlation matrix? VIF table?)
- [ ] Link to repository for specs/code

---

## Notes / Additional Context to Mention

*(These can go in speaker notes or as brief asides during presentation)*

- This is pilot mode (fish_activity only) — full run will cover all 9 responses
- MEANt has no real variation in raw data (numerical noise) — model handled it via shrinkage
- Depth coefficient flips sign between GLMM (+) and GAMM (-) — suggests GLMM mis-specified
- We want commentary that walks through logic and decisions conversationally (not just "here's a table")

---

## Estimated Slide Count

| Section | Slides |
|---------|--------|
| Context | 2 |
| Methods | 2 |
| Index Reduction | 3 |
| Modeling Results | 9 |
| Open Questions | 1 |
| **Total** | **17** |

This is on the longer side. Could trim by combining some zoom slides or cutting GLMM diagnostics if needed. 


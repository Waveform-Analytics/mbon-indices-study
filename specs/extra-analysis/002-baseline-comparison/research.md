# Research: Baseline Model Comparison & VIF Validation

**Created**: 2026-02-02
**Status**: Complete (straightforward extension of existing pipeline)

## Summary

This feature extends existing GAMM modeling code with minimal research needed - all techniques are standard R/mgcv patterns already used in the project.

## Technical Decisions

### 1. Baseline Formula Structure

**Decision**: Use same environmental + temporal + random effects as full model, just remove index terms.

```r
response ~ s(temperature, k=5) + s(depth, k=5) +
           s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) +
           s(station, bs='re') + s(month_id, bs='re')
```

**Rationale**: This is the "null" hypothesis - what can we explain without acoustic indices?

### 2. AR1 Correlation Handling

**Decision**: Use rho from existing full model when fitting baseline.

**Rationale**:
- Spec FR-003 requires this for fair comparison
- Re-estimating rho would change the autocorrelation structure
- Different rho values would confound the AIC comparison

**Implementation**: Extract from full model via `model$AR1.rho` or model object inspection.

### 3. 60-Index Data Source

**Decision**: Join from `data/interim/aligned_indices.parquet`

**Verified**: File contains 62 acoustic indices (vs 17 in analysis_ready.parquet)

**Columns available**: ZCR, MEANt, VARt, SKEWt, KURTt, LEQt, BGNt, SNRt, MED, Ht, ACTtFraction, ACTtCount, ACTtMean, EVNtFraction, EVNtMean, EVNtCount, MEANf, VARf, SKEWf, KURTf, NBPEAKS, LEQf, ENRf, BGNf, SNRf, Hf, EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW, ACI, NDSI, rBA, AnthroEnergy, BioEnergy, BI, ROU, ADI, AEI, LFC, MFC, HFC, ACTspFract, ACTspCount, ACTspMean, EVNspFract, EVNspMean, EVNspCount, TFSD, H_Havrda, H_Renyi, H_pairedShannon, H_gamma, H_GiniSimpson, RAOQ, AGI, nROI, aROI

### 4. 60-Index Model Convergence

**Decision**: Use k=3 instead of k=5 for 60-index smooth terms

**Rationale**:
- 60 smooth terms × k=5 = 300 basis functions (very high)
- k=3 reduces to 180 basis functions
- Still captures non-linearity while improving convergence odds

**Fallback**: If k=3 fails, record as "failed to converge" and continue

### 5. AIC Comparison Direction

**Decision**: ΔAIC = baseline_AIC - full_AIC

**Interpretation**:
- Positive ΔAIC: Full model (with indices) is better
- Negative ΔAIC: Baseline model is better (indices don't help)
- Threshold: ΔAIC > 10 = strong evidence (Burnham & Anderson, 2002)

## No Research Needed

The following use standard patterns already in the codebase:
- `mgcv::bam()` for model fitting
- `AIC()` for AIC extraction
- `summary(model)$dev.expl` for deviance explained
- `arrow::read_parquet()` for data loading
- Error handling and logging patterns from stage05c

## References

- Burnham, K. P., & Anderson, D. R. (2002). Model Selection and Multimodel Inference. Springer.
- Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.). CRC Press.

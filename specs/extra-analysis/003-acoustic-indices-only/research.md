# Research: Acoustic Indices Only Model Comparison

**Feature**: 003-acoustic-indices-only
**Date**: 2026-02-05

## Overview

This analysis is the inverse of the baseline comparison (002-baseline-comparison). While baseline tested whether env/temporal variables alone can predict responses, this tests whether acoustic indices alone can predict responses.

## Key Decisions

### 1. Formula Structure

**Decision**: `response ~ s(index1, k=3) + ... + s(index61, k=3) + s(station, bs='re') + s(month_id, bs='re')`

**Rationale**:
- Excludes: temperature, depth, hour_of_day, day_of_year (these are the "env/temporal" terms)
- Keeps: Random effects (station, month_id) because these capture grouping structure, not confounders
- Uses k=3 for all index smooth terms (matches current full model configuration)

**Alternatives Considered**:
- Exclude random effects entirely → Rejected: would change the model class entirely, not comparable
- Use k=5 for index terms → Rejected: k=3 is current standard for 60+ index models

### 2. AR1 Correlation

**Decision**: Use rho from the full model

**Rationale**: Same approach as baseline comparison ensures fair comparison. The autocorrelation structure should be held constant across comparisons.

### 3. ΔAIC Direction

**Decision**: Calculate as `acoustic_only_AIC - full_AIC`

**Rationale**: Positive values mean the full model (with env/temporal) fits better. This is the expected direction - env/temporal variables should improve fit.

### 4. Index Count

**Confirmed**: 61 acoustic indices available in `data/interim/aligned_indices.parquet`

Columns identified as indices (excluding: station, datetime, datetime_local, date, hour, Filename):
- ZCR, MEANt, VARt, SKEWt, KURTt, LEQt, BGNt, SNRt, MED, Ht, ...
- Total: 61 indices

### 5. Script Naming

**Decision**: `stage05e_acoustic_only.R`

**Rationale**: Follows convention (stage05d is baseline, stage05e continues the pattern).

## Relationship to Baseline Comparison

| Model | Indices | Env/Temporal | Random Effects |
|-------|---------|--------------|----------------|
| Full | ✓ 61 | ✓ temp, depth, hour, day | ✓ station, month |
| Baseline | ✗ | ✓ temp, depth, hour, day | ✓ station, month |
| Acoustic-only | ✓ 61 | ✗ | ✓ station, month |

## Expected Results

Possible outcomes for each response variable:

1. **Full >> Acoustic-only >> Baseline**: Indices are the main driver, env/temporal adds some value
2. **Full >> Baseline >> Acoustic-only**: Env/temporal are the main driver, indices add some value
3. **Full ≈ Acoustic-only >> Baseline**: Indices capture almost everything, env/temporal redundant
4. **Full ≈ Baseline >> Acoustic-only**: Env/temporal capture almost everything, indices redundant
5. **Acoustic-only > Full**: Unexpected - would suggest overfitting in full model or indices capture confounded relationships

## No Open Questions

All technical decisions resolved based on precedent from baseline comparison and current modeling approach.

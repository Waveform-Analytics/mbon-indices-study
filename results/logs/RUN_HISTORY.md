# Analysis Run History

This file tracks run-specific outcomes for each pipeline stage. Entries are auto-appended by stage scripts.

For methodology and rationale, see the stage specs in `specs/stages/`.

---

## 2025-12-08 11:10 — Stage 01: Index Reduction

- **Config**: correlation_r=0.6, vif=2, vif_fallback=5
- **Result**: 60 → 17 (correlation) → 14 (VIF) indices
- **Final indices**: ACI, ACTtCount, ADI, BI, BioEnergy, EAS, EPS_KURT, EVNspMean, KURTt, MEANt, NBPEAKS, SKEWt, TFSD, VARt
- **Categories**: All 5 preserved (Amplitude, Complexity, Diversity, Spectral, Temporal)
- **Max VIF**: 1.72 (KURTt)
- **Notes**: Tightened from r=0.7/VIF=5 per Zuur et al. 2010 ecological best practices

---

## 2025-12-08 12:52 — Stage 05: Modeling

- **Config**: pilot_mode=TRUE, n_responses=1
- **Indices**: 14 predictors from Stage 01
- **Results**:
  - fish_activity: GAMM (ΔAIC=NA)
- **Notes**:

---


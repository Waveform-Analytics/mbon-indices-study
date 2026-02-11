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
- **Notes**: The GLMM convergence issue persists even with 14 indices. This likely requires investigating the AR1 structure or simplifying random effects - a separate modeling concern from index reduction (though index reduction is still a concern as 14 indices is still a lot).

---

## 2025-12-09 13:42 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GAMM (ΔAIC=NA)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2025-12-09 15:31 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GAMM (ΔAIC=485.2)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2025-12-09 17:43 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GLMM ok (AIC=34903.5, 3.0min) | GAMM ok (AIC=34418.3, 0.0min) | Selected: GAMM (dAIC=485.2)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2025-12-10 10:44 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GLMM ok (AIC=34903.5, 47.1min) | GAMM ok (AIC=34418.3, 0.0min) | Selected: GAMM (dAIC=485.2)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2025-12-10 11:07 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GLMM ok (AIC=34903.5, 3.0min) | GAMM ok (AIC=28832.8, 0.0min) | Selected: GAMM (dAIC=6070.7)
- **Log**: results/logs/modeling_summary.json
- **Notes**: 
  - MEANt shows no meaningful variation (values basically around numerical noise). This is present all the way back to the raw data. Model correctly shrunk this term away. Consider excluding it from stage 01 index list in future runs. 

---

## 2025-12-11 10:05 — Stage 05: Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 14
- **Results**:
  - fish_activity: GLMM ok (AIC=34903.5, 3.1min) | GAMM ok (AIC=28832.8, 0.0min) | Selected: GAMM (dAIC=6070.7)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2025-12-15 14:19 — Stage 01: Index Reduction

- **Config**:
  - correlation_r: 0.6
  - vif: 2
  - vif_fallback: 5
- **Results**:
  - n_start: 60
  - n_after_corr: 17
  - n_final: 14
  - final_indices: ACI, ACTtCount, ADI, BI, BioEnergy, EAS, EPS_KURT, EVNspMean, KURTt, MEANt, NBPEAKS, SKEWt, TFSD, VARt
  - categories: 5 (Amplitude Indices, Complexity Indices, Diversity Indices, Spectral Indices, Temporal Indices)
  - max_vif: 1.72 (KURTt)
- **Log**: results/logs/stage01_index_reduction_20251215_141459.txt
- **Notes**: 

---

## 2025-12-16 13:44 — Stage 01: Index Reduction

- **Config**:
  - method: VIF-only
  - vif: 2
  - vif_fallback: 5
- **Results**:
  - n_start: 60
  - n_final: 17
  - n_removed: 43
  - final_indices: ACI, BI, BioEnergy, ECV, EPS_KURT, EVNtCount, EVNtMean, H_Havrda, KURTt, MEANt, NBPEAKS, ROU, SKEWt, TFSD, VARt, ZCR, nROI
  - categories: 5 (Amplitude Indices, Complexity Indices, Diversity Indices, Spectral Indices, Temporal Indices)
  - max_vif: 1.91 (KURTt)
- **Log**: results/logs/stage01_index_reduction_20251216_134359.txt
- **Notes**: 

---

## 2025-12-16 13:44 — Stage 01: Index Reduction

- **Config**:
  - method: VIF-only
  - vif: 2
  - vif_fallback: 5
- **Results**:
  - n_start: 60
  - n_final: 17
  - n_removed: 43
  - final_indices: ACI, BI, BioEnergy, ECV, EPS_KURT, EVNtCount, EVNtMean, H_Havrda, KURTt, MEANt, NBPEAKS, ROU, SKEWt, TFSD, VARt, ZCR, nROI
  - categories: 5 (Amplitude Indices, Complexity Indices, Diversity Indices, Spectral Indices, Temporal Indices)
  - max_vif: 1.91 (KURTt)
- **Log**: results/logs/stage01_index_reduction_20251216_134431.txt
- **Notes**: 

---

## 2025-12-26 19:29 — Stage 00-b: Alignment

- **Config**:
  - resolution_hours: 2
  - stations: 9M, 14M, 37M
  - years: 2018, 2021
- **Results**:
  - rows_detections: 26250
  - rows_environment: 26284
  - rows_indices: 13102
  - rows_base: 26250
- **Log**: (none)
- **Notes**: 

---

## 2025-12-26 19:29 — Stage 00-c: QA Artifacts

- **Config**:
  - artifacts_checked: 5
- **Results**:
  - schema_columns: 178
  - missing_temp_frac: 0.00%
  - missing_depth_frac: 5.84%
- **Log**: (none)
- **Notes**: 

---

## 2025-12-26 19:30 — Stage 01: Index Reduction

- **Config**:
  - method: VIF-only
  - vif: 2
  - vif_fallback: 5
- **Results**:
  - n_start: 60
  - n_final: 17
  - n_removed: 43
  - final_indices: ACI, BI, BioEnergy, ECV, EPS_KURT, EVNtCount, EVNtMean, H_Havrda, KURTt, MEANt, NBPEAKS, ROU, SKEWt, TFSD, VARt, ZCR, nROI
  - categories: 5 (Amplitude Indices, Complexity Indices, Diversity Indices, Spectral Indices, Temporal Indices)
  - max_vif: 1.91 (KURTt)
- **Log**: results/logs/stage01_index_reduction_20251226_193011.txt
- **Notes**: 

---

## 2025-12-26 19:31 — Stage 02: Community Metrics

- **Config**:
  - fish_species: 8
  - dolphin_cols: 3
  - vessel_cols: 1
- **Results**:
  - rows: 26250
  - stations: 14M, 37M, 9M
  - fish_presence_mean: 51.8%
  - dolphin_presence_mean: 22.5%
  - vessel_presence_mean: 19.0%
- **Log**: results/logs/stage02_community_metrics_20251226_193100.txt
- **Notes**: 

---

## 2025-12-26 19:31 — Stage 03: Feature Engineering

- **Config**:
  - final_indices: 17
  - scale_covariates: False
- **Results**:
  - rows: 13102
  - columns: 39
  - unique_day_ids: 1095
  - unique_month_ids: 12
  - stations: 14M, 37M, 9M
- **Log**: results/logs/stage03_feature_engineering_20251226_193129.txt
- **Notes**: 

---

## 2025-12-26 19:32 — Stage 04: Exploratory Visualization

- **Config**:
  - responses: 9
  - indices: 17
  - covariates: 2
  - heatmap_cmap: viridis
- **Results**:
  - rows: 13102
  - scatter_plots: 9
  - heatmap_vars: 28
- **Log**: results/logs/stage04_exploratory_viz_20251226_193153.txt
- **Notes**: 

---

## 2025-12-26 19:34 — Stage 05a: GAMM Modeling

- **Config**:
  - pilot_mode: TRUE
  - scaling_enabled: TRUE
  - n_responses: 1
  - n_indices: 17
- **Results**:
  - fish_activity: converged (rho=0.20, AIC=33897.1, 0.0min)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2026-01-10 10:18 — Stage 05a: GAMM Modeling

- **Config**:
  - pilot_mode: FALSE
  - scaling_enabled: TRUE
  - n_responses: 9
  - n_indices: 17
- **Results**:
  - fish_activity: converged (rho=0.20, AIC=33897.1, 0.0min)
  - fish_richness: converged (rho=0.18, AIC=22471.6, 0.0min)
  - fish_presence: converged (rho=0.14, AIC=9412.7, 0.0min)
  - dolphin_burst_pulse: converged (rho=0.06, AIC=8187.1, 0.1min)
  - dolphin_echolocation: converged (rho=0.13, AIC=30765.0, 0.1min)
  - dolphin_whistle: converged (rho=0.11, AIC=3189.0, 0.0min)
  - dolphin_activity: converged (rho=0.13, AIC=32367.1, 0.0min)
  - dolphin_presence: converged (rho=0.11, AIC=10949.0, 0.0min)
  - vessel_presence: converged (rho=0.08, AIC=6961.0, 0.0min)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2026-02-03 14:35 — Stage 01: Index Reduction

- **Config**:
  - method: All indices (VIF disabled)
  - vif_enabled: False
  - vif: 2
  - vif_fallback: 5
- **Results**:
  - n_start: 60
  - n_final: 60
  - n_removed: 0
  - final_indices: ACI, ACTspCount, ACTspFract, ACTspMean, ACTtCount, ACTtFraction, ACTtMean, ADI, AEI, AGI, AnthroEnergy, BGNf, BGNt, BI, BioEnergy, EAS, ECU, ECV, ENRf, EPS, EPS_KURT, EPS_SKEW, EVNspCount, EVNspFract, EVNspMean, EVNtCount, EVNtFraction, EVNtMean, HFC, H_GiniSimpson, H_Havrda, H_Renyi, H_gamma, H_pairedShannon, Hf, Ht, KURTf, KURTt, LEQf, LEQt, LFC, MEANf, MEANt, MED, MFC, NBPEAKS, NDSI, RAOQ, ROU, SKEWf, SKEWt, SNRf, SNRt, TFSD, VARf, VARt, ZCR, aROI, nROI, rBA
  - categories: 5 (Amplitude Indices, Complexity Indices, Diversity Indices, Spectral Indices, Temporal Indices)
  - max_vif: N/A (VIF disabled)
- **Log**: results/logs/stage01_index_reduction_20260203_143532.txt
- **Notes**: 

---

## 2026-02-03 14:35 — Stage 02: Community Metrics

- **Config**:
  - fish_species: 8
  - dolphin_cols: 3
  - vessel_cols: 1
- **Results**:
  - rows: 26250
  - stations: 14M, 37M, 9M
  - fish_presence_mean: 51.8%
  - dolphin_presence_mean: 22.5%
  - vessel_presence_mean: 19.0%
- **Log**: results/logs/stage02_community_metrics_20260203_143537.txt
- **Notes**: 

---

## 2026-02-03 14:35 — Stage 03: Feature Engineering

- **Config**:
  - final_indices: 60
  - scale_covariates: False
- **Results**:
  - rows: 13102
  - columns: 82
  - unique_day_ids: 1095
  - unique_month_ids: 12
  - stations: 14M, 37M, 9M
- **Log**: results/logs/stage03_feature_engineering_20260203_143540.txt
- **Notes**: 

---

## 2026-02-03 14:37 — Stage 04: Exploratory Visualization

- **Config**:
  - responses: 9
  - indices: 60
  - covariates: 2
  - heatmap_cmap: viridis
- **Results**:
  - rows: 13102
  - scatter_plots: 9
  - heatmap_vars: 71
- **Log**: results/logs/stage04_exploratory_viz_20260203_143545.txt
- **Notes**: 

---

## 2026-02-03 14:40 — Stage 05a: GAMM Modeling

- **Config**:
  - pilot_mode: FALSE
  - scaling_enabled: TRUE
  - n_responses: 9
  - n_indices: 60
- **Results**:
  - fish_activity: converged (rho=0.17, AIC=33450.2, 0.2min)
  - fish_richness: converged (rho=0.16, AIC=22334.2, 0.3min)
  - fish_presence: converged (rho=0.12, AIC=9076.9, 0.2min)
  - dolphin_burst_pulse: converged (rho=0.06, AIC=8046.1, 0.3min)
  - dolphin_echolocation: converged (rho=0.10, AIC=30564.6, 0.5min)
  - dolphin_whistle: converged (rho=0.11, AIC=3117.5, 0.3min)
  - dolphin_activity: converged (rho=0.10, AIC=32128.3, 0.4min)
  - dolphin_presence: converged (rho=0.09, AIC=10692.8, 0.3min)
  - vessel_presence: converged (rho=0.06, AIC=6434.6, 0.3min)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

## 2026-02-03 17:30 — Stage 05a: GAMM Modeling

- **Config**:
  - pilot_mode: FALSE
  - scaling_enabled: TRUE
  - n_responses: 9
  - n_indices: 60
- **Results**:
  - fish_activity: converged (rho=0.17, AIC=33450.2, 0.2min)
  - fish_richness: converged (rho=0.16, AIC=22334.2, 0.3min)
  - fish_presence: converged (rho=0.12, AIC=9076.9, 0.2min)
  - dolphin_burst_pulse: converged (rho=0.06, AIC=8046.1, 0.3min)
  - dolphin_echolocation: converged (rho=0.10, AIC=30564.6, 0.5min)
  - dolphin_whistle: converged (rho=0.11, AIC=3117.5, 0.3min)
  - dolphin_activity: converged (rho=0.10, AIC=32128.3, 0.3min)
  - dolphin_presence: converged (rho=0.09, AIC=10692.8, 0.3min)
  - vessel_presence: converged (rho=0.06, AIC=6434.6, 0.3min)
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---


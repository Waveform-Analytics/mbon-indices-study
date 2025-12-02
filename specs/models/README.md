# Per‑Metric Specs

Purpose
- Each response metric (e.g., `fish_activity`, `dolphin_presence`, `vessel_presence`) has a dedicated spec per model type (GLMM or GAMM).

Naming
- `specs/models/<metric>-glmm-spec.md`
- `specs/models/<metric>-gamm-spec.md`

Content
- Inputs: analysis_ready, indices list, config keys
- Formula/Structure: fixed effects, random effects, family/link
- Diagnostics: DHARMa or GAMM summary, AIC
- Acceptance Criteria: convergence, dispersion, autocorrelation, edf interpretation
- Change Record: dated decisions and rationale
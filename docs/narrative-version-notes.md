# Notes: Narrative Version of Results Viewer

From a readability review of `results-viewer.html` (2026-02-09).

The current version works well as a **technical reference** — thorough, honest about failures, good practical callouts. The goal is to create a companion **narrative version** that tells the story of the findings without requiring the reader to grind through every metric.

## What to keep from the current version

- The elevator pitch opening ("Can acoustic indices predict biological activity? Yes for presence, less reliably for activity levels")
- The conversational tone and "what this means" translations
- Honesty about what doesn't work (zero-inflation, count model failures)
- "Practical implication" callouts — these are the best part

## Structural problems to fix

### 1. Everything has equal weight

Vessel detection (AUC 0.93, the big win) gets the same real estate as dolphin_whistle (R² = -infinity). The reader can't tell what matters without reading everything.

**Fix:** Build a clear hierarchy. Lead with the headline findings, use detail as supporting evidence.

### 2. The 9 metric-by-metric sections are repetitive

Identical template (response type → top indices → CV performance → interpretation) x9 puts the brain on autopilot. By dolphin_echolocation you're skimming.

**Fix options:**
- Group by story (fish / dolphins / vessels) with shared narrative
- Lead with the finding ("presence works, counts don't") and use individual metrics as evidence
- Single comparison table + short narrative, move per-metric details to expandable sections or link back to the reference version

### 3. Too many tables, not enough synthesis

Data is laid out well but the reader does the interpretive work of comparing across tables.

**Fix:** A few well-designed figures would replace 3-4 tables each. Examples:
- Dot plot: AUC (binary models) vs R² (count models) — instantly communicates "presence works, counts don't"
- Grouped bar chart: index contribution (ΔAIC) across all 9 metrics
- Heatmap: which index families matter for which organism groups

### 4. The most interesting parts are buried

Zero-inflation, concurvity, 17-vs-60 index evolution — all at the bottom, after the reader is spent.

**Fix:** Pull key insights up. The "why count models fail" story is more interesting than the individual count model results.

### 5. Audience calibration is uneven

AR1 autocorrelation is explained from scratch (audience probably knows this). Effect size table mixes fold changes and percentage points without much guidance on comparing them.

**Fix:** Pick one audience level and be consistent. For the narrative version, probably assume ecological/acoustic literacy but not statistical expertise.

## Proposed structure for narrative version

1. **The question** (1 paragraph)
2. **The answer** (the elevator pitch, expanded slightly)
3. **The wins** — vessel detection, then biological presence. Why they work. Key indices. Practical applications.
4. **The honest failures** — count models. Why they fail (zero-inflation, the structural problem). What this means for monitoring.
5. **The circularity question** — ACTspFract and dolphin presence. Is this prediction or detection?
6. **What surprised us** — 60 indices > 17 (regularization > pre-filtering). Station-level variation. No universal "best index."
7. **Recommendations** — what to use, what not to use, what to calibrate per-site
8. **Link to full reference** for per-metric details, tables, appendices

## Visual ideas

- **Figure 1:** Binary vs count model performance (AUC / R²) — the core finding in one image
- **Figure 2:** Which index families map to which organisms (heatmap or network diagram)
- **Figure 3:** Station-level variation for the top 3 metrics — shows "no universal best index"
- Keep the smooth plots in the reference version, not here

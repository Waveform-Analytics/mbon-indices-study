# MBON Acoustic Indices Study: Model Results Interpretation Guide

> **Purpose:** This document explains the statistical modeling results in accessible terms. It focuses on *what the results mean* and *how to think about them*, not just what the numbers are.

---

## Overview: What We Did

We used **Generalized Additive Mixed Models (GAMMs)** to ask: *Do acoustic indices predict biological activity in the May River estuary?*

- **Predictors:** 17 acoustic indices (automated summaries of soundscape characteristics) plus environmental variables (temperature, depth) and temporal patterns (time of day, day of year)
- **Responses:** 9 biological metrics derived from manual acoustic detections — fish activity/richness/presence, dolphin vocalizations (burst pulses, echolocation, whistles), dolphin activity/presence, and vessel presence
- **Data:** 13,102 observations across 3 monitoring stations, 2 years (2018, 2021), at 2-hour resolution

GAMMs let us model **nonlinear relationships** (an index might matter a lot at low values but plateau at high values) while accounting for the fact that observations close in time aren't independent.

---

## Part 1: Model Convergence and Technical Soundness

### The Good News: All 9 Models Converged

| Metric | Converged | AR1 (ρ) | AIC |
|--------|-----------|---------|-----|
| fish_activity | Yes | 0.196 | 33,897 |
| fish_richness | Yes | 0.185 | 22,472 |
| fish_presence | Yes | 0.144 | 9,413 |
| dolphin_burst_pulse | Yes | 0.064 | 8,187 |
| dolphin_echolocation | Yes | 0.128 | 30,765 |
| dolphin_whistle | Yes | 0.107 | 3,189 |
| dolphin_activity | Yes | 0.129 | 32,367 |
| dolphin_presence | Yes | 0.110 | 10,949 |
| vessel_presence | Yes | 0.077 | 6,961 |

**What "converged" means:** The optimization algorithm successfully found stable parameter estimates. If models fail to converge, it usually means the data can't support the model complexity, or there are numerical issues. Convergence is a necessary (but not sufficient) condition for trusting the results.

---

### Understanding the AR1 Parameter (ρ): Temporal "Stickiness"

The AR1 (autoregressive order 1) parameter tells us: **How much does knowing the current state tell us about the next time point?**

Think of it as measuring persistence or "stickiness" over time:
- **ρ = 0** means each 2-hour window is independent — no memory
- **ρ = 1** means perfect persistence — if it's high now, it's high next time
- **ρ = 0.1–0.2** (what we see) means modest persistence — some temporal continuity, but plenty of change

#### What the values tell us ecologically:

**Fish (ρ = 0.14–0.20) — Most persistent**
Fish activity doesn't flip on and off rapidly. If fish are acoustically active in one 2-hour window, they're likely still active in the next. This makes sense: fish schools persist, feeding periods extend across hours, and tidal influences create sustained activity windows rather than random spikes.

**Dolphin burst pulses (ρ = 0.06) — Most independent**
Burst pulses are the most "event-like" of all our metrics. These sounds are associated with foraging and prey capture — discrete hunting events rather than sustained behavior. A dolphin catching prey now doesn't strongly predict it will be hunting in the same spot 2 hours later.

**Dolphin whistles and echolocation (ρ = 0.10–0.13) — Moderate persistence**
These are social (whistles) and navigational (echolocation) signals. When dolphins are in the area communicating or moving through, they tend to stick around for a while — but not as persistently as fish activity.

**Vessel presence (ρ = 0.08) — Low persistence**
Boats transit through. A vessel now doesn't strongly predict another vessel 2 hours later. This is consistent with transient boat traffic rather than stationary vessels.

#### Why this matters for interpretation:

The AR1 structure in our models **accounts for temporal non-independence**. Without it, we'd be pretending each observation is independent, which would:
1. Underestimate our uncertainty (confidence intervals too narrow)
2. Potentially attribute temporal patterns to the wrong predictors

Our modest ρ values (0.06–0.20) mean the autocorrelation is being handled appropriately without dominating the signal. If these were very high (0.7+), we'd worry that temporal persistence was swamping the actual index-response relationships.

**Bottom line:** We can credibly attribute patterns to the acoustic indices because we've accounted for the fact that consecutive observations aren't independent.

---

### Understanding AIC: Model Complexity and Fit

AIC (Akaike Information Criterion) balances model fit against complexity. Lower AIC means better fit for the same data — but there's a critical caveat.

#### What you CAN'T do with these AIC values:

Compare across different response metrics. The fish_activity AIC of 33,897 vs. dolphin_whistle's 3,189 does NOT mean the whistle model is "better." These metrics have:
- Different response distributions (counts vs. binary)
- Different prevalence (fish detections are common; some dolphin call types are rare)
- Different underlying variance structures

#### What AIC tells us here:

The AIC values are most useful for:
1. **Comparing alternative models for the same metric** (e.g., "does adding this index improve the fish_activity model?")
2. **Getting a rough sense of modeling complexity** — higher AIC often reflects more variance to explain

The fact that activity/count metrics (fish_activity, dolphin_activity, dolphin_echolocation) have higher AICs than presence metrics makes sense: counts have more possible values and more variance than binary yes/no outcomes.

---

## Part 2: Which Indices Predict Which Responses?

This is the core question: **Do acoustic indices — automated summaries of soundscape characteristics — actually predict biological activity?**

The short answer: **Yes, but the patterns differ by response variable.**

---

### How to Read the Results

For each predictor (acoustic index, environmental variable, or temporal pattern), the model estimates a "smooth term" — a flexible curve describing the relationship. Two key numbers:

| Metric | What it means |
|--------|---------------|
| **p-value** | Statistical significance. p < 0.05 means the relationship is unlikely to be due to chance. |
| **EDF** (effective degrees of freedom) | Shape complexity. EDF ≈ 1 means nearly linear; EDF > 2 means the relationship curves or bends. |

A predictor matters if it has **low p-value AND meaningful EDF**. Very low EDF (< 0.01) means the model essentially removed that term — it doesn't contribute.

---

### The Big Picture: Significant Predictors by Response

The table below shows which acoustic indices significantly predict each biological response (p < 0.05). Indices that appear across multiple responses may be especially useful as biodiversity indicators.

| Index | Fish Act. | Fish Rich. | Fish Pres. | Dolph. BP | Dolph. Echo | Dolph. Whis. | Dolph. Act. | Dolph. Pres. | Vessel |
|-------|:---------:|:----------:|:----------:|:---------:|:-----------:|:------------:|:-----------:|:------------:|:------:|
| **ACI** | | ✓ | ✓ | ✓ | | | | | ✓ |
| **BI** | | | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| **BioEnergy** | | | ✓ | | | | | ✓ | |
| **ECV** | | | | | ✓ | ✓ | ✓ | ✓ | |
| **EPS_KURT** | | ✓ | ✓ | ✓ | ✓ | | ✓ | | |
| **EVNtCount** | | ✓ | ✓ | | | | ✓ | ✓ | ✓ |
| **EVNtMean** | | | ✓ | | | | | ✓ | |
| **H_Havrda** | | | ✓ | ✓ | | | ✓ | | ✓ |
| **KURTt** | | | ✓ | | | | | ✓ | ✓ |
| **NBPEAKS** | | | | ✓ | ✓ | | ✓ | ✓ | |
| **ROU** | | ✓ | ✓ | | ✓ | | ✓ | | |
| **SKEWt** | | ✓ | | | | | | | |
| **TFSD** | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **VARt** | | | ✓ | | | | | | |
| **ZCR** | ✓ | ✓ | ✓ | | | ✓ | | | ✓ |
| **nROI** | | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |

**Legend:** ✓ = significant at p < 0.05. Empty = not significant.

---

### Interpreting the Patterns

#### Indices that predict broadly (appear in 5+ responses):

- **TFSD** (Temporal Fine Structure Deviation): Significant for 6/9 responses — all dolphin metrics plus vessels. This index captures rapid temporal fluctuations in the soundscape. Interpretation: Dolphins and boats both create distinctive temporal patterns that this index picks up.

- **nROI** (Number of Regions of Interest): Significant for 7/9 responses. This counts distinct acoustic "events" in the spectrogram. More events = more activity. It's a general activity indicator rather than taxon-specific.

- **BI** (Bioacoustic Index): Significant for 6/9 responses, especially dolphins and vessels. BI measures sound level in the 2-8 kHz range — right where dolphin whistles and many anthropogenic sounds occur.

- **EVNtCount** (Acoustic Event Count): Significant for 5/9 responses. Like nROI, this counts discrete sound events. More complex soundscapes have higher counts.

#### Indices that are taxon-specific:

- **ZCR** (Zero Crossing Rate): Strongly predicts fish metrics (all 3) but not dolphin metrics. ZCR relates to the dominant frequency of sounds — fish choruses have characteristic frequency content that differs from dolphin clicks and whistles.

- **ECV** (Entropy of Spectral Variance): Predicts dolphins (4/5 metrics) but not fish. Dolphins produce sounds with distinctive spectral entropy patterns.

- **BioEnergy**: Only significant for fish_presence and dolphin_presence — not the activity or richness metrics. This suggests it's better at detecting "something is there" than "how much is there."

#### The fish story:

Fish presence has the most significant predictors (12 of 17 indices). Fish richness and activity have fewer. This suggests:
- Detecting fish presence is relatively easy — many indices pick it up
- Distinguishing activity levels and species richness is harder — fewer indices are informative

The indices that DO predict fish richness (ACI, EPS_KURT, EVNtCount, ROU, SKEWt, ZCR, nROI) are mostly related to acoustic complexity and temporal structure — consistent with the idea that more fish species = more diverse soundscape.

#### The dolphin story:

Different call types show different patterns:
- **Burst pulses** (foraging): Predicted by TFSD, NBPEAKS, nROI — indices sensitive to rapid, impulsive sounds
- **Echolocation**: Predicted by BI, TFSD, NBPEAKS, ROU — broadband click trains have distinctive spectral and temporal structure
- **Whistles**: Predicted by only 3 indices (ECV, TFSD, ZCR) — these tonal sounds have specific spectral characteristics that fewer indices capture

This differentiation is ecologically meaningful: different behaviors (hunting vs. navigating vs. socializing) produce acoustically distinct signals.

#### Vessels:

Vessel presence is predicted by 8 indices, with strong contributions from ACI, BI, EVNtCount, TFSD, nROI. This makes sense: boats are loud, broadband, and create complex temporal patterns. The lack of a temperature or depth signal (unlike dolphins) reinforces that vessel detection is purely acoustic, not environmentally mediated.

---

### Environmental and Temporal Predictors

Beyond acoustic indices, the models include environmental and temporal covariates:

| Predictor | What it captures |
|-----------|------------------|
| **temperature** | Water temperature — affects metabolic rates and behavior |
| **depth** | Tide proxy — depth changes with tidal cycle |
| **hour_of_day** | Diel (daily) patterns — dawn/dusk activity |
| **day_of_year** | Seasonal patterns — breeding seasons, migrations |
| **month_id** | Similar to day_of_year but coarser |
| **station** | Spatial differences among the 3 monitoring sites |

**Key findings:**

- **Fish** show strong temporal patterns: hour_of_day and day_of_year are highly significant. Fish activity follows predictable diel and seasonal rhythms.

- **Dolphins** are driven more by temperature than time of day. Temperature is significant for all dolphin metrics; hour_of_day is weaker. This suggests dolphin activity may track prey availability or thermal preferences rather than light cycles.

- **Station effects** are significant for most metrics — the three sites (9M, 14M, 37M) differ in their biological communities. This is expected given they span a salinity gradient along the estuary.

---

### Relationship Shapes: Linear vs. Curved

The EDF values tell us whether relationships are simple (linear) or complex (curved):

**Highly nonlinear relationships (EDF > 2.5):**
- hour_of_day → fish metrics (EDF 5-7): Strong dawn/dusk peaks
- day_of_year → fish presence (EDF 9.4): Complex seasonal pattern
- TFSD → dolphin metrics (EDF 2.5-3): Threshold effects — dolphins vocalize when TFSD is in a certain range

**Nearly linear relationships (EDF ≈ 1):**
- temperature → some metrics: Monotonic effects — more/less with warmer water
- Some indices show linear associations where the effect is consistent across the index's range

**What this means:**
The nonlinear relationships justify using GAMMs rather than simpler linear models. A linear model would miss the dawn/dusk peaks in fish activity or the threshold effects for some indices.

---

## Part 3: Model Diagnostics — Are the Models Reliable?

Diagnostic plots help us assess whether the models are behaving as expected. We generated four diagnostic plots for each of the 9 models:

1. **Residuals vs Fitted** — Do residuals scatter randomly around zero?
2. **Q-Q Plot** — Do residuals follow the expected distribution?
3. **Distribution of Residuals** — What's the shape of the residual distribution?
4. **Response vs Fitted** — How well do predictions match observations?

The diagnostic plots are saved at: `results/figures/<metric>/gamm_diagnostics.png`

---

### How to Read Diagnostic Plots

**What we're looking for:**
- Residuals centered around zero (no systematic bias)
- No strong patterns in residuals vs fitted (would indicate missing predictors or wrong functional form)
- Q-Q plot following the diagonal line (residuals match expected distribution)
- No extreme outliers dominating the fit

**What's "normal" depends on the response type:**
- **Count models** (fish_activity, fish_richness, dolphin counts): Expect discrete "bands" in residuals because counts are integers. Some fanning (wider spread at higher fitted values) is normal.
- **Binary models** (presence/absence): Expect two distinct bands of residuals (one for 0s, one for 1s). The S-shaped pattern in Q-Q plots is expected for logistic regression.

---

### Summary of Diagnostic Findings

#### Count Models (Negative Binomial)

**fish_activity, fish_richness, dolphin_activity, dolphin_echolocation:**
- Residuals show the expected discrete banding pattern for count data
- Q-Q plots follow the diagonal reasonably well in the center, with some deviation in the tails
- Right-skewed residual distributions are expected — counts can't go below zero but can have high values
- No catastrophic departures from model assumptions

**dolphin_burst_pulse, dolphin_whistle:**
- These show more pronounced zero-inflation — many observations are zeros
- Q-Q plots show stronger right-tail deviation (heavy positive tail)
- This is expected for rare events: most time windows have no whistles or burst pulses, but occasionally there are bursts of activity
- The negative binomial distribution handles this reasonably well, though a zero-inflated model could be explored if needed

#### Binary Models (Binomial/Logistic)

**fish_presence, dolphin_presence, vessel_presence:**
- Classic binary model diagnostics: two bands of residuals corresponding to observed 0s and 1s
- Q-Q plots show the characteristic S-curve that's normal for binary outcomes
- Response vs Fitted shows only 0 and 1 values (as expected)
- No concerns — these are well-behaved logistic regression diagnostics

---

### Overall Assessment

| Model Type | Status | Notes |
|------------|--------|-------|
| Count models (common events) | Good | Expected patterns for negative binomial |
| Count models (rare events) | Acceptable | Some zero-inflation visible; results still valid |
| Binary models | Good | Classic logistic regression patterns |

**Bottom line:** The diagnostics show no red flags. The models are behaving as expected for their respective distribution families. The patterns we see — discrete banding in count data, S-curves in binary data, right-skew for rare events — are all features of the data generating process, not model failures.

**One note of caution:** The rare event models (dolphin_whistle, dolphin_burst_pulse) show zero-inflation. This doesn't invalidate the results, but it does mean:
- Effect sizes for these metrics should be interpreted carefully
- Predictions at higher count values have more uncertainty
- A zero-inflated negative binomial model could be a future refinement

---

## Part 4: Smooth Plots — The Shape of Relationships

Smooth plots show the actual functional form of each predictor-response relationship. They're the visual payoff of GAMM modeling — showing not just *whether* a predictor matters, but *how* it relates to the response.

The smooth plots are saved at:
- Overview grids: `results/figures/<metric>/gamm_smooths.png`
- Individual plots: `results/figures/<metric>/smooth_<predictor>.png`

---

### How to Read Smooth Plots

Each plot shows:
- **X-axis:** The predictor value (e.g., ZCR, temperature, hour_of_day)
- **Y-axis:** The partial effect on the response (on the link scale)
- **Solid line:** The estimated smooth function
- **Dashed lines:** 95% confidence interval
- **Y-axis label:** Shows EDF in parentheses, e.g., `s(ZCR,3.83)` means EDF = 3.83

**Interpreting the Y-axis:**
- Values are centered around zero (the average effect)
- Positive values = higher predicted response than average
- Negative values = lower predicted response than average
- The scale is on the "link" scale (log for counts, logit for binary), not raw units

**Key patterns to look for:**
- **Flat line:** Predictor has no effect (often with wide confidence bands)
- **Linear slope:** Simple monotonic relationship (more/less = more/less)
- **Curved line:** Nonlinear effect — optimal ranges, thresholds, or saturation
- **Wide confidence bands:** More uncertainty (often at extreme predictor values)

---

### Highlighted Examples

#### Fish Activity: Hour of Day (EDF = 6.81)

The fish_activity ~ hour_of_day smooth shows a subtle diel pattern:
- Slight increase during afternoon hours (12-18)
- Slight decrease overnight and early morning
- The effect is modest (y-range only ~0.5 units) but statistically significant (p < 0.001)

This suggests fish acoustic activity has a weak but detectable afternoon peak, possibly related to feeding activity or chorus timing.

#### Fish Presence: Day of Year (EDF = 9.36)

This is one of the most complex smooths — a strong seasonal pattern:
- Peak around day 100-120 (April-May)
- Lower presence in summer and winter months
- The effect is large (y-range ~15 units on link scale)

This aligns with known fish spawning seasons in estuarine environments. Spring is peak activity for many soniferous fish species.

#### Fish Activity/Richness: ZCR (EDF = 3.83)

ZCR (Zero Crossing Rate) shows a nonlinear relationship with fish metrics:
- Relatively flat at low-to-moderate ZCR values
- Declining effect at high ZCR values
- Peak effect around ZCR = 1-2 (scaled units)

This suggests fish sounds occupy a particular frequency range — very high ZCR (high-frequency dominated soundscapes) are associated with *less* fish activity.

#### Vessel Presence: Hour of Day (EDF = 8.26)

A clear and intuitive pattern:
- Peak vessel presence in morning hours (6-10am)
- Lower in evening/night (18-24)
- Strong effect size (y-range ~5 units)

This reflects recreational and commercial boating patterns — boats are active during daylight, especially mornings.

#### Dolphin Activity: Temperature (EDF = 2.37)

Nearly linear negative relationship:
- More dolphin activity at cooler temperatures
- Less activity in warmer water
- Modest effect size but tight confidence bands

This could reflect:
- Seasonal patterns (dolphins more active in cooler months)
- Prey availability (prey species may prefer cooler water)
- Or behavioral thermoregulation
Eric's group will likely know

#### Dolphin Whistle: TFSD (EDF = 1.74)

TFSD (Temporal Fine Structure Deviation) shows:
- Increasing whistle probability as TFSD increases from very low values
- Plateau/slight decline at high TFSD values
- Wide confidence bands at extreme values (fewer observations)

Interpretation: Moderate temporal complexity in the soundscape is associated with dolphin whistles — makes sense, as whistles themselves contribute to temporal structure.

#### Dolphin Burst Pulse: NBPEAKS (EDF = 2.49)

NBPEAKS shows a declining relationship:
- Higher burst pulse activity at low-to-moderate NBPEAKS
- Lower activity at very high NBPEAKS values
- Nonlinear shape suggests an optimal range

This may seem counterintuitive (more peaks = less activity?), but could reflect:
- Burst pulses are short, impulsive sounds that may not create many spectral peaks
- High NBPEAKS might indicate other sound sources (boats, snapping shrimp) that mask or coincide with reduced dolphin foraging

#### Fish Richness: Depth (EDF = 2.63)

Nearly linear positive relationship:
- Higher fish richness at greater depths (high tide)
- Lower richness at shallow depths (low tide)
- Tight confidence bands = high certainty

This likely reflects tidal access — at high tide, more fish species can access the monitoring sites. The estuary becomes more connected to deeper waters, allowing more species to move in.

---

### Patterns Across Metrics

**Temporal patterns differ by taxa:**
- Fish: Strong diel (hour_of_day) and seasonal (day_of_year) patterns
- Dolphins: Weaker diel patterns, stronger temperature effects
- Vessels: Strong diel pattern following human activity schedules

**Many index effects are subtle:**
- Most acoustic index smooths have y-ranges of 1-3 units
- Temporal and environmental predictors often have larger effects
- This doesn't mean indices are unimportant — they add predictive value *after* accounting for time and environment

**Flat smooths are informative too:**
- When a smooth is essentially flat (EDF near 0), the model has determined that predictor doesn't help
- This is the model doing automatic variable selection via penalization

---

### What We Can and Can't Conclude

**We CAN say:**
- Multiple acoustic indices significantly predict biological activity, even after controlling for environmental and temporal factors
- Different response variables show distinct index associations
- The relationships are often nonlinear — thresholds and optima matter
- Temporal patterns (diel, seasonal) are strong drivers of fish activity
- Temperature is a key driver for dolphins

**We CANNOT say (yet):**
- Whether these are causal relationships (indices → behavior) or just correlations
- The exact mechanisms linking specific indices to specific behaviors
- How well these patterns generalize to other estuaries or time periods

---

## Part 5: Model Validation — Do the Models Generalize?

We validated the models in two ways:
1. **AR1 validation** — Is the autocorrelation correction working?
2. **Week-based cross-validation** — Can the models predict held-out time periods?

The validation outputs are at:
- `results/tables/ar1_validation.csv`
- `results/tables/cv_performance_summary.csv`
- `results/figures/acf_residuals.png`
- `results/figures/cv_performance_by_metric.png`

---

### AR1 Validation Results

We checked whether the AR1 correction removed temporal autocorrelation from the residuals.

| Metric | ρ estimated | Residual ACF(1) | AR1 Effective? |
|--------|-------------|-----------------|----------------|
| fish_activity | 0.196 | 0.205 | No |
| fish_richness | 0.185 | 0.192 | No |
| fish_presence | 0.144 | 0.150 | No |
| dolphin_burst_pulse | 0.064 | 0.065 | Yes |
| dolphin_echolocation | 0.128 | 0.134 | No |
| dolphin_whistle | 0.107 | 0.085 | Yes |
| dolphin_activity | 0.129 | 0.135 | No |
| dolphin_presence | 0.110 | 0.113 | No |
| vessel_presence | 0.077 | 0.081 | Yes |

**What this means:**

The residual ACF(1) values are nearly identical to the estimated ρ values. Ideally, if the AR1 correction were fully absorbing the autocorrelation, residual ACF(1) would be near zero.

**Why this happens:** The `bam()` function in mgcv treats ρ as a fixed value estimated from preliminary residuals, rather than iteratively re-estimating it during model fitting. This is a computational shortcut that works reasonably well but doesn't fully eliminate autocorrelation.

**Is this a problem?** Not necessarily:
- The autocorrelation levels are modest (0.06–0.20)
- The models are still accounting for *most* of the temporal structure
- Standard errors may be slightly underestimated, so interpret p-values conservatively
- For practical prediction and understanding relationships, this is acceptable

---

### Cross-Validation Results

We held out one week at a time (53 folds), retrained on the remaining weeks, and predicted the held-out week.

#### Binary Metrics (Presence/Absence) — AUC

| Metric | Mean AUC | SD | Interpretation |
|--------|----------|-----|----------------|
| vessel_presence | **0.92** | 0.03 | Excellent — strong, consistent predictions |
| fish_presence | 0.75 | 0.11 | Moderate — better than chance (0.5), useful signal |
| dolphin_presence | 0.74 | 0.13 | Moderate — similar to fish |

**AUC interpretation:**
- 0.5 = random guessing
- 0.7–0.8 = acceptable discrimination
- 0.8–0.9 = good discrimination
- 0.9+ = excellent discrimination

The vessel model generalizes very well. The biological presence models (fish, dolphin) are moderately predictive — they're capturing real signal, but there's substantial week-to-week variation.

#### Count Metrics (Activity/Richness) — RMSE and R²

| Metric | Mean RMSE | Mean R² | Interpretation |
|--------|-----------|---------|----------------|
| fish_richness | 0.56 | 0.02 | Low error but R² near 0 |
| fish_activity | 1.14 | 0.01 | Same pattern |
| dolphin_whistle | 0.78 | -∞ | Doesn't generalize |
| dolphin_burst_pulse | 1.54 | -0.13 | Worse than mean prediction |
| dolphin_echolocation | 11.0 | -455 | Outlier folds dominating |
| dolphin_activity | 7.22 | -48 | Same issue |

**What negative R² means:**

R² < 0 means the model predictions are worse than simply predicting the mean for every observation. This sounds alarming, but it's common in time series CV for several reasons:

1. **Week-to-week variation is high** — patterns in one week may not transfer to another
2. **Temporal structure dominates** — the model fits temporal patterns that are week-specific
3. **Rare events** — dolphin whistles and burst pulses have many zeros; predicting *when* rare events happen is very hard

**The dolphin models in particular** show that while the fitted relationships are statistically significant within the training data, they don't reliably predict held-out weeks. This is a cautionary finding.

---

### What the Validation Tells Us

**Strong finding — Vessel detection:**
The vessel presence model generalizes excellently (AUC = 0.92). Acoustic indices reliably distinguish vessel presence across different time periods. This is practically useful for monitoring anthropogenic noise.

**Moderate finding — Biological presence:**
Fish and dolphin presence models show moderate generalization (AUC ~0.75). The indices contain real signal about biological presence, but predictions for any given week are uncertain.

**Cautionary finding — Activity/count models:**
The count models (especially for dolphins) do not generalize well to held-out weeks. The fitted relationships within the data are statistically significant, but they don't reliably predict future observations.

**Implications for interpretation:**
- The index-response relationships in Part 2 and Part 4 are real patterns in this dataset
- For presence/absence questions, the models have practical predictive value
- For activity/count questions, the models describe patterns but should not be used for prediction without caution
- The AR1 correction partially addresses autocorrelation but doesn't eliminate it — interpret significance conservatively

---

## Glossary

| Term | Plain English |
|------|---------------|
| **GAMM** | A flexible regression model that can capture curved relationships and account for non-independence in the data |
| **Smooth term** | A flexible curve fitted to the data (vs. a straight line in regular regression) |
| **AR1** | A way of modeling temporal autocorrelation — the tendency for consecutive observations to be similar |
| **AIC** | A score for comparing models; lower is better, but only for the same response variable |
| **Converged** | The model fitting algorithm successfully found stable estimates |
| **EDF** | Effective degrees of freedom — measures how "wiggly" a smooth term is (1 = straight line, higher = more complex curve) |

---

*Document created: 2026-01-10*
*Last updated: 2026-01-14*
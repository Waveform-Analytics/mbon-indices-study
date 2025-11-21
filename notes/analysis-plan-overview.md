Analysis Plan: Acoustic Indices Predicting Biological Patterns
Core Research Question
Can a reduced set of acoustic indices predict biological activity patterns in estuarine environments without manual annotation?

Phase 1: Index Selection and Data Preparation
Finalize index reduction (from 60 to ~10):
Step 1: Correlation-based reduction
Calculate pairwise correlations among all 60 indices
Remove one index from each pair with r > 0.7
Retain ~20-30 indices representing distinct acoustic features
Step 2: Variance Inflation Factor (VIF) analysis
Calculate VIF for remaining indices
Remove indices with VIF > 5-10 until all below threshold
Final set: ~10 indices
Step 3: Domain knowledge check
Verify final set represents distinct aspects of soundscape (spectral, temporal, complexity)
Document which aspects each index captures
Create temporal variables:
# Cyclic time terms for GLMMs
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

# Temporal variables for GAMMs
df['day_of_year'] = df['date'].dt.dayofyear
df['hour_of_day'] = df['datetime'].dt.hour

# Grouping factors for random effects
df['day_id'] = df['date'].astype(str) + '_' + df['station']
df['month_id'] = df['date'].dt.to_period('M').astype(str)

# Time sequence for AR1 (sequential within each station-day)
df['time_within_day'] = df.groupby('day_id').cumcount()

Community metrics to model:
Fish metrics:
Fish activity (sum of intensity values) - negative binomial
Fish richness (species count) - Poisson or negative binomial
Fish presence (any species) - binomial
Dolphin metrics: 4. Dolphin burst pulse counts - negative binomial 5. Dolphin click counts - negative binomial 6. Dolphin whistle counts - negative binomial 7. Dolphin activity (sum across call types) - negative binomial 8. Dolphin presence (any calls) - binomial
Vessel metric: 9. Vessel presence - binomial

Phase 2: GLMM Analysis
Model structure:
Fit separate models for each of the 9 community metrics using the ~10 indices as predictors.
For count data (fish activity, richness, dolphin counts):
library(glmmTMB)

model <- glmmTMB(
  response ~ 
    # Acoustic indices (~10 predictors)
    index1 + index2 + index3 + ... + index10 +
    # Environmental variables
    temperature + depth +
    # Diel pattern (cyclic terms)
    sin_hour + cos_hour +
    # Random effects (nested structure)
    (1|station) + (1|month_id) + (1|day_id) +
    # Temporal autocorrelation
    ar1(time_within_day + 0 | day_id),
  family = nbinom2,  # or poisson if no overdispersion
  data = dat
)

For presence/absence (fish presence, dolphin presence, vessel presence):
model <- glmmTMB(
  response ~ 
    index1 + index2 + ... + index10 +
    temperature + depth +
    sin_hour + cos_hour +
    (1|station) + (1|month_id) + (1|day_id) +
    ar1(time_within_day + 0 | day_id),
  family = binomial(link = "logit"),
  data = dat
)

Diagnostics for each model:
library(DHARMa)

# Residual diagnostics
sim_res <- simulateResiduals(model)
plot(sim_res)
testDispersion(sim_res)
testTemporalAutocorrelation(sim_res, time = dat$time_within_day)

# Check for overdispersion in count models
# If present, use nbinom2; if not, poisson is sufficient

# Examine random effects structure
summary(model)
ranef(model)  # Variation by station/month/day

# Check fixed effects
confint(model)  # Which index CIs don't overlap zero?

Model refinement:
For each community metric:
Identify indices with significant effects (CIs don't overlap zero)
Consider dropping non-significant indices for cleaner final models
Document which indices matter for which biological responses
Note: It's fine if different indices are important for different metrics - this is biologically meaningful (e.g., fish vs dolphin calls occupy different frequency ranges).

Phase 3: GAMM Analysis
Refit models with smooth terms:
library(mgcv)

# For count data
model_gam <- bam(
  response ~ 
    # Smooth terms for indices (k=5 allows moderate non-linearity)
    s(index1, k=5) + s(index2, k=5) + ... + s(index10, k=5) +
    # Smooth environmental variables
    s(temperature, k=5) + s(depth, k=5) +
    # Cyclic smooths for temporal patterns
    s(hour_of_day, bs="cc", k=12) +  # diel pattern
    s(day_of_year, bs="cc", k=12) +  # seasonal pattern
    # Random effects
    s(station, bs="re") +
    s(month_id, bs="re") +
    s(day_id, bs="re"),
  family = nb(),  # or binomial for presence/absence
  data = dat,
  method = "fREML",
  discrete = TRUE  # computational efficiency
)

Examine smooth functions:
# Visualize smooth relationships
plot(model_gam, pages=1, scheme=2)

# Check effective degrees of freedom (edf)
summary(model_gam)

Interpreting edf:
edf ≈ 1: Relationship is essentially linear (GLMM likely sufficient)
edf >> 1: Clear non-linear relationship (GAMM captures complexity GLMM misses)
Compare GLMM vs GAMM using AIC:
AIC(model_glmm, model_gam)

Decision criteria:
ΔAIC < 4: Models equivalent, GLMM may be preferred (simpler, more interpretable)
ΔAIC 4-10: Moderate support for GAMM, check if non-linearities are ecologically meaningful
ΔAIC > 10: Strong support for GAMM
Inspect non-linearities:
If GAMMs show better AIC, examine smooth plots to identify:
Threshold effects (index only matters above/below certain value)
Saturation (biological response plateaus at high index values)
Unexpected patterns that inform ecology

Phase 4: Cross-Validation
Challenge with temporal validation:
The dataset contains some seasonal phenomena (e.g., spawning calls occurring only in specific months). This creates challenges for standard temporal cross-validation approaches, as holding out certain periods means the model never encounters those phenomena during training.
Potential validation approaches:
Option 1: Leave-one-station-out CV
For each station, train on other 2 stations, test on held-out station
Advantage: Tests spatial generalization (critical for deployment to new sites)
Limitation: Only 3 stations means limited replication
Interpretation: Assesses whether relationships are site-specific or generalizable
Option 2: Within-month temporal split
Train on first 3 weeks of every month (days 1-21), test on fourth week (days 22-end)
Advantage: All seasonal phenomena represented in both sets; tests if 3 weeks predicts remainder of month; mimics realistic deployment
Limitation: Training and test periods temporally close; doesn't test year-to-year generalization
Interpretation: Shows whether initial monthly monitoring predicts ongoing patterns within established seasonal context
Cross-validation implementation:
Apply the same validation approach(es) to both GLMM and GAMM models for fair comparison.
# Example structure for any CV approach
cv_results = []
for train_set, test_set in cv_splits:
    # Fit both GLMM and GAMM on training data
    # Predict on test set with both models
    # Calculate performance metrics
    
    cv_results.append({
        'split': split_id,
        'glmm_rmse': glmm_rmse,
        'gamm_rmse': gamm_rmse,
        'glmm_r2': glmm_r2,
        'gamm_r2': gamm_r2,
        # Additional metrics as appropriate
    })

Performance metrics by response type:
Count data: RMSE, MAE, R² (or pseudo-R²)
Binary data: AUC, accuracy, sensitivity, specificity
Final model selection:
Choose between GLMM and GAMM for each community metric based on:
AIC comparison (in-sample fit)
Cross-validation performance (out-of-sample prediction)
Ecological interpretability of non-linearities (if GAMM shows them)
If AIC and CV both favor GAMM, use GAMM. If results are mixed or equivalent, GLMM may be preferred for interpretability.

Phase 5: Final Model Selection and Interpretation
For each of the 9 community metrics:
1. Select best model (GLMM vs GAMM based on AIC + CV performance)
2. Identify key predictive indices:
Which indices significantly predict this response?
What are effect sizes and directions?
Can model be simplified further (drop non-significant indices)?
3. Assess spatial consistency:
Examine station random effects - are they large or small?
Small random effects = relationships generalize across stations
Large random effects = station-specific calibration needed
4. Create biological interpretation:
Example structure:
Fish activity is predicted by:
- ACI (positive, indicates broadband biotic sound)
- Acoustic Entropy (negative, decreases with structured calls)
- Temperature (positive, metabolic activity)

Dolphin presence is predicted by:
- [Different subset of indices]
- [Effect directions and magnitudes]

Vessel presence is predicted by:
- [Different subset, likely low-frequency indices]

Summarize index importance across all metrics:
Which indices are universally important vs metric-specific?
This informs the "rapid assessment tool" recommendation
E.g., "ACI and BI together predict most biological patterns"

Phase 6: Results Presentation
Key analyses and figures:
1. Index selection documentation
Correlation heatmap of final 10 indices (shows they're distinct)
Table showing which aspect of soundscape each index represents
2. Model comparison
Table of AIC values: GLMM vs GAMM for each metric
Cross-validation performance comparison: GLMM vs GAMM
Justification for model choice per metric
3. Model coefficients/effects
Forest plots showing effect sizes + confidence intervals (GLMMs)
Smooth function plots showing relationships (GAMMs)
Separate panel for each community metric
4. Cross-validation performance
Performance metrics (RMSE, R², AUC) for each community metric
Visualization of predicted vs observed values
Discussion of where models perform well vs poorly (e.g., seasonal patterns)
5. Predicted vs observed
Scatter plots or time series showing model fit
Separate for each metric or show key examples
6. Index importance summary
Heat map: which indices matter for which metrics
Identifies shared vs unique predictors
7. Spatial variation
Random effects by station (if substantial)
Or state that relationships are consistent across stations

Manuscript Structure
Introduction
Manual annotation is labor-intensive and limits acoustic monitoring scalability
Acoustic indices offer potential automation but need validation
Previous work (Alcocer et al.) identified temporal pseudoreplication issues
This approach: proper temporal structure + ground-truthed annotations
Methods
Study system:
3 estuarine stations, year-long deployment, 2-hour resolution
Manual annotations: fish calls, dolphin calls, vessel detections
Index reduction:
60 indices → ~10 via correlation and VIF analysis
Ensures predictors represent distinct soundscape features
Statistical approach:
GLMMs and GAMMs with nested random effects
AR1 temporal autocorrelation structure
Model selection via AIC and cross-validation
[Specify chosen CV approach(es)]
Results
Model selection:
Present whether GLMM or GAMM was better for each metric (with justification)
AIC and CV performance comparison
Predictive indices by metric:
Fish activity predicted by: [indices, effects, CIs]
Fish richness predicted by: [...]
[etc. for all 9 metrics]
Predictive performance:
CV results showing RMSE/R²/AUC for each metric
Overall: indices predict biological patterns with X accuracy
Discuss where prediction is strong vs weak (e.g., seasonal limitations)
Spatial consistency:
Relationships hold across stations OR some station-specific variation
Discussion
What works:
Specific indices reliably predict specific biological patterns
E.g., "ACI predicts fish activity (R² = 0.6), making it suitable for rapid assessment"
Why it makes sense:
Connect index properties to biological/acoustic mechanisms
E.g., "ACI captures broadband energy typical of fish choruses"
How to apply:
Recommend index combinations for monitoring programs
"To track fish activity and vessel presence, monitor these 3-5 indices"
Acknowledge need for seasonal training data (or spatial calibration, depending on CV results)
Limitations:
Models may require training from all seasons of interest (if using temporal CV)
Some station-specific calibration may be needed (if spatial random effects are large)
Environmental context (temperature, depth) also important
Advances over previous work:
Properly addresses temporal autocorrelation (Alcocer critique)
Ground-truthed with extensive manual annotations
Multi-station, year-long validation

Key Messages for Publication
Acoustic indices CAN predict biological patterns when analyzed with appropriate statistical structure (addressing temporal autocorrelation)


Different indices predict different components of the biological soundscape (fish vs dolphins vs vessels require different acoustic features)


Practical application: Monitoring programs can use a small subset of indices (3-5) to assess biological activity without extensive manual annotation


Honest limitations: [Based on CV results - may include seasonal training requirements, spatial calibration needs, or other constraints]


Methodological contribution: Demonstrates proper approach to index validation using nested random effects and temporal structure, directly addressing critiques in recent literature




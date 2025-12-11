# Stage 05 Modeling Review — For Annotation

## Overall Assessment: Solid Foundation 👍

You've made good choices on the fundamentals:
- **Negative binomial for counts** — correct, ecological count data is almost always overdispersed
- **Binomial for presence/absence** — appropriate
- **Random effects for station and month** — accounts for spatial and temporal non-independence
- **AR1 autocorrelation** — handles the fact that consecutive 2-hour bins aren't independent
- **Predictor scaling** — essential for model convergence and coefficient comparability

MW:

---

## Things Worth Discussing

### 1. AIC Comparison Between GLMM and GAMM

You're comparing AIC between `glmmTMB` (GLMM) and `mgcv::bam` (GAMM). This is a bit nuanced:

- **Different likelihood calculations**: glmmTMB uses ML (you set `REML = FALSE`), but bam uses `method = "fREML"` (fast REML). REML and ML likelihoods aren't directly comparable via AIC.
- **Recommendation**: Either set `method = "ML"` in bam(), or acknowledge in your write-up that this comparison is approximate.

That said, with ΔAIC = 485, the difference is so large it probably doesn't change your conclusion.

MW: I really would rather have them directly comparable... but is there a downside to switching the bam to use ML?

---

### 2. The AR1 Structure in GLMM vs GAMM

Your GLMM has explicit AR1 autocorrelation:
```r
ar1(time_within_day + 0 | day_id)
```

But your GAMM doesn't have an equivalent autocorrelation structure — `bam()` with `discrete = TRUE` doesn't model temporal autocorrelation the same way.

This is a **structural difference** between the models, not just "linear vs smooth." The GAMM might be winning partly because the smooth terms are soaking up variance that the GLMM attributes to autocorrelation.

**Options to consider:**
- Accept this as a known limitation (fine for a learning project)
- Use `bam(..., rho = X)` to add AR1 to the GAMM (requires estimating rho)
- Mention in your write-up that the models differ in autocorrelation structure

MW: hm. my goal is to follow best practices and to align with approaches used in other papers in this field. I do not want to do anything new and groundbreaking in terms of analysis. I will say that we initially started with GLMM and then collaborators noted that many ecological papers are moving to gamms. we are implementing both here but if there is a "best practice" for gamms, like if we simply just went with gamms directly, what would that be? would you typically include somthing like the AR1 term to account for temporal autocorrelation? 

---

### 3. Cyclic Spline Boundaries

For `s(hour_of_day, bs='cc', k=12)` — cyclic splines need the data to span the full cycle. Your `hour_of_day` should range from 0 to 23 (or close to it). If you have gaps (e.g., no data at certain hours), the cyclic spline might behave unexpectedly.

Same for `s(day_of_year, bs='cc', k=12)` — check that you have reasonable coverage across the year.

MW: in earlier steps we have confirmed that we have good coverage across the year. I think we're alright here. 

---

### 4. Random Effects in GAMM

You're using:
```r
s(station, bs='re')
s(month_id, bs='re')
```

This is correct for random intercepts. Just note that `station` needs to be a **factor** for this to work — which you've done correctly in your data prep.

MW: ok, no action needed here. 

---

### 5. Small Detail: The `%||%` Operator

Your comment on line 350 is spot on:
```r
# MW: Should we even have "backup" values here?
```

I agree — for a reproducible analysis, failing loudly on missing config is better than silent fallbacks. Your instinct is right.

MW:

---

## Things You Did Well

1. **Excellent documentation** — the comments explaining AR1, family choices, EDF interpretation, etc. are genuinely helpful

2. **Clean slate approach** — deleting old outputs before each run prevents confusion

3. **DHARMa diagnostics** — this is the right way to check GLMM residuals (standard residual plots don't work well for GLMMs) 

4. **Scaling parameters saved** — you can back-transform coefficients if needed

5. **Run history tracking** — great for reproducibility

MW:

---

## Questions for You to Consider

### Q1: Do you expect linear or non-linear relationships?

If you have ecological reason to expect smooth, monotonic relationships (e.g., "more X → more fish activity"), the GLMM might be more interpretable even if GAMM fits better statistically.

MW: colleagues have indicated gamms are increasingly preferred for this type of study.

---

### Q2: What will you do with the GAMM if it's selected?

The smooth plots are harder to summarize in a paper than "coefficient ± SE, p-value." Have you thought about how you'll report these?

MW: nope - i'm totally new to this and not sure how it's normally reported.

---

### Q3: Have you looked at the smooth plots yet?

They'll tell you whether the non-linearity is ecologically meaningful or just fitting noise.

MW: not yet. 

---

## Your Additional Notes / Questions

(Add anything else here)

MW:

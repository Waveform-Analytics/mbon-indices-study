# ==============================================================================
# Stage 05a: GAMM Modeling
# ==============================================================================
#
# Purpose:
#   Fit Generalized Additive Mixed Models (GAMMs) for each response metric to
#   predict biological community metrics from acoustic indices. GAMMs capture
#   non-linear relationships that initial modeling showed exist between indices
#   and responses.
#
# Inputs:
#   - data/processed/analysis_ready.parquet
#   - data/processed/indices_final.csv
#   - config/analysis.yml
#
# Outputs (per response):
#   - results/models/<metric>/gamm.rds
#   - results/tables/<metric>/gamm_summary.csv
#   - results/tables/<metric>/scaling_params.csv
#   - results/figures/<metric>/gamm_diagnostics.png
#   - results/figures/<metric>/gamm_smooths.png (overview grid)
#   - results/figures/<metric>/smooth_<term>.png (individual smooth plots)
#
# Summary outputs:
#   - results/tables/model_summary.csv
#   - results/logs/modeling_summary.json
#
# Usage:
#   Rscript scripts/stage05_modeling.R [--pilot]
#
#   --pilot: Run only fish_activity (default behavior for now)
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SETUP: Load packages
# ------------------------------------------------------------------------------

# We suppress startup messages for cleaner output
suppressPackageStartupMessages({
  library(arrow)      # Read parquet files
  library(yaml)       # Read config files
  library(dplyr)      # Data manipulation
  library(tidyr)      # Data reshaping
  library(mgcv)       # Fit GAMMs
  library(ggplot2)    # Plotting
  library(jsonlite)   # Write JSON logs
})

# Set a seed for reproducibility
set.seed(1234)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Read the analysis config
config <- yaml::read_yaml("config/analysis.yml")

# Get GAMM settings from config
smooth_k <- config$gamm$smooth_k %||% 5
cyclic_k <- config$gamm$cyclic_k %||% 12
cat(sprintf("GAMM smooth_k: %d (from config)\n", smooth_k))

# Define which responses to model and their distribution families
# Family determines the likelihood function:
#   - nbinom2: Negative binomial (for overdispersed counts)
#   - binomial: For binary (0/1) presence/absence data
responses <- list(
  fish_activity = list(family = "nbinom2", type = "count"),
  fish_richness = list(family = "nbinom2", type = "count"),
  fish_presence = list(family = "binomial", type = "binary"),
  dolphin_burst_pulse = list(family = "nbinom2", type = "count"),
  dolphin_echolocation = list(family = "nbinom2", type = "count"),
  dolphin_whistle = list(family = "nbinom2", type = "count"),
  dolphin_activity = list(family = "nbinom2", type = "count"),
  dolphin_presence = list(family = "binomial", type = "binary"),
  vessel_presence = list(family = "binomial", type = "binary")
)

# Check if we're in pilot mode (run subset of responses for testing)
pilot_mode <- config$run$pilot_mode %||% FALSE
pilot_responses <- config$run$pilot_responses %||% c("fish_activity")

if (pilot_mode) {
  responses <- responses[names(responses) %in% pilot_responses]
  cat(sprintf("Running in PILOT MODE: %s only\n", paste(pilot_responses, collapse = ", ")))
} else {
  cat(sprintf("Running FULL MODE: all %d responses\n", length(responses)))
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

#' Clean output directories for a specific metric
#'
#' We use a "clean slate" approach: delete old outputs before each run to avoid
#' confusion from stale results. This is safe because:
#' 1. We can always re-run the script
#' 2. Git history preserves our code
#' 3. Results are derived from data (reproducible)
#'
#' @param metric Character string, e.g., "fish_activity"
clean_metric_outputs <- function(metric) {
  dirs <- c(
    file.path("results/models", metric),
    file.path("results/tables", metric),
    file.path("results/figures", metric)
  )

  for (dir in dirs) {
    if (dir.exists(dir)) {
      unlink(dir, recursive = TRUE)
      cat(sprintf("  Cleaned: %s\n", dir))
    }
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
}

#' Ensure output directories exist
ensure_dirs <- function() {
  dirs <- c(
    "results/models",
    "results/tables",
    "results/figures",
    "results/logs"
  )
  for (dir in dirs) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  }
}

#' Scale predictors using z-score standardization
#'
#' Transforms each predictor to mean=0, SD=1. This is critical for:
#' 1. Numerical stability: Optimizers struggle with parameters on vastly different scales
#' 2. Interpretability: Coefficients become "effect per 1-SD change", directly comparable
#'
#' @param data Data frame containing the predictors
#' @param predictors Character vector of column names to scale
#' @return List with:
#'   - data: Data frame with scaled predictors (original columns replaced)
#'   - params: Data frame with mean and sd for each predictor (for back-transformation)
scale_predictors <- function(data, predictors) {
  params <- data.frame(
    predictor = predictors,
    mean = NA_real_,
    sd = NA_real_
  )

  for (i in seq_along(predictors)) {
    col <- predictors[i]
    col_mean <- mean(data[[col]], na.rm = TRUE)
    col_sd <- sd(data[[col]], na.rm = TRUE)

    # Store parameters
    params$mean[i] <- col_mean
    params$sd[i] <- col_sd

    # Scale the column
    data[[col]] <- (data[[col]] - col_mean) / col_sd
  }

  list(data = data, params = params)
}

#' Build the GAMM formula
#'
#' The GAMM formula uses smooth terms to capture non-linear relationships:
#'
#' response ~ s(index1, k=K) + s(index2, k=K) + ... +
#'            s(temperature, k=K) + s(depth, k=K) +
#'            s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) +
#'            s(station, bs="re") + s(month_id, bs="re")
#'
#' Key components:
#' - s(x, k=K): Smooth function of x with up to ~K-1 degrees of wiggliness
#'   If the true relationship is linear, the smooth will estimate a line
#' - bs="cc": Cyclic cubic spline (wraps around, so hour 23 connects to hour 0)
#' - bs="re": Random effect smooth (equivalent to random intercept)
#'
#' Note: We use bam() instead of gam() for speed on larger datasets
#'
#' @param response Character string, the response variable name
#' @param indices Character vector of index column names
#' @param smooth_k Integer, basis dimension for smooth terms (from config)
#' @param cyclic_k Integer, basis dimension for cyclic terms (from config)
#' @return A formula object
build_gamm_formula <- function(response, indices, smooth_k = 5, cyclic_k = 12) {
  # Smooth terms for indices
  index_terms <- sapply(indices, function(idx) {
    sprintf("s(%s, k=%d)", idx, smooth_k)
  })


  # Smooth terms for covariates
  covariate_terms <- c(
    sprintf("s(temperature, k=%d)", smooth_k),
    sprintf("s(depth, k=%d)", smooth_k)
  )

  # Cyclic smooths for temporal terms
  # bs="cc" means cyclic cubic spline - the curve wraps around
  temporal_terms <- c(
    sprintf("s(hour_of_day, bs='cc', k=%d)", cyclic_k),
    sprintf("s(day_of_year, bs='cc', k=%d)", cyclic_k)
  )

  # Random effects as smooth terms
  # bs="re" is equivalent to (1|x) in mixed model notation
  random_terms <- c(
    "s(station, bs='re')",
    "s(month_id, bs='re')"
  )

  # Combine all terms
  all_terms <- c(index_terms, covariate_terms, temporal_terms, random_terms)
  formula_str <- sprintf("%s ~ %s", response, paste(all_terms, collapse = " + "))

  as.formula(formula_str)
}

#' Get the mgcv family object
#'
#' Different response types need different distribution families:
#'
#' - nb: Negative binomial (for overdispersed counts)
#'   Used for count data where variance > mean
#'   Ecological count data is almost always overdispersed!
#'
#' - binomial: For binary (0/1) data
#'   Models the log-odds of the event occurring
#'
#' @param family_name Character string, either "nbinom2" or "binomial"
#' @return A family object compatible with mgcv::bam
get_gam_family <- function(family_name) {
  switch(family_name,
    "nbinom2" = mgcv::nb(),  # mgcv uses nb() for negative binomial
    "binomial" = binomial(),
    stop(sprintf("Unknown family: %s", family_name))
  )
}

#' Estimate AR1 rho parameter from preliminary model residuals
#'
#' Rather than using an arbitrary fixed value for rho, we estimate it from data:
#' 1. Fit a preliminary model without AR1 correlation (rho = 0)
#' 2. Extract deviance residuals
#' 3. Compute lag-1 autocorrelation (ACF)
#'
#' This gives a data-driven estimate of temporal autocorrelation strength.
#'
#' @param formula The model formula
#' @param data The model data
#' @param family The distribution family
#' @return List with:
#'   - rho: Estimated AR1 correlation (0 to 1, clamped)
#'   - preliminary_fit: The preliminary model object (can be discarded)
estimate_rho <- function(formula, data, family) {
  cat("  Step 1: Fitting preliminary model (rho=0) to estimate AR1 correlation...\n")

  # Fit preliminary model without AR1
  preliminary_fit <- tryCatch({
    bam(
      formula = formula,
      data = data,
      family = family,
      method = "fREML",
      discrete = TRUE,
      select = TRUE,
      rho = 0  # No AR1 in preliminary fit
    )
  }, error = function(e) {
    cat(sprintf("    WARNING: Preliminary fit failed: %s\n", e$message))
    return(NULL)
  })

  if (is.null(preliminary_fit)) {
    cat("    Using default rho = 0.5 due to preliminary fit failure\n")
    return(list(rho = 0.5, preliminary_fit = NULL))
  }

  # Extract deviance residuals
  resids <- residuals(preliminary_fit, type = "deviance")

  # Compute lag-1 autocorrelation
  # acf() returns correlations at lags 0, 1, 2, ...
  # We want lag 1, which is the second element (index 2)
  acf_result <- acf(resids, lag.max = 1, plot = FALSE)
  rho_raw <- acf_result$acf[2]  # Lag-1 correlation

  # Clamp rho to valid range [0, 1)
  # Negative rho is theoretically possible but rare and often indicates model issues
  # Values >= 1 would cause numerical problems
  rho_estimated <- max(0, min(rho_raw, 0.99))

  cat(sprintf("    Lag-1 ACF of residuals: %.3f\n", rho_raw))
  cat(sprintf("    Using rho = %.3f for final model\n", rho_estimated))

  list(rho = rho_estimated, preliminary_fit = preliminary_fit)
}

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("\n=== Loading Data ===\n")

# Read the analysis-ready dataset
# This contains:
# - Keys: datetime, datetime_local, date, station
# - Temporal: hour_of_day, day_of_year
# - Grouping: month_id
# - Predictors: acoustic indices
# - Covariates: temperature, depth
# - Responses: 9 community metrics
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded %d observations\n", nrow(data)))

# Read the list of final indices (after VIF pruning in Stage 01)
indices_df <- read.csv("data/processed/indices_final.csv")
indices <- indices_df$index_name[indices_df$kept == "True"]
cat(sprintf("  Using %d acoustic indices as predictors\n", length(indices)))

# Convert grouping variables to factors (required for random effects)
# Factors tell R these are categorical, not continuous
data <- data %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  )

# ------------------------------------------------------------------------------
# SCALING CONFIGURATION
# ------------------------------------------------------------------------------

# Get scaling settings from config
# MW: Should we even have "backup" values here? seems like we should just throw an error if the config is incomplete. 
scaling_enabled <- config$scaling$enabled %||% TRUE
scaling_include <- config$scaling$include %||% c("indices", "covariates")
scaling_exclude <- config$scaling$exclude %||% c("sin_hour", "cos_hour")

# Build list of predictors to scale based on config
predictors_to_scale <- c()
if ("indices" %in% scaling_include) {
  predictors_to_scale <- c(predictors_to_scale, indices)
}
if ("covariates" %in% scaling_include) {
  predictors_to_scale <- c(predictors_to_scale, "temperature", "depth")
}
# Remove any excluded predictors
predictors_to_scale <- setdiff(predictors_to_scale, scaling_exclude)

cat(sprintf("  Scaling enabled: %s\n", scaling_enabled))
cat(sprintf("  Predictors to scale: %d (%s)\n",
            length(predictors_to_scale),
            paste(predictors_to_scale, collapse = ", ")))

# Check for missing data in predictors
# GAMMs will fail if there are NAs in the model matrix
missing_check <- data %>%
  select(all_of(c(indices, "temperature", "depth"))) %>%
  summarise(across(everything(), ~sum(is.na(.))))

if (any(missing_check > 0)) {
  cat("  WARNING: Missing values detected in predictors:\n")
  print(missing_check[, colSums(missing_check) > 0])
  cat("  Rows with any missing predictor will be dropped during model fitting.\n")
}

# ------------------------------------------------------------------------------
# MAIN MODELING LOOP
# ------------------------------------------------------------------------------

cat("\n=== Starting Model Fitting ===\n")

# Ensure output directories exist
ensure_dirs()

# Store results for summary
all_results <- list()

for (metric in names(responses)) {
  cat(sprintf("\n--- Modeling: %s ---\n", metric))

  metric_info <- responses[[metric]]

  # Clean previous outputs for this metric
  cat("Cleaning previous outputs...\n")
  clean_metric_outputs(metric)

  # Prepare data for this metric
  # Drop rows where the response is NA
  # MW: note for ppl new to R like me! model_data is supposed to include both the responses and the predictors
  model_data <- data %>%
    filter(!is.na(.data[[metric]]))

  cat(sprintf("  Using %d observations (after dropping NA responses)\n",
              nrow(model_data)))

  # Scale predictors if enabled in config
  # Coefficients will represent "effect per 1-SD change"
  if (scaling_enabled && length(predictors_to_scale) > 0) {
    cat("  Scaling predictors (z-score standardization)...\n")
    scaled_result <- scale_predictors(model_data, predictors_to_scale)
    model_data <- scaled_result$data
    scaling_params <- scaled_result$params

    # Save scaling parameters for back-transformation if needed
    write.csv(scaling_params,
              file.path("results/tables", metric, "scaling_params.csv"),
              row.names = FALSE)
    cat("  Saved: results/tables/", metric, "/scaling_params.csv\n", sep = "")
  }

  # --------------------------------------------------------------------------
  # FIT GAMM (two-step: estimate rho, then fit final model)
  # --------------------------------------------------------------------------

  cat("\nFitting GAMM...\n")

  # Build formula
  gamm_formula <- build_gamm_formula(metric, indices, smooth_k, cyclic_k)
  cat(sprintf("  Formula: %s\n", deparse(gamm_formula, width.cutoff = 500)))

  # Get the appropriate family
  gamm_family <- get_gam_family(metric_info$family)

  gamm_start <- Sys.time()

  # Step 1: Estimate rho from preliminary model residuals
  rho_result <- estimate_rho(gamm_formula, model_data, gamm_family)
  rho_estimated <- rho_result$rho

  # Step 2: Fit final model with estimated rho
  # bam() is optimized for large datasets
  # select=TRUE enables automatic smoothness selection (shrinks unneeded wiggles)
  cat("  Step 2: Fitting final model with estimated rho...\n")

  gamm_fit <- tryCatch({
    bam(
      formula = gamm_formula,
      data = model_data,
      family = gamm_family,
      method = "fREML",   # Fast restricted maximum likelihood
      discrete = TRUE,    # Discretization for speed
      select = TRUE,      # Shrinkage selection (penalizes unnecessary complexity)
      rho = rho_estimated # Data-driven AR1 correlation
    )
  }, error = function(e) {
    cat(sprintf("  ERROR fitting GAMM: %s\n", e$message))
    NULL
  })

  gamm_time <- difftime(Sys.time(), gamm_start, units = "mins")

  if (!is.null(gamm_fit)) {
    cat(sprintf("  GAMM fitted in %.2f minutes\n", as.numeric(gamm_time)))

    # Save the model object
    saveRDS(gamm_fit, file.path("results/models", metric, "gamm.rds"))
    cat("  Saved: results/models/", metric, "/gamm.rds\n", sep = "")

    # Extract and save smooth term summary
    # EDF (effective degrees of freedom) tells us how non-linear each term is:
    # - EDF ≈ 1: essentially linear
    # - EDF > 1: increasingly non-linear
    gamm_summary <- as.data.frame(summary(gamm_fit)$s.table)
    gamm_summary$term <- rownames(gamm_summary)

    # Column names vary by family: Chi.sq for some, F for others (e.g., nb)
    # We'll rename whatever statistic column exists to "statistic"
    stat_col <- intersect(c("Chi.sq", "F"), names(gamm_summary))
    if (length(stat_col) > 0) {
      names(gamm_summary)[names(gamm_summary) == stat_col[1]] <- "statistic"
    }

    gamm_summary <- gamm_summary %>%
      select(term, everything()) %>%
      rename(
        edf = edf,
        ref_df = Ref.df,
        p_value = `p-value`
      )

    write.csv(gamm_summary,
              file.path("results/tables", metric, "gamm_summary.csv"),
              row.names = FALSE)
    cat("  Saved: results/tables/", metric, "/gamm_summary.csv\n", sep = "")

    # Generate smooth plots
    cat("  Generating GAMM smooth plots...\n")

    # Main overview plot (all smooths in a grid, no repeated title)
    png(file.path("results/figures", metric, "gamm_smooths.png"),
        width = 2000, height = 1500, res = 120)
    plot(gamm_fit, pages = 1, all.terms = FALSE, shade = TRUE)
    dev.off()
    cat("  Saved: results/figures/", metric, "/gamm_smooths.png\n", sep = "")

    # Generate individual smooth plots for key terms
    # Extract smooth term names from the model
    smooth_terms <- sapply(gamm_fit$smooth, function(s) s$label)

    for (i in seq_along(smooth_terms)) {
      term_label <- smooth_terms[i]
      # Create a clean filename from the term label
      # e.g., "s(hour_of_day)" -> "smooth_hour_of_day.png"
      term_name <- gsub("s\\(|\\)|,.*", "", term_label)  # Remove s(), ), and anything after comma
      filename <- paste0("smooth_", term_name, ".png")

      png(file.path("results/figures", metric, filename),
          width = 800, height = 600, res = 120)
      plot(gamm_fit, select = i, shade = TRUE, main = term_label)
      dev.off()
    }
    cat("  Saved:", length(smooth_terms), "individual smooth plots\n")

    # Generate standard diagnostic plot (4-panel: QQ, histogram, residuals vs fitted, response vs fitted)
    png(file.path("results/figures", metric, "gamm_diagnostics.png"),
        width = 1200, height = 1000, res = 120)
    par(mfrow = c(2, 2))
    gam.check(gamm_fit, pch = 20, cex = 0.5)
    dev.off()
    cat("  Saved: results/figures/", metric, "/gamm_diagnostics.png\n", sep = "")

    # Get AIC
    gamm_aic <- AIC(gamm_fit)
    cat(sprintf("  GAMM AIC: %.2f\n", gamm_aic))

  } else {
    gamm_aic <- NA
    gamm_time <- NA
    rho_estimated <- NA
  }

  # Store results for summary (including estimated rho for transparency)
  all_results[[metric]] <- data.frame(
    metric = metric,
    converged = !is.null(gamm_fit),
    rho = rho_estimated,
    aic = gamm_aic,
    time_mins = as.numeric(gamm_time)
  )
}

# ------------------------------------------------------------------------------
# GENERATE SUMMARY OUTPUTS
# ------------------------------------------------------------------------------

cat("\n=== Generating Summary Outputs ===\n")

# Combine all results into summary table
summary_df <- bind_rows(all_results)
write.csv(summary_df, "results/tables/model_summary.csv", row.names = FALSE)
cat("Saved: results/tables/model_summary.csv\n")

# Generate JSON log with metadata
log_data <- list(
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  pilot_mode = pilot_mode,
  scaling_enabled = scaling_enabled,
  predictors_scaled = predictors_to_scale,
  n_responses_modeled = length(responses),
  responses_modeled = names(responses),
  n_indices = length(indices),
  indices_used = indices,
  n_observations = nrow(data),
  results_summary = summary_df
)

write_json(log_data, "results/logs/modeling_summary.json", pretty = TRUE)
cat("Saved: results/logs/modeling_summary.json\n")

# ------------------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------------------

cat("\n=== Stage 05a Modeling Complete ===\n\n")

cat("Summary:\n")
print(summary_df %>% select(metric, converged, rho, aic, time_mins))

cat("\nNext steps:\n")
cat("1. Review smooth plots in results/figures/<metric>/\n")
cat("2. Check EDF values in results/tables/<metric>/gamm_summary.csv\n")
cat("3. Run `quarto render results/results_summary.qmd` to generate slides\n")

# ------------------------------------------------------------------------------
# APPEND TO RUN HISTORY
# ------------------------------------------------------------------------------

# Build a concise summary for each modeled response
model_summaries <- sapply(names(all_results), function(m) {
  res <- all_results[[m]]

  if (res$converged) {
    sprintf("%s: converged (rho=%.2f, AIC=%.1f, %.1fmin)", m, res$rho, res$aic, res$time_mins)
  } else {
    sprintf("%s: FAILED", m)
  }
})

# Create the run history entry
run_entry <- sprintf(
  "## %s — Stage 05a: GAMM Modeling

- **Config**:
  - pilot_mode: %s
  - scaling_enabled: %s
  - n_responses: %d
  - n_indices: %d
- **Results**:
%s
- **Log**: results/logs/modeling_summary.json
- **Notes**:

---

",
  format(Sys.time(), "%Y-%m-%d %H:%M"),
  ifelse(pilot_mode, "TRUE", "FALSE"),
  ifelse(scaling_enabled, "TRUE", "FALSE"),
  length(responses),
  length(indices),
  paste("  -", model_summaries, collapse = "\n")
)

# Append to RUN_HISTORY.md
history_path <- "results/logs/RUN_HISTORY.md"
cat(run_entry, file = history_path, append = TRUE)
cat(sprintf("Appended to run history: %s\n", history_path))

cat("\n")

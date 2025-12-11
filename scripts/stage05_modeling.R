# ==============================================================================
# Stage 05: GLMM and GAMM Modeling with AIC-based Model Selection
# ==============================================================================
#
# Purpose:
#   Fit both GLMM and GAMM for each response metric, compare via AIC, and select
#   the better-fitting model. Goal is inference (understanding relationships
#   between acoustic indices and community metrics). May expand validation
#   later.
#
# Inputs:
#   - data/processed/analysis_ready.parquet
#   - data/processed/indices_final.csv
#   - config/analysis.yml
#
# Outputs (per response):
#   - results/models/<metric>/glmm.rds
#   - results/models/<metric>/gamm.rds
#   - results/tables/<metric>/glmm_summary.csv
#   - results/tables/<metric>/gamm_summary.csv
#   - results/tables/<metric>/model_comparison.csv
#   - results/tables/<metric>/scaling_params.csv
#   - results/figures/<metric>/glmm_diagnostics.png
#   - results/figures/<metric>/gamm_smooths.png (overview grid)
#   - results/figures/<metric>/smooth_<term>.png (individual smooth plots)
#
# Summary outputs:
#   - results/tables/model_selection_summary.csv
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
  library(glmmTMB)    # Fit GLMMs with AR1 correlation
  library(mgcv)       # Fit GAMMs
  library(DHARMa)     # GLMM diagnostics via simulation
  library(ggplot2)    # Plotting
  library(jsonlite)   # Write JSON logs
})

# Set a seed for reproducibility (affects DHARMa simulations)
set.seed(1234)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Read the analysis config
config <- yaml::read_yaml("config/analysis.yml")

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

# For pilot mode, we only run fish_activity
# TODO: Add command-line argument parsing to control this
pilot_mode <- TRUE
if (pilot_mode) {
  responses <- responses["fish_activity"]
  cat("Running in PILOT MODE: fish_activity only\n")
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

#' Build the GLMM formula
#'
#' The GLMM formula has these components:
#'
#' response ~ indices + covariates + sin_hour + cos_hour +
#'            (1|station) + (1|month_id) + ar1(time_within_day + 0 | day_id)
#'
#' Breaking this down:
#' - indices: The acoustic indices (our predictors of interest)
#' - covariates: temperature + depth (environmental controls)
#' - sin_hour + cos_hour: Cyclic encoding of time of day (captures diel patterns)
#' - (1|station): Random intercept for station (accounts for site differences)
#' - (1|month_id): Random intercept for month (accounts for seasonal baselines)
#' - ar1(...): First-order autoregressive correlation within each day
#'
#' The AR1 term handles temporal autocorrelation: observations close in time
#' within the same day are correlated. At midnight, correlation "resets" because
#' each day is treated independently. This is OK because:
#' - The cyclic terms (sin_hour, cos_hour) handle the wrap-around diel pattern
#' - AR1 models short-term environmental continuity, not the diel cycle itself
#'
#' @param response Character string, the response variable name
#' @param indices Character vector of index column names
#' @param include_ar1 Logical, whether to include AR1 term
#' @return A formula object
build_glmm_formula <- function(response, indices, include_ar1 = TRUE) {
  # Fixed effects: all indices + covariates + cyclic time terms
  # (hard-coded bc they're the same for every model)
  fixed_terms <- c(
    indices,
    "temperature", "depth",
    "sin_hour", "cos_hour"
  )
  fixed_part <- paste(fixed_terms, collapse = " + ") # like "+".join(list_of_strings)


  # Random effects
  # (1|station): Different stations have different baseline levels
  # (1|month_id): Different months have different baseline levels
  random_part <- "(1|station) + (1|month_id)"

  # AR1 autocorrelation structure (optional)
  # ar1(time_within_day + 0 | day_id) means:
  #   - Within each day_id group, observations are AR1 correlated
  #   - time_within_day gives the ordering (0, 1, 2, ... for 2-hour bins)
  #   - The "+ 0" removes the random intercept (we only want the correlation)
  if (include_ar1) {
    ar1_part <- " + ar1(time_within_day + 0 | day_id)"
  } else {
    ar1_part <- ""
  }

  # Combine into full formula
  formula_str <- sprintf("%s ~ %s + %s%s",
                         response, fixed_part, random_part, ar1_part)

  as.formula(formula_str)
}

#' Build the GAMM formula
#'
#' The GAMM formula uses smooth terms instead of linear terms:
#'
#' response ~ s(index1, k=5) + s(index2, k=5) + ... +
#'            s(temperature, k=5) + s(depth, k=5) +
#'            s(hour_of_day, bs="cc", k=12) + s(day_of_year, bs="cc", k=12) +
#'            s(station, bs="re") + s(month_id, bs="re")
#'
#' Key differences from GLMM:
#' - s(x, k=5): Smooth function of x with up to ~4 degrees of wiggliness
#'   If the true relationship is linear, the smooth will estimate a line
#' - bs="cc": Cyclic cubic spline (wraps around, so hour 23 connects to hour 0)
#' - bs="re": Random effect smooth (equivalent to random intercept in GLMM)
#'
#' Note: We use bam() instead of gam() for speed on larger datasets
#'
#' @param response Character string, the response variable name
#' @param indices Character vector of index column names
#' @return A formula object
build_gamm_formula <- function(response, indices) {
  # Smooth terms for indices (k=5 allows moderate non-linearity)
  index_terms <- sapply(indices, function(idx) {
    sprintf("s(%s, k=5)", idx)
  })


  # Smooth terms for covariates
  covariate_terms <- c(
    "s(temperature, k=5)",
    "s(depth, k=5)"
  )

  # Cyclic smooths for temporal terms
  # bs="cc" means cyclic cubic spline - the curve wraps around
  # k=12 allows more flexibility for diel/seasonal patterns
  temporal_terms <- c(
    "s(hour_of_day, bs='cc', k=12)",
    "s(day_of_year, bs='cc', k=12)"
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

#' Get the glmmTMB family object
#'
#' Different response types need different distribution families:
#'
#' - nbinom2: Negative binomial with quadratic variance (var = mu + mu^2/k)
#'   Used for count data that is overdispersed (variance > mean)
#'   Ecological count data is almost always overdispersed!
#'
#' - binomial: For binary (0/1) data
#'   Models the log-odds of the event occurring
#'
#' @param family_name Character string, either "nbinom2" or "binomial"
#' @return A glmmTMB family object
get_glmm_family <- function(family_name) {
  switch(family_name,
    "nbinom2" = glmmTMB::nbinom2(),
    "binomial" = binomial(),
    stop(sprintf("Unknown family: %s", family_name))
  )
}

#' Get the mgcv family object
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

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("\n=== Loading Data ===\n")

# Read the analysis-ready dataset
# This contains:
# - Keys: datetime, datetime_local, date, station
# - Temporal: hour_of_day, sin_hour, cos_hour, day_of_year
# - Grouping: day_id, month_id
# - Sequence: time_within_day (for AR1)
# - Predictors: acoustic indices
# - Covariates: temperature, depth
# - Responses: 9 community metrics
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded %d observations\n", nrow(data)))

# Read the list of final indices (after correlation/VIF pruning in Stage 01)
indices_df <- read.csv("data/processed/indices_final.csv")
indices <- indices_df$index_name[indices_df$kept == "True"]
cat(sprintf("  Using %d acoustic indices as predictors\n", length(indices)))

# Convert grouping variables to factors (required for random effects)
# Factors tell R these are categorical, not continuous
# time_within_day must also be a factor for glmmTMB's ar1() structure
data <- data %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id),
    day_id = as.factor(day_id),
    time_within_day = as.factor(time_within_day)
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
# GLMMs will fail if there are NAs in the model matrix
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
  # FIT GLMM
  # --------------------------------------------------------------------------

  cat("\nFitting GLMM...\n")

  # Build formula
  glmm_formula <- build_glmm_formula(metric, indices, include_ar1 = TRUE)
  cat(sprintf("  Formula: %s\n", deparse(glmm_formula, width.cutoff = 500)))

  # Get the appropriate family
  glmm_family <- get_glmm_family(metric_info$family)
  cat(sprintf("  Family: %s\n", metric_info$family))

  # Fit the model
  # glmmTMB is used because it supports:
  # - Negative binomial distributions
  # - Complex random effect structures
  # - AR1 autocorrelation
  glmm_start <- Sys.time()

  glmm_fit <- tryCatch({
    glmmTMB(
      formula = glmm_formula,
      data = model_data,
      family = glmm_family,
      # REML = FALSE for AIC comparison (must use ML, not REML)
      REML = FALSE,
      # Control settings for convergence
      control = glmmTMBControl(
        optimizer = optim,
        optArgs = list(method = "BFGS")
      )
    )
  }, error = function(e) {
    cat(sprintf("  ERROR fitting GLMM: %s\n", e$message))
    NULL
  })

  glmm_time <- difftime(Sys.time(), glmm_start, units = "mins")

  if (!is.null(glmm_fit)) {
    cat(sprintf("  GLMM fitted in %.2f minutes\n", as.numeric(glmm_time)))

    # Check for convergence warnings
    if (length(glmm_fit$fit$message) > 0) {
      cat(sprintf("  Convergence message: %s\n", glmm_fit$fit$message))
    }

    # Save the model object
    saveRDS(glmm_fit, file.path("results/models", metric, "glmm.rds"))
    cat("  Saved: results/models/", metric, "/glmm.rds\n", sep = "")

    # Extract and save fixed effects summary
    glmm_summary <- as.data.frame(summary(glmm_fit)$coefficients$cond)
    glmm_summary$term <- rownames(glmm_summary)
    glmm_summary <- glmm_summary %>%
      select(term, everything()) %>%
      rename(
        estimate = Estimate,
        std_error = `Std. Error`,
        z_value = `z value`,
        p_value = `Pr(>|z|)`
      )

    write.csv(glmm_summary,
              file.path("results/tables", metric, "glmm_summary.csv"),
              row.names = FALSE)
    cat("  Saved: results/tables/", metric, "/glmm_summary.csv\n", sep = "")

    # Generate DHARMa diagnostics
    # DHARMa uses simulation to create "standardized residuals" that should
    # look uniform if the model is correct
    cat("  Generating GLMM diagnostics...\n")

    dharma_res <- tryCatch({
      simulateResiduals(glmm_fit, n = 250, refit = FALSE)
    }, error = function(e) {
      cat(sprintf("  WARNING: DHARMa simulation failed: %s\n", e$message))
      NULL
    })

    if (!is.null(dharma_res)) {
      png(file.path("results/figures", metric, "glmm_diagnostics.png"),
          width = 1200, height = 800, res = 120)
      plot(dharma_res, main = paste(metric, "- GLMM Diagnostics"))
      dev.off()
      cat("  Saved: results/figures/", metric, "/glmm_diagnostics.png\n", sep = "")
    }

    # Get AIC
    glmm_aic <- AIC(glmm_fit)
    cat(sprintf("  GLMM AIC: %.2f\n", glmm_aic))

  } else {
    glmm_aic <- NA
    glmm_time <- NA
  }

  # --------------------------------------------------------------------------
  # FIT GAMM
  # --------------------------------------------------------------------------

  cat("\nFitting GAMM...\n")

  # Build formula
  gamm_formula <- build_gamm_formula(metric, indices)
  cat(sprintf("  Formula: %s\n", deparse(gamm_formula, width.cutoff = 500)))

  # Get the appropriate family
  gamm_family <- get_gam_family(metric_info$family)

  # Fit the model using bam() for speed
  # bam() is optimized for large datasets - MW: I think results are equivalent to gam() but need to confirm
  # select=TRUE enables automatic smoothness selection (shrinks unneeded wiggles)
  gamm_start <- Sys.time()

  gamm_fit <- tryCatch({
    bam(
      formula = gamm_formula,
      data = model_data,
      family = gamm_family,
      # method = "ML",  # Switched to ML for direct comparison w GLMM (from fREML)
      method = "fREML",
      discrete = TRUE,   # Discretization for speed
      select = TRUE,      # Shrinkage selection (penalizes unnecessary complexity)
      rho = 0.6 # include rho to add AR1 correlation, to match GLMM more closely - MW: how much does this parameter matter? 
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

    # Get AIC
    gamm_aic <- AIC(gamm_fit)
    cat(sprintf("  GAMM AIC: %.2f\n", gamm_aic))

  } else {
    gamm_aic <- NA
    gamm_time <- NA
  }

  # --------------------------------------------------------------------------
  # COMPARE MODELS VIA AIC
  # --------------------------------------------------------------------------

  cat("\nComparing models via AIC...\n")

  # Calculate delta AIC
  # Lower AIC = better balance of fit and parsimony
  if (!is.na(glmm_aic) && !is.na(gamm_aic)) {
    delta_aic <- glmm_aic - gamm_aic  # Negative means GLMM is better

    # Decision rules from spec:
    # ΔAIC < 2: Models essentially equivalent
    # ΔAIC 2-4: Weak preference
    # ΔAIC 4-10: Moderate preference
    # ΔAIC > 10: Strong preference
    # When equivalent, prefer GLMM (simpler interpretation)

    if (abs(delta_aic) < 4) {
      selected_model <- "glmm"  # Prefer simpler model when equivalent
      selection_reason <- "Models equivalent (|ΔAIC| < 4); GLMM preferred for interpretability"
    } else if (delta_aic < 0) {
      selected_model <- "glmm"
      selection_reason <- sprintf("GLMM has lower AIC (ΔAIC = %.1f)", delta_aic)
    } else {
      selected_model <- "gamm"
      selection_reason <- sprintf("GAMM has lower AIC (ΔAIC = %.1f)", delta_aic)
    }

    cat(sprintf("  ΔAIC (GLMM - GAMM): %.2f\n", delta_aic))
    cat(sprintf("  Selected model: %s\n", toupper(selected_model)))
    cat(sprintf("  Reason: %s\n", selection_reason))

  } else {
    delta_aic <- NA
    if (is.na(glmm_aic) && !is.na(gamm_aic)) {
      selected_model <- "gamm"
      selection_reason <- "GLMM failed to fit"
    } else if (!is.na(glmm_aic) && is.na(gamm_aic)) {
      selected_model <- "glmm"
      selection_reason <- "GAMM failed to fit"
    } else {
      selected_model <- NA
      selection_reason <- "Both models failed to fit"
    }
    cat(sprintf("  Selected model: %s\n", ifelse(is.na(selected_model), "NONE", toupper(selected_model))))
    cat(sprintf("  Reason: %s\n", selection_reason))
  }

  # Save comparison results
  comparison <- data.frame(
    metric = metric,
    glmm_aic = glmm_aic,
    gamm_aic = gamm_aic,
    delta_aic = delta_aic,
    selected_model = selected_model,
    selection_reason = selection_reason,
    glmm_time_mins = as.numeric(glmm_time),
    gamm_time_mins = as.numeric(gamm_time)
  )

  write.csv(comparison,
            file.path("results/tables", metric, "model_comparison.csv"),
            row.names = FALSE)
  cat("  Saved: results/tables/", metric, "/model_comparison.csv\n", sep = "")

  # Store for summary
  all_results[[metric]] <- comparison
}

# ------------------------------------------------------------------------------
# GENERATE SUMMARY OUTPUTS
# ------------------------------------------------------------------------------

cat("\n=== Generating Summary Outputs ===\n")

# Combine all results into summary table
summary_df <- bind_rows(all_results)
write.csv(summary_df, "results/tables/model_selection_summary.csv", row.names = FALSE)
cat("Saved: results/tables/model_selection_summary.csv\n")

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

cat("\n=== Stage 05 Modeling Complete ===\n\n")

cat("Summary of model selection:\n")
print(summary_df %>% select(metric, glmm_aic, gamm_aic, delta_aic, selected_model))

cat("\nNext steps:\n")
cat("1. Review diagnostic plots in results/figures/<metric>/\n")
cat("2. Check coefficient tables in results/tables/<metric>/\n")
cat("3. Coming soon: Run `quarto render results/results_summary.qmd` to generate slides\n")

# ------------------------------------------------------------------------------
# APPEND TO RUN HISTORY
# ------------------------------------------------------------------------------

# Build a concise summary for each modeled response
# Format: metric: GLMM ✓/✗ (AIC=X) | GAMM ✓/✗ (AIC=Y) | Selected: Z (ΔAIC=W)
model_summaries <- sapply(names(all_results), function(m) {
  res <- all_results[[m]]

  # GLMM status
  if (is.na(res$glmm_aic)) {
    glmm_str <- "GLMM x (failed)"
  } else {
    glmm_str <- sprintf("GLMM ok (AIC=%.1f, %.1fmin)", res$glmm_aic, res$glmm_time_mins)
  }

  # GAMM status
  if (is.na(res$gamm_aic)) {
    gamm_str <- "GAMM x (failed)"
  } else {
    gamm_str <- sprintf("GAMM ok (AIC=%.1f, %.1fmin)", res$gamm_aic, res$gamm_time_mins)
  }

  # Selection summary
  if (is.na(res$selected_model)) {
    select_str <- "Selected: NONE"
  } else if (is.na(res$delta_aic)) {
    select_str <- sprintf("Selected: %s", toupper(res$selected_model))
  } else {
    select_str <- sprintf("Selected: %s (dAIC=%.1f)", toupper(res$selected_model), res$delta_aic)
  }

  sprintf("%s: %s | %s | %s", m, glmm_str, gamm_str, select_str)
})

# Create the run history entry
run_entry <- sprintf(
  "## %s — Stage 05: Modeling

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

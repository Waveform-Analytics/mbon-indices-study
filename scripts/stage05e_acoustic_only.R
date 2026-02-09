# ==============================================================================
# Stage 05e: Acoustic-Only Model Comparison
# ==============================================================================
#
# Purpose:
#   Compare GAMM performance using ONLY acoustic indices (no environmental or
#   temporal covariates) vs full models. This is the inverse of baseline
#   comparison - tests "How well do indices predict alone?"
#
# Inputs:
#   - results/models/<metric>/gamm.rds (9 fitted full models)
#   - data/processed/analysis_ready.parquet
#   - data/interim/aligned_indices.parquet
#   - config/analysis.yml
#
# Outputs:
#   - results/models/<metric>/gamm_acoustic_only.rds (acoustic-only models)
#   - results/tables/acoustic_only_comparison.csv (full vs acoustic-only comparison)
#
# Usage:
#   Rscript scripts/stage05e_acoustic_only.R                       # All metrics
#   Rscript scripts/stage05e_acoustic_only.R --metric fish_presence  # Single metric
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SETUP: Load packages
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(arrow)      # Read parquet files
  library(yaml)       # Read config files
  library(dplyr)      # Data manipulation
  library(tidyr)      # Data reshaping
  library(mgcv)       # Fit GAMMs
})

# Set seed for reproducibility
set.seed(1234)

# ------------------------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

# Parse --metric argument
single_metric <- NULL

for (i in seq_along(args)) {
  if (args[i] == "--metric" && i < length(args)) {
    single_metric <- args[i + 1]
  }
}

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Read the analysis config
config <- yaml::read_yaml("config/analysis.yml")

# Define response metrics and their types
responses <- list(
  fish_activity = list(family = "nbinom2", type = "count"),
  fish_richness = list(family = "nbinom2", type = "count"),
  fish_presence = list(family = "binomial", type = "presence"),
  dolphin_burst_pulse = list(family = "nbinom2", type = "count"),
  dolphin_echolocation = list(family = "nbinom2", type = "count"),
  dolphin_whistle = list(family = "nbinom2", type = "count"),
  dolphin_activity = list(family = "nbinom2", type = "count"),
  dolphin_presence = list(family = "binomial", type = "presence"),
  vessel_presence = list(family = "binomial", type = "presence")
)

# Filter to single metric if specified
if (!is.null(single_metric)) {
  if (!single_metric %in% names(responses)) {
    stop(sprintf("Unknown metric: %s. Available: %s",
                 single_metric, paste(names(responses), collapse = ", ")))
  }
  responses <- responses[single_metric]
  cat(sprintf("Running acoustic-only comparison for single metric: %s\n", single_metric))
} else {
  cat(sprintf("Running acoustic-only comparison for all %d metrics\n", length(responses)))
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS (reused from stage05d_baseline.R)
# ------------------------------------------------------------------------------

#' Load a fitted full GAMM model for a specific metric
#'
#' @param metric Character string, e.g., "fish_activity"
#' @return The fitted model object, or NULL if file doesn't exist
load_full_model <- function(metric) {
  model_path <- file.path("results/models", metric, "gamm.rds")

  if (!file.exists(model_path)) {
    warning(sprintf("Full model not found: %s", model_path))
    return(NULL)
  }

  tryCatch({
    readRDS(model_path)
  }, error = function(e) {
    warning(sprintf("Error loading model %s: %s", metric, e$message))
    return(NULL)
  })
}

#' Extract AR1 rho parameter from a fitted model
#'
#' @param model A fitted bam() model object
#' @return Numeric rho value (0 to 1)
extract_rho <- function(model) {
  if (!is.null(model$AR1.rho)) {
    return(model$AR1.rho)
  }
  if (!is.null(model$rho)) {
    return(model$rho)
  }
  warning("Could not extract rho from model, using 0")
  return(0)
}

#' Get the mgcv family object for a given family name
#'
#' @param family_name Character string, either "nbinom2" or "binomial"
#' @return A family object compatible with mgcv::bam
get_gam_family <- function(family_name) {
  switch(family_name,
    "nbinom2" = mgcv::nb(),
    "binomial" = binomial(),
    stop(sprintf("Unknown family: %s", family_name))
  )
}

#' Extract key metrics from a fitted model
#'
#' @param model A fitted bam() model object
#' @return List with aic, deviance_explained, converged
extract_model_metrics <- function(model) {
  if (is.null(model)) {
    return(list(
      aic = NA_real_,
      deviance_explained = NA_real_,
      converged = FALSE
    ))
  }

  model_summary <- summary(model)

  list(
    aic = AIC(model),
    deviance_explained = model_summary$dev.expl,
    converged = model$converged %||% TRUE
  )
}

#' Fit a GAMM model with error handling
#'
#' @param formula Model formula
#' @param data Data frame
#' @param family Distribution family
#' @param rho AR1 correlation parameter
#' @return Fitted model or NULL on failure
fit_model_safe <- function(formula, data, family, rho) {
  tryCatch({
    bam(
      formula = formula,
      data = data,
      family = family,
      method = "fREML",
      discrete = TRUE,
      select = TRUE,
      rho = rho
    )
  }, error = function(e) {
    warning(sprintf("Model fitting failed: %s", e$message))
    return(NULL)
  })
}

#' Get all acoustic index column names from data
#'
#' @param data Data frame containing all indices
#' @return Character vector of index column names
get_all_index_columns <- function(data) {
  non_index_cols <- c(
    # Response variables
    "fish_activity", "fish_richness", "fish_presence",
    "dolphin_burst_pulse", "dolphin_echolocation", "dolphin_whistle",
    "dolphin_activity", "dolphin_presence", "vessel_presence",
    # Temporal/grouping
    "datetime", "datetime_local", "date", "station", "month_id", "day_id",
    "hour_of_day", "day_of_year", "sin_hour", "cos_hour", "time_within_day",
    "hour", "Filename",
    # Environmental
    "temperature", "depth"
  )

  all_cols <- names(data)
  numeric_cols <- all_cols[sapply(data, is.numeric)]
  index_cols <- setdiff(numeric_cols, non_index_cols)

  return(index_cols)
}

#' Ensure output directories exist for a metric
#'
#' @param metric Character string, e.g., "fish_activity"
ensure_output_dirs <- function(metric) {
  dirs <- c(
    file.path("results/models", metric),
    file.path("results/tables"),
    file.path("results/figures")
  )

  for (dir in dirs) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    }
  }
}

# ------------------------------------------------------------------------------
# ACOUSTIC-ONLY SPECIFIC: Formula Builder (T004)
# ------------------------------------------------------------------------------

#' Build the acoustic-only model formula (indices + random effects only)
#'
#' The acoustic-only model excludes environmental and temporal covariates.
#' This tests "How well do indices alone predict the response?"
#'
#' @param response Character string, the response variable name
#' @param index_cols Character vector of acoustic index column names
#' @return A formula object
build_acoustic_only_formula <- function(response, index_cols) {
  # Index terms with k=3 (matches full model configuration)
  index_terms <- paste0("s(", index_cols, ", k=3)", collapse = " + ")

  # Random effects only (NO temperature, depth, hour_of_day, day_of_year)
  random_terms <- "s(station, bs='re') + s(month_id, bs='re')"

  formula_str <- sprintf("%s ~ %s + %s", response, index_terms, random_terms)
  as.formula(formula_str)
}

# ==============================================================================
# LOAD DATA (T005)
# ==============================================================================

cat("\n=== Loading Data ===\n")

# Read the analysis-ready dataset (for response variables and grouping)
analysis_data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded analysis_ready: %d observations\n", nrow(analysis_data)))

# Read aligned indices (for all 61 acoustic indices)
aligned_indices <- arrow::read_parquet("data/interim/aligned_indices.parquet")
cat(sprintf("  Loaded aligned_indices: %d observations\n", nrow(aligned_indices)))

# Merge: join indices with response variables and grouping factors
data <- analysis_data %>%
  select(station, datetime, all_of(names(responses)),
         month_id) %>%
  left_join(
    aligned_indices %>% select(-any_of(c("datetime_local", "date", "hour", "Filename"))),
    by = c("station", "datetime")
  ) %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  ) %>%
  as.data.frame()

cat(sprintf("  Merged data: %d observations\n", nrow(data)))

# Get all index columns
index_cols <- get_all_index_columns(data)
cat(sprintf("  Found %d acoustic indices\n", length(index_cols)))

# ==============================================================================
# ACOUSTIC-ONLY COMPARISON (T006-T010)
# ==============================================================================

cat("\n=== Acoustic-Only Model Comparison ===\n")
cat("Comparing full models (indices + env/temporal) vs acoustic-only models (indices only)\n\n")

# Storage for results
comparison_results <- data.frame(
  metric = character(),
  full_aic = numeric(),
  acoustic_only_aic = numeric(),
  delta_aic = numeric(),
  full_dev_explained = numeric(),
  acoustic_only_dev_explained = numeric(),
  acoustic_only_converged = logical(),
  interpretation = character(),
  stringsAsFactors = FALSE
)

# Track processing statistics
models_processed <- 0
models_converged <- 0
models_failed <- 0

for (metric in names(responses)) {
  cat(sprintf("Processing: %s\n", metric))
  models_processed <- models_processed + 1

  # Ensure output directories exist
  ensure_output_dirs(metric)

  # Get family for this metric
  family <- get_gam_family(responses[[metric]]$family)

  # T006: Load full model and extract rho
  full_model <- load_full_model(metric)
  if (is.null(full_model)) {
    cat(sprintf("  SKIPPED: Full model not found for %s\n", metric))
    models_failed <- models_failed + 1
    next
  }

  rho <- extract_rho(full_model)
  cat(sprintf("  Using rho = %.3f from full model\n", rho))

  # Build acoustic-only formula and fit
  acoustic_only_formula <- build_acoustic_only_formula(metric, index_cols)
  cat(sprintf("  Fitting acoustic-only model (%d index terms + random effects)...\n", length(index_cols)))

  acoustic_only_model <- fit_model_safe(acoustic_only_formula, data, family, rho)

  # T007 & T011: Save acoustic-only model with validation
  if (!is.null(acoustic_only_model)) {
    # T011: Validate model before saving
    validation_ok <- tryCatch({
      summary(acoustic_only_model)
      TRUE
    }, error = function(e) {
      cat(sprintf("  WARNING: Model validation failed: %s\n", e$message))
      FALSE
    })

    if (validation_ok) {
      model_path <- file.path("results/models", metric, "gamm_acoustic_only.rds")
      saveRDS(acoustic_only_model, model_path)
      # T012: Confirm save
      cat(sprintf("  Saved acoustic-only model: %s\n", model_path))
      models_converged <- models_converged + 1
    } else {
      models_failed <- models_failed + 1
    }
  } else {
    models_failed <- models_failed + 1
  }

  # T008: Extract metrics
  full_metrics <- extract_model_metrics(full_model)
  acoustic_only_metrics <- extract_model_metrics(acoustic_only_model)

  # Calculate delta AIC (acoustic_only_AIC - full_AIC)
  # Positive = full model better (env/temporal add value)
  delta_aic <- acoustic_only_metrics$aic - full_metrics$aic

  # T013: Interpret results based on ΔAIC thresholds
  interpretation <- if (is.na(delta_aic)) {
    "Failed to fit acoustic-only"
  } else if (delta_aic > 10) {
    "Full model much better - env/temporal add significant value"
  } else if (delta_aic > 4) {
    "Full model moderately better"
  } else if (delta_aic > 0) {
    "Similar performance"
  } else if (delta_aic > -4) {
    "Similar performance"
  } else if (delta_aic > -10) {
    "Acoustic-only moderately better (unexpected)"
  } else {
    "Acoustic-only much better (unexpected - review model)"
  }

  # Store results
  comparison_results <- rbind(comparison_results, data.frame(
    metric = metric,
    full_aic = full_metrics$aic,
    acoustic_only_aic = acoustic_only_metrics$aic,
    delta_aic = delta_aic,
    full_dev_explained = full_metrics$deviance_explained,
    acoustic_only_dev_explained = acoustic_only_metrics$deviance_explained,
    acoustic_only_converged = acoustic_only_metrics$converged,
    interpretation = interpretation,
    stringsAsFactors = FALSE
  ))

  # T010: Console summary
  cat(sprintf("  Full AIC: %.1f | Acoustic-only AIC: %.1f | ΔAIC: %.1f\n",
              full_metrics$aic,
              ifelse(is.na(acoustic_only_metrics$aic), NA, acoustic_only_metrics$aic),
              ifelse(is.na(delta_aic), NA, delta_aic)))
  cat(sprintf("  Full Dev Expl: %.1f%% | Acoustic-only Dev Expl: %.1f%%\n",
              full_metrics$deviance_explained * 100,
              ifelse(is.na(acoustic_only_metrics$deviance_explained), NA,
                     acoustic_only_metrics$deviance_explained * 100)))
  cat(sprintf("  Interpretation: %s\n\n", interpretation))
}

# T009: Export comparison table
output_path <- "results/tables/acoustic_only_comparison.csv"
write.csv(comparison_results, output_path, row.names = FALSE)
cat(sprintf("Saved comparison table: %s\n", output_path))

# ==============================================================================
# SUMMARY OUTPUT (T014-T015, T017)
# ==============================================================================

cat("\n=== ACOUSTIC-ONLY COMPARISON SUMMARY ===\n")
cat(sprintf("Total metrics processed: %d\n", models_processed))
cat(sprintf("Models converged: %d\n", models_converged))
cat(sprintf("Models failed: %d\n", models_failed))

# Count by interpretation
if (nrow(comparison_results) > 0) {
  full_better_strong <- sum(grepl("Full model much better", comparison_results$interpretation))
  full_better_moderate <- sum(grepl("Full model moderately", comparison_results$interpretation))
  similar <- sum(grepl("Similar", comparison_results$interpretation))
  acoustic_better <- sum(grepl("Acoustic-only", comparison_results$interpretation))
  failed <- sum(grepl("Failed", comparison_results$interpretation))

  cat("\nInterpretation breakdown:\n")
  cat(sprintf("  Full model much better (ΔAIC > 10): %d metrics\n", full_better_strong))
  cat(sprintf("  Full model moderately better (ΔAIC 4-10): %d metrics\n", full_better_moderate))
  cat(sprintf("  Similar performance (|ΔAIC| < 4): %d metrics\n", similar))
  cat(sprintf("  Acoustic-only better (ΔAIC < -4): %d metrics\n", acoustic_better))
  if (failed > 0) cat(sprintf("  Failed to fit: %d metrics\n", failed))
}

# T14: Compare with baseline if available
baseline_path <- "results/tables/baseline_comparison.csv"
if (file.exists(baseline_path)) {
  cat("\n=== COMPARISON WITH BASELINE RESULTS ===\n")
  baseline_results <- read.csv(baseline_path)

  # T15: Show which component (indices vs env/temporal) contributes more
  cat("\nComponent contribution analysis:\n")
  cat("(Higher dev_explained = greater contribution)\n\n")

  cat(sprintf("%-25s %10s %12s %15s %s\n",
              "Metric", "Full Dev%", "Baseline Dev%", "Acoustic-only%", "Primary Driver"))

  for (metric in comparison_results$metric) {
    full_dev <- comparison_results$full_dev_explained[comparison_results$metric == metric] * 100
    baseline_dev <- baseline_results$baseline_dev_explained[baseline_results$metric == metric] * 100
    acoustic_dev <- comparison_results$acoustic_only_dev_explained[comparison_results$metric == metric] * 100

    if (length(baseline_dev) > 0 && !is.na(baseline_dev) && !is.na(acoustic_dev)) {
      # Determine primary driver
      driver <- if (acoustic_dev > baseline_dev) {
        "Acoustic indices"
      } else if (baseline_dev > acoustic_dev) {
        "Env/temporal"
      } else {
        "Equal"
      }

      cat(sprintf("%-25s %10.1f %12.1f %15.1f %s\n",
                  metric, full_dev, baseline_dev, acoustic_dev, driver))
    }
  }
}

# T17: Files created summary
cat("\n=== FILES CREATED ===\n")
cat(sprintf("Comparison table: %s\n", output_path))
for (metric in comparison_results$metric[comparison_results$acoustic_only_converged]) {
  cat(sprintf("Model: results/models/%s/gamm_acoustic_only.rds\n", metric))
}

cat("\n=== Acoustic-Only Comparison Complete ===\n")

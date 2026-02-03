# ==============================================================================
# Stage 05d: Baseline Model Comparison & VIF Validation
# ==============================================================================
#
# Purpose:
#   Compare GAMM performance with and without acoustic indices to answer:
#   "Do indices add explanatory power beyond environmental and temporal
#   variables alone?" Also validates VIF filtering by comparing 17-index
#   vs 60-index models.
#
# Inputs:
#   - results/models/<metric>/gamm.rds (9 fitted full models)
#   - data/processed/analysis_ready.parquet
#   - data/interim/aligned_indices.parquet (for 60-index comparison)
#   - config/analysis.yml
#
# Outputs:
#   - results/models/<metric>/gamm_baseline.rds (baseline models)
#   - results/tables/baseline_comparison.csv (full vs baseline comparison)
#   - results/models/<metric>/gamm_60index.rds (60-index models, if not --baseline-only)
#   - results/tables/vif_validation.csv (17-index vs 60-index comparison)
#   - results/tables/vif_significance_comparison.csv (significant terms comparison)
#
# Usage:
#   Rscript scripts/stage05d_baseline.R                    # Full analysis
#   Rscript scripts/stage05d_baseline.R --baseline-only    # Baseline comparison only
#   Rscript scripts/stage05d_baseline.R --metric fish_presence  # Single metric
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
  library(ggplot2)    # Plotting
  library(jsonlite)   # Write JSON logs
})

# Set seed for reproducibility
set.seed(1234)

# ------------------------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

# Parse --metric argument
single_metric <- NULL
baseline_only <- FALSE

for (i in seq_along(args)) {
  if (args[i] == "--metric" && i < length(args)) {
    single_metric <- args[i + 1]
  }
  if (args[i] == "--baseline-only") {
    baseline_only <- TRUE
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
  cat(sprintf("Running baseline comparison for single metric: %s\n", single_metric))
} else {
  cat(sprintf("Running baseline comparison for all %d metrics\n", length(responses)))
}

if (baseline_only) {
  cat("Mode: --baseline-only (skipping VIF validation)\n")
} else {
  cat("Mode: Full analysis (baseline + VIF validation)\n")
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Model Loading (T003)
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

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: AR1 Extraction (T004)
# ------------------------------------------------------------------------------

#' Extract AR1 rho parameter from a fitted model
#'
#' The rho parameter controls temporal autocorrelation in the residuals.
#' We use the full model's rho for baseline to ensure fair comparison.
#'
#' @param model A fitted bam() model object
#' @return Numeric rho value (0 to 1)
extract_rho <- function(model) {
  # bam() stores rho in the model object
  # Check multiple possible locations
  if (!is.null(model$AR1.rho)) {
    return(model$AR1.rho)
  }
  if (!is.null(model$rho)) {
    return(model$rho)
  }
  # If not found, return 0 (no autocorrelation)
  warning("Could not extract rho from model, using 0")
  return(0)
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Family (T005)
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Model Metrics (T006)
# ------------------------------------------------------------------------------

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

  # Get model summary
  model_summary <- summary(model)

  list(
    aic = AIC(model),
    deviance_explained = model_summary$dev.expl,
    converged = model$converged %||% TRUE  # bam doesn't always set this
  )
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Baseline Formula (T007)
# ------------------------------------------------------------------------------

#' Build the baseline model formula (no acoustic indices)
#'
#' The baseline model includes only environmental + temporal + random effects.
#' This is the "null hypothesis" - what we can explain without indices.
#'
#' @param response Character string, the response variable name
#' @return A formula object
build_baseline_formula <- function(response) {
  # Environmental + temporal + random effects (no indices)
  # Matches FR-001 from spec
  formula_str <- sprintf(
    "%s ~ s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) + s(station, bs='re') + s(month_id, bs='re')",
    response
  )
  as.formula(formula_str)
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Safe Model Fitting (T008)
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Output Directory (T009)
# ------------------------------------------------------------------------------

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
# HELPER FUNCTIONS: VIF Validation (T022, T023, T028)
# ------------------------------------------------------------------------------

#' Get all acoustic index column names from data
#'
#' Identifies numeric columns that are acoustic indices (not response variables,
#' temporal variables, or environmental covariates).
#'
#' @param data Data frame containing all indices
#' @return Character vector of index column names
get_all_index_columns <- function(data) {
  # Columns that are NOT indices
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

  # Get all numeric columns that aren't in the exclusion list
  all_cols <- names(data)
  numeric_cols <- all_cols[sapply(data, is.numeric)]
  index_cols <- setdiff(numeric_cols, non_index_cols)

  return(index_cols)
}

#' Build 60-index model formula
#'
#' Creates formula with all available indices using k=3 (reduced from k=5)
#' to help with convergence given the large number of smooth terms.
#'
#' @param response Character string, the response variable name
#' @param index_cols Character vector of index column names
#' @return A formula object
build_60index_formula <- function(response, index_cols) {
  # Index terms with k=3 (reduced for convergence)
  index_terms <- paste0("s(", index_cols, ", k=3)", collapse = " + ")

  # Environmental + temporal + random effects (same as full model)
  other_terms <- "s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) + s(station, bs='re') + s(month_id, bs='re')"

  formula_str <- sprintf("%s ~ %s + %s", response, index_terms, other_terms)
  as.formula(formula_str)
}

#' Extract significant smooth terms from a fitted model
#'
#' @param model A fitted bam() model object
#' @param threshold P-value threshold for significance (default 0.05)
#' @return Data frame with term, edf, p_value, significant columns
extract_significant_terms <- function(model, threshold = 0.05) {
  if (is.null(model)) {
    return(data.frame(
      term = character(),
      edf = numeric(),
      p_value = numeric(),
      significant = logical(),
      stringsAsFactors = FALSE
    ))
  }

  # Get model summary
  model_summary <- summary(model)

  # Extract smooth term table
  smooth_table <- model_summary$s.table

  if (is.null(smooth_table) || nrow(smooth_table) == 0) {
    return(data.frame(
      term = character(),
      edf = numeric(),
      p_value = numeric(),
      significant = logical(),
      stringsAsFactors = FALSE
    ))
  }

  data.frame(
    term = rownames(smooth_table),
    edf = smooth_table[, "edf"],
    p_value = smooth_table[, "p-value"],
    significant = smooth_table[, "p-value"] < threshold,
    stringsAsFactors = FALSE
  )
}

# ==============================================================================
# LOAD DATA
# ==============================================================================

cat("\n=== Loading Data ===\n")

# Read the analysis-ready dataset
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded %d observations\n", nrow(data)))

# Convert grouping variables to factors (required for random effects)
data <- data %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  )

# ==============================================================================
# BASELINE COMPARISON (User Story 1: T010-T015)
# ==============================================================================

cat("\n=== Baseline Model Comparison ===\n")
cat("Comparing full models (with indices) vs baseline models (no indices)\n\n")

# Storage for results
baseline_results <- data.frame(
  metric = character(),
  full_aic = numeric(),
  baseline_aic = numeric(),
  delta_aic = numeric(),
  full_dev_explained = numeric(),
  baseline_dev_explained = numeric(),
  baseline_converged = logical(),
  interpretation = character(),
  stringsAsFactors = FALSE
)

for (metric in names(responses)) {
  cat(sprintf("Processing: %s\n", metric))

  # Ensure output directories exist
  ensure_output_dirs(metric)

  # Get family for this metric
  family <- get_gam_family(responses[[metric]]$family)

  # T010: Load full model and extract rho
  full_model <- load_full_model(metric)
  if (is.null(full_model)) {
    cat(sprintf("  SKIPPED: Full model not found for %s\n", metric))
    next
  }

  rho <- extract_rho(full_model)
  cat(sprintf("  Using rho = %.3f from full model\n", rho))

  # Build baseline formula and fit
  baseline_formula <- build_baseline_formula(metric)
  cat("  Fitting baseline model (no indices)...\n")

  baseline_model <- fit_model_safe(baseline_formula, data, family, rho)

  # T011: Save baseline model
  if (!is.null(baseline_model)) {
    baseline_path <- file.path("results/models", metric, "gamm_baseline.rds")
    saveRDS(baseline_model, baseline_path)
    cat(sprintf("  Saved baseline model: %s\n", baseline_path))
  }

  # T012: Extract metrics
  full_metrics <- extract_model_metrics(full_model)
  baseline_metrics <- extract_model_metrics(baseline_model)

  # Calculate delta AIC (positive = indices help)
  delta_aic <- baseline_metrics$aic - full_metrics$aic

  # T013: Interpret results based on Burnham & Anderson thresholds
  interpretation <- if (is.na(delta_aic)) {
    "Failed to fit baseline"
  } else if (delta_aic > 10) {
    "Strong evidence indices add value"
  } else if (delta_aic > 4) {
    "Moderate evidence indices add value"
  } else if (delta_aic > 0) {
    "Weak evidence indices add value"
  } else {
    "No evidence indices add value"
  }

  # Store results
  baseline_results <- rbind(baseline_results, data.frame(
    metric = metric,
    full_aic = full_metrics$aic,
    baseline_aic = baseline_metrics$aic,
    delta_aic = delta_aic,
    full_dev_explained = full_metrics$deviance_explained,
    baseline_dev_explained = baseline_metrics$deviance_explained,
    baseline_converged = baseline_metrics$converged,
    interpretation = interpretation,
    stringsAsFactors = FALSE
  ))

  cat(sprintf("  Full AIC: %.1f | Baseline AIC: %.1f | ΔAIC: %.1f\n",
              full_metrics$aic, baseline_metrics$aic, delta_aic))
  cat(sprintf("  Full Dev Expl: %.1f%% | Baseline Dev Expl: %.1f%%\n",
              full_metrics$deviance_explained * 100,
              baseline_metrics$deviance_explained * 100))
  cat(sprintf("  Interpretation: %s\n\n", interpretation))
}

# T014: Export comparison table
output_path <- "results/tables/baseline_comparison.csv"
write.csv(baseline_results, output_path, row.names = FALSE)
cat(sprintf("Saved comparison table: %s\n", output_path))

# T015: Summary output
cat("\n=== BASELINE COMPARISON SUMMARY ===\n")
cat(sprintf("Total metrics processed: %d\n", nrow(baseline_results)))

# Count by interpretation
if (nrow(baseline_results) > 0) {
  strong <- sum(grepl("Strong", baseline_results$interpretation))
  moderate <- sum(grepl("Moderate", baseline_results$interpretation))
  weak <- sum(grepl("Weak evidence", baseline_results$interpretation))
  none <- sum(grepl("No evidence", baseline_results$interpretation))
  failed <- sum(grepl("Failed", baseline_results$interpretation))

  cat(sprintf("  Strong evidence indices help: %d metrics\n", strong))
  cat(sprintf("  Moderate evidence: %d metrics\n", moderate))
  cat(sprintf("  Weak evidence: %d metrics\n", weak))
  cat(sprintf("  No evidence: %d metrics\n", none))
  if (failed > 0) cat(sprintf("  Failed to fit: %d metrics\n", failed))

  # List metrics with strong evidence
  strong_metrics <- baseline_results$metric[grepl("Strong", baseline_results$interpretation)]
  if (length(strong_metrics) > 0) {
    cat(sprintf("\nMetrics where indices clearly add value: %s\n",
                paste(strong_metrics, collapse = ", ")))
  }
}

cat("\n=== Baseline Comparison Complete ===\n")

# ==============================================================================
# VIF VALIDATION (User Story 4: T021-T030 - only if not --baseline-only)
# ==============================================================================

if (!baseline_only) {
  cat("\n=== VIF Validation: 17-index vs 60-index comparison ===\n")
  cat("Comparing models with 17 VIF-filtered indices vs all ~60 indices\n\n")

  # T021: Load 60 indices from aligned_indices.parquet
  cat("Loading all indices from aligned_indices.parquet...\n")
  aligned_indices <- arrow::read_parquet("data/interim/aligned_indices.parquet") %>%
    mutate(station = as.factor(station))

  # Join with analysis data (need response variables + temporal/grouping)
  # Use datetime and station as join keys
  # IMPORTANT: Convert to data.frame - tibble causes issues with bam()
  data_60 <- data %>%
    select(station, datetime, all_of(names(responses)),
           temperature, depth, hour_of_day, day_of_year, month_id) %>%
    left_join(
      aligned_indices %>% select(-any_of(c("datetime_local", "date", "hour", "Filename"))),
      by = c("station", "datetime")
    ) %>%
    as.data.frame()

  cat(sprintf("  Joined data: %d rows\n", nrow(data_60)))

  # T022: Get all index columns
  all_index_cols <- get_all_index_columns(data_60)
  cat(sprintf("  Found %d acoustic indices for 60-index models\n", length(all_index_cols)))

  # Storage for VIF validation results
  vif_results <- data.frame(
    metric = character(),
    full17_aic = numeric(),
    full60_aic = numeric(),
    delta_aic = numeric(),
    full17_dev_explained = numeric(),
    full60_dev_explained = numeric(),
    converged_60index = logical(),
    stringsAsFactors = FALSE
  )

  # Storage for significance comparison
  sig_comparison <- data.frame(
    metric = character(),
    index = character(),
    significant_17 = logical(),
    significant_60 = logical(),
    pval_17 = numeric(),
    pval_60 = numeric(),
    stringsAsFactors = FALSE
  )

  for (metric in names(responses)) {
    cat(sprintf("\nProcessing 60-index model: %s\n", metric))

    # Get family for this metric
    family <- get_gam_family(responses[[metric]]$family)

    # Load existing 17-index (full) model
    full_model <- load_full_model(metric)
    if (is.null(full_model)) {
      cat(sprintf("  SKIPPED: Full model not found for %s\n", metric))
      next
    }

    rho <- extract_rho(full_model)
    cat(sprintf("  Using rho = %.3f from full model\n", rho))

    # T023: Build 60-index formula
    formula_60 <- build_60index_formula(metric, all_index_cols)

    # T024: Fit 60-index model with enhanced error handling
    cat("  Fitting 60-index model (this may take a while)...\n")

    model_60 <- tryCatch({
      bam(
        formula = formula_60,
        data = data_60,
        family = family,
        method = "fREML",
        discrete = TRUE,
        select = TRUE,
        rho = rho
      )
    }, error = function(e) {
      cat(sprintf("  WARNING: 60-index model failed: %s\n", e$message))
      return(NULL)
    })

    # T025: Save 60-index model if successful
    if (!is.null(model_60)) {
      model_60_path <- file.path("results/models", metric, "gamm_60index.rds")
      saveRDS(model_60, model_60_path)
      cat(sprintf("  Saved 60-index model: %s\n", model_60_path))
    }

    # T026: Extract metrics for comparison
    full17_metrics <- extract_model_metrics(full_model)
    full60_metrics <- extract_model_metrics(model_60)

    # Calculate delta AIC (negative = 60-index is better)
    delta_aic <- full60_metrics$aic - full17_metrics$aic

    vif_results <- rbind(vif_results, data.frame(
      metric = metric,
      full17_aic = full17_metrics$aic,
      full60_aic = full60_metrics$aic,
      delta_aic = delta_aic,
      full17_dev_explained = full17_metrics$deviance_explained,
      full60_dev_explained = full60_metrics$deviance_explained,
      converged_60index = full60_metrics$converged,
      stringsAsFactors = FALSE
    ))

    cat(sprintf("  17-index AIC: %.1f | 60-index AIC: %.1f | ΔAIC: %.1f\n",
                full17_metrics$aic,
                ifelse(is.na(full60_metrics$aic), NA, full60_metrics$aic),
                ifelse(is.na(delta_aic), NA, delta_aic)))

    # T028-T029: Extract and compare significant terms
    if (!is.null(model_60)) {
      sig_17 <- extract_significant_terms(full_model)
      sig_60 <- extract_significant_terms(model_60)

      # Filter to only index terms (exclude environmental/temporal)
      sig_17_idx <- sig_17[grepl("^s\\([A-Z]|^s\\(n", sig_17$term), ]
      sig_60_idx <- sig_60[grepl("^s\\([A-Z]|^s\\(n", sig_60$term), ]

      # Clean term names to just index names
      sig_17_idx$index <- gsub("s\\(|\\)", "", sig_17_idx$term)
      sig_60_idx$index <- gsub("s\\(|\\)", "", sig_60_idx$term)

      # Get the 17 indices used in full model
      indices_17 <- sig_17_idx$index

      # For each 17-index term, compare significance
      for (idx in indices_17) {
        sig_in_17 <- sig_17_idx$significant[sig_17_idx$index == idx]
        pval_17 <- sig_17_idx$p_value[sig_17_idx$index == idx]

        # Find matching term in 60-index model
        sig_in_60 <- sig_60_idx$significant[sig_60_idx$index == idx]
        pval_60 <- sig_60_idx$p_value[sig_60_idx$index == idx]

        if (length(sig_in_17) > 0) {
          sig_comparison <- rbind(sig_comparison, data.frame(
            metric = metric,
            index = idx,
            significant_17 = ifelse(length(sig_in_17) > 0, sig_in_17[1], NA),
            significant_60 = ifelse(length(sig_in_60) > 0, sig_in_60[1], NA),
            pval_17 = ifelse(length(pval_17) > 0, pval_17[1], NA),
            pval_60 = ifelse(length(pval_60) > 0, pval_60[1], NA),
            stringsAsFactors = FALSE
          ))
        }
      }
    }
  }

  # T027: Export VIF validation table
  vif_output_path <- "results/tables/vif_validation.csv"
  write.csv(vif_results, vif_output_path, row.names = FALSE)
  cat(sprintf("\nSaved VIF validation table: %s\n", vif_output_path))

  # T030: Export significance comparison
  sig_output_path <- "results/tables/vif_significance_comparison.csv"
  write.csv(sig_comparison, sig_output_path, row.names = FALSE)
  cat(sprintf("Saved significance comparison: %s\n", sig_output_path))

  # Summary
  cat("\n=== VIF VALIDATION SUMMARY ===\n")
  cat(sprintf("Total metrics processed: %d\n", nrow(vif_results)))

  converged <- sum(vif_results$converged_60index, na.rm = TRUE)
  failed <- sum(!vif_results$converged_60index | is.na(vif_results$converged_60index))
  cat(sprintf("  60-index models converged: %d\n", converged))
  cat(sprintf("  60-index models failed: %d\n", failed))

  # Interpretation
  if (any(!is.na(vif_results$delta_aic))) {
    avg_delta <- mean(vif_results$delta_aic, na.rm = TRUE)
    cat(sprintf("\nAverage ΔAIC (60-index minus 17-index): %.1f\n", avg_delta))

    if (avg_delta < -10) {
      cat("Interpretation: 60-index models perform notably better - VIF filtering may have been too aggressive\n")
    } else if (avg_delta > 10) {
      cat("Interpretation: 17-index models perform better - VIF filtering improved parsimony without losing predictive power\n")
    } else {
      cat("Interpretation: Similar performance - VIF filtering removed redundant (not useful) information\n")
    }
  }

  cat("\n=== VIF Validation Complete ===\n")
}

cat("\nScript complete.\n")

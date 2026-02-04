# ==============================================================================
# Stage 05c: Model Diagnostics & Effect Interpretation
# ==============================================================================
#
# Purpose:
#   Generate diagnostic checks and effect size calculations for fitted GAMM
#   models. This extends Stage 05 with standard biostatistical validation
#   (concurvity, zero-inflation, residual diagnostics) and practical effect
#   size quantification for stakeholder communication.
#
# Inputs:
#   - results/models/<metric>/gamm.rds (9 fitted GAMM objects)
#   - data/processed/analysis_ready.parquet
#   - config/analysis.yml
#
# Outputs:
#   - results/tables/effect_sizes.csv (consolidated effect sizes)
#   - results/tables/zero_inflation_check.csv (zero-inflation comparison)
#   - results/tables/<metric>/concurvity.csv (per-model concurvity)
#   - results/figures/<metric>/residuals_by_station.png (station-faceted residuals)
#   - results/figures/<metric>/random_effects_qq.png (RE normality check)
#
# Usage:
#   Rscript scripts/stage05c_diagnostics.R
#   Rscript scripts/stage05c_diagnostics.R --metric fish_presence
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
  library(mgcv)       # GAM diagnostics
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
if (length(args) > 0) {
  for (i in seq_along(args)) {
    if (args[i] == "--metric" && i < length(args)) {
      single_metric <- args[i + 1]
    }
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
  cat(sprintf("Running diagnostics for single metric: %s\n", single_metric))
} else {
  cat(sprintf("Running diagnostics for all %d metrics\n", length(responses)))
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Model Loading
# ------------------------------------------------------------------------------

#' Load a fitted GAMM model for a specific metric
#'
#' @param metric Character string, e.g., "fish_activity"
#' @return The loaded model object, or NULL if not found
load_model <- function(metric) {
  model_path <- file.path("results", "models", metric, "gamm.rds")

  if (!file.exists(model_path)) {
    warning(sprintf("Model file not found: %s", model_path))
    return(NULL)
  }

  model <- readRDS(model_path)
  cat(sprintf("  Loaded model from %s\n", model_path))
  return(model)
}

#' Check if a model converged successfully
#'
#' @param model A fitted GAMM object
#' @return TRUE if converged, FALSE otherwise
check_convergence <- function(model) {
  # For mgcv models, check the convergence status
  if (is.null(model)) return(FALSE)

  # Check if model fitting converged
  converged <- model$converged
  if (is.null(converged)) {
    # Fallback: check if there are any NA coefficients
    converged <- !any(is.na(coef(model)))
  }

  return(isTRUE(converged))
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS: Utilities
# ------------------------------------------------------------------------------

#' Get acoustic index column names from the final index list
#'
#' @return Character vector of index column names
get_index_columns <- function() {
  indices_path <- "data/processed/indices_final.csv"
  if (!file.exists(indices_path)) {
    stop("indices_final.csv not found at ", indices_path)
  }

  indices_df <- read.csv(indices_path, stringsAsFactors = FALSE)
  # Filter to kept indices only
  kept_indices <- indices_df$index_name[indices_df$kept == TRUE | indices_df$kept == "True"]
  return(kept_indices)
}

#' Get the model type (presence or count) for a metric
#'
#' @param metric Character string, e.g., "fish_activity"
#' @return "presence" or "count"
get_model_type <- function(metric) {
  if (!metric %in% names(responses)) {
    stop(sprintf("Unknown metric: %s", metric))
  }
  return(responses[[metric]]$type)
}

#' Ensure output directories exist
#'
#' @param metric Character string, e.g., "fish_activity"
ensure_output_dirs <- function(metric) {
  dirs <- c(
    file.path("results", "tables", metric),
    file.path("results", "figures", metric)
  )

  for (d in dirs) {
    if (!dir.exists(d)) {
      dir.create(d, recursive = TRUE)
      cat(sprintf("  Created directory: %s\n", d))
    }
  }
}

# ------------------------------------------------------------------------------
# EFFECT SIZE CALCULATION (User Story 1)
# ------------------------------------------------------------------------------

#' Calculate effect sizes for all indices in a model
#'
#' For each index, calculates the predicted response change from the 10th to
#' 90th percentile while holding other predictors at their median values.
#' Accounts for z-score standardization used during model fitting.
#'
#' @param model A fitted GAMM object
#' @param data The original data used for fitting
#' @param index_cols Character vector of index column names
#' @param model_type "presence" or "count"
#' @param metric Name of the metric (to load scaling params)
#' @return data.frame with effect sizes
calculate_effect_sizes <- function(model, data, index_cols, model_type, metric) {
  results <- list()

  # Load scaling parameters (model was fitted on z-scored data)
  scaling_path <- file.path("results", "tables", metric, "scaling_params.csv")
  if (file.exists(scaling_path)) {
    scaling_params <- read.csv(scaling_path, stringsAsFactors = FALSE)
  } else {
    warning(sprintf("Scaling params not found at %s, using raw values", scaling_path))
    scaling_params <- NULL
  }

  # Helper to z-score a value
  zscore <- function(x, predictor) {
    if (is.null(scaling_params)) return(x)
    row <- scaling_params[scaling_params$predictor == predictor, ]
    if (nrow(row) == 0) return(x)
    (x - row$mean) / row$sd
  }

  # Get model terms to know which predictors to include
  model_terms <- attr(terms(model), "term.labels")

  for (idx in index_cols) {
    # Skip if index not in data or model
    if (!idx %in% names(data)) next
    if (!idx %in% model_terms) next

    # Calculate 10th and 90th percentiles for this index (raw scale)
    idx_values <- data[[idx]]
    idx_values <- idx_values[!is.na(idx_values)]

    if (length(idx_values) < 10) next

    low_val_raw <- quantile(idx_values, 0.10)
    high_val_raw <- quantile(idx_values, 0.90)

    # Create newdata with scaled values
    newdata_base <- data.frame(matrix(ncol = 0, nrow = 2))

    for (col in model_terms) {
      if (col %in% c("station", "month_id")) next  # Handle factors separately

      if (!col %in% names(data)) next
      if (!is.numeric(data[[col]])) next

      if (col == idx) {
        # Scale the low and high values
        newdata_base[[col]] <- c(zscore(low_val_raw, col), zscore(high_val_raw, col))
      } else {
        # Use median, then scale
        med_val <- median(data[[col]], na.rm = TRUE)
        newdata_base[[col]] <- rep(zscore(med_val, col), 2)
      }
    }

    # Add factor columns at reference level
    newdata_base$station <- factor(rep(levels(data$station)[1], 2),
                                   levels = levels(data$station))
    newdata_base$month_id <- factor(rep(levels(data$month_id)[1], 2),
                                    levels = levels(data$month_id))

    # Predict at low and high values
    tryCatch({
      preds <- predict(model, newdata = newdata_base, type = "response", se.fit = FALSE)

      low_pred <- preds[1]
      high_pred <- preds[2]

      # Calculate effect size based on model type
      if (model_type == "presence") {
        effect_size <- high_pred - low_pred  # Probability change
        effect_type <- "probability_change"
      } else {
        # For counts, use fold-change (avoid division by zero)
        if (low_pred > 0.01) {
          effect_size <- high_pred / low_pred
        } else {
          effect_size <- NA
        }
        effect_type <- "fold_change"
      }

      results[[idx]] <- data.frame(
        index = idx,
        low_value = low_val_raw,
        high_value = high_val_raw,
        low_pred = low_pred,
        high_pred = high_pred,
        effect_size = effect_size,
        effect_type = effect_type,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      warning(sprintf("Could not calculate effect size for %s: %s", idx, e$message))
    })
  }

  if (length(results) == 0) {
    return(data.frame())
  }

  return(do.call(rbind, results))
}

# ------------------------------------------------------------------------------
# CONCURVITY CHECK (User Story 2)
# ------------------------------------------------------------------------------

#' Check concurvity for all smooth terms in a model
#'
#' Concurvity measures how much each smooth could be approximated by a
#' combination of the others. High values (>0.8) indicate potential issues.
#'
#' @param model A fitted GAMM object
#' @param threshold Threshold for flagging (default 0.8)
#' @return data.frame with concurvity values
check_concurvity <- function(model, threshold = 0.8) {
  # Get concurvity matrix
  # Returns a matrix with rows: worst, observed, estimate
  # Columns are the smooth terms
  conc <- concurvity(model, full = TRUE)

  # Extract values from matrix rows
  terms <- colnames(conc)
  worst_vals <- conc["worst", ]
  observed_vals <- conc["observed", ]

  # Create results data frame
  results <- data.frame(
    term = terms,
    worst = as.numeric(worst_vals),
    observed = as.numeric(observed_vals),
    flagged = as.numeric(worst_vals) > threshold,
    stringsAsFactors = FALSE
  )

  return(results)
}

# ------------------------------------------------------------------------------
# ZERO-INFLATION CHECK (User Story 2)
# ------------------------------------------------------------------------------

#' Check for zero-inflation in count models
#'
#' Compares the observed proportion of zeros to the model-predicted proportion.
#' A large gap may indicate need for a zero-inflated model.
#'
#' @param model A fitted GAMM object
#' @param data The original data
#' @param response_col Name of the response column
#' @param gap_threshold Threshold for flagging (default 0.10)
#' @return data.frame with zero-inflation check results
check_zero_inflation <- function(model, data, response_col, gap_threshold = 0.10) {
  # Get observed values
  y <- data[[response_col]]
  y <- y[!is.na(y)]

  # Observed zero proportion
  obs_zero_prop <- mean(y == 0)

  # Predicted zero proportion
  # For negative binomial, use the fitted values to estimate P(Y=0)
  fitted_vals <- fitted(model)

  # For nbinom2 family, the probability of zero depends on mu and theta
  # P(Y=0) = (theta / (theta + mu))^theta
  # We approximate using the fitted means

  # Simple approximation: count predicted values that round to 0
  pred_zero_prop <- mean(fitted_vals < 0.5)

  # Calculate gap
  gap <- abs(obs_zero_prop - pred_zero_prop)

  return(data.frame(
    observed_zero_prop = obs_zero_prop,
    predicted_zero_prop = pred_zero_prop,
    gap = gap,
    flagged = gap > gap_threshold,
    stringsAsFactors = FALSE
  ))
}

# ------------------------------------------------------------------------------
# RANDOM EFFECTS QQ PLOTS (User Story 2)
# ------------------------------------------------------------------------------

#' Generate QQ plots for random effects
#'
#' Extracts random effect estimates (BLUPs) and creates QQ plots to
#' assess the normality assumption.
#'
#' @param model A fitted GAMM object (bam/gam class)
#' @param metric Name of the metric (for plot title)
#' @return A ggplot object
plot_random_effects_qq <- function(model, metric) {
  # For mgcv bam/gam models, random effects (bs="re") are stored as coefficients
  # with names like "s(station).1", "s(station).2", etc.

  coef_names <- names(coef(model))

  # Extract random effect coefficients by pattern matching
  station_idx <- grep("s[(]station[)]", coef_names)
  month_idx <- grep("s[(]month_id[)]", coef_names)

  station_re <- if (length(station_idx) > 0) coef(model)[station_idx] else numeric(0)
  month_re <- if (length(month_idx) > 0) coef(model)[month_idx] else numeric(0)

  # Create data for plotting
  plot_data <- rbind(
    if (length(station_re) > 0) {
      data.frame(re_type = "Station", value = as.numeric(station_re))
    } else {
      data.frame(re_type = character(0), value = numeric(0))
    },
    if (length(month_re) > 0) {
      data.frame(re_type = "Month", value = as.numeric(month_re))
    } else {
      data.frame(re_type = character(0), value = numeric(0))
    }
  )

  if (nrow(plot_data) == 0) {
    # Return empty plot with message
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "No random effects found") +
      theme_void() +
      ggtitle(sprintf("%s: Random Effects QQ", metric))
    return(p)
  }

  # Create QQ plot
  p <- ggplot(plot_data, aes(sample = value)) +
    stat_qq() +
    stat_qq_line(color = "red") +
    facet_wrap(~re_type, scales = "free") +
    theme_bw() +
    labs(
      title = sprintf("%s: Random Effects QQ Plot", metric),
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    )

  return(p)
}

# ------------------------------------------------------------------------------
# TOP-N SMOOTH PLOTS (Filter to most important indices)
# ------------------------------------------------------------------------------

#' Generate smooth plots filtered to top N indices by effect size
#'
#' Creates a readable overview of smooth terms by showing only the most
#' impactful indices rather than all 60.
#'
#' @param model A fitted GAMM object
#' @param effect_sizes data.frame with effect sizes for this metric
#' @param metric Name of the metric
#' @param top_n Number of top indices to show (default 10)
#' @return Path to saved plot, or NULL if failed
plot_top_smooths <- function(model, effect_sizes, metric, top_n = 10) {
  if (nrow(effect_sizes) == 0) {
    warning("No effect sizes available for filtering")
    return(NULL)
  }

  # Get model type to determine how to rank effect sizes
  model_type <- get_model_type(metric)

  # Calculate absolute effect magnitude
  # For fold_change: abs(log(effect_size)) captures deviation from 1
  # For probability_change: abs(effect_size) directly
  if (model_type == "count") {
    effect_sizes$abs_effect <- abs(log(pmax(effect_sizes$effect_size, 0.001)))
  } else {
    effect_sizes$abs_effect <- abs(effect_sizes$effect_size)
  }

  # Sort by absolute effect and get top N
  effect_sizes <- effect_sizes[order(effect_sizes$abs_effect, decreasing = TRUE), ]
  top_indices <- head(effect_sizes$index, top_n)

  cat(sprintf("    Top %d indices: %s\n", length(top_indices), paste(top_indices, collapse = ", ")))

  # Get smooth term indices in the model
  smooth_terms <- sapply(model$smooth, function(s) s$label)

  # Find which smooth indices correspond to our top indices
  # Smooth labels look like "s(ACI)" so extract the variable name
  smooth_vars <- gsub("s\\(([^,)]+).*", "\\1", smooth_terms)

  # Get indices of smooths to plot (top indices + always include temporal/environmental)
  always_include <- c("temperature", "depth", "hour_of_day", "day_of_year")
  vars_to_plot <- union(top_indices, always_include)

  # Find which smooth term indices to include
  select_indices <- which(smooth_vars %in% vars_to_plot)

  if (length(select_indices) == 0) {
    warning("No matching smooth terms found")
    return(NULL)
  }

  # Calculate grid layout
  n_plots <- length(select_indices)
  n_cols <- min(4, ceiling(sqrt(n_plots)))
  n_rows <- ceiling(n_plots / n_cols)

  # Create the plot
  out_path <- file.path("results", "figures", metric, "gamm_smooths_top10.png")

  png(out_path, width = 400 * n_cols, height = 350 * n_rows, res = 120)
  par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 2, 1))

  for (i in select_indices) {
    plot(model, select = i, shade = TRUE, main = smooth_terms[i])
  }

  dev.off()

  return(out_path)
}

# ------------------------------------------------------------------------------
# RESIDUALS BY STATION (User Story 3)
# ------------------------------------------------------------------------------

#' Generate residual plots faceted by station
#'
#' Creates plots of deviance residuals vs fitted values, separated by station.
#'
#' @param model A fitted GAMM object
#' @param data The original data (must contain 'station' column)
#' @param metric Name of the metric (for plot title)
#' @return A ggplot object
plot_residuals_by_station <- function(model, data, metric) {
  # Extract deviance residuals
  resids <- residuals(model, type = "deviance")
  fitted_vals <- fitted(model)

  # Get station information
  # The model was fitted on the same data, so indices should match
  if (!"station" %in% names(data)) {
    warning("No 'station' column found in data")
    return(NULL)
  }

  # Handle case where lengths don't match (due to NA removal during fitting)
  n_model <- length(resids)
  n_data <- nrow(data)

  if (n_model != n_data) {
    # Use model frame to get the actual data used
    mf <- model.frame(model)
    if ("station" %in% names(mf)) {
      stations <- mf$station
    } else {
      # Try to match by complete cases
      stations <- data$station[complete.cases(data[, names(mf)])]
    }
  } else {
    stations <- data$station
  }

  # Ensure lengths match
  if (length(stations) != length(resids)) {
    warning(sprintf("Station length mismatch: %d stations, %d residuals",
                    length(stations), length(resids)))
    stations <- rep(NA, length(resids))
  }

  plot_data <- data.frame(
    fitted = fitted_vals,
    residuals = resids,
    station = stations
  )

  # Remove rows with NA station
  plot_data <- plot_data[!is.na(plot_data$station), ]

  if (nrow(plot_data) == 0) {
    warning("No valid data for residual plot")
    return(NULL)
  }

  # Create faceted plot
  p <- ggplot(plot_data, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE, color = "blue", linewidth = 0.5) +
    facet_wrap(~station, ncol = 3) +
    theme_bw() +
    labs(
      title = sprintf("%s: Residuals by Station", metric),
      x = "Fitted Values",
      y = "Deviance Residuals"
    )

  return(p)
}

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

cat("\n==================================================\n")
cat("Stage 05c: Model Diagnostics & Effect Interpretation\n")
cat("==================================================\n\n")

# Load the analysis data
cat("Loading data...\n")
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
data <- as.data.frame(data)

# Convert character columns to factors (needed for prediction)
if (is.character(data$station)) {
  data$station <- factor(data$station)
}
if (is.character(data$month_id)) {
  data$month_id <- factor(data$month_id)
}

cat(sprintf("  Loaded %d rows from analysis_ready.parquet\n", nrow(data)))

# Get index columns
index_cols <- get_index_columns()
cat(sprintf("  Found %d acoustic indices\n", length(index_cols)))

# Initialize results collectors
all_effect_sizes <- list()
all_zero_inflation <- list()

# Counters for summary
models_processed <- 0
flags_raised <- 0
files_created <- 0

# Process each metric
for (metric in names(responses)) {
  cat(sprintf("\n--- Processing: %s ---\n", metric))

  # Ensure output directories exist
  ensure_output_dirs(metric)

  # Load model
  model <- load_model(metric)
  if (is.null(model)) {
    cat("  SKIPPED: Model not found\n")
    next
  }

  # Check convergence
  if (!check_convergence(model)) {
    cat("  SKIPPED: Model did not converge\n")
    next
  }

  models_processed <- models_processed + 1
  model_type <- get_model_type(metric)
  cat(sprintf("  Model type: %s\n", model_type))

  # --- Effect Sizes ---
  cat("  Calculating effect sizes...\n")
  effect_sizes <- calculate_effect_sizes(model, data, index_cols, model_type, metric)
  if (nrow(effect_sizes) > 0) {
    effect_sizes$metric <- metric
    all_effect_sizes[[metric]] <- effect_sizes

    # Report top effect sizes
    top_effects <- effect_sizes[order(abs(effect_sizes$effect_size), decreasing = TRUE), ]
    top_effects <- head(top_effects, 3)
    for (i in seq_len(nrow(top_effects))) {
      cat(sprintf("    Top %d: %s (effect = %.3f %s)\n",
                  i, top_effects$index[i], top_effects$effect_size[i], top_effects$effect_type[i]))
    }

    # Generate filtered smooth plot (top 10 indices)
    cat("  Generating top-10 smooth plot...\n")
    top_smooth_path <- plot_top_smooths(model, effect_sizes, metric, top_n = 10)
    if (!is.null(top_smooth_path)) {
      files_created <- files_created + 1
      cat(sprintf("    Saved: %s\n", top_smooth_path))
    }
  }

  # --- Concurvity ---
  cat("  Checking concurvity...\n")
  concurvity_results <- check_concurvity(model)
  concurvity_path <- file.path("results", "tables", metric, "concurvity.csv")
  write.csv(concurvity_results, concurvity_path, row.names = FALSE)
  files_created <- files_created + 1

  flagged_terms <- sum(concurvity_results$flagged)
  if (flagged_terms > 0) {
    cat(sprintf("    WARNING: %d terms with concurvity > 0.8\n", flagged_terms))
    flags_raised <- flags_raised + flagged_terms
  } else {
    cat("    All terms OK\n")
  }

  # --- Zero-Inflation (count models only) ---
  if (model_type == "count") {
    cat("  Checking zero-inflation...\n")
    zi_results <- check_zero_inflation(model, data, metric)
    zi_results$metric <- metric
    all_zero_inflation[[metric]] <- zi_results

    if (zi_results$flagged) {
      cat(sprintf("    WARNING: Zero-inflation gap = %.1f%%\n", zi_results$gap * 100))
      flags_raised <- flags_raised + 1
    } else {
      cat(sprintf("    OK: Zero-inflation gap = %.1f%%\n", zi_results$gap * 100))
    }
  }

  # --- Random Effects QQ ---
  cat("  Generating random effects QQ plot...\n")
  qq_plot <- plot_random_effects_qq(model, metric)
  qq_path <- file.path("results", "figures", metric, "random_effects_qq.png")
  ggsave(qq_path, qq_plot, width = 8, height = 4, dpi = 150)
  files_created <- files_created + 1
  cat(sprintf("    Saved: %s\n", qq_path))

  # --- Residuals by Station ---
  cat("  Generating residuals by station plot...\n")
  resid_plot <- plot_residuals_by_station(model, data, metric)
  if (!is.null(resid_plot)) {
    resid_path <- file.path("results", "figures", metric, "residuals_by_station.png")
    ggsave(resid_path, resid_plot, width = 10, height = 4, dpi = 150)
    files_created <- files_created + 1
    cat(sprintf("    Saved: %s\n", resid_path))

    # Summary stats per station
    mf <- model.frame(model)
    if ("station" %in% names(mf)) {
      resids <- residuals(model, type = "deviance")
      stations <- mf$station
      for (st in unique(stations)) {
        st_resids <- resids[stations == st]
        cat(sprintf("    Station %s: mean=%.3f, SD=%.3f\n",
                    st, mean(st_resids), sd(st_resids)))
      }
    }
  }
}

# --- Save consolidated outputs ---
cat("\n--- Saving consolidated outputs ---\n")

# Effect sizes
if (length(all_effect_sizes) > 0) {
  effect_sizes_df <- do.call(rbind, all_effect_sizes)
  # Reorder columns
  effect_sizes_df <- effect_sizes_df[, c("metric", "index", "low_value", "high_value",
                                          "low_pred", "high_pred", "effect_size", "effect_type")]
  effect_sizes_path <- "results/tables/effect_sizes.csv"
  write.csv(effect_sizes_df, effect_sizes_path, row.names = FALSE)
  files_created <- files_created + 1
  cat(sprintf("Saved: %s (%d rows)\n", effect_sizes_path, nrow(effect_sizes_df)))
}

# Zero-inflation
if (length(all_zero_inflation) > 0) {
  zi_df <- do.call(rbind, all_zero_inflation)
  zi_df <- zi_df[, c("metric", "observed_zero_prop", "predicted_zero_prop", "gap", "flagged")]
  zi_path <- "results/tables/zero_inflation_check.csv"
  write.csv(zi_df, zi_path, row.names = FALSE)
  files_created <- files_created + 1
  cat(sprintf("Saved: %s (%d rows)\n", zi_path, nrow(zi_df)))
}

# --- Save run log ---
log_data <- list(
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  metrics_processed = models_processed,
  flags_raised = flags_raised,
  files_created = files_created,
  single_metric = single_metric
)

log_dir <- "results/logs"
if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE)
log_path <- file.path(log_dir, "diagnostics_summary.json")
write_json(log_data, log_path, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("Saved: %s\n", log_path))

# --- Final Summary ---
cat("\n==================================================\n")
cat("SUMMARY\n")
cat("==================================================\n")
cat(sprintf("Models processed: %d\n", models_processed))
cat(sprintf("Flags raised: %d\n", flags_raised))
cat(sprintf("Files created: %d\n", files_created))
cat("\nDone!\n")

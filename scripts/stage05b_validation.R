# ==============================================================================
# Stage 05b: Model Validation
# ==============================================================================
#
# Purpose:
#   1. Validate AR1 autocorrelation structure (confirm rho estimates are adequate)
#   2. Assess predictive performance via week-based k-fold cross-validation
#
# Inputs:
#   - Fitted GAMM models: results/models/<metric>/gamm.rds
#   - Analysis-ready data: data/processed/analysis_ready.parquet
#   - Model summary: results/tables/model_summary.csv
#   - CV config: config/cv.yml
#
# Outputs:
#   - results/tables/ar1_validation.csv
#   - results/figures/acf_residuals.png
#   - results/tables/cv_performance_summary.csv
#   - results/figures/cv_performance_by_metric.png
#   - results/logs/validation_log.json
#
# Usage:
#   Rscript scripts/stage05b_validation.R
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
library(arrow)
library(yaml)
library(dplyr)
library(tidyr)
library(mgcv)
library(ggplot2)
library(patchwork)
library(jsonlite)
library(pROC)        # For AUC calculation
library(lubridate)   # For ISO week
})

set.seed(2025)

cat("=== Stage 05b: Model Validation ===\n\n")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Load configs
analysis_config <- yaml::read_yaml("config/analysis.yml")
cv_config <- yaml::read_yaml("config/cv.yml")

# Response definitions (same as Stage 05a)
responses <- list(
  fish_activity = list(family = "nb", type = "count"),
  fish_richness = list(family = "nb", type = "count"),
  fish_presence = list(family = "binomial", type = "binary"),
  dolphin_burst_pulse = list(family = "nb", type = "count"),
  dolphin_echolocation = list(family = "nb", type = "count"),
  dolphin_whistle = list(family = "nb", type = "count"),
  dolphin_activity = list(family = "nb", type = "count"),
  dolphin_presence = list(family = "binomial", type = "binary"),
  vessel_presence = list(family = "binomial", type = "binary")
)

# CV parameters
min_fold_rows <- cv_config$primary_strategy$min_fold_rows

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("Loading data...\n")

# Load analysis-ready data
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")

# Load final indices list
indices_final <- read.csv("data/processed/indices_final.csv")$index

# Add ISO week for CV folds and ensure station is factor
data <- data %>%
  mutate(
    station = as.factor(station),
    iso_week = isoweek(datetime),
    iso_year = isoyear(datetime),
    fold_id = paste(iso_year, sprintf("%02d", iso_week), sep = "-W")
  )

cat(sprintf("  Loaded %d observations\n", nrow(data)))
cat(sprintf("  %d unique week-folds\n", length(unique(data$fold_id))))

# Load model summary for rho values
model_summary <- read.csv("results/tables/model_summary.csv")

# ------------------------------------------------------------------------------
# PART 1: AR1 VALIDATION
# ------------------------------------------------------------------------------

cat("\n--- Part 1: AR1 Autocorrelation Validation ---\n\n")

ar1_results <- list()

for (metric in names(responses)) {
  cat(sprintf("Processing %s...\n", metric))

  # Load fitted model
  model_path <- file.path("results/models", metric, "gamm.rds")
  if (!file.exists(model_path)) {
    cat(sprintf("  WARNING: Model not found, skipping\n"))
    next
  }

  model <- readRDS(model_path)

  # Extract GAM component
  if (inherits(model, "list") && "gam" %in% names(model)) {
    gam_obj <- model$gam
  } else {
    gam_obj <- model
  }

  # Get deviance residuals
  resids <- residuals(gam_obj, type = "deviance")

  # Compute ACF (up to lag 12 = 24 hours)
  acf_result <- acf(resids, lag.max = 12, plot = FALSE)

  # Extract lag-1 ACF (index 2 because index 1 is lag 0)
  lag1_acf <- acf_result$acf[2]
  lag2_acf <- acf_result$acf[3]
  lag3_acf <- acf_result$acf[4]

  # Get rho from model summary
  rho_estimated <- model_summary %>%
    filter(metric == !!metric) %>%
    pull(rho)

  ar1_results[[metric]] <- data.frame(
    metric = metric,
    rho_estimated = rho_estimated,
    residual_acf_lag1 = lag1_acf,
    residual_acf_lag2 = lag2_acf,
    residual_acf_lag3 = lag3_acf,
    ar1_effective = abs(lag1_acf) < 0.1  # Is residual ACF near zero?
  )

  cat(sprintf("  Rho: %.3f, Residual ACF(1): %.3f\n", rho_estimated, lag1_acf))
}

# Combine results
ar1_df <- bind_rows(ar1_results)

# Save AR1 validation table
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)
write.csv(ar1_df, "results/tables/ar1_validation.csv", row.names = FALSE)
cat("\nSaved: results/tables/ar1_validation.csv\n")

# Create ACF plot for all metrics
cat("\nGenerating ACF plots...\n")

acf_plots <- list()
for (metric in names(responses)) {
  model_path <- file.path("results/models", metric, "gamm.rds")
  if (!file.exists(model_path)) next

  model <- readRDS(model_path)
  if (inherits(model, "list") && "gam" %in% names(model)) {
    gam_obj <- model$gam
  } else {
    gam_obj <- model
  }

  resids <- residuals(gam_obj, type = "deviance")
  acf_result <- acf(resids, lag.max = 12, plot = FALSE)

  acf_data <- data.frame(
    lag = acf_result$lag[-1],  # Remove lag 0
    acf = acf_result$acf[-1]
  )

  # Confidence interval (approximate)
  ci <- 1.96 / sqrt(length(resids))

  acf_plots[[metric]] <- ggplot(acf_data, aes(x = lag, y = acf)) +
    geom_hline(yintercept = 0, linetype = "solid", color = "gray50") +
    geom_hline(yintercept = c(-ci, ci), linetype = "dashed", color = "blue", alpha = 0.5) +
    geom_segment(aes(xend = lag, yend = 0), linewidth = 1) +
    geom_point(size = 2) +
    labs(title = metric, x = "Lag (2-hour units)", y = "ACF") +
    ylim(-0.3, 0.3) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, face = "bold"))
}

# Combine into grid
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)
combined_acf <- wrap_plots(acf_plots, ncol = 3) +
  plot_annotation(
    title = "Residual Autocorrelation After AR1 Correction",
    subtitle = "Dashed blue lines show 95% CI; values within bands indicate AR1 is effective"
  )

ggsave("results/figures/acf_residuals.png", combined_acf,
       width = 12, height = 10, dpi = 150, bg = "white")
cat("Saved: results/figures/acf_residuals.png\n")

# ------------------------------------------------------------------------------
# PART 2: WEEK-BASED CROSS-VALIDATION
# ------------------------------------------------------------------------------

cat("\n--- Part 2: Week-Based Cross-Validation ---\n\n")

# Get formula components - simplified for CV stability
# Use regular thin plate splines instead of cyclic to avoid edge issues when subsetting
build_formula_cv <- function(response, indices, env_vars = c("temperature", "depth"),
                              temporal_vars = c("hour_of_day", "day_of_year")) {

  index_terms <- paste0("s(", indices, ", k=5)", collapse = " + ")
  env_terms <- paste0("s(", env_vars, ", k=5)", collapse = " + ")
  # Use regular splines for temporal vars in CV (more robust to subsetting)
  temporal_terms <- paste0("s(", temporal_vars, ", k=8)", collapse = " + ")
  # Station as random effect (same as Stage 05a)
  station_term <- "s(station, bs='re')"

  formula_str <- paste(response, "~", index_terms, "+", env_terms, "+",
                       temporal_terms, "+", station_term)
  as.formula(formula_str)
}

# CV function for a single metric
run_cv_for_metric <- function(metric, response_info, data, indices, min_rows = 20) {

  cat(sprintf("\nRunning CV for %s...\n", metric))

  # Get unique folds
  fold_ids <- unique(data$fold_id)
  fold_ids <- sort(fold_ids)

  # Storage for fold results
  fold_results <- list()

  for (i in seq_along(fold_ids)) {
    fold <- fold_ids[i]

    # Split data
    test_data <- data %>% filter(fold_id == fold)
    train_data <- data %>% filter(fold_id != fold)

    # Skip sparse folds
    if (nrow(test_data) < min_rows) {
      cat(sprintf("  Fold %s: skipped (n=%d < %d)\n", fold, nrow(test_data), min_rows))
      next
    }

    # Build formula (simplified for CV)
    formula <- build_formula_cv(metric, indices)

    # Fit model on training data
    tryCatch({
      if (response_info$type == "count") {
        fit <- bam(
          formula,
          data = train_data,
          family = nb(),
          method = "fREML",
          discrete = TRUE,
          control = gam.control(trace = FALSE)
        )
      } else {
        fit <- bam(
          formula,
          data = train_data,
          family = binomial(),
          method = "fREML",
          discrete = TRUE,
          control = gam.control(trace = FALSE)
        )
      }

      # Predict on test data
      if (response_info$type == "count") {
        preds <- predict(fit, newdata = test_data, type = "response")
        observed <- test_data[[metric]]

        # Compute metrics
        rmse <- sqrt(mean((observed - preds)^2, na.rm = TRUE))
        mae <- mean(abs(observed - preds), na.rm = TRUE)
        ss_res <- sum((observed - preds)^2, na.rm = TRUE)
        ss_tot <- sum((observed - mean(observed, na.rm = TRUE))^2, na.rm = TRUE)
        r2 <- 1 - ss_res / ss_tot

        fold_results[[fold]] <- data.frame(
          fold = fold,
          n_test = nrow(test_data),
          rmse = rmse,
          mae = mae,
          r2 = r2,
          auc = NA,
          accuracy = NA,
          sensitivity = NA,
          specificity = NA
        )

        cat(sprintf("  Fold %s: n=%d, RMSE=%.2f\n", fold, nrow(test_data), rmse))

      } else {
        preds_prob <- predict(fit, newdata = test_data, type = "response")
        preds_class <- as.integer(preds_prob > 0.5)
        observed <- test_data[[metric]]

        # Compute metrics
        roc_obj <- pROC::roc(observed, preds_prob, quiet = TRUE)
        auc_val <- as.numeric(pROC::auc(roc_obj))

        accuracy <- mean(preds_class == observed, na.rm = TRUE)

        # Sensitivity and specificity
        tp <- sum(preds_class == 1 & observed == 1, na.rm = TRUE)
        tn <- sum(preds_class == 0 & observed == 0, na.rm = TRUE)
        fp <- sum(preds_class == 1 & observed == 0, na.rm = TRUE)
        fn <- sum(preds_class == 0 & observed == 1, na.rm = TRUE)

        sensitivity <- tp / (tp + fn)
        specificity <- tn / (tn + fp)

        fold_results[[fold]] <- data.frame(
          fold = fold,
          n_test = nrow(test_data),
          rmse = NA,
          mae = NA,
          r2 = NA,
          auc = auc_val,
          accuracy = accuracy,
          sensitivity = sensitivity,
          specificity = specificity
        )

        cat(sprintf("  Fold %s: n=%d, AUC=%.3f\n", fold, nrow(test_data), auc_val))
      }

    }, error = function(e) {
      cat(sprintf("  Fold %s: ERROR - %s\n", fold, e$message))
    })
  }

  # Combine fold results
  if (length(fold_results) == 0) {
    return(NULL)
  }

  fold_df <- bind_rows(fold_results)
  fold_df$metric <- metric
  fold_df$type <- response_info$type

  return(fold_df)
}

# Run CV for all metrics
cv_results <- list()

for (metric in names(responses)) {
  result <- run_cv_for_metric(
    metric = metric,
    response_info = responses[[metric]],
    data = data,
    indices = indices_final,
    min_rows = min_fold_rows
  )

  if (!is.null(result)) {
    cv_results[[metric]] <- result
  }
}

# Combine all CV results
cv_all <- bind_rows(cv_results)

# Check if we got any results
if (nrow(cv_all) == 0) {
  cat("\nWARNING: No CV folds completed successfully.\n")
  cat("Skipping CV summary and plots.\n")
  cv_summary <- data.frame()
} else {
  # Save detailed fold-level results
  write.csv(cv_all, "results/tables/cv_fold_results.csv", row.names = FALSE)
  cat("\nSaved: results/tables/cv_fold_results.csv\n")

  # Create summary by metric
  cv_summary <- cv_all %>%
    group_by(metric, type) %>%
    summarise(
      n_folds = n(),
      rmse_mean = mean(rmse, na.rm = TRUE),
      rmse_sd = sd(rmse, na.rm = TRUE),
      mae_mean = mean(mae, na.rm = TRUE),
      r2_mean = mean(r2, na.rm = TRUE),
      auc_mean = mean(auc, na.rm = TRUE),
      auc_sd = sd(auc, na.rm = TRUE),
      accuracy_mean = mean(accuracy, na.rm = TRUE),
      sensitivity_mean = mean(sensitivity, na.rm = TRUE),
      specificity_mean = mean(specificity, na.rm = TRUE),
      .groups = "drop"
    )

  write.csv(cv_summary, "results/tables/cv_performance_summary.csv", row.names = FALSE)
  cat("Saved: results/tables/cv_performance_summary.csv\n")
}

# ------------------------------------------------------------------------------
# CV VISUALIZATION
# ------------------------------------------------------------------------------

if (nrow(cv_all) > 0) {
  cat("\nGenerating CV performance plots...\n")

  # Plot for count metrics (RMSE)
  count_metrics <- cv_all %>% filter(type == "count")
  p_count <- NULL

  if (nrow(count_metrics) > 0) {
    p_count <- ggplot(count_metrics, aes(x = metric, y = rmse)) +
      geom_boxplot(fill = "steelblue", alpha = 0.7) +
      labs(
        title = "Cross-Validation Performance: Count Metrics",
        subtitle = "RMSE across week-based folds",
        x = "Response Metric",
        y = "RMSE"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }

  # Plot for binary metrics (AUC)
  binary_metrics <- cv_all %>% filter(type == "binary")
  p_binary <- NULL

  if (nrow(binary_metrics) > 0) {
    p_binary <- ggplot(binary_metrics, aes(x = metric, y = auc)) +
      geom_boxplot(fill = "coral", alpha = 0.7) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
      labs(
        title = "Cross-Validation Performance: Binary Metrics",
        subtitle = "AUC across week-based folds (red line = random chance)",
        x = "Response Metric",
        y = "AUC"
      ) +
      ylim(0.4, 1.0) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }

  # Combine plots
  if (!is.null(p_count) && !is.null(p_binary)) {
    combined_cv <- p_count / p_binary
  } else if (!is.null(p_count)) {
    combined_cv <- p_count
  } else if (!is.null(p_binary)) {
    combined_cv <- p_binary
  } else {
    combined_cv <- NULL
  }

  if (!is.null(combined_cv)) {
    ggsave("results/figures/cv_performance_by_metric.png", combined_cv,
           width = 10, height = 10, dpi = 150, bg = "white")
    cat("Saved: results/figures/cv_performance_by_metric.png\n")
  }
} else {
  cat("\nSkipping CV plots (no data).\n")
}

# ------------------------------------------------------------------------------
# SAVE LOG
# ------------------------------------------------------------------------------

dir.create("results/logs", showWarnings = FALSE, recursive = TRUE)

log_data <- list(
  timestamp = Sys.time(),
  stage = "05b-validation",
  ar1_validation = list(
    metrics_validated = nrow(ar1_df),
    all_effective = all(ar1_df$ar1_effective)
  ),
  cross_validation = list(
    strategy = "week_based_kfold",
    min_fold_rows = min_fold_rows,
    total_folds_run = nrow(cv_all),
    metrics_validated = length(unique(cv_all$metric))
  ),
  outputs = list(
    tables = c(
      "results/tables/ar1_validation.csv",
      "results/tables/cv_fold_results.csv",
      "results/tables/cv_performance_summary.csv"
    ),
    figures = c(
      "results/figures/acf_residuals.png",
      "results/figures/cv_performance_by_metric.png"
    )
  )
)

write_json(log_data, "results/logs/validation_log.json", pretty = TRUE, auto_unbox = TRUE)
cat("\nSaved: results/logs/validation_log.json\n")

# ------------------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------------------

cat("\n=== Stage 05b Complete ===\n\n")

cat("AR1 Validation Summary:\n")
print(ar1_df %>% select(metric, rho_estimated, residual_acf_lag1, ar1_effective))

cat("\nCV Performance Summary:\n")
print(cv_summary)

cat("\nDone!\n")
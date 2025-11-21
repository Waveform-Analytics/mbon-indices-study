#!/usr/bin/env Rscript
# 05b_individual_index_models.R
# Test each acoustic index individually in separate GLMMs
#
# Purpose: Validate multi-index selection by testing if filtered indices
# (like ACTtFraction) had strong effects when tested alone, and discover
# clustering patterns based on prediction similarity
#
# Output: 600 models (60 indices × 10 response variables)

library(glmmTMB)
library(broom.mixed)
library(parallel)

cat("============================================================\n")
cat("SCRIPT 5b: INDIVIDUAL INDEX GLMMS\n")
cat("============================================================\n\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Response variable families
RESPONSE_FAMILIES <- list(
  fish_activity = "poisson",
  fish_richness = "poisson",
  fish_present = "binomial",
  dolphin_activity = "nbinom2",
  dolphin_whistles = "nbinom2",
  dolphin_echolocation = "nbinom2",
  dolphin_burst_pulses = "nbinom2",
  dolphin_present = "binomial",
  vessel_present = "binomial",
  vessel_count = "poisson"
)

OUTPUT_DIR <- "../../../data/processed/glmm_results/individual_indices"

# Use parallel processing?
USE_PARALLEL <- TRUE
N_CORES <- max(1, detectCores() - 1)  # Leave one core free

# ============================================================================
# LOAD DATA
# ============================================================================

cat("Loading data...\n")
cat("----------------------------------------\n")

# Load the dataset with all 60 acoustic indices (from lookup table)
# This excludes individual species detections (which would be circular predictors)
data <- read.csv("../../../data/processed/model_data_all_indices.csv")

cat(sprintf("✓ Data loaded: %d observations\n", nrow(data)))
cat(sprintf("✓ Variables: %d columns\n", ncol(data)))

# Convert categorical variables
data$month <- as.factor(data$month)
data$station <- as.factor(data$station)
data$hour <- as.factor(data$hour)

# ============================================================================
# IDENTIFY VARIABLES
# ============================================================================

cat("\nIdentifying variables...\n")
cat("----------------------------------------\n")

# Response variables
response_vars <- names(RESPONSE_FAMILIES)[names(RESPONSE_FAMILIES) %in% colnames(data)]
cat(sprintf("Found %d response variables\n", length(response_vars)))

# Identify ALL acoustic indices (not just the final 5)
all_cols <- colnames(data)
response_cols <- response_vars
metadata_cols <- c("Date", "month", "hour", "station", "temp", "depth")

# All numeric columns that aren't responses or metadata
acoustic_indices <- setdiff(all_cols, c(response_cols, metadata_cols))
acoustic_indices <- acoustic_indices[sapply(data[acoustic_indices], is.numeric)]

cat(sprintf("Found %d acoustic indices to test individually\n", length(acoustic_indices)))
cat(sprintf("Total models to fit: %d\n", length(acoustic_indices) * length(response_vars)))

if (USE_PARALLEL) {
  cat(sprintf("\n✓ Using parallel processing with %d cores\n", N_CORES))
}

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat(sprintf("✓ Created output directory: %s\n", OUTPUT_DIR))
}

cat("\n")

# ============================================================================
# DEFINE MODEL FITTING FUNCTION
# ============================================================================

fit_single_index_model <- function(index_name, response_var, data, families) {
  # Fit GLMM with single index + covariates
  # Returns: List with index_name, response, coefficient, std_error, p_value, AIC, converged

  # Get family
  family_name <- families[[response_var]]

  # Build formula: response ~ INDEX + temp + depth + month + hour + (1|station)
  formula_str <- sprintf("%s ~ %s + temp + depth + month + hour + (1|station)",
                         response_var, index_name)
  formula_obj <- as.formula(formula_str)

  # Select family function
  family_func <- switch(family_name,
    "poisson" = poisson(),
    "nbinom2" = nbinom2(),
    "binomial" = binomial(),
    poisson()  # default
  )

  # Fit model with error handling
  result <- tryCatch({
    model <- glmmTMB(formula_obj, data = data, family = family_func)

    # Extract coefficient for the acoustic index
    coef_tidy <- tidy(model, conf.int = FALSE)
    index_coef <- coef_tidy[coef_tidy$term == index_name & coef_tidy$effect == "fixed", ]

    if (nrow(index_coef) == 0) {
      # Index not in model (shouldn't happen but handle it)
      return(list(
        index = index_name,
        response = response_var,
        family = family_name,
        coefficient = NA,
        std_error = NA,
        statistic = NA,
        p_value = NA,
        AIC = NA,
        converged = FALSE,
        error = "Index coefficient not found"
      ))
    }

    list(
      index = index_name,
      response = response_var,
      family = family_name,
      coefficient = index_coef$estimate,
      std_error = index_coef$std.error,
      statistic = index_coef$statistic,
      p_value = index_coef$p.value,
      AIC = AIC(model),
      converged = model$sdr$pdHess,
      error = NA
    )

  }, error = function(e) {
    list(
      index = index_name,
      response = response_var,
      family = family_name,
      coefficient = NA,
      std_error = NA,
      statistic = NA,
      p_value = NA,
      AIC = NA,
      converged = FALSE,
      error = as.character(e$message)
    )
  })

  return(result)
}

# ============================================================================
# FIT ALL MODELS
# ============================================================================

cat("============================================================\n")
cat("FITTING INDIVIDUAL INDEX MODELS\n")
cat("============================================================\n\n")

start_time <- Sys.time()

# Create all combinations of index × response
model_combinations <- expand.grid(
  index = acoustic_indices,
  response = response_vars,
  stringsAsFactors = FALSE
)

cat(sprintf("Fitting %d models...\n", nrow(model_combinations)))
cat("This will take a while (est. 10-30 minutes depending on your machine)\n\n")

# Fit models (with or without parallel)
if (USE_PARALLEL) {
  # Parallel processing
  cl <- makeCluster(N_CORES)
  clusterExport(cl, c("fit_single_index_model", "data", "RESPONSE_FAMILIES"))
  clusterEvalQ(cl, {
    library(glmmTMB)
    library(broom.mixed)
  })

  results_list <- parApply(cl, model_combinations, 1, function(row) {
    fit_single_index_model(row["index"], row["response"], data, RESPONSE_FAMILIES)
  })

  stopCluster(cl)

} else {
  # Sequential processing
  results_list <- apply(model_combinations, 1, function(row) {
    if (as.numeric(rownames(row)) %% 50 == 0) {
      cat(sprintf("  Progress: %d / %d models\n",
                  as.numeric(rownames(row)), nrow(model_combinations)))
    }
    fit_single_index_model(row["index"], row["response"], data, RESPONSE_FAMILIES)
  })
}

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "mins")

cat(sprintf("\n✓ Completed in %.1f minutes\n\n", elapsed))

# ============================================================================
# COMPILE RESULTS
# ============================================================================

cat("Compiling results...\n")
cat("----------------------------------------\n")

# Convert list to dataframe
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    index = x$index,
    response = x$response,
    family = x$family,
    coefficient = x$coefficient,
    std_error = x$std_error,
    statistic = x$statistic,
    p_value = x$p_value,
    AIC = x$AIC,
    converged = x$converged,
    error = ifelse(is.na(x$error), "", x$error),
    stringsAsFactors = FALSE
  )
}))

# Save full results
full_results_path <- file.path(OUTPUT_DIR, "all_single_index_models.csv")
write.csv(results_df, full_results_path, row.names = FALSE)
cat(sprintf("✓ Saved full results: %s\n", full_results_path))

# ============================================================================
# CONVERGENCE SUMMARY
# ============================================================================

n_total <- nrow(results_df)
n_converged <- sum(results_df$converged, na.rm = TRUE)
n_failed <- sum(!results_df$converged, na.rm = TRUE)

cat(sprintf("\nConvergence summary:\n"))
cat(sprintf("  Total models: %d\n", n_total))
cat(sprintf("  Converged: %d (%.1f%%)\n", n_converged, 100*n_converged/n_total))
cat(sprintf("  Failed: %d (%.1f%%)\n", n_failed, 100*n_failed/n_total))

# ============================================================================
# CREATE SUMMARY TABLES
# ============================================================================

cat("\nCreating summary tables...\n")
cat("----------------------------------------\n")

# Only use converged models for summaries
results_converged <- results_df[results_df$converged, ]

# 1. Top indices by response variable (by absolute coefficient)
cat("  Creating top indices by response...\n")

top_by_response <- lapply(response_vars, function(resp) {
  resp_data <- results_converged[results_converged$response == resp, ]
  resp_data$abs_coef <- abs(resp_data$coefficient)
  resp_data <- resp_data[order(-resp_data$abs_coef), ]
  head(resp_data[, c("index", "coefficient", "std_error", "statistic", "AIC")], 20)
})
names(top_by_response) <- response_vars

# Save
for (resp in response_vars) {
  path <- file.path(OUTPUT_DIR, sprintf("top_indices_%s.csv", resp))
  write.csv(top_by_response[[resp]], path, row.names = FALSE)
}
cat(sprintf("    ✓ Saved top 20 indices for each response variable\n"))

# 2. Performance by index across all responses
cat("  Creating index performance summary...\n")

index_summary <- aggregate(
  cbind(abs_coef = abs(coefficient)) ~ index,
  data = results_converged,
  FUN = function(x) c(
    mean = mean(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    max = max(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE)
  )
)

# Flatten the matrix columns
index_summary <- data.frame(
  index = index_summary$index,
  mean_abs_coef = index_summary$abs_coef[, "mean"],
  median_abs_coef = index_summary$abs_coef[, "median"],
  max_abs_coef = index_summary$abs_coef[, "max"],
  sd_abs_coef = index_summary$abs_coef[, "sd"]
)

index_summary <- index_summary[order(-index_summary$mean_abs_coef), ]

index_summary_path <- file.path(OUTPUT_DIR, "index_performance_summary.csv")
write.csv(index_summary, index_summary_path, row.names = FALSE)
cat(sprintf("    ✓ Saved index performance summary\n"))

# 3. Comparison with final 5-index model
cat("  Creating comparison with multi-index model...\n")

# Load multi-index results
multi_index_results <- read.csv("../../../data/processed/glmm_results/glmm_all_coefficients.csv")

# Filter to just acoustic indices
final_indices <- c("BI", "EAS", "EPS_KURT", "EVNtMean", "nROI")
multi_coefs <- multi_index_results[
  multi_index_results$term %in% final_indices &
  multi_index_results$effect == "fixed",
  c("term", "response", "estimate", "std.error")
]
names(multi_coefs) <- c("index", "response", "multi_coef", "multi_se")

# Get single-index results for these same indices
single_coefs <- results_converged[
  results_converged$index %in% final_indices,
  c("index", "response", "coefficient", "std_error")
]
names(single_coefs) <- c("index", "response", "single_coef", "single_se")

# Merge
comparison <- merge(multi_coefs, single_coefs, by = c("index", "response"))
comparison$coef_ratio <- comparison$single_coef / comparison$multi_coef
comparison$difference <- comparison$single_coef - comparison$multi_coef

comparison_path <- file.path(OUTPUT_DIR, "single_vs_multi_comparison.csv")
write.csv(comparison, comparison_path, row.names = FALSE)
cat(sprintf("    ✓ Saved single vs. multi-index comparison\n"))

# ============================================================================
# PREDICTION SIMILARITY MATRIX (for clustering)
# ============================================================================

cat("\n  Creating prediction similarity matrix...\n")

# Create wide matrix: indices × responses (coefficients as values)
coef_matrix <- reshape2::dcast(
  results_converged,
  index ~ response,
  value.var = "coefficient"
)
rownames(coef_matrix) <- coef_matrix$index
coef_matrix$index <- NULL

# Remove any indices with missing values
coef_matrix <- coef_matrix[complete.cases(coef_matrix), ]

# Calculate correlation between indices based on their prediction patterns
# (which indices make similar predictions across response variables?)
similarity_matrix <- cor(t(coef_matrix), use = "pairwise.complete.obs")

similarity_path <- file.path(OUTPUT_DIR, "prediction_similarity_matrix.csv")
write.csv(similarity_matrix, similarity_path)
cat(sprintf("    ✓ Saved prediction similarity matrix (%d × %d)\n",
            nrow(similarity_matrix), ncol(similarity_matrix)))

# ============================================================================
# KEY FINDINGS REPORT
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("KEY FINDINGS\n")
cat("============================================================\n\n")

# 1. ACTtFraction specifically
cat("1. ACTtFraction Performance:\n")
act_results <- results_converged[results_converged$index == "ACTtFraction", ]
if (nrow(act_results) > 0) {
  act_results <- act_results[order(-abs(act_results$coefficient)), ]
  cat("   Top 3 responses by effect size:\n")
  for (i in 1:min(3, nrow(act_results))) {
    cat(sprintf("     %s: coef = %.4f (SE = %.4f)\n",
                act_results$response[i],
                act_results$coefficient[i],
                act_results$std_error[i]))
  }

  # Compare to multi-index if ACTtFraction was in final 5
  if ("ACTtFraction" %in% comparison$index) {
    cat("\n   Comparison to multi-index model:\n")
    cat("   (ACTtFraction was in final model)\n")
  } else {
    cat("\n   ACTtFraction was NOT in final 5-index model\n")
    cat("   (filtered out at Stage 2: Feature Importance)\n")
  }
} else {
  cat("   ⚠️ ACTtFraction models did not converge\n")
}

cat("\n2. Top 5 Indices by Mean Absolute Effect Size:\n")
for (i in 1:5) {
  cat(sprintf("   %d. %s: mean |coef| = %.4f (max = %.4f)\n",
              i,
              index_summary$index[i],
              index_summary$mean_abs_coef[i],
              index_summary$max_abs_coef[i]))
}

cat("\n3. Comparison: Single vs. Multi-Index (Final 5):\n")
for (idx in final_indices) {
  idx_comp <- comparison[comparison$index == idx, ]
  if (nrow(idx_comp) > 0) {
    mean_ratio <- mean(abs(idx_comp$coef_ratio), na.rm = TRUE)
    cat(sprintf("   %s: single/multi ratio = %.2f (averaged across responses)\n",
                idx, mean_ratio))
    if (mean_ratio > 1.5) {
      cat("     → Effect STRONGER when tested alone (suppression in multi-index?)\n")
    } else if (mean_ratio < 0.7) {
      cat("     → Effect WEAKER when tested alone (enhancement in multi-index?)\n")
    }
  }
}

# ============================================================================
# COMPLETION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================================\n\n")

cat("Output files created:\n")
cat(sprintf("  • %s/all_single_index_models.csv\n", OUTPUT_DIR))
cat(sprintf("  • %s/top_indices_[response].csv (10 files)\n", OUTPUT_DIR))
cat(sprintf("  • %s/index_performance_summary.csv\n", OUTPUT_DIR))
cat(sprintf("  • %s/single_vs_multi_comparison.csv\n", OUTPUT_DIR))
cat(sprintf("  • %s/prediction_similarity_matrix.csv\n", OUTPUT_DIR))

cat("\n✅ Individual index analysis completed!\n")
cat("📊 Ready for clustering analysis and validation\n\n")

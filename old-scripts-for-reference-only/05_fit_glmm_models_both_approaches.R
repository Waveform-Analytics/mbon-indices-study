#!/usr/bin/env Rscript
# 05_fit_glmm_models_both_approaches.R
# Fit GLMMs for both universal and taxa-specific approaches
#
# Purpose: Compare universal (5 indices for all taxa) vs taxa-specific approaches
#
# Output: Model results for all approaches + performance comparison

library(glmmTMB)
library(broom.mixed)

cat("============================================================\n")
cat("SCRIPT 5: FIT GLMMS (UNIVERSAL + TAXA-SPECIFIC)\n")
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
  vessel_present = "binomial"
)

# Target groups
TARGET_GROUPS <- list(
  fish = c("fish_activity", "fish_richness", "fish_present"),
  dolphin = c("dolphin_activity", "dolphin_whistles", "dolphin_echolocation",
              "dolphin_burst_pulses", "dolphin_present"),
  vessel = c("vessel_present")
)

OUTPUT_DIR <- "data/processed/glmm_results"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

fit_glmms_for_dataset <- function(data_path, response_vars, approach_name) {
  cat("\n")
  cat(rep("=", 60), "\n", sep="")
  cat(sprintf("FITTING: %s\n", approach_name))
  cat(rep("=", 60), "\n", sep="")

  # Load data
  data <- read.csv(data_path)
  data$month <- as.factor(data$month)
  data$station <- as.factor(data$station)
  data$hour <- as.factor(data$hour)

  cat(sprintf("✓ Data: %d observations\n", nrow(data)))
  cat(sprintf("✓ Responses: %d\n", length(response_vars)))

  # Identify acoustic indices (columns that aren't responses, covariates, or metadata)
  metadata_cols <- c("Date", "month", "hour", "station", "temp", "depth")
  acoustic_indices <- setdiff(colnames(data), c(response_vars, metadata_cols))
  acoustic_indices <- acoustic_indices[sapply(data[acoustic_indices], is.numeric)]

  cat(sprintf("✓ Acoustic indices: %d\n", length(acoustic_indices)))

  # Build formula
  predictors <- c(acoustic_indices, "temp", "depth", "month", "hour")
  formula_base <- paste(predictors, collapse=" + ")
  formula_base <- paste("~", formula_base, "+ (1|station)")

  # Fit models
  results_list <- list()
  model_stats <- data.frame()

  for (response_var in response_vars) {
    cat(sprintf("\nFitting: %s...", response_var))

    # Get family
    family_name <- RESPONSE_FAMILIES[[response_var]]
    family_func <- switch(family_name,
      "poisson" = poisson(),
      "nbinom2" = nbinom2(),
      "binomial" = binomial(),
      poisson()
    )

    # Fit model
    formula_full <- as.formula(paste(response_var, formula_base))

    model <- tryCatch({
      glmmTMB(formula_full, data = data, family = family_func)
    }, error = function(e) {
      cat(" FAILED\n")
      return(NULL)
    })

    if (!is.null(model)) {
      converged <- model$sdr$pdHess
      cat(if(converged) " ✓\n" else " ⚠ (convergence issue)\n")

      # Extract coefficients
      coef_tidy <- tidy(model, conf.int = TRUE)
      coef_tidy$response <- response_var
      coef_tidy$approach <- approach_name
      results_list[[response_var]] <- coef_tidy

      # Model stats
      model_stats <- rbind(model_stats, data.frame(
        approach = approach_name,
        response = response_var,
        family = family_name,
        AIC = AIC(model),
        logLik = as.numeric(logLik(model)),
        converged = converged
      ))
    }
  }

  # Combine coefficients
  all_coefs <- do.call(rbind, results_list)

  return(list(
    coefficients = all_coefs,
    model_stats = model_stats
  ))
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

cat("📥 FITTING GLMMS FOR ALL APPROACHES\n")
cat(rep("-", 40), "\n", sep="")

all_coefficients <- list()
all_model_stats <- list()

# 1. Universal approach
results_universal <- fit_glmms_for_dataset(
  "data/processed/model_data_universal.csv",
  unlist(TARGET_GROUPS, use.names=FALSE),
  "universal"
)
all_coefficients$universal <- results_universal$coefficients
all_model_stats$universal <- results_universal$model_stats

# 2-4. Taxa-specific approaches
for (group in names(TARGET_GROUPS)) {
  data_path <- sprintf("data/processed/model_data_%s.csv", group)
  results <- fit_glmms_for_dataset(
    data_path,
    TARGET_GROUPS[[group]],
    group
  )
  all_coefficients[[group]] <- results$coefficients
  all_model_stats[[group]] <- results$model_stats
}

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("\n")
cat(rep("=", 60), "\n", sep="")
cat("SAVING RESULTS\n")
cat(rep("=", 60), "\n\n")

# Create output directory
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# Save all coefficients
all_coefs_combined <- do.call(rbind, all_coefficients)
coefs_path <- file.path(OUTPUT_DIR, "glmm_all_coefficients_both_approaches.csv")
write.csv(all_coefs_combined, coefs_path, row.names = FALSE)
cat(sprintf("✓ Saved coefficients: %s\n", basename(coefs_path)))

# Save model statistics
all_stats_combined <- do.call(rbind, all_model_stats)
stats_path <- file.path(OUTPUT_DIR, "glmm_model_stats_both_approaches.csv")
write.csv(all_stats_combined, stats_path, row.names = FALSE)
cat(sprintf("✓ Saved model stats: %s\n", basename(stats_path)))

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

cat("\n")
cat(rep("=", 60), "\n", sep="")
cat("MODEL COMPARISON\n")
cat(rep("=", 60), "\n\n")

# Compare AICs for each response variable
cat("AIC Comparison (lower is better):\n")
cat(rep("-", 60), "\n", sep="")

for (response_var in names(RESPONSE_FAMILIES)) {
  # Get AICs for this response from all approaches
  aics <- all_stats_combined[all_stats_combined$response == response_var, c("approach", "AIC")]

  if (nrow(aics) > 0) {
    aics <- aics[order(aics$AIC), ]
    best_approach <- aics$approach[1]

    cat(sprintf("\n%s:\n", response_var))
    for (i in 1:nrow(aics)) {
      marker <- if(i == 1) " <-- BEST" else ""
      cat(sprintf("  %12s: AIC=%.2f%s\n", aics$approach[i], aics$AIC[i], marker))
    }
  }
}

cat("\n")
cat(rep("=", 60), "\n", sep="")
cat("✅ GLMM FITTING COMPLETED\n")
cat(rep("=", 60), "\n\n")

cat("📊 Next step: Script 06 - Visualize comparison\n\n")

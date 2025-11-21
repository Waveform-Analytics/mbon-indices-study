#!/usr/bin/env Rscript
# 05_fit_glmm_models.R
# Flexible GLMM fitting for multiple taxa (fish, dolphins, vessels)
#
# Key Features:
# - Auto-detects response variables from CSV
# - Handles multiple taxa simultaneously
# - Includes hour (time-of-day) as covariate
# - Auto-selects appropriate family (Poisson, binomial)
# - Saves results systematically for all models

library(glmmTMB)
library(broom.mixed)

cat("============================================================\n")
cat("SCRIPT 5: FIT GLMMS FOR MULTIPLE TAXA\n")
cat("============================================================\n\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Response variable families (auto-detected or manually specified)
RESPONSE_FAMILIES <- list(
  # Fish metrics
  fish_activity = "poisson",
  fish_richness = "poisson",
  fish_present = "binomial",

  # Dolphin metrics
  dolphin_activity = "nbinom2",  # Negative binomial for potentially overdispersed counts
  dolphin_whistles = "nbinom2",
  dolphin_echolocation = "nbinom2",
  dolphin_burst_pulses = "nbinom2",
  dolphin_present = "binomial",

  # Vessel metrics (presence-only)
  vessel_present = "binomial"
)

# Output directory
OUTPUT_DIR <- "../../../data/processed/glmm_results"

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

cat("Loading data...\n")
cat("----------------------------------------\n")

data <- read.csv("../../../data/processed/model_data_for_r.csv")

cat(sprintf("✓ Data loaded: %d observations\n", nrow(data)))
cat(sprintf("✓ Variables: %d columns\n", ncol(data)))

# Convert categorical variables
data$month <- as.factor(data$month)
data$station <- as.factor(data$station)
data$hour <- as.factor(data$hour)  # Treat hour as factor to capture non-linear diel patterns

cat(sprintf("✓ Stations: %s\n", paste(unique(data$station), collapse=", ")))
cat(sprintf("✓ Months: %s\n", paste(sort(unique(data$month)), collapse=", ")))
if ("hour" %in% colnames(data)) {
  cat(sprintf("✓ Hours: %d unique values (treated as factor for non-linear diel patterns)\n",
              length(unique(data$hour))))
}

cat("\n")

# ============================================================================
# IDENTIFY RESPONSE VARIABLES AND PREDICTORS
# ============================================================================

cat("Identifying variables...\n")
cat("----------------------------------------\n")

# Response variables are those defined in RESPONSE_FAMILIES that exist in data
response_vars <- names(RESPONSE_FAMILIES)[names(RESPONSE_FAMILIES) %in% colnames(data)]
response_vars <- setdiff(response_vars, c("vessel_count"))

cat(sprintf("Found %d response variables:\n", length(response_vars)))
for (rv in response_vars) {
  cat(sprintf("  • %s (%s family)\n", rv, RESPONSE_FAMILIES[[rv]]))
}

# Extract acoustic indices (columns between response vars and covariates)
# This assumes a specific column order from Script 04
all_cols <- colnames(data)
response_cols <- response_vars
metadata_cols <- c("Date", "month", "hour", "station", "temp", "depth")
acoustic_indices <- setdiff(all_cols, c(response_cols, metadata_cols))

# Filter to only numeric columns that look like acoustic indices
acoustic_indices <- acoustic_indices[sapply(data[acoustic_indices], is.numeric)]

cat(sprintf("\nFound %d acoustic indices:\n", length(acoustic_indices)))
if (length(acoustic_indices) <= 10) {
  for (idx in acoustic_indices) {
    cat(sprintf("  • %s\n", idx))
  }
} else {
  cat(sprintf("  %s ... (and %d more)\n",
              paste(head(acoustic_indices, 3), collapse=", "),
              length(acoustic_indices) - 3))
}

cat("\n")

# ============================================================================
# BUILD MODEL FORMULA
# ============================================================================

cat("Building model formula...\n")
cat("----------------------------------------\n")

# Build predictor string
predictors <- c(
  acoustic_indices,
  "temp",
  "depth",
  "month"
)

# Add hour if available
if ("hour" %in% colnames(data)) {
  predictors <- c(predictors, "hour")
}

# Formula template
formula_base <- paste(predictors, collapse=" + ")
formula_base <- paste("~", formula_base, "+ (1|station)")

cat("Model formula:\n")
cat(sprintf("  response %s\n", formula_base))
cat("\n")

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat(sprintf("✓ Created output directory: %s\n\n", OUTPUT_DIR))
}

# ============================================================================
# FIT MODELS FOR ALL RESPONSE VARIABLES
# ============================================================================

cat("============================================================\n")
cat("FITTING GLMMS\n")
cat("============================================================\n\n")

# Storage for models and results
models <- list()
coefficients <- list()
summaries <- list()
model_stats <- data.frame(
  response = character(),
  family = character(),
  AIC = numeric(),
  logLik = numeric(),
  converged = logical(),
  stringsAsFactors = FALSE
)

for (response_var in response_vars) {
  cat(rep("=", 60), "\n", sep="")
  cat(sprintf("Fitting: %s\n", response_var))
  cat(rep("=", 60), "\n", sep="")

  # Get family
  family_name <- RESPONSE_FAMILIES[[response_var]]

  # Build formula
  formula_full <- as.formula(paste(response_var, formula_base))

  # Select family function
  family_func <- switch(family_name,
    "poisson" = poisson(),
    "nbinom2" = nbinom2(),
    "binomial" = binomial(),
    poisson()  # default
  )

  cat(sprintf("  Family: %s\n", family_name))
  cat(sprintf("  Formula: %s %s\n", response_var, formula_base))

  # Fit model with error handling
  model <- tryCatch({
    glmmTMB(formula_full, data = data, family = family_func)
  }, error = function(e) {
    cat(sprintf("  ✗ Error fitting model: %s\n", e$message))
    return(NULL)
  })

  if (!is.null(model)) {
    # Check convergence
    converged <- model$sdr$pdHess

    if (converged) {
      cat("  ✓ Model converged successfully\n")
    } else {
      cat("  ⚠ Model may not have converged properly\n")
    }

    # Store model
    models[[response_var]] <- model

    # Extract summary
    model_summary <- summary(model)
    summaries[[response_var]] <- model_summary

    # Print key info
    cat(sprintf("  AIC: %.2f\n", AIC(model)))
    cat(sprintf("  Log-likelihood: %.2f\n", as.numeric(logLik(model))))

    # Extract and save coefficients
    coef_tidy <- tidy(model, conf.int = TRUE)
    coefficients[[response_var]] <- coef_tidy

    # Save individual coefficient table
    coef_path <- file.path(OUTPUT_DIR, sprintf("glmm_coef_%s.csv", response_var))
    write.csv(coef_tidy, coef_path, row.names = FALSE)
    cat(sprintf("  ✓ Coefficients saved: %s\n", basename(coef_path)))

    # Add to model stats
    model_stats <- rbind(model_stats, data.frame(
      response = response_var,
      family = family_name,
      AIC = AIC(model),
      logLik = as.numeric(logLik(model)),
      converged = converged
    ))
  }

  cat("\n")
}

# ============================================================================
# SAVE CONSOLIDATED RESULTS
# ============================================================================

cat(rep("=", 60), "\n", sep="")
cat("SAVING CONSOLIDATED RESULTS\n")
cat(rep("=", 60), "\n", sep="")

# Save model statistics
stats_path <- file.path(OUTPUT_DIR, "glmm_model_stats.csv")
write.csv(model_stats, stats_path, row.names = FALSE)
cat(sprintf("✓ Model statistics: %s\n", stats_path))

# Save all summaries to text file
summaries_path <- file.path(OUTPUT_DIR, "glmm_summaries.txt")
sink(summaries_path)

for (response_var in names(summaries)) {
  cat("\n")
  cat(rep("=", 60), "\n")
  cat(sprintf("MODEL: %s (%s)\n", response_var, RESPONSE_FAMILIES[[response_var]]))
  cat(rep("=", 60), "\n\n")
  print(summaries[[response_var]])
}

sink()
cat(sprintf("✓ Model summaries: %s\n", summaries_path))

# ============================================================================
# EXTRACT AND SAVE KEY RESULTS
# ============================================================================

# Combine all coefficients into one table for easy comparison
all_coefs <- do.call(rbind, lapply(names(coefficients), function(rv) {
  coef_df <- coefficients[[rv]]
  coef_df$response <- rv
  coef_df$family <- RESPONSE_FAMILIES[[rv]]
  coef_df
}))

all_coefs_path <- file.path(OUTPUT_DIR, "glmm_all_coefficients.csv")
write.csv(all_coefs, all_coefs_path, row.names = FALSE)
cat(sprintf("✓ Combined coefficients: %s\n", all_coefs_path))

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

cat("\n")
cat(rep("=", 60), "\n", sep="")
cat("SUMMARY REPORT\n")
cat(rep("=", 60), "\n\n")

cat(sprintf("Models fitted: %d\n", nrow(model_stats)))
cat(sprintf("Models converged: %d\n", sum(model_stats$converged)))

if (sum(!model_stats$converged) > 0) {
  cat("\nModels with convergence issues:\n")
  for (i in which(!model_stats$converged)) {
    cat(sprintf("  ⚠ %s\n", model_stats$response[i]))
  }
}

cat("\nModel comparison (by AIC):\n")
model_stats_sorted <- model_stats[order(model_stats$AIC), ]
for (i in 1:nrow(model_stats_sorted)) {
  cat(sprintf("  %d. %s: AIC=%.2f\n",
              i,
              model_stats_sorted$response[i],
              model_stats_sorted$AIC[i]))
}

cat("\n")
cat(rep("=", 60), "\n", sep="")
cat("ALL MODELS COMPLETE!\n")
cat(rep("=", 60), "\n\n")

cat("Output files created in:", OUTPUT_DIR, "\n")
cat("  - glmm_model_stats.csv\n")
cat("  - glmm_summaries.txt\n")
cat("  - glmm_all_coefficients.csv\n")
cat(sprintf("  - %d individual coefficient CSVs\n", length(coefficients)))

cat("\n✅ GLMM fitting completed successfully!\n")
cat("🔄 Ready for results visualization and interpretation\n\n")

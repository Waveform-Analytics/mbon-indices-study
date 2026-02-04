# ==============================================================================
# Stage 05e: By-Station GAMM Effects Analysis
# ==============================================================================
#
# Purpose:
#   Fit separate GAMM models for each station to understand which acoustic
#   indices are significant predictors at each location. This addresses the
#   question: "Which indices are significant at which stations?"
#
# Inputs:
#   - data/processed/analysis_ready.parquet
#   - data/processed/indices_final.csv
#   - config/analysis.yml
#
# Outputs:
#   - results/tables/bystation_index_significance.csv (full results)
#   - results/tables/bystation_significance_summary.csv (summary table)
#
# Usage:
#   Rscript scripts/stage05e_bystation_effects.R
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

set.seed(1234)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

cat("\n=== Stage 05e: By-Station GAMM Effects Analysis ===\n")

# Read the analysis config
config <- yaml::read_yaml("config/analysis.yml")

# Get GAMM settings from config
smooth_k <- config$gamm$smooth_k %||% 3
cyclic_k <- config$gamm$cyclic_k %||% 12
cat(sprintf("GAMM smooth_k: %d (from config)\n", smooth_k))

# Define which responses to model and their distribution families
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

# Stations to analyze
stations <- c("9M", "14M", "37M")

# Fixed rho for speed (skip AR1 estimation)
FIXED_RHO <- 0.1

cat(sprintf("Stations: %s\n", paste(stations, collapse = ", ")))
cat(sprintf("Metrics: %d\n", length(responses)))
cat(sprintf("Fixed rho: %.2f (for speed)\n", FIXED_RHO))

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

#' Scale predictors using z-score standardization
scale_predictors <- function(data, predictors) {
  for (col in predictors) {
    col_mean <- mean(data[[col]], na.rm = TRUE)
    col_sd <- sd(data[[col]], na.rm = TRUE)
    if (col_sd > 0) {
      data[[col]] <- (data[[col]] - col_mean) / col_sd
    }
  }
  data
}

#' Build the GAMM formula for per-station analysis
#' Note: No station random effect since we're fitting per-station
build_bystation_formula <- function(response, indices, smooth_k = 3, cyclic_k = 12) {
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
  temporal_terms <- c(
    sprintf("s(hour_of_day, bs='cc', k=%d)", cyclic_k),
    sprintf("s(day_of_year, bs='cc', k=%d)", cyclic_k)
  )

  # Random effects - only month_id (no station since per-station)
  random_terms <- c(
    "s(month_id, bs='re')"
  )

  # Combine all terms
  all_terms <- c(index_terms, covariate_terms, temporal_terms, random_terms)
  formula_str <- sprintf("%s ~ %s", response, paste(all_terms, collapse = " + "))

  as.formula(formula_str)
}

#' Get the mgcv family object
get_gam_family <- function(family_name) {
  switch(family_name,
    "nbinom2" = mgcv::nb(),
    "binomial" = binomial(),
    stop(sprintf("Unknown family: %s", family_name))
  )
}

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("\n=== Loading Data ===\n")

# Read the analysis-ready dataset
data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded %d observations\n", nrow(data)))

# Read the list of final indices
indices_df <- read.csv("data/processed/indices_final.csv")
indices <- indices_df$index_name[indices_df$kept == "True"]
cat(sprintf("  Using %d acoustic indices as predictors\n", length(indices)))

# Convert grouping variables to factors
data <- data %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  )

# Build list of predictors to scale
predictors_to_scale <- c(indices, "temperature", "depth")

# Ensure output directory exists
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# MAIN ANALYSIS LOOP
# ------------------------------------------------------------------------------

cat("\n=== Fitting Per-Station Models ===\n")
cat(sprintf("Total models to fit: %d (3 stations x %d metrics)\n\n",
            3 * length(responses), length(responses)))

# Store all results
all_results <- list()
model_count <- 0
total_models <- length(stations) * length(responses)

for (station_name in stations) {
  cat(sprintf("\n--- Station: %s ---\n", station_name))

  # Subset data to this station
  station_data <- data %>%
    filter(station == station_name)

  cat(sprintf("  Observations: %d\n", nrow(station_data)))

  for (metric in names(responses)) {
    model_count <- model_count + 1
    cat(sprintf("\n  [%d/%d] %s @ %s\n", model_count, total_models, metric, station_name))

    metric_info <- responses[[metric]]

    # Prepare data for this metric
    model_data <- station_data %>%
      filter(!is.na(.data[[metric]]))

    n_obs <- nrow(model_data)
    cat(sprintf("    Observations (after NA drop): %d\n", n_obs))

    if (n_obs < 100) {
      cat("    WARNING: Too few observations, skipping\n")
      next
    }

    # Scale predictors
    model_data <- scale_predictors(model_data, predictors_to_scale)

    # Build formula
    gamm_formula <- build_bystation_formula(metric, indices, smooth_k, cyclic_k)

    # Get the appropriate family
    gamm_family <- get_gam_family(metric_info$family)

    # Fit the model
    fit_start <- Sys.time()

    gamm_fit <- tryCatch({
      bam(
        formula = gamm_formula,
        data = model_data,
        family = gamm_family,
        method = "fREML",
        discrete = TRUE,
        select = TRUE,
        rho = FIXED_RHO
      )
    }, error = function(e) {
      cat(sprintf("    ERROR: %s\n", e$message))
      NULL
    })

    fit_time <- difftime(Sys.time(), fit_start, units = "secs")

    if (!is.null(gamm_fit)) {
      cat(sprintf("    Fitted in %.1f seconds\n", as.numeric(fit_time)))

      # Extract smooth term summary
      smooth_table <- as.data.frame(summary(gamm_fit)$s.table)
      smooth_table$term <- rownames(smooth_table)

      # Standardize column names
      stat_col <- intersect(c("Chi.sq", "F"), names(smooth_table))
      if (length(stat_col) > 0) {
        names(smooth_table)[names(smooth_table) == stat_col[1]] <- "statistic"
      }

      # Clean up the term names to extract just the variable
      smooth_table <- smooth_table %>%
        mutate(
          station = station_name,
          metric = metric,
          # Extract variable name from s(variable, k=3) or s(variable)
          variable = gsub("s\\(([^,)]+).*", "\\1", term),
          is_index = variable %in% indices
        ) %>%
        rename(
          p_value = `p-value`,
          ref_df = Ref.df
        ) %>%
        select(station, metric, term, variable, is_index, edf, ref_df, statistic, p_value)

      all_results[[paste(station_name, metric, sep = "_")]] <- smooth_table

      # Report significant indices
      sig_indices <- smooth_table %>%
        filter(is_index & p_value < 0.05) %>%
        nrow()
      cat(sprintf("    Significant indices (p<0.05): %d/%d\n", sig_indices, length(indices)))

    } else {
      cat("    Model fitting FAILED\n")
    }
  }
}

# ------------------------------------------------------------------------------
# COMBINE AND SAVE RESULTS
# ------------------------------------------------------------------------------

cat("\n\n=== Saving Results ===\n")

# Combine all results
results_df <- bind_rows(all_results)

# Save full results
write.csv(results_df,
          "results/tables/bystation_index_significance.csv",
          row.names = FALSE)
cat("Saved: results/tables/bystation_index_significance.csv\n")

# Create summary: which indices are significant at which stations for each metric
# Filter to just indices (not covariates or temporal terms)
index_results <- results_df %>%
  filter(is_index)

# Create a summary with significance indicators
summary_df <- index_results %>%
  mutate(
    significant = p_value < 0.05,
    sig_marker = ifelse(significant, "*", "")
  ) %>%
  select(station, metric, variable, edf, p_value, significant) %>%
  arrange(metric, variable, station)

# Create a wide-format summary showing station comparisons
summary_wide <- index_results %>%
  mutate(
    sig_level = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      p_value < 0.1 ~ ".",
      TRUE ~ ""
    )
  ) %>%
  select(metric, variable, station, edf, p_value, sig_level) %>%
  pivot_wider(
    id_cols = c(metric, variable),
    names_from = station,
    values_from = c(edf, p_value, sig_level),
    names_glue = "{station}_{.value}"
  ) %>%
  # Reorder columns for readability
  select(
    metric, variable,
    `9M_edf`, `9M_p_value`, `9M_sig_level`,
    `14M_edf`, `14M_p_value`, `14M_sig_level`,
    `37M_edf`, `37M_p_value`, `37M_sig_level`
  ) %>%
  arrange(metric, variable)

write.csv(summary_wide,
          "results/tables/bystation_significance_summary.csv",
          row.names = FALSE)
cat("Saved: results/tables/bystation_significance_summary.csv\n")

# ------------------------------------------------------------------------------
# PRINT SUMMARY
# ------------------------------------------------------------------------------

cat("\n=== Summary: Significant Indices by Station ===\n")
cat("(* p<0.05, ** p<0.01, *** p<0.001)\n\n")

# Count significant indices per station per metric
sig_counts <- index_results %>%
  group_by(metric, station) %>%
  summarise(
    n_significant = sum(p_value < 0.05),
    n_total = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = station,
    values_from = n_significant,
    names_prefix = "sig_"
  )

cat("Significant indices (p<0.05) by station:\n")
print(as.data.frame(sig_counts))

# Which indices are consistently significant across stations?
cat("\n\nIndices significant at ALL 3 stations (by metric):\n")
consistent_indices <- index_results %>%
  filter(p_value < 0.05) %>%
  group_by(metric, variable) %>%
  summarise(n_stations = n(), .groups = "drop") %>%
  filter(n_stations == 3)

if (nrow(consistent_indices) > 0) {
  for (m in unique(consistent_indices$metric)) {
    idx <- consistent_indices$variable[consistent_indices$metric == m]
    cat(sprintf("  %s: %s\n", m, paste(idx, collapse = ", ")))
  }
} else {
  cat("  (None)\n")
}

# Which indices are significant at ONLY some stations?
cat("\n\nIndices significant at 1-2 stations (station-specific effects):\n")
partial_indices <- index_results %>%
  filter(p_value < 0.05) %>%
  group_by(metric, variable) %>%
  summarise(
    n_stations = n(),
    which_stations = paste(station, collapse = ", "),
    .groups = "drop"
  ) %>%
  filter(n_stations < 3 & n_stations > 0)

if (nrow(partial_indices) > 0) {
  for (m in unique(partial_indices$metric)) {
    cat(sprintf("\n  %s:\n", m))
    subset <- partial_indices %>% filter(metric == m)
    for (i in 1:nrow(subset)) {
      cat(sprintf("    %s: %s\n", subset$variable[i], subset$which_stations[i]))
    }
  }
} else {
  cat("  (None)\n")
}

cat("\n=== Stage 05e Complete ===\n\n")

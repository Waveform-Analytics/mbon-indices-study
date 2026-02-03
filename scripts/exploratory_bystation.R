# ==============================================================================
# Exploratory: By-Station Baseline Comparison
# ==============================================================================
#
# Purpose:
#   Run the "do indices add value?" comparison independently for each station.
#   Supplementary analysis to stage05d_baseline.R (main pooled comparison).
#
# Context:
#   Liz asked whether the index-presence relationships hold at each site
#   independently, or if pooling data across stations masks site differences.
#
# Approach:
#   - Fit full model (with indices) per station
#   - Fit baseline model (no indices) per station
#   - Compare ΔAIC at each station
#
# Note: Station random effect removed since we're fitting per-station.
#       Month random effect kept to handle temporal structure.
#
# Output: results/tables/baseline_comparison_bystation.csv
#
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(mgcv)
})

set.seed(1234)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Response metrics to test (start with a subset for speed)
test_metrics <- c("fish_presence", "vessel_presence", "dolphin_presence")

# Families
families <- list(
  fish_presence = binomial(),
  vessel_presence = binomial(),
  dolphin_presence = binomial()
)

# 17 VIF-filtered index columns (from data/processed/analysis_ready.parquet)
index_cols <- c(
  "ACI", "BI", "BioEnergy", "ECV", "EPS_KURT", "EVNtCount", "EVNtMean",
  "H_Havrda", "KURTt", "MEANt", "NBPEAKS", "ROU", "SKEWt", "TFSD",
  "VARt", "ZCR", "nROI"
)

# ------------------------------------------------------------------------------
# FORMULAS (adapted for single-station - no station random effect)
# ------------------------------------------------------------------------------

build_full_formula <- function(response, index_cols) {
  index_terms <- paste0("s(", index_cols, ", k=5)", collapse = " + ")
  formula_str <- sprintf(
    "%s ~ %s + s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) + s(month_id, bs='re')",
    response, index_terms
  )
  as.formula(formula_str)
}

build_baseline_formula <- function(response) {
  formula_str <- sprintf(
    "%s ~ s(temperature, k=5) + s(depth, k=5) + s(hour_of_day, bs='cc', k=12) + s(day_of_year, bs='cc', k=12) + s(month_id, bs='re')",
    response
  )
  as.formula(formula_str)
}

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("Loading data...\n")
data <- arrow::read_parquet("data/processed/analysis_ready.parquet") %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  )

stations <- levels(data$station)
cat(sprintf("Stations: %s\n", paste(stations, collapse = ", ")))
cat(sprintf("Observations per station: ~%.0f\n\n", nrow(data) / length(stations)))

# ------------------------------------------------------------------------------
# RUN BY-STATION COMPARISON
# ------------------------------------------------------------------------------

results <- data.frame(

  station = character(),
  metric = character(),
  n_obs = integer(),
  full_aic = numeric(),
  baseline_aic = numeric(),
  delta_aic = numeric(),
  full_dev_expl = numeric(),
  baseline_dev_expl = numeric(),
  interpretation = character(),
  stringsAsFactors = FALSE
)

for (stn in stations) {
  cat(sprintf("=== Station: %s ===\n", stn))

  # Filter to this station
  data_stn <- data %>% filter(station == stn)
  cat(sprintf("  Observations: %d\n", nrow(data_stn)))

  for (metric in test_metrics) {
    cat(sprintf("  Metric: %s\n", metric))

    family <- families[[metric]]

    # Fit full model (with indices)
    full_formula <- build_full_formula(metric, index_cols)
    full_model <- tryCatch({
      bam(full_formula, data = data_stn, family = family,
          method = "fREML", discrete = TRUE, select = TRUE)
    }, error = function(e) {
      cat(sprintf("    FAILED full model: %s\n", e$message))
      NULL
    })

    # Fit baseline model (no indices)
    baseline_formula <- build_baseline_formula(metric)
    baseline_model <- tryCatch({
      bam(baseline_formula, data = data_stn, family = family,
          method = "fREML", discrete = TRUE, select = TRUE)
    }, error = function(e) {
      cat(sprintf("    FAILED baseline model: %s\n", e$message))
      NULL
    })

    # Extract metrics
    if (!is.null(full_model) && !is.null(baseline_model)) {
      full_aic <- AIC(full_model)
      baseline_aic <- AIC(baseline_model)
      delta_aic <- baseline_aic - full_aic

      full_dev <- summary(full_model)$dev.expl
      baseline_dev <- summary(baseline_model)$dev.expl

      interpretation <- if (delta_aic > 10) {
        "Strong"
      } else if (delta_aic > 4) {
        "Moderate"
      } else if (delta_aic > 0) {
        "Weak"
      } else {
        "None"
      }

      results <- rbind(results, data.frame(
        station = stn,
        metric = metric,
        n_obs = nrow(data_stn),
        full_aic = round(full_aic, 1),
        baseline_aic = round(baseline_aic, 1),
        delta_aic = round(delta_aic, 1),
        full_dev_expl = round(full_dev * 100, 1),
        baseline_dev_expl = round(baseline_dev * 100, 1),
        interpretation = interpretation,
        stringsAsFactors = FALSE
      ))

      cat(sprintf("    ΔAIC: %.1f (%s)\n", delta_aic, interpretation))
    } else {
      results <- rbind(results, data.frame(
        station = stn,
        metric = metric,
        n_obs = nrow(data_stn),
        full_aic = NA, baseline_aic = NA, delta_aic = NA,
        full_dev_expl = NA, baseline_dev_expl = NA,
        interpretation = "Failed",
        stringsAsFactors = FALSE
      ))
    }
  }
  cat("\n")
}

# ------------------------------------------------------------------------------
# OUTPUT RESULTS
# ------------------------------------------------------------------------------

cat("\n=== BY-STATION BASELINE COMPARISON ===\n\n")

# Pivot wider for easy reading
library(tidyr)
summary_wide <- results %>%
  select(station, metric, delta_aic, interpretation) %>%
  pivot_wider(
    names_from = station,
    values_from = c(delta_aic, interpretation),
    names_glue = "{station}_{.value}"
  )

print(summary_wide)

# Save full results
output_path <- "results/tables/baseline_comparison_bystation.csv"
write.csv(results, output_path, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", output_path))

# Summary interpretation
cat("\n=== INTERPRETATION ===\n")
for (m in test_metrics) {
  m_results <- results %>% filter(metric == m)
  all_strong <- all(m_results$interpretation == "Strong")
  all_positive <- all(m_results$delta_aic > 0, na.rm = TRUE)

  if (all_strong) {
    cat(sprintf("%s: Strong evidence at ALL stations - robust finding\n", m))
  } else if (all_positive) {
    cat(sprintf("%s: Positive evidence at all stations (varying strength)\n", m))
  } else {
    cat(sprintf("%s: Mixed results across stations\n", m))
  }
}

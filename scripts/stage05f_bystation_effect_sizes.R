# ==============================================================================
# Stage 05f: By-Station Effect Sizes
# ==============================================================================
#
# Purpose:
#   Calculate effect sizes for each acoustic index at each station.
#   This extends stage05e by computing effect magnitudes, not just significance.
#
# Inputs:
#   - data/processed/analysis_ready.parquet
#   - data/processed/indices_final.csv
#   - config/analysis.yml
#
# Outputs:
#   - results/tables/bystation_effect_sizes.csv
#
# Usage:
#   Rscript scripts/stage05f_bystation_effect_sizes.R
#
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(yaml)
  library(dplyr)
  library(tidyr)
  library(mgcv)
})

set.seed(1234)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

cat("\n=== Stage 05f: By-Station Effect Sizes ===\n")

config <- yaml::read_yaml("config/analysis.yml")
smooth_k <- config$gamm$smooth_k %||% 3
cyclic_k <- config$gamm$cyclic_k %||% 12

# Focus on presence models (most interpretable effect sizes)
responses <- list(
  fish_presence = list(family = "binomial", type = "presence"),
  dolphin_presence = list(family = "binomial", type = "presence"),
  vessel_presence = list(family = "binomial", type = "presence")
)

stations <- c("9M", "14M", "37M")
FIXED_RHO <- 0.1

cat(sprintf("Metrics: %s\n", paste(names(responses), collapse = ", ")))
cat(sprintf("Stations: %s\n", paste(stations, collapse = ", ")))

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

scale_predictors <- function(data, predictors) {
  scaling_info <- list()
  for (col in predictors) {
    col_mean <- mean(data[[col]], na.rm = TRUE)
    col_sd <- sd(data[[col]], na.rm = TRUE)
    scaling_info[[col]] <- list(mean = col_mean, sd = col_sd)
    if (col_sd > 0) {
      data[[col]] <- (data[[col]] - col_mean) / col_sd
    }
  }
  attr(data, "scaling_info") <- scaling_info
  data
}

build_bystation_formula <- function(response, indices, smooth_k = 3, cyclic_k = 12) {
  index_terms <- sapply(indices, function(idx) sprintf("s(%s, k=%d)", idx, smooth_k))
  covariate_terms <- c(
    sprintf("s(temperature, k=%d)", smooth_k),
    sprintf("s(depth, k=%d)", smooth_k)
  )
  temporal_terms <- c(
    sprintf("s(hour_of_day, bs='cc', k=%d)", cyclic_k),
    sprintf("s(day_of_year, bs='cc', k=%d)", cyclic_k)
  )
  random_terms <- "s(month_id, bs='re')"
  all_terms <- c(index_terms, covariate_terms, temporal_terms, random_terms)
  as.formula(sprintf("%s ~ %s", response, paste(all_terms, collapse = " + ")))
}

get_gam_family <- function(family_name) {
  switch(family_name,
    "binomial" = binomial(),
    "nbinom2" = mgcv::nb(),
    stop(sprintf("Unknown family: %s", family_name))
  )
}

#' Calculate effect sizes for a fitted model
calculate_effect_sizes <- function(model, data, index_cols, model_type, scaling_info) {
  results <- list()

  zscore <- function(x, predictor) {
    info <- scaling_info[[predictor]]
    if (is.null(info)) return(x)
    (x - info$mean) / info$sd
  }

  model_terms <- attr(terms(model), "term.labels")

  for (idx in index_cols) {
    if (!idx %in% names(data)) next
    if (!any(grepl(idx, model_terms))) next

    idx_values <- data[[idx]]
    idx_values <- idx_values[!is.na(idx_values)]
    if (length(idx_values) < 10) next

    # Use RAW data percentiles (before scaling)
    raw_data <- data
    # Unscale to get raw values
    if (!is.null(scaling_info[[idx]])) {
      raw_values <- idx_values * scaling_info[[idx]]$sd + scaling_info[[idx]]$mean
    } else {
      raw_values <- idx_values
    }

    low_val_raw <- quantile(raw_values, 0.10, na.rm = TRUE)
    high_val_raw <- quantile(raw_values, 0.90, na.rm = TRUE)

    # Create newdata with scaled values
    newdata_base <- data.frame(matrix(ncol = 0, nrow = 2))

    for (col in model_terms) {
      if (col %in% c("month_id")) next
      if (!col %in% names(data)) next
      if (!is.numeric(data[[col]])) next

      if (col == idx) {
        newdata_base[[col]] <- c(zscore(low_val_raw, col), zscore(high_val_raw, col))
      } else {
        # Use median of SCALED data (which is ~0 for z-scored)
        newdata_base[[col]] <- rep(0, 2)
      }
    }

    newdata_base$month_id <- factor(rep(levels(data$month_id)[1], 2),
                                    levels = levels(data$month_id))

    tryCatch({
      preds <- predict(model, newdata = newdata_base, type = "response", se.fit = FALSE)

      low_pred <- preds[1]
      high_pred <- preds[2]

      if (model_type == "presence") {
        effect_size <- high_pred - low_pred
        effect_type <- "probability_change"
      } else {
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
      # Skip silently
    })
  }

  if (length(results) == 0) return(data.frame())
  do.call(rbind, results)
}

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

cat("\n=== Loading Data ===\n")

data <- arrow::read_parquet("data/processed/analysis_ready.parquet")
cat(sprintf("  Loaded %d observations\n", nrow(data)))

indices_df <- read.csv("data/processed/indices_final.csv")
indices <- indices_df$index_name[indices_df$kept == "True"]
cat(sprintf("  Using %d acoustic indices\n", length(indices)))

data <- data %>%
  mutate(
    station = as.factor(station),
    month_id = as.factor(month_id)
  )

predictors_to_scale <- c(indices, "temperature", "depth")
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------------------------

cat("\n=== Calculating Per-Station Effect Sizes ===\n")

all_results <- list()
model_count <- 0
total_models <- length(stations) * length(responses)

for (station_name in stations) {
  cat(sprintf("\n--- Station: %s ---\n", station_name))

  station_data <- data %>% filter(station == station_name)
  cat(sprintf("  Observations: %d\n", nrow(station_data)))

  for (metric in names(responses)) {
    model_count <- model_count + 1
    cat(sprintf("\n  [%d/%d] %s @ %s\n", model_count, total_models, metric, station_name))

    metric_info <- responses[[metric]]

    model_data <- station_data %>% filter(!is.na(.data[[metric]]))
    n_obs <- nrow(model_data)

    if (n_obs < 100) {
      cat("    Skipping (too few observations)\n")
      next
    }

    # Scale predictors and save scaling info
    model_data <- scale_predictors(model_data, predictors_to_scale)
    scaling_info <- attr(model_data, "scaling_info")

    gamm_formula <- build_bystation_formula(metric, indices, smooth_k, cyclic_k)
    gamm_family <- get_gam_family(metric_info$family)

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

      # Calculate effect sizes
      effect_sizes <- calculate_effect_sizes(
        gamm_fit, model_data, indices, metric_info$type, scaling_info
      )

      if (nrow(effect_sizes) > 0) {
        effect_sizes$station <- station_name
        effect_sizes$metric <- metric
        all_results[[paste(station_name, metric, sep = "_")]] <- effect_sizes

        # Report top 3
        top3 <- effect_sizes[order(abs(effect_sizes$effect_size), decreasing = TRUE), ]
        top3 <- head(top3, 3)
        cat("    Top 3 by effect size:\n")
        for (i in 1:nrow(top3)) {
          cat(sprintf("      %s: %+.1f pp\n", top3$index[i], top3$effect_size[i] * 100))
        }
      }
    }
  }
}

# ------------------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------------------

cat("\n\n=== Saving Results ===\n")

results_df <- bind_rows(all_results)

# Reorder columns
results_df <- results_df[, c("metric", "station", "index", "low_value", "high_value",
                              "low_pred", "high_pred", "effect_size", "effect_type")]

write.csv(results_df, "results/tables/bystation_effect_sizes.csv", row.names = FALSE)
cat(sprintf("Saved: results/tables/bystation_effect_sizes.csv (%d rows)\n", nrow(results_df)))

# ------------------------------------------------------------------------------
# CREATE SUMMARY TABLE
# ------------------------------------------------------------------------------

cat("\n=== Top 5 Indices by Station (by effect size) ===\n")

for (m in names(responses)) {
  cat(sprintf("\n%s:\n", m))
  mdf <- results_df[results_df$metric == m, ]

  for (s in stations) {
    sdf <- mdf[mdf$station == s, ]
    sdf <- sdf[order(abs(sdf$effect_size), decreasing = TRUE), ]
    top5 <- head(sdf, 5)

    top5_str <- paste(sprintf("%s (%+.0f pp)", top5$index, top5$effect_size * 100), collapse = ", ")
    cat(sprintf("  %s: %s\n", s, top5_str))
  }
}

cat("\n=== Stage 05f Complete ===\n\n")

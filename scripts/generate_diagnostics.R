# ==============================================================================
# Generate GAMM Diagnostic Plots
# ==============================================================================
#
# Purpose:
#   Load fitted GAMM models and generate diagnostic plots for each metric.
#   Produces residual vs fitted, Q-Q plot, histogram of residuals, and
#   response vs fitted plots.
#
# Outputs:
#   - results/figures/<metric>/gamm_diagnostics.png
#
# ==============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(ggplot2)
  library(gridExtra)
  library(dplyr)
})

# Define metrics (same as main modeling script)
metrics <- c(

"fish_activity",
  "fish_richness",
  "fish_presence",
  "dolphin_burst_pulse",
  "dolphin_echolocation",
  "dolphin_whistle",
  "dolphin_activity",
  "dolphin_presence",
  "vessel_presence"
)

cat("Generating diagnostic plots for all metrics...\n\n")

for (metric in metrics) {
  cat(sprintf("Processing %s...\n", metric))

  # Load the model
model_path <- file.path("results/models", metric, "gamm.rds")

  if (!file.exists(model_path)) {
    cat(sprintf("  WARNING: Model not found at %s, skipping\n", model_path))
    next
  }

  model <- readRDS(model_path)

  # Extract the gam component (gamm returns a list with $gam and $lme)
  if (inherits(model, "list") && "gam" %in% names(model)) {
    gam_obj <- model$gam
  } else {
    gam_obj <- model
  }

  # Get residuals and fitted values
  resids <- residuals(gam_obj, type = "deviance")
  fitted_vals <- fitted(gam_obj)
  response_vals <- gam_obj$y

  # Create a data frame for plotting
  diag_df <- data.frame(
    fitted = fitted_vals,
    residuals = resids,
    response = response_vals
  )

  # Plot 1: Residuals vs Fitted
  p1 <- ggplot(diag_df, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE, color = "blue", linewidth = 0.8) +
    labs(
      title = "Residuals vs Fitted",
      x = "Fitted values",
      y = "Deviance residuals"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))

  # Plot 2: Q-Q plot of residuals
  p2 <- ggplot(diag_df, aes(sample = residuals)) +
    stat_qq(alpha = 0.3, size = 0.5) +
    stat_qq_line(color = "red", linetype = "dashed") +
    labs(
      title = "Q-Q Plot of Residuals",
      x = "Theoretical quantiles",
      y = "Sample quantiles"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))

  # Plot 3: Histogram of residuals
  p3 <- ggplot(diag_df, aes(x = residuals)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50,
                   fill = "steelblue", alpha = 0.7, color = "white") +
    geom_density(color = "red", linewidth = 0.8) +
    labs(
      title = "Distribution of Residuals",
      x = "Deviance residuals",
      y = "Density"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))

  # Plot 4: Response vs Fitted
  p4 <- ggplot(diag_df, aes(x = fitted, y = response)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    labs(
      title = "Response vs Fitted",
      x = "Fitted values",
      y = "Observed response"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))

  # Combine into a 2x2 grid
  combined_plot <- grid.arrange(
    p1, p2, p3, p4,
    ncol = 2,
    top = grid::textGrob(
      paste0("GAMM Diagnostics: ", metric),
      gp = grid::gpar(fontsize = 14, fontface = "bold")
    )
  )

  # Save the plot
  output_path <- file.path("results/figures", metric, "gamm_diagnostics.png")

  ggsave(
    output_path,
    plot = combined_plot,
    width = 10,
    height = 8,
    dpi = 150,
    bg = "white"
  )

  cat(sprintf("  Saved: %s\n", output_path))
}

cat("\nDiagnostic plot generation complete!\n")
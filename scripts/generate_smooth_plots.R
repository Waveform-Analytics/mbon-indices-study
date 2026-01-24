# ==============================================================================
# Generate Improved Smooth Plots
# ==============================================================================
#
# Creates publication-quality smooth term visualizations:
#   - Larger panels with minimal spacing
#   - Index name and EDF as annotations inside plot area
#   - Clean, readable fonts
#   - Common y-axis scale for comparability
#
# Outputs:
#   - results/figures/<metric>/gamm_smooths.png (improved grid)
#
# ==============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(gratia)  # For GAM visualization
})

# Define metrics
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

cat("=== Generating Improved Smooth Plots ===\n\n")

# Check if gratia is available, install if needed
if (!requireNamespace("gratia", quietly = TRUE)) {
  cat("Installing gratia package...\n")
  install.packages("gratia", repos = "https://cloud.r-project.org", quiet = TRUE)
  library(gratia)
}

for (metric in metrics) {
  cat(sprintf("Processing %s...\n", metric))

  # Load the model
  model_path <- file.path("results/models", metric, "gamm.rds")

  if (!file.exists(model_path)) {
    cat(sprintf("  WARNING: Model not found at %s, skipping\n", model_path))
    next
  }

  model <- readRDS(model_path)

  # Extract GAM component
  if (inherits(model, "list") && "gam" %in% names(model)) {
    gam_obj <- model$gam
  } else {
    gam_obj <- model
  }

  # Get smooth terms info
  smooth_names <- gratia::smooths(gam_obj)

  # If it's a character vector, convert to data frame
  if (is.character(smooth_names)) {
    smooth_info <- data.frame(smooth = smooth_names, stringsAsFactors = FALSE)
  } else {
    smooth_info <- smooth_names
  }

  # Skip random effects (station, month_id as RE)
  # We want to plot continuous smooths only
  smooth_info <- smooth_info %>%
    filter(!grepl("s\\(station", smooth) & !grepl("bs='re'", smooth, fixed = TRUE))

  # Calculate grid dimensions
  n_plots <- nrow(smooth_info)
  n_cols <- 4
  n_rows <- ceiling(n_plots / n_cols)

  # ===========================================================================
  # FIRST PASS: Gather all smooth data to calculate common y-axis limits
  # ===========================================================================
  all_smooth_data <- list()
  all_edf <- list()

  for (i in seq_len(n_plots)) {
    smooth_name <- smooth_info$smooth[i]
    var_name <- gsub("s\\(([^,\\)]+).*", "\\1", smooth_name)

    sm_data <- tryCatch({
      gratia::smooth_estimates(gam_obj, select = smooth_name, n = 100)
    }, error = function(e) {
      NULL
    })

    if (!is.null(sm_data)) {
      # Get EDF
      sm_summary <- summary(gam_obj)$s.table
      edf_row <- grep(var_name, rownames(sm_summary), fixed = TRUE)
      edf_val <- if (length(edf_row) > 0) round(sm_summary[edf_row[1], "edf"], 2) else NA

      all_smooth_data[[var_name]] <- sm_data
      all_edf[[var_name]] <- edf_val
    }
  }

  # Calculate common y-axis limits across all smooths (including CI)
  y_min <- Inf
  y_max <- -Inf
  for (sm_data in all_smooth_data) {
    ci_low <- sm_data$.estimate - 1.96 * sm_data$.se
    ci_high <- sm_data$.estimate + 1.96 * sm_data$.se
    y_min <- min(y_min, min(ci_low, na.rm = TRUE))
    y_max <- max(y_max, max(ci_high, na.rm = TRUE))
  }
  # Add 10% padding for labels
  y_range <- y_max - y_min
  y_min <- y_min - 0.05 * y_range
  y_max <- y_max + 0.1 * y_range  # More padding at top for label

  # ===========================================================================
  # SECOND PASS: Create plots with common y-axis
  # ===========================================================================
  plot_list <- list()

  for (var_name in names(all_smooth_data)) {
    sm_data <- all_smooth_data[[var_name]]
    edf_val <- all_edf[[var_name]]

    # Determine x variable column name
    x_col <- var_name
    if (!x_col %in% names(sm_data)) {
      possible_cols <- setdiff(names(sm_data), c(".smooth", ".type", ".by", ".estimate", ".se"))
      if (length(possible_cols) > 0) {
        x_col <- possible_cols[1]
      }
    }

    # Skip if x column is a factor (can't plot continuous smooth)
    if (is.factor(sm_data[[x_col]]) || is.character(sm_data[[x_col]])) {
      next
    }

    # Get x range for positioning labels
    x_min_data <- min(sm_data[[x_col]], na.rm = TRUE)
    x_max_data <- max(sm_data[[x_col]], na.rm = TRUE)
    x_range_data <- x_max_data - x_min_data

    # Position labels inside plot area (upper-left quadrant)
    label_x <- x_min_data + 0.05 * x_range_data
    label_y_name <- y_max - 0.15 * y_range  # Index name
    label_y_edf <- y_max - 0.28 * y_range   # EDF below it

    # Create the plot with common y-axis
    p <- ggplot(sm_data, aes(x = .data[[x_col]], y = .estimate)) +
      geom_ribbon(aes(ymin = .estimate - 1.96 * .se,
                      ymax = .estimate + 1.96 * .se),
                  fill = "grey70", alpha = 0.5) +
      geom_line(linewidth = 0.8) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.3) +
      # Annotate with variable name (upper-left, inside plot)
      annotate("text", x = label_x, y = label_y_name,
               label = var_name,
               hjust = 0, vjust = 0.5, fontface = "bold", size = 5.5) +
      # Annotate with EDF (below index name)
      annotate("text", x = label_x, y = label_y_edf,
               label = ifelse(!is.na(edf_val), paste0("EDF=", edf_val), ""),
               hjust = 0, vjust = 0.5, size = 4.5, color = "grey40") +
      labs(x = NULL, y = NULL) +
      coord_cartesian(ylim = c(y_min, y_max)) +
      theme_minimal(base_size = 14) +
      theme(
        plot.margin = margin(6, 6, 6, 6),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(linewidth = 0.2, color = "grey90"),
        axis.text = element_text(size = 12),
        axis.ticks = element_line(linewidth = 0.2)
      )

    plot_list[[var_name]] <- p
  }

  # Combine plots
  if (length(plot_list) > 0) {
    # Calculate common y-axis limits across all plots for this metric
    # (already on link scale, so comparable)

    combined <- wrap_plots(plot_list, ncol = n_cols) +
      plot_annotation(
        title = paste0(metric, " — Smooth Terms"),
        theme = theme(
          plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
        )
      )

    # Save with larger dimensions and higher DPI
    output_path <- file.path("results/figures", metric, "gamm_smooths.png")
    ggsave(
      output_path,
      plot = combined,
      width = 14,
      height = 3.5 * n_rows,
      dpi = 150,
      bg = "white"
    )

    cat(sprintf("  Saved: %s (%d smooths)\n", output_path, length(plot_list)))
  }
}

cat("\n=== Done! ===\n")
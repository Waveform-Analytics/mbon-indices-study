#!/usr/bin/env Rscript
# visualize_diagnostics.R
# Generate figures for diagnostic checks section of results-viewer.qmd

library(ggplot2)
library(dplyr)
library(tidyr)

# Output directory (results/ is source of truth; rebuild-docs.sh copies to docs/)
out_dir <- "results/figures"

# =============================================================================
# Figure 1: Effect Sizes for Presence Models
# =============================================================================

cat("Creating effect sizes plot for presence models...\n")

effect_sizes <- read.csv("results/tables/effect_sizes.csv", stringsAsFactors = FALSE)

# Filter to presence models only (probability_change)
presence_effects <- effect_sizes %>%
  filter(effect_type == "probability_change") %>%
  mutate(
    metric_label = case_when(
      metric == "fish_presence" ~ "Fish Presence",
      metric == "dolphin_presence" ~ "Dolphin Presence",
      metric == "vessel_presence" ~ "Vessel Presence",
      TRUE ~ metric
    ),
    # Color by direction
    direction = ifelse(effect_size > 0, "Positive", "Negative"),
    # Flag large effects
    is_large = abs(effect_size) > 0.1
  )

# Order metrics by best performance
presence_effects$metric_label <- factor(
  presence_effects$metric_label,
  levels = c("Vessel Presence", "Fish Presence", "Dolphin Presence")
)

# Create dot plot
p1 <- ggplot(presence_effects, aes(x = effect_size, y = reorder(index, effect_size))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = c(-0.1, 0.1), linetype = "dotted", color = "gray70") +
  geom_point(aes(color = direction, size = is_large), alpha = 0.8) +
  scale_color_manual(values = c("Positive" = "#2166ac", "Negative" = "#b2182b")) +
  scale_size_manual(values = c("TRUE" = 3, "FALSE" = 2), guide = "none") +
  facet_wrap(~metric_label, ncol = 1, scales = "free_y") +
  labs(
    title = "Effect Sizes: Presence Models",
    subtitle = "Change in probability when index moves from 10th to 90th percentile",
    x = "Probability Change",
    y = NULL,
    color = "Direction"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 9)
  ) +
  annotate("text", x = 0.35, y = 1, label = "Dotted lines: ±0.10 threshold",
           size = 3, color = "gray50", hjust = 1)

ggsave(file.path(out_dir, "effect_sizes_presence.png"), p1,
       width = 8, height = 10, dpi = 150, bg = "white")
cat("  Saved:", file.path(out_dir, "effect_sizes_presence.png"), "\n")


# =============================================================================
# Figure 2: Effect Sizes for Count Models
# =============================================================================

cat("Creating effect sizes plot for count models...\n")

count_effects <- effect_sizes %>%
  filter(effect_type == "fold_change") %>%
  filter(!is.na(effect_size)) %>%
  mutate(
    metric_label = case_when(
      metric == "fish_activity" ~ "Fish Activity",
      metric == "fish_richness" ~ "Fish Richness",
      metric == "dolphin_burst_pulse" ~ "Dolphin Burst Pulse",
      metric == "dolphin_echolocation" ~ "Dolphin Echolocation",
      metric == "dolphin_whistle" ~ "Dolphin Whistle",
      metric == "dolphin_activity" ~ "Dolphin Activity",
      TRUE ~ metric
    ),
    # Log transform for visualization (fold change is multiplicative)
    log_effect = log2(effect_size),
    direction = ifelse(effect_size > 1, "Increase", "Decrease"),
    is_large = abs(log_effect) > 0.3  # ~1.23x fold change
  )

p2 <- ggplot(count_effects, aes(x = effect_size, y = reorder(index, effect_size))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = c(0.8, 1.25), linetype = "dotted", color = "gray70") +
  geom_point(aes(color = direction, size = is_large), alpha = 0.8) +
  scale_color_manual(values = c("Increase" = "#2166ac", "Decrease" = "#b2182b")) +
  scale_size_manual(values = c("TRUE" = 3, "FALSE" = 2), guide = "none") +
  scale_x_continuous(breaks = c(0.5, 0.75, 1, 1.25, 1.5), limits = c(0.4, 1.7)) +
  facet_wrap(~metric_label, ncol = 2, scales = "free_y") +
  labs(
    title = "Effect Sizes: Count Models",
    subtitle = "Fold-change in expected count (10th to 90th percentile of index)",
    x = "Fold Change (1 = no effect)",
    y = NULL,
    color = "Direction"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 10),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 8)
  )

ggsave(file.path(out_dir, "effect_sizes_count.png"), p2,
       width = 10, height = 8, dpi = 150, bg = "white")
cat("  Saved:", file.path(out_dir, "effect_sizes_count.png"), "\n")


# =============================================================================
# Figure 3: Zero-Inflation Gap
# =============================================================================

cat("Creating zero-inflation gap plot...\n")

zero_inf <- read.csv("results/tables/zero_inflation_check.csv", stringsAsFactors = FALSE)

# Reshape for plotting
zero_inf_long <- zero_inf %>%
  mutate(
    metric_label = case_when(
      metric == "fish_activity" ~ "Fish Activity",
      metric == "fish_richness" ~ "Fish Richness",
      metric == "dolphin_burst_pulse" ~ "Dolphin Burst Pulse",
      metric == "dolphin_echolocation" ~ "Dolphin Echolocation",
      metric == "dolphin_whistle" ~ "Dolphin Whistle",
      metric == "dolphin_activity" ~ "Dolphin Activity",
      TRUE ~ metric
    )
  ) %>%
  select(metric_label, observed_zero_prop, predicted_zero_prop, gap, flagged) %>%
  pivot_longer(
    cols = c(observed_zero_prop, predicted_zero_prop),
    names_to = "type",
    values_to = "proportion"
  ) %>%
  mutate(
    type_label = ifelse(type == "observed_zero_prop", "Observed", "Model Predicts")
  )

# Order by gap size
metric_order <- zero_inf %>%
  arrange(desc(gap)) %>%
  mutate(metric_label = case_when(
    metric == "fish_activity" ~ "Fish Activity",
    metric == "fish_richness" ~ "Fish Richness",
    metric == "dolphin_burst_pulse" ~ "Dolphin Burst Pulse",
    metric == "dolphin_echolocation" ~ "Dolphin Echolocation",
    metric == "dolphin_whistle" ~ "Dolphin Whistle",
    metric == "dolphin_activity" ~ "Dolphin Activity",
    TRUE ~ metric
  )) %>%
  pull(metric_label)

zero_inf_long$metric_label <- factor(zero_inf_long$metric_label, levels = metric_order)

p3 <- ggplot(zero_inf_long, aes(x = metric_label, y = proportion * 100, fill = type_label)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(
    data = zero_inf %>% mutate(
      metric_label = factor(case_when(
        metric == "fish_activity" ~ "Fish Activity",
        metric == "fish_richness" ~ "Fish Richness",
        metric == "dolphin_burst_pulse" ~ "Dolphin Burst Pulse",
        metric == "dolphin_echolocation" ~ "Dolphin Echolocation",
        metric == "dolphin_whistle" ~ "Dolphin Whistle",
        metric == "dolphin_activity" ~ "Dolphin Activity",
        TRUE ~ metric
      ), levels = metric_order)
    ),
    aes(x = metric_label, y = observed_zero_prop * 100 + 3,
        label = sprintf("Gap: %.0f%%", gap * 100)),
    inherit.aes = FALSE, size = 3.5, fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("Observed" = "#4393c3", "Model Predicts" = "#d6604d"),
    name = NULL
  ) +
  labs(
    title = "Zero-Inflation: Observed vs. Predicted Zeros",
    subtitle = "Models underpredict zeros — they expect more activity than actually occurs",
    x = NULL,
    y = "Percentage of Zeros"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 10)
  ) +
  coord_cartesian(ylim = c(0, 105))

ggsave(file.path(out_dir, "zero_inflation_gap.png"), p3,
       width = 9, height = 6, dpi = 150, bg = "white")
cat("  Saved:", file.path(out_dir, "zero_inflation_gap.png"), "\n")


# =============================================================================
# Figure 4: Combined summary - Top effects only
# =============================================================================

cat("Creating top effects summary plot...\n")

# Get top 3 effects per metric for presence models
top_presence <- presence_effects %>%
  group_by(metric_label) %>%
  arrange(desc(abs(effect_size))) %>%
  slice_head(n = 3) %>%
  ungroup() %>%
  mutate(rank = rep(1:3, times = 3))

p4 <- ggplot(top_presence, aes(x = effect_size, y = metric_label, color = direction)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 4, alpha = 0.9) +
  geom_text(aes(label = index), vjust = -0.8, size = 3, show.legend = FALSE) +
  scale_color_manual(values = c("Positive" = "#2166ac", "Negative" = "#b2182b")) +
  labs(
    title = "Top 3 Index Effects per Presence Model",
    subtitle = "Change in detection probability (10th to 90th percentile)",
    x = "Probability Change",
    y = NULL,
    color = "Direction"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 11, face = "bold")
  ) +
  scale_x_continuous(limits = c(-0.5, 0.6), breaks = seq(-0.4, 0.6, 0.2))

ggsave(file.path(out_dir, "top_effects_presence.png"), p4,
       width = 8, height = 4, dpi = 150, bg = "white")
cat("  Saved:", file.path(out_dir, "top_effects_presence.png"), "\n")


cat("\nDone! Created 4 figures in", out_dir, "\n")

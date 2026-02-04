#!/bin/bash
# ==============================================================================
# Rebuild Documentation
# ==============================================================================
#
# This script:
#   1. Copies figures from results/ to docs/ for web serving
#   2. Renders the Quarto results viewer
#
# Usage:
#   ./rebuild-docs.sh
#
# ==============================================================================

set -e  # Exit on error

echo "=== Rebuilding Documentation ==="

# Navigate to project root
cd "$(dirname "$0")"

# Step 1: Copy only essential figures (not exploratory)
echo ""
echo "Copying essential figures to docs/..."
rm -rf docs/figures  # Clean old copies
mkdir -p docs/figures

# Copy per-metric figures (smooths and diagnostics)
for metric in fish_activity fish_richness fish_presence \
              dolphin_burst_pulse dolphin_echolocation dolphin_whistle \
              dolphin_activity dolphin_presence vessel_presence; do
  mkdir -p "docs/figures/$metric"
  cp "results/figures/$metric/gamm_smooths_top12.png" "docs/figures/$metric/" 2>/dev/null || true
  cp "results/figures/$metric/gamm_smooths.png" "docs/figures/$metric/" 2>/dev/null || true
  cp "results/figures/$metric/gamm_diagnostics.png" "docs/figures/$metric/" 2>/dev/null || true
  cp "results/figures/$metric/random_effects_qq.png" "docs/figures/$metric/" 2>/dev/null || true
  cp "results/figures/$metric/residuals_by_station.png" "docs/figures/$metric/" 2>/dev/null || true
done

# Copy validation/summary figures
cp results/figures/acf_residuals.png docs/figures/ 2>/dev/null || true
cp results/figures/cv_performance_by_metric.png docs/figures/ 2>/dev/null || true

# Copy diagnostic figures
cp results/figures/effect_sizes_presence.png docs/figures/ 2>/dev/null || true
cp results/figures/zero_inflation_gap.png docs/figures/ 2>/dev/null || true

echo "  Copied $(find docs/figures -name '*.png' | wc -l | tr -d ' ') PNG files"

# Step 2: Copy tables (for reference/download)
echo ""
echo "Copying key tables to docs/..."
mkdir -p docs/tables
cp results/tables/model_summary.csv docs/tables/
cp results/tables/cv_performance_summary.csv docs/tables/
cp results/tables/ar1_validation.csv docs/tables/
cp results/tables/effect_sizes.csv docs/tables/ 2>/dev/null || true
cp results/tables/zero_inflation_check.csv docs/tables/ 2>/dev/null || true
cp results/tables/baseline_comparison.csv docs/tables/ 2>/dev/null || true
cp results/tables/baseline_comparison_bystation.csv docs/tables/ 2>/dev/null || true
cp results/tables/bystation_significance_summary.csv docs/tables/ 2>/dev/null || true
cp results/tables/vif_validation.csv docs/tables/ 2>/dev/null || true
echo "  Copied summary tables"

# Step 3: Render Quarto
echo ""
echo "Rendering Quarto document..."
quarto render docs/results-viewer.qmd

echo ""
echo "=== Done! ==="
echo "View locally: open docs/results-viewer.html"
echo "After pushing, view at: https://<username>.github.io/mbon-indices-study/results-viewer.html"
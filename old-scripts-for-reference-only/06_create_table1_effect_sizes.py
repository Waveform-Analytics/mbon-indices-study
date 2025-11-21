#!/usr/bin/env python3
"""
Script 6: Create Table 1 - Effect Size Comparison
=================================================

Purpose: Compare effect sizes of acoustic indices vs environmental covariates
Input: glmm_coef_*.csv files
Output: table1_effect_sizes.csv

Table shows:
- Standardized coefficients (or raw coefficients with SE)
- For the 5 acoustic indices vs key environmental covariates
- Across all response variables
- Highlights which indices have large vs small effects
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import pipeline utilities
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models to include (exclude vessel_count)
RESPONSE_VARS = [
    'fish_present', 'fish_richness', 'fish_activity',
    'dolphin_present', 'dolphin_activity', 'dolphin_whistles',
    'dolphin_burst_pulses', 'dolphin_echolocation',
    'vessel_present'
]

# Response variable name mapping (abbreviated for table)
RESPONSE_NAMES = {
    'fish_present': 'Fish Pres.',
    'fish_richness': 'Fish Rich.',
    'fish_activity': 'Fish Act.',
    'dolphin_present': 'Dol. Pres.',
    'dolphin_activity': 'Dol. Act.',
    'dolphin_whistles': 'Dol. Whist.',
    'dolphin_burst_pulses': 'Dol. BP',
    'dolphin_echolocation': 'Dol. Echol.',
    'vessel_present': 'Vessel Pres.'
}

# Predictors to extract
ACOUSTIC_INDICES = ['BI', 'EAS', 'EPS_KURT', 'EVNtMean', 'nROI']
ENVIRONMENTAL_COVARIATES = ['temp', 'depth']
ALL_PREDICTORS = ACOUSTIC_INDICES + ENVIRONMENTAL_COVARIATES


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 6: CREATE TABLE 1 - EFFECT SIZE COMPARISON")
    print("=" * 60)
    print()

    # Setup paths
    paths = ProjectPaths()
    glmm_dir = paths.processed_data / "glmm_results"

    # ========================================================================
    # LOAD COEFFICIENTS
    # ========================================================================

    print("📥 LOADING COEFFICIENTS")
    print("-" * 40)

    all_coefs = []

    for response_var in RESPONSE_VARS:
        coef_file = glmm_dir / f"glmm_coef_{response_var}.csv"

        if not coef_file.exists():
            print(f"⚠️  Missing: {response_var}")
            continue

        coef_df = pd.read_csv(coef_file)

        # Filter to fixed effects only
        coef_df = coef_df[coef_df['effect'] == 'fixed']

        # Extract predictors of interest
        coef_df = coef_df[coef_df['term'].isin(ALL_PREDICTORS)]

        # Add response variable
        coef_df['response'] = response_var

        all_coefs.append(coef_df[['response', 'term', 'estimate', 'std.error', 'statistic']])
        print(f"✓ Loaded: {response_var}")

    # Combine all coefficients
    coef_combined = pd.concat(all_coefs, ignore_index=True)
    print(f"\n✓ Combined: {len(coef_combined)} coefficients across {len(RESPONSE_VARS)} models")
    print()

    # ========================================================================
    # CREATE EFFECT SIZE TABLE
    # ========================================================================

    print("🔨 BUILDING TABLE 1")
    print("-" * 40)

    # Pivot to wide format: rows = predictors, columns = response variables
    table1_estimates = coef_combined.pivot(
        index='term',
        columns='response',
        values='estimate'
    )

    # Reorder rows: indices first, then environmental
    table1_estimates = table1_estimates.reindex(ALL_PREDICTORS)

    # Reorder columns to match response variable order
    table1_estimates = table1_estimates[RESPONSE_VARS]

    # Rename columns for publication
    table1_estimates.columns = [RESPONSE_NAMES[col] for col in table1_estimates.columns]

    # Add predictor type column
    predictor_types = []
    for pred in ALL_PREDICTORS:
        if pred in ACOUSTIC_INDICES:
            predictor_types.append('Acoustic Index')
        else:
            predictor_types.append('Environmental')

    table1_estimates.insert(0, 'Type', predictor_types)

    # Reset index to make 'term' a column
    table1_estimates = table1_estimates.reset_index()
    table1_estimates = table1_estimates.rename(columns={'term': 'Predictor'})

    print(f"✓ Created table with {len(table1_estimates)} predictors × {len(RESPONSE_VARS)} responses")
    print()

    # ========================================================================
    # CREATE MAGNITUDE SUMMARY
    # ========================================================================

    print("📊 COMPUTING EFFECT MAGNITUDE SUMMARY")
    print("-" * 40)

    # For each predictor, compute mean absolute effect size across responses
    magnitude_summary = []

    for predictor in ALL_PREDICTORS:
        pred_coefs = coef_combined[coef_combined['term'] == predictor]
        mean_abs_effect = pred_coefs['estimate'].abs().mean()
        max_abs_effect = pred_coefs['estimate'].abs().max()
        n_large_effects = (pred_coefs['estimate'].abs() > 1.0).sum()

        magnitude_summary.append({
            'Predictor': predictor,
            'Mean |Coef|': f"{mean_abs_effect:.3f}",
            'Max |Coef|': f"{max_abs_effect:.3f}",
            'N Large (>1)': n_large_effects
        })

    magnitude_df = pd.DataFrame(magnitude_summary)
    print(magnitude_df.to_string(index=False))
    print()

    # ========================================================================
    # DISPLAY TABLE
    # ========================================================================

    print("📋 TABLE 1 PREVIEW (coefficients)")
    print("-" * 60)

    # Round for display
    table1_display = table1_estimates.copy()
    numeric_cols = table1_display.select_dtypes(include=[np.number]).columns
    table1_display[numeric_cols] = table1_display[numeric_cols].round(3)

    print(table1_display.to_string(index=False))
    print()

    # ========================================================================
    # IDENTIFY KEY FINDINGS
    # ========================================================================

    print("🔍 KEY FINDINGS")
    print("-" * 40)

    # Find largest effects for each predictor
    for predictor in ACOUSTIC_INDICES:
        pred_data = coef_combined[coef_combined['term'] == predictor]
        max_idx = pred_data['estimate'].abs().idxmax()
        max_row = pred_data.loc[max_idx]
        print(f"{predictor:12s}: Largest effect for {max_row['response']:20s} ({max_row['estimate']:+.3f})")

    print()

    # Compare index vs environmental covariate magnitudes
    print("Effect size comparison (mean absolute coefficient):")
    index_mean = coef_combined[coef_combined['term'].isin(ACOUSTIC_INDICES)]['estimate'].abs().mean()
    env_mean = coef_combined[coef_combined['term'].isin(ENVIRONMENTAL_COVARIATES)]['estimate'].abs().mean()
    print(f"  Acoustic indices:  {index_mean:.3f}")
    print(f"  Environmental:     {env_mean:.3f}")
    print(f"  Ratio (index/env): {index_mean/env_mean:.2f}×")
    print()

    # ========================================================================
    # SAVE TABLES
    # ========================================================================

    print("💾 SAVING TABLES")
    print("-" * 40)

    # Save main coefficient table
    output_path = paths.processed_data / "table1_effect_sizes.csv"
    table1_estimates.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    # Save magnitude summary
    magnitude_path = paths.processed_data / "table1_magnitude_summary.csv"
    magnitude_df.to_csv(magnitude_path, index=False)
    print(f"✓ Saved: {magnitude_path}")

    # Save with standard errors (for supplementary material)
    table1_with_se = create_table_with_se(coef_combined, RESPONSE_VARS, RESPONSE_NAMES, ALL_PREDICTORS)
    se_path = paths.processed_data / "table1_effect_sizes_with_se.csv"
    table1_with_se.to_csv(se_path, index=False)
    print(f"✓ Saved (with SE): {se_path}")

    print()

    # ========================================================================
    # GENERATE LATEX VERSION
    # ========================================================================

    print("📝 GENERATING LATEX VERSION")
    print("-" * 40)

    latex_output = paths.processed_data / "table1_effect_sizes_latex.txt"
    with open(latex_output, 'w') as f:
        f.write("% LaTeX table code\n")
        f.write("% Use booktabs and rotating packages\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Coefficient estimates comparing effect sizes of acoustic indices ")
        f.write("versus environmental covariates across response variables. Values represent ")
        f.write("log-odds ratios (binomial models) or log-rate ratios (count models). ")
        f.write("Large absolute values indicate stronger associations.}\n")
        f.write("\\label{tab:effect-sizes}\n")
        f.write(table1_display.to_latex(index=False, escape=False, float_format="%.3f"))
        f.write("\\end{table}\n")

    print(f"✓ Saved LaTeX version: {latex_output}")
    print()

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print("=" * 60)
    print("✅ TABLE 1 CREATED SUCCESSFULLY")
    print("=" * 60)
    print()

    return table1_estimates, magnitude_df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_table_with_se(coef_df, response_vars, response_names, predictors):
    """Create version of table with estimate ± SE format."""

    table_rows = []

    for predictor in predictors:
        row = {'Predictor': predictor}

        for response_var in response_vars:
            # Get coefficient for this predictor-response combo
            match = coef_df[(coef_df['term'] == predictor) &
                           (coef_df['response'] == response_var)]

            if len(match) == 0:
                cell_value = "—"
            else:
                est = match.iloc[0]['estimate']
                se = match.iloc[0]['std.error']
                cell_value = f"{est:.3f} ± {se:.3f}"

            row[response_names[response_var]] = cell_value

        table_rows.append(row)

    return pd.DataFrame(table_rows)


if __name__ == "__main__":
    table1, magnitude = main()

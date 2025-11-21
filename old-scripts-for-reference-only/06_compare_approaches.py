#!/usr/bin/env python3
"""
Script 06: Compare Universal vs Taxa-Specific Approaches
=========================================================

Purpose: Compare model performance between universal and taxa-specific
         acoustic index selection approaches.

Key Questions:
1. Does taxa-specific selection improve model fit (lower AIC)?
2. Which response variables benefit most from taxa-specific indices?
3. What are the trade-offs between approaches?

Inputs:
- glmm_model_stats_both_approaches.csv - AIC and performance metrics
- glmm_all_coefficients_both_approaches.csv - Model coefficients
- comparison_universal_vs_taxa_specific.json - Index selections

Outputs:
- Comparison summary table
- Visualization comparing approaches
- Recommendations for each response variable
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_GROUPS = {
    'fish': ['fish_activity', 'fish_richness', 'fish_present'],
    'dolphin': ['dolphin_activity', 'dolphin_whistles', 'dolphin_echolocation',
                'dolphin_burst_pulses', 'dolphin_present'],
    'vessel': ['vessel_present']
}

# Reverse lookup: response -> group
RESPONSE_TO_GROUP = {}
for group, responses in TARGET_GROUPS.items():
    for response in responses:
        RESPONSE_TO_GROUP[response] = group

# Validation: no vessel_count responses
if 'vessel_count' in RESPONSE_TO_GROUP:
    raise ValueError('vessel_count present in response mapping; presence-only policy violated')

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 60)
    print("SCRIPT 06: COMPARE UNIVERSAL VS TAXA-SPECIFIC APPROACHES")
    print("=" * 60)
    print()

    # Setup paths
    paths = ProjectPaths()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Model statistics
    stats_path = paths.processed_data / "glmm_results" / "glmm_model_stats_both_approaches.csv"
    if not stats_path.exists():
        print("⚠️  Model results not found yet. Please wait for Script 05 to complete.")
        print(f"   Expected: {stats_path}")
        return

    model_stats = pd.read_csv(stats_path)
    print(f"✓ Model statistics: {len(model_stats)} models")

    # Coefficients
    coefs_path = paths.processed_data / "glmm_results" / "glmm_all_coefficients_both_approaches.csv"
    coefficients = pd.read_csv(coefs_path)
    print(f"✓ Coefficients: {len(coefficients)} estimates")

    # Index selections
    comparison_path = paths.processed_data / "comparison_universal_vs_taxa_specific.json"
    with open(comparison_path, 'r') as f:
        index_comparison = json.load(f)
    print(f"✓ Index selections loaded")

    print()

    # ========================================================================
    # AIC COMPARISON
    # ========================================================================

    print("📊 AIC COMPARISON")
    print("-" * 40)
    print()

    # For each response, compare universal vs taxa-specific AIC
    comparison_results = []

    for response in model_stats['response'].unique():
        # Get universal AIC
        universal_row = model_stats[(model_stats['response'] == response) &
                                   (model_stats['approach'] == 'universal')]

        # Get taxa-specific AIC
        group = RESPONSE_TO_GROUP[response]
        taxa_row = model_stats[(model_stats['response'] == response) &
                              (model_stats['approach'] == group)]

        if len(universal_row) > 0 and len(taxa_row) > 0:
            aic_universal = universal_row['AIC'].values[0]
            aic_taxa = taxa_row['AIC'].values[0]
            delta_aic = aic_universal - aic_taxa  # Positive = taxa-specific is better

            # Get number of indices
            n_indices_universal = len(index_comparison['universal_approach']['indices'])
            n_indices_taxa = len(index_comparison['taxa_specific_approach'][group]['indices'])

            # Determine winner
            if abs(delta_aic) < 2:
                winner = 'tie'
            elif delta_aic > 0:
                winner = 'taxa-specific'
            else:
                winner = 'universal'

            comparison_results.append({
                'response': response,
                'group': group,
                'AIC_universal': aic_universal,
                'AIC_taxa_specific': aic_taxa,
                'delta_AIC': delta_aic,
                'n_indices_universal': n_indices_universal,
                'n_indices_taxa': n_indices_taxa,
                'winner': winner,
                'converged_universal': universal_row['converged'].values[0],
                'converged_taxa': taxa_row['converged'].values[0]
            })

    comparison_df = pd.DataFrame(comparison_results)

    # Print summary
    print("AIC Comparison (delta_AIC = Universal - Taxa-specific):")
    print("Positive delta_AIC means taxa-specific is better\n")

    for _, row in comparison_df.iterrows():
        marker = ""
        if row['winner'] == 'taxa-specific':
            marker = " ← TAXA-SPECIFIC WINS"
        elif row['winner'] == 'universal':
            marker = " ← UNIVERSAL WINS"
        else:
            marker = " ← TIE (|ΔAIC| < 2)"

        print(f"{row['response']:30s} | ΔA IC={row['delta_AIC']:7.2f} | "
              f"Indices: {row['n_indices_universal']:2d} vs {row['n_indices_taxa']:2d}{marker}")

    print()

    # Overall summary
    wins_taxa = (comparison_df['winner'] == 'taxa-specific').sum()
    wins_universal = (comparison_df['winner'] == 'universal').sum()
    ties = (comparison_df['winner'] == 'tie').sum()

    print(f"Overall: Taxa-specific wins: {wins_taxa}, Universal wins: {wins_universal}, Ties: {ties}")
    print()

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    print("📈 CREATING VISUALIZATIONS")
    print("-" * 40)

    # Create figure directory
    fig_dir = paths.get_figures_dir()
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: AIC comparison bar chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel A: Delta AIC
    ax = axes[0]
    colors = comparison_df['winner'].map({
        'taxa-specific': '#2ecc71',
        'universal': '#e74c3c',
        'tie': '#95a5a6'
    })

    bars = ax.barh(comparison_df['response'], comparison_df['delta_AIC'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('ΔAIC (Universal - Taxa-specific)', fontsize=11)
    ax.set_ylabel('')
    ax.set_title('A) Model Performance Comparison\n'
                 'Positive = Taxa-specific better | Negative = Universal better',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Taxa-specific wins'),
        Patch(facecolor='#e74c3c', label='Universal wins'),
        Patch(facecolor='#95a5a6', label='Tie (|ΔAIC| < 2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Panel B: Number of indices comparison
    ax = axes[1]
    x = np.arange(len(comparison_df))
    width = 0.35

    ax.barh(x - width/2, comparison_df['n_indices_universal'], width,
            label='Universal', color='#3498db', alpha=0.8)
    ax.barh(x + width/2, comparison_df['n_indices_taxa'], width,
            label='Taxa-specific', color='#e67e22', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df['response'])
    ax.set_xlabel('Number of Acoustic Indices', fontsize=11)
    ax.set_title('B) Number of Indices Used by Each Approach',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig_path = fig_dir / "06_universal_vs_taxa_specific_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path.name}")
    plt.close()

    # ========================================================================
    # SAVE COMPARISON TABLE
    # ========================================================================

    print()
    print("💾 SAVING RESULTS")
    print("-" * 40)

    # Save comparison table
    output_path = paths.processed_data / "model_approach_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"✓ Saved comparison table: {output_path.name}")

    # Create summary report
    summary = {
        'overall_summary': {
            'taxa_specific_wins': int(wins_taxa),
            'universal_wins': int(wins_universal),
            'ties': int(ties),
            'total_responses': len(comparison_df)
        },
        'by_group': {},
        'recommendations': []
    }

    # Summary by group
    for group in TARGET_GROUPS.keys():
        group_data = comparison_df[comparison_df['group'] == group]
        summary['by_group'][group] = {
            'n_responses': len(group_data),
            'taxa_specific_wins': int((group_data['winner'] == 'taxa-specific').sum()),
            'universal_wins': int((group_data['winner'] == 'universal').sum()),
            'ties': int((group_data['winner'] == 'tie').sum()),
            'mean_delta_aic': float(group_data['delta_AIC'].mean()),
            'n_indices_universal': int(group_data['n_indices_universal'].iloc[0]),
            'n_indices_taxa': int(group_data['n_indices_taxa'].iloc[0])
        }

    # Generate recommendations
    for group in TARGET_GROUPS.keys():
        group_data = comparison_df[comparison_df['group'] == group]
        group_summary = summary['by_group'][group]

        if group_summary['taxa_specific_wins'] > group_summary['universal_wins']:
            recommendation = f"Use taxa-specific indices for {group} ({group_summary['n_indices_taxa']} indices)"
        elif group_summary['universal_wins'] > group_summary['taxa_specific_wins']:
            recommendation = f"Use universal indices for {group} (simpler with {group_summary['n_indices_universal']} indices)"
        else:
            recommendation = f"Either approach works for {group} (performance similar)"

        summary['recommendations'].append({
            'group': group,
            'recommendation': recommendation,
            'mean_aic_improvement': float(group_summary['mean_delta_aic'])
        })

    # Save summary
    summary_path = paths.processed_data / "approach_comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path.name}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 60)
    print("✅ COMPARISON COMPLETED")
    print("=" * 60)
    print()

    print("RECOMMENDATIONS:")
    for rec in summary['recommendations']:
        print(f"\n{rec['group'].upper()}:")
        print(f"  {rec['recommendation']}")
        print(f"  Mean AIC improvement: {rec['mean_aic_improvement']:.2f}")

    print()
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()

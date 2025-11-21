#!/usr/bin/env python3
"""
Script 3b: Feature Importance Analysis
======================================

Purpose: Identify which acoustic indices predict each taxa group
Key Question: Which indices work best for fish vs. dolphins vs. vessels?

This script uses Random Forest to rank acoustic indices by their predictive
power for each response variable. This helps identify:
- Taxon-specific indices (e.g., indices that predict dolphins but not fish)
- Universally useful indices (predict multiple taxa)
- Indices to prioritize for final VIF selection

Input: indices_reduced_correlation.json (~37 indices), taxa_metrics.parquet
Output: feature_importance_*.csv for each response variable

Design Philosophy:
- Exploratory: Random Forest used as screening tool, not for inference
- Multi-taxa: Tests all taxa separately to identify taxon-specific patterns
- Transparent: Saves importance scores for all indices and all response variables
- Flagged indices: Specifically tracks ACTtFraction performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Import pipeline utilities
import os
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Response variables to test
RESPONSE_VARIABLES = {
    # Fish metrics
    'fish_activity': {'type': 'regression', 'description': 'Fish acoustic activity (sum of intensity scores)'},
    'fish_richness': {'type': 'regression', 'description': 'Fish species richness (count)'},
    'fish_present': {'type': 'classification', 'description': 'Fish presence (binary)'},

    # Dolphin metrics
    'dolphin_activity': {'type': 'regression', 'description': 'Dolphin activity (sum of all call types)'},
    'dolphin_present': {'type': 'classification', 'description': 'Dolphin presence (binary)'},

    # Vessel metrics
    'vessel_present': {'type': 'classification', 'description': 'Vessel presence (binary)'},
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 20,
    'random_state': 42,
    'n_jobs': -1
}

# Indices to specifically flag in output
FLAGGED_INDICES = ['ACTtFraction', 'BI', 'BioEnergy', 'nROI']


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 3b: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    print(f"Response variables to test: {len(RESPONSE_VARIABLES)}")
    print(f"Flagged indices: {', '.join(FLAGGED_INDICES)}")
    print()

    # Setup paths
    paths = ProjectPaths()
    variant = os.getenv('MBON_INDICES_VARIANT', 'v2')

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load combined dataset (for acoustic indices)
    combined_path = paths.processed_data / "df_combined.parquet"
    df_combined = pd.read_parquet(combined_path)
    print(f"✓ Loaded combined dataset: {len(df_combined)} records")

    # Load reduced indices list
    # Prefer variant-specific correlation reduction file if available
    indices_path_variant = paths.processed_data / f"indices_reduced_correlation_{variant}.json"
    indices_path = indices_path_variant if indices_path_variant.exists() else paths.processed_data / "indices_reduced_correlation.json"
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
    selected_indices = indices_data['selected_indices']
    print(f"✓ Loaded reduced indices: {len(selected_indices)} indices")

    # Load taxa metrics
    metrics_path = paths.processed_data / "taxa_metrics.parquet"
    df_metrics = pd.read_parquet(metrics_path)
    print(f"✓ Loaded taxa metrics: {len(df_metrics)} records")
    print()

    # ========================================================================
    # PREPARE DATA
    # ========================================================================

    print("🧹 PREPARING DATA FOR ANALYSIS")
    print("-" * 40)

    # Extract acoustic indices
    df_indices = df_combined[selected_indices].copy()

    # Clean data
    df_indices_clean = df_indices.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Original rows: {len(df_indices)}")
    print(f"After cleaning: {len(df_indices_clean)} rows")

    # Align metrics with cleaned indices (same index)
    df_metrics_aligned = df_metrics.loc[df_indices_clean.index]
    print(f"Aligned metrics: {len(df_metrics_aligned)} rows")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_indices_clean)
    print(f"✓ Features standardized: {X_scaled.shape}")
    print()

    # ========================================================================
    # RUN FEATURE IMPORTANCE FOR EACH RESPONSE VARIABLE
    # ========================================================================

    print("🌲 RUNNING RANDOM FOREST FEATURE IMPORTANCE")
    print("-" * 60)

    all_importance_results = {}

    for response_var, config in RESPONSE_VARIABLES.items():
        if response_var not in df_metrics_aligned.columns:
            print(f"⚠️  Skipping {response_var} - not found in metrics")
            continue

        print(f"\nAnalyzing: {response_var}")
        print(f"  Type: {config['type']}")
        print(f"  Description: {config['description']}")

        # Get response variable
        y = df_metrics_aligned[response_var]

        # Run Random Forest
        if config['type'] == 'regression':
            importance_df, model, cv_score = run_random_forest_regression(
                X_scaled, y, selected_indices, response_var
            )
        else:
            importance_df, model, cv_score = run_random_forest_classification(
                X_scaled, y, selected_indices, response_var
            )

        all_importance_results[response_var] = {
            'importance_df': importance_df,
            'cv_score': cv_score,
            'type': config['type']
        }

        # Save individual results
        output_path = paths.processed_data / f"feature_importance_{response_var}.csv"
        importance_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path}")

    print()

    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================

    print("📊 COMPARATIVE ANALYSIS ACROSS TAXA")
    print("-" * 60)

    comparative_results = create_comparative_analysis(
        all_importance_results, selected_indices, FLAGGED_INDICES
    )

    # Save comparative results
    comp_path = paths.processed_data / "feature_importance_comparison.csv"
    comp_variant_path = paths.processed_data / f"feature_importance_comparison_{variant}.csv"
    comparative_results.to_csv(comp_path, index=False)
    print(f"✓ Saved comparative analysis: {comp_path}")
    comparative_results.to_csv(comp_variant_path, index=False)
    print(f"✓ Saved comparative analysis (variant): {comp_variant_path}")
    print()

    # ========================================================================
    # IDENTIFY TOP CANDIDATES FOR VIF SELECTION
    # ========================================================================

    print("🎯 IDENTIFYING TOP CANDIDATE INDICES FOR VIF SELECTION")
    print("-" * 60)

    candidates = identify_top_candidates(
        all_importance_results, selected_indices, top_n=15
    )

    print(f"\nTop {len(candidates)} candidate indices for VIF selection:")
    for i, idx in enumerate(candidates, 1):
        print(f"  {i:2d}. {idx}")

    # Save candidates list
    candidates_path = paths.processed_data / "vif_candidate_indices.json"
    candidates_variant_path = paths.processed_data / f"vif_candidate_indices_{variant}.json"
    with open(candidates_path, 'w') as f:
        json.dump({
            'candidates': candidates,
            'method': 'Random Forest feature importance (top performing across all taxa)',
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\n✓ Saved candidates list: {candidates_path}")
    with open(candidates_variant_path, 'w') as f:
        json.dump({
            'candidates': candidates,
            'method': 'Random Forest feature importance (top performing across all taxa)',
            'timestamp': datetime.now().isoformat(),
            'indices_variant': variant
        }, f, indent=2)
    print(f"✓ Saved candidates list (variant): {candidates_variant_path}")
    print()

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    print("📈 GENERATING VISUALIZATIONS")
    print("-" * 40)

    create_importance_heatmap(all_importance_results, selected_indices, paths)
    create_taxon_specificity_plot(comparative_results, paths)

    print("✓ Feature importance heatmap saved")
    print("✓ Taxon specificity plot saved")
    print()

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================

    print("📋 SUMMARY REPORT")
    print("-" * 60)

    generate_summary_report(all_importance_results, comparative_results,
                           candidates, FLAGGED_INDICES, paths)

    print()

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print("=" * 60)
    print("🎉 FEATURE IMPORTANCE ANALYSIS COMPLETED")
    print("=" * 60)
    print()
    print("📋 KEY OUTPUTS:")
    print(f"   • Feature importance CSVs for {len(all_importance_results)} response variables")
    print(f"   • {comp_path}")
    print(f"   • {candidates_path}")
    print(f"   • {paths.get_figure_path('03b_importance_heatmap.png')}")
    print(f"   • {paths.get_figure_path('03b_taxon_specificity.png')}")
    print()
    print("🔄 READY FOR SCRIPT 3c: VIF-Based Final Selection")

    return all_importance_results, comparative_results, candidates


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_random_forest_regression(X, y, feature_names, response_name):
    """Run Random Forest regression and return feature importances."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)

    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_score = cv_scores.mean()

    # Get feature importances
    importance_df = pd.DataFrame({
        'index': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    train_score = model.score(X, y)
    print(f"    R² (training): {train_score:.3f}")
    print(f"    R² (5-fold CV): {cv_score:.3f}")
    print(f"    Top 3 indices: {', '.join(importance_df.head(3)['index'].tolist())}")

    return importance_df, model, cv_score


def run_random_forest_classification(X, y, feature_names, response_name):
    """Run Random Forest classification and return feature importances."""
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_score = cv_scores.mean()

    # Get feature importances
    importance_df = pd.DataFrame({
        'index': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    train_score = model.score(X, y)
    print(f"    Accuracy (training): {train_score:.3f}")
    print(f"    ROC-AUC (5-fold CV): {cv_score:.3f}")
    print(f"    Top 3 indices: {', '.join(importance_df.head(3)['index'].tolist())}")

    return importance_df, model, cv_score


def create_comparative_analysis(all_results, selected_indices, flagged_indices):
    """Create comparative analysis showing importance across all response variables."""
    # Build wide-format dataframe
    comp_data = {'index': selected_indices}

    for response_var, results in all_results.items():
        importance_dict = dict(zip(results['importance_df']['index'],
                                  results['importance_df']['importance']))
        comp_data[response_var] = [importance_dict.get(idx, 0) for idx in selected_indices]

    comp_df = pd.DataFrame(comp_data)

    # Add summary columns
    response_cols = [col for col in comp_df.columns if col != 'index']
    comp_df['mean_importance'] = comp_df[response_cols].mean(axis=1)
    comp_df['max_importance'] = comp_df[response_cols].max(axis=1)
    comp_df['n_top10'] = (comp_df[response_cols].rank(ascending=False) <= 10).sum(axis=1)

    # Flag indices
    comp_df['flagged'] = comp_df['index'].isin(flagged_indices)

    # Sort by mean importance
    comp_df = comp_df.sort_values('mean_importance', ascending=False)

    return comp_df


def identify_top_candidates(all_results, selected_indices, top_n=15):
    """Identify top candidate indices based on performance across all taxa."""
    # Count how many times each index appears in top rankings
    index_scores = {idx: 0 for idx in selected_indices}

    for response_var, results in all_results.items():
        importance_df = results['importance_df']

        # Give higher scores to higher-ranked indices
        for rank, row in enumerate(importance_df.head(20).itertuples(), 1):
            score = 21 - rank  # Top index gets 20 points, 20th gets 1 point
            index_scores[row.index] += score

    # Sort by score and return top N
    sorted_indices = sorted(index_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [idx for idx, score in sorted_indices[:top_n]]

    return top_candidates


def generate_summary_report(all_results, comparative_results, candidates,
                           flagged_indices, paths):
    """Generate text summary report."""

    print("\nPERFORMANCE BY RESPONSE VARIABLE:")
    for response_var, results in all_results.items():
        print(f"\n{response_var}:")
        print(f"  Cross-validation score: {results['cv_score']:.3f}")
        top5 = results['importance_df'].head(5)
        print(f"  Top 5 indices:")
        for i, row in enumerate(top5.itertuples(), 1):
            flag = " ⭐" if row.index in flagged_indices else ""
            print(f"    {i}. {row.index}: {row.importance:.4f}{flag}")

    print("\nFLAGGED INDEX PERFORMANCE:")
    for flagged_idx in flagged_indices:
        if flagged_idx in comparative_results['index'].values:
            row = comparative_results[comparative_results['index'] == flagged_idx].iloc[0]
            print(f"\n{flagged_idx}:")
            print(f"  Mean importance: {row['mean_importance']:.4f}")
            print(f"  Times in top 10: {row['n_top10']}")
            print(f"  Best for: ", end="")
            response_cols = [col for col in comparative_results.columns
                           if col not in ['index', 'mean_importance', 'max_importance', 'n_top10', 'flagged']]
            best_response = max(response_cols, key=lambda col: row[col])
            print(f"{best_response} ({row[best_response]:.4f})")

    print("\nTAXON-SPECIFIC INSIGHTS:")

    # Find indices that work well for fish but not dolphins
    fish_specific = comparative_results[
        (comparative_results['fish_activity'] > 0.02) &
        (comparative_results['dolphin_activity'] < 0.01)
    ].head(3)
    if len(fish_specific) > 0:
        print("\nFish-specific indices:")
        for idx in fish_specific['index']:
            print(f"  • {idx}")

    # Find indices that work well for dolphins but not fish
    dolphin_specific = comparative_results[
        (comparative_results['dolphin_activity'] > 0.02) &
        (comparative_results['fish_activity'] < 0.01)
    ].head(3)
    if len(dolphin_specific) > 0:
        print("\nDolphin-specific indices:")
        for idx in dolphin_specific['index']:
            print(f"  • {idx}")

    # Find universal indices (good for both)
    universal = comparative_results[
        (comparative_results['fish_activity'] > 0.02) &
        (comparative_results['dolphin_activity'] > 0.02)
    ].head(3)
    if len(universal) > 0:
        print("\nUniversal indices (good for multiple taxa):")
        for idx in universal['index']:
            print(f"  • {idx}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_importance_heatmap(all_results, selected_indices, paths):
    """Create heatmap showing feature importance across all response variables."""
    # Prepare data for heatmap
    heatmap_data = pd.DataFrame({'index': selected_indices})

    for response_var, results in all_results.items():
        importance_dict = dict(zip(results['importance_df']['index'],
                                  results['importance_df']['importance']))
        heatmap_data[response_var] = [importance_dict.get(idx, 0) for idx in selected_indices]

    # Sort by mean importance
    response_cols = [col for col in heatmap_data.columns if col != 'index']
    heatmap_data['mean'] = heatmap_data[response_cols].mean(axis=1)
    heatmap_data = heatmap_data.sort_values('mean', ascending=False).drop('mean', axis=1)

    # Take top 30 indices for visualization
    heatmap_data = heatmap_data.head(30)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))

    data_to_plot = heatmap_data.set_index('index')[response_cols]
    sns.heatmap(data_to_plot, cmap='YlOrRd', annot=False, fmt='.3f',
                cbar_kws={'label': 'Feature Importance'}, ax=ax)

    ax.set_title('Feature Importance Across Response Variables\n(Top 30 Indices)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Response Variable', fontsize=12)
    ax.set_ylabel('Acoustic Index', fontsize=12)

    plt.tight_layout()
    figure_path = paths.get_figure_path('03b_importance_heatmap.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_taxon_specificity_plot(comparative_results, paths):
    """Create plot showing taxon-specific vs universal indices."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot fish vs dolphin importance
    scatter_data = comparative_results.head(30)  # Top 30 by mean importance

    ax.scatter(scatter_data['fish_activity'], scatter_data['dolphin_activity'],
               s=100, alpha=0.6)

    # Add labels for top indices
    for _, row in scatter_data.head(10).iterrows():
        ax.annotate(row['index'],
                   (row['fish_activity'], row['dolphin_activity']),
                   fontsize=8, alpha=0.7)

    # Add diagonal line (universal indices)
    max_val = max(scatter_data['fish_activity'].max(),
                  scatter_data['dolphin_activity'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Universal (equal importance)')

    ax.set_xlabel('Fish Activity Importance', fontsize=12)
    ax.set_ylabel('Dolphin Activity Importance', fontsize=12)
    ax.set_title('Taxon Specificity of Acoustic Indices\n(Top 30 Indices by Mean Importance)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figure_path = paths.get_figure_path('03b_taxon_specificity.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results, comparative, candidates = main()

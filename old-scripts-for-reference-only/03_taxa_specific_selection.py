#!/usr/bin/env python3
"""
Script 03: Taxa-Specific Index Selection
========================================

Purpose: Select optimal acoustic indices for each target group (fish, dolphins, vessels)
Key Question: Do different taxa require different acoustic indices for optimal prediction?

This script performs the complete 3-stage selection process independently for each
target group, allowing indices to be optimized for specific detection tasks.

Target Groups:
- Fish: fish_activity, fish_richness, fish_present
- Dolphins: dolphin_activity, dolphin_whistles, dolphin_echolocation,
           dolphin_burst_pulses, dolphin_present
- Vessels: vessel_present, vessel_count

Three-Stage Selection Process (per group):
1. Correlation-based reduction (remove highly correlated indices)
2. Feature importance ranking (Random Forest across group's response variables)
3. VIF-based selection (remove multicollinear indices)

Key Outputs:
- indices_final_{group}.json - Final indices for each target group
- comparison_universal_vs_taxa_specific.json - Compare approaches
- figures/03_taxa_specific_selection.png - Visualization

Design Philosophy:
- Uses standardized indices for fair comparison
- Same thresholds as universal approach (correlation=0.85, VIF<5)
- Data-driven: number of final indices emerges naturally per group
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For correlation reduction
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# For feature importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# For VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import pipeline utilities
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target groups and their response variables
TARGET_GROUPS = {
    'fish': ['fish_activity', 'fish_richness', 'fish_present'],
    'dolphin': ['dolphin_activity', 'dolphin_whistles', 'dolphin_echolocation',
                'dolphin_burst_pulses', 'dolphin_present'],
    'vessel': ['vessel_present']
}

# Selection thresholds (same as universal approach)
CORRELATION_THRESHOLD = 0.85
VIF_THRESHOLD = 5.0
TOP_N_CANDIDATES = 15  # After feature importance, before VIF

# ============================================================================
# STAGE 1: CORRELATION-BASED REDUCTION
# ============================================================================

def reduce_by_correlation(indices_data, acoustic_indices, group_name, threshold=0.85):
    """
    Reduce acoustic indices by removing highly correlated pairs.
    Uses hierarchical clustering to find groups of correlated indices.

    Args:
        indices_data: DataFrame with acoustic indices
        acoustic_indices: List of acoustic index column names
        group_name: Name of target group (for logging)
        threshold: Correlation threshold (default 0.85)

    Returns:
        list: Selected indices after correlation reduction
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: CORRELATION REDUCTION - {group_name.upper()}")
    print(f"{'='*60}\n")

    # Compute correlation matrix
    corr_matrix = indices_data[acoustic_indices].corr(method='pearson').abs()

    # Convert to distance matrix for hierarchical clustering
    distance_matrix = 1 - corr_matrix

    # Handle any NaN or Inf values that might occur with standardized data
    distance_matrix = distance_matrix.fillna(0)  # Perfect correlation
    distance_matrix = distance_matrix.replace([np.inf, -np.inf], 0)

    # Add small epsilon to avoid zero distances that can cause clustering issues
    np.fill_diagonal(distance_matrix.values, 0)  # Keep diagonal as 0
    distance_matrix = distance_matrix + 1e-10
    np.fill_diagonal(distance_matrix.values, 0)  # Restore diagonal

    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(condensed_dist, method='average')

    # Cut tree at correlation threshold
    clusters = hierarchy.fcluster(linkage, 1 - threshold, criterion='distance')

    # Select one representative from each cluster (the one with lowest mean correlation)
    selected_indices = []
    for cluster_id in np.unique(clusters):
        cluster_members = [acoustic_indices[i] for i in range(len(clusters)) if clusters[i] == cluster_id]

        # Select index with lowest mean absolute correlation with others in cluster
        mean_corrs = []
        for idx in cluster_members:
            other_members = [m for m in cluster_members if m != idx]
            if other_members:
                mean_corr = corr_matrix.loc[idx, other_members].mean()
            else:
                mean_corr = 0
            mean_corrs.append(mean_corr)

        best_idx = cluster_members[np.argmin(mean_corrs)]
        selected_indices.append(best_idx)

    print(f"✓ Original indices: {len(acoustic_indices)}")
    print(f"✓ Correlation threshold: {threshold}")
    print(f"✓ Selected indices: {len(selected_indices)}")
    print(f"✓ Removed: {len(acoustic_indices) - len(selected_indices)}")

    return selected_indices

# ============================================================================
# STAGE 2: FEATURE IMPORTANCE RANKING
# ============================================================================

def rank_by_feature_importance(indices_data, taxa_data, selected_indices,
                               response_vars, group_name, top_n=15):
    """
    Rank indices by predictive importance using Random Forest.

    Args:
        indices_data: DataFrame with acoustic indices
        taxa_data: DataFrame with response variables
        selected_indices: List of indices from Stage 1
        response_vars: List of response variable names for this group
        group_name: Name of target group
        top_n: Number of top candidates to select

    Returns:
        list: Top N indices by feature importance
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: FEATURE IMPORTANCE - {group_name.upper()}")
    print(f"{'='*60}\n")

    # Merge data
    data = pd.merge(indices_data[['Date', 'station'] + selected_indices],
                    taxa_data[['Date', 'station'] + response_vars],
                    on=['Date', 'station'])

    # Remove any rows with missing values
    data = data.dropna()

    print(f"✓ Training data: {len(data)} observations")
    print(f"✓ Response variables: {len(response_vars)}")

    # Train Random Forest for each response variable
    importance_scores = {idx: [] for idx in selected_indices}

    X = data[selected_indices].values

    for response_var in response_vars:
        y = data[response_var].values

        # Determine if classification or regression
        n_unique = len(np.unique(y))
        if n_unique <= 10:  # Treat as classification
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # Treat as regression
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)

        # Store feature importances
        for idx, importance in zip(selected_indices, model.feature_importances_):
            importance_scores[idx].append(importance)

    # Aggregate importances across response variables
    mean_importance = {idx: np.mean(scores) for idx, scores in importance_scores.items()}

    # Rank and select top N
    ranked_indices = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in ranked_indices[:top_n]]

    print(f"✓ Top {top_n} candidates selected by importance")
    print(f"\nTop 5 indices:")
    for i, (idx, importance) in enumerate(ranked_indices[:5], 1):
        print(f"  {i}. {idx}: {importance:.4f}")

    return top_indices, mean_importance

# ============================================================================
# STAGE 3: VIF-BASED SELECTION
# ============================================================================

def select_by_vif(indices_data, candidate_indices, group_name, threshold=5.0):
    """
    Iteratively remove indices with high VIF until all remaining have VIF < threshold.

    Args:
        indices_data: DataFrame with acoustic indices
        candidate_indices: List of candidate indices from Stage 2
        group_name: Name of target group
        threshold: VIF threshold

    Returns:
        tuple: (final_indices, final_vif_values, removal_history)
    """
    print(f"\n{'='*60}")
    print(f"STAGE 3: VIF SELECTION - {group_name.upper()}")
    print(f"{'='*60}\n")

    # Prepare data
    X = indices_data[candidate_indices].dropna()
    remaining_indices = candidate_indices.copy()
    removal_history = []

    print(f"✓ Starting with {len(remaining_indices)} candidate indices")
    print(f"✓ VIF threshold: {threshold}")

    iteration = 0
    while True:
        iteration += 1

        # Calculate VIF for all remaining indices
        X_current = X[remaining_indices].values
        vif_values = {}

        for i, idx in enumerate(remaining_indices):
            try:
                vif = variance_inflation_factor(X_current, i)
                vif_values[idx] = vif
            except:
                vif_values[idx] = np.inf

        # Find max VIF
        max_vif_idx = max(vif_values, key=vif_values.get)
        max_vif = vif_values[max_vif_idx]

        # Check if all VIF values are below threshold
        if max_vif < threshold:
            print(f"\n✓ All remaining indices have VIF < {threshold}")
            break

        # Remove index with highest VIF
        removal_history.append({
            'iteration': iteration,
            'removed': max_vif_idx,
            'vif': float(max_vif)
        })

        print(f"  Iteration {iteration}: Removing {max_vif_idx} (VIF={max_vif:.2f})")
        remaining_indices.remove(max_vif_idx)

        # Safety check
        if len(remaining_indices) < 2:
            print(f"\n⚠️ Warning: Only {len(remaining_indices)} indices remaining")
            break

    # Final VIF values
    X_final = X[remaining_indices].values
    final_vif_values = {}
    for i, idx in enumerate(remaining_indices):
        try:
            final_vif_values[idx] = float(variance_inflation_factor(X_final, i))
        except:
            final_vif_values[idx] = None

    print(f"\n✓ Final selection: {len(remaining_indices)} indices")
    print(f"\nFinal VIF values:")
    for idx in remaining_indices:
        vif = final_vif_values[idx]
        if vif is not None:
            print(f"  {idx}: VIF={vif:.2f}")

    return remaining_indices, final_vif_values, removal_history

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 03: TAXA-SPECIFIC INDEX SELECTION")
    print("=" * 60)
    print(f"Target groups: {list(TARGET_GROUPS.keys())}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"VIF threshold: {VIF_THRESHOLD}")
    print()

    # Setup paths
    paths = ProjectPaths()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load standardized combined dataset
    df_standardized = pd.read_parquet(paths.processed_data / "df_combined_standardized.parquet")
    print(f"✓ Loaded standardized dataset: {len(df_standardized)} records")

    # Load taxa metrics
    taxa = pd.read_parquet(paths.processed_data / "taxa_metrics.parquet")
    print(f"✓ Loaded taxa metrics: {len(taxa)} records")

    # Load lookup table to identify acoustic indices
    with open(paths.processed_data / "lookup_table.json", 'r') as f:
        lookup_table = json.load(f)

    acoustic_indices = [col for col, cat in lookup_table.items() if cat == "acoustic index"]
    acoustic_indices = [col for col in acoustic_indices if col in df_standardized.columns]
    print(f"✓ Found {len(acoustic_indices)} acoustic indices")
    print()

    # ========================================================================
    # RUN SELECTION FOR EACH TARGET GROUP
    # ========================================================================

    results = {}

    for group_name, response_vars in TARGET_GROUPS.items():
        print(f"\n{'#'*60}")
        print(f"# PROCESSING GROUP: {group_name.upper()}")
        print(f"# Response variables: {', '.join(response_vars)}")
        print(f"{'#'*60}")

        # Stage 1: Correlation reduction
        stage1_indices = reduce_by_correlation(
            df_standardized, acoustic_indices, group_name, CORRELATION_THRESHOLD
        )

        # Stage 2: Feature importance
        stage2_indices, importance_scores = rank_by_feature_importance(
            df_standardized, taxa, stage1_indices, response_vars,
            group_name, TOP_N_CANDIDATES
        )

        # Stage 3: VIF selection
        final_indices, final_vif, removal_history = select_by_vif(
            df_standardized, stage2_indices, group_name, VIF_THRESHOLD
        )

        # Store results
        results[group_name] = {
            'response_variables': response_vars,
            'n_original': len(acoustic_indices),
            'stage1_selected': stage1_indices,
            'n_stage1': len(stage1_indices),
            'stage2_candidates': stage2_indices,
            'n_stage2': len(stage2_indices),
            'final_indices': final_indices,
            'n_final': len(final_indices),
            'final_vif_values': final_vif,
            'removal_history': removal_history,
            'feature_importance_scores': {idx: float(importance_scores[idx])
                                         for idx in stage2_indices},
            'timestamp': datetime.now().isoformat()
        }

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}\n")

    # Save individual group results
    for group_name, group_results in results.items():
        output_path = paths.processed_data / f"indices_final_{group_name}.json"
        with open(output_path, 'w') as f:
            json.dump(group_results, f, indent=2)
        print(f"✓ Saved {group_name} results: {output_path.name}")

    # Create comparison summary
    comparison = {
        'universal_approach': {
            'description': 'Single set of indices for all response variables',
            'indices': ['BI', 'EAS', 'EPS_KURT', 'EVNtMean', 'nROI'],  # From previous run
            'n_indices': 5
        },
        'taxa_specific_approach': {
            group: {
                'indices': results[group]['final_indices'],
                'n_indices': results[group]['n_final'],
                'response_variables': results[group]['response_variables']
            }
            for group in TARGET_GROUPS.keys()
        },
        'timestamp': datetime.now().isoformat()
    }

    comparison_path = paths.processed_data / "comparison_universal_vs_taxa_specific.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"✓ Saved comparison: {comparison_path.name}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print("Taxa-Specific Selection Results:")
    for group_name in TARGET_GROUPS.keys():
        group_results = results[group_name]
        print(f"\n{group_name.upper()}:")
        print(f"  Response variables: {len(group_results['response_variables'])}")
        print(f"  Final indices: {group_results['n_final']}")
        print(f"  Indices: {', '.join(group_results['final_indices'])}")

    print(f"\n{'='*60}")
    print("✅ TAXA-SPECIFIC SELECTION COMPLETED")
    print(f"{'='*60}\n")

    print("📊 Next steps:")
    print("  • Script 04: Prepare taxa-specific model datasets")
    print("  • Script 05: Fit taxa-specific GLMMs")
    print("  • Script 06: Compare universal vs. taxa-specific performance")
    print()

if __name__ == "__main__":
    main()

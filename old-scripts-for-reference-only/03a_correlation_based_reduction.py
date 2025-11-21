#!/usr/bin/env python3
"""
Script 3a: Correlation-Based Index Reduction
============================================

Purpose: Reduce acoustic indices by removing highly correlated (redundant) indices
Key Question: Which acoustic indices measure unique information?

This script performs unsupervised feature reduction using hierarchical clustering
to identify and remove redundant acoustic indices that are highly correlated (r > 0.85).

Input: df_combined.parquet (60 acoustic indices)
Output: indices_reduced_correlation.json (~37 indices)

Design Philosophy:
- Unsupervised: Does not depend on response variables
- Conservative: Only removes clearly redundant indices
- Transparent: Saves clustering dendrogram and correlation matrix
- Reusable: Can adjust correlation threshold easily
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

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Import pipeline utilities
import os
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

CORRELATION_THRESHOLD = 0.85  # Correlation threshold for clustering
CORRELATION_METHOD = 'pearson'  # Correlation method


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 3a: CORRELATION-BASED INDEX REDUCTION")
    print("=" * 60)
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"Correlation method: {CORRELATION_METHOD}")
    print()

    # Setup paths
    paths = ProjectPaths()
    variant = os.getenv('MBON_INDICES_VARIANT', 'v2')

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load combined dataset
    combined_path = paths.processed_data / "df_combined.parquet"
    df_combined = pd.read_parquet(combined_path)
    print(f"✓ Loaded combined dataset: {len(df_combined)} records")

    # Load lookup table to identify acoustic indices
    lookup_path = paths.processed_data / "lookup_table.json"
    with open(lookup_path, 'r') as f:
        lookup_table = json.load(f)

    # Extract acoustic index columns
    acoustic_index_cols = [col for col, col_type in lookup_table.items()
                           if col_type == 'acoustic index']
    print(f"✓ Found {len(acoustic_index_cols)} acoustic indices")
    print()

    # ========================================================================
    # PREPARE DATA
    # ========================================================================

    print("🧹 PREPARING ACOUSTIC INDEX DATA")
    print("-" * 40)

    # Extract acoustic indices
    df_indices = df_combined[acoustic_index_cols].copy()

    # Clean data: replace infinities with NaN, then drop rows with NaN
    df_indices_clean = df_indices.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Original data: {len(df_indices)} rows")
    print(f"After cleaning: {len(df_indices_clean)} rows")
    print(f"Removed: {len(df_indices) - len(df_indices_clean)} rows with inf/NaN")
    print()

    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================

    print("📊 COMPUTING CORRELATION MATRIX")
    print("-" * 40)

    # Calculate correlation matrix
    corr_matrix = df_indices_clean.corr(method=CORRELATION_METHOD)

    # Count highly correlated pairs
    high_corr_mask = (corr_matrix.abs() > CORRELATION_THRESHOLD) & (corr_matrix.abs() < 1.0)
    n_high_corr = high_corr_mask.sum().sum() // 2  # Divide by 2 because matrix is symmetric

    print(f"✓ Correlation matrix computed: {len(corr_matrix)} x {len(corr_matrix)}")
    print(f"✓ Found {n_high_corr} pairs with |r| > {CORRELATION_THRESHOLD}")
    print()

    # ========================================================================
    # HIERARCHICAL CLUSTERING
    # ========================================================================

    print("🌳 PERFORMING HIERARCHICAL CLUSTERING")
    print("-" * 40)

    # Convert correlation to distance and handle numerical issues
    corr_matrix_clipped = corr_matrix.abs().clip(0, 1)
    distance_matrix = 1 - corr_matrix_clipped
    distance_condensed = squareform(distance_matrix, checks=False)
    distance_condensed = np.clip(distance_condensed, 0, None)  # Ensure non-negative

    # Perform hierarchical clustering
    linkage = hierarchy.ward(distance_condensed)

    # Cut dendrogram to form clusters
    cluster_ids = hierarchy.fcluster(linkage, 1 - CORRELATION_THRESHOLD, criterion='distance')

    # Group indices by cluster
    cluster_dict = {}
    for idx, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(acoustic_index_cols[idx])

    print(f"✓ Formed {len(cluster_dict)} clusters")
    print()

    # ========================================================================
    # SELECT REPRESENTATIVE INDICES
    # ========================================================================

    print("✅ SELECTING REPRESENTATIVE INDICES")
    print("-" * 40)

    # Select first index from each cluster (could be improved with domain knowledge)
    selected_indices = [indices[0] for indices in cluster_dict.values()]
    selected_indices = sorted(selected_indices)  # Sort for consistency

    print(f"Original indices: {len(acoustic_index_cols)}")
    print(f"Selected indices: {len(selected_indices)}")
    print(f"Reduction: {len(acoustic_index_cols) - len(selected_indices)} indices removed")
    print(f"Reduction rate: {(1 - len(selected_indices)/len(acoustic_index_cols))*100:.1f}%")
    print()

    # ========================================================================
    # SHOW CLUSTER DETAILS
    # ========================================================================

    print("📋 CLUSTER DETAILS (showing clusters with >1 index)")
    print("-" * 60)

    multi_index_clusters = {k: v for k, v in cluster_dict.items() if len(v) > 1}
    for cluster_id, indices in sorted(multi_index_clusters.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\nCluster {cluster_id} ({len(indices)} indices):")
        print(f"  Selected: {indices[0]} ✓")
        for idx in indices[1:]:
            # Show correlation with selected index
            corr_with_selected = corr_matrix.loc[indices[0], idx]
            print(f"  Removed:  {idx} (r={corr_with_selected:.3f})")

    print()

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("💾 SAVING RESULTS")
    print("-" * 40)

    # Save selected indices list
    # Save with variant-aware filename for comparison reproducibility
    output_path = paths.processed_data / "indices_reduced_correlation.json"
    variant_output_path = paths.processed_data / f"indices_reduced_correlation_{variant}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'selected_indices': selected_indices,
            'n_original': len(acoustic_index_cols),
            'n_selected': len(selected_indices),
            'correlation_threshold': CORRELATION_THRESHOLD,
            'correlation_method': CORRELATION_METHOD,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"✓ Saved: {output_path}")

    # Also save variant-specific copy
    with open(variant_output_path, 'w') as f:
        json.dump({
            'selected_indices': selected_indices,
            'n_original': len(acoustic_index_cols),
            'n_selected': len(selected_indices),
            'correlation_threshold': CORRELATION_THRESHOLD,
            'correlation_method': CORRELATION_METHOD,
            'timestamp': datetime.now().isoformat(),
            'indices_variant': variant
        }, f, indent=2)
    print(f"✓ Saved: {variant_output_path}")

    # Save detailed cluster assignments
    cluster_assignments = {
        idx: {'cluster_id': int(cluster_id), 'selected': idx in selected_indices}
        for idx, cluster_id in zip(acoustic_index_cols, cluster_ids)
    }
    cluster_path = paths.processed_data / "correlation_clusters.json"
    cluster_variant_path = paths.processed_data / f"correlation_clusters_{variant}.json"
    with open(cluster_path, 'w') as f:
        json.dump(cluster_assignments, f, indent=2)
    print(f"✓ Saved: {cluster_path}")
    with open(cluster_variant_path, 'w') as f:
        json.dump(cluster_assignments, f, indent=2)
    print(f"✓ Saved: {cluster_variant_path}")

    # Save correlation matrix
    corr_matrix_path = paths.processed_data / "correlation_matrix.csv"
    corr_matrix_variant_path = paths.processed_data / f"correlation_matrix_{variant}.csv"
    corr_matrix.to_csv(corr_matrix_path)
    print(f"✓ Saved: {corr_matrix_path}")
    corr_matrix.to_csv(corr_matrix_variant_path)
    print(f"✓ Saved: {corr_matrix_variant_path}")
    print()

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    print("📈 GENERATING VISUALIZATIONS")
    print("-" * 40)

    create_dendrogram(linkage, acoustic_index_cols, CORRELATION_THRESHOLD, paths)
    create_correlation_heatmap(corr_matrix, selected_indices, paths)

    print("✓ Dendrogram saved")
    print("✓ Correlation heatmap saved")
    print()

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print("=" * 60)
    print("🎉 CORRELATION-BASED REDUCTION COMPLETED")
    print("=" * 60)
    print()
    print("📋 KEY OUTPUTS:")
    print(f"   • {output_path}")
    print(f"   • {cluster_path}")
    print(f"   • {corr_matrix_path}")
    print(f"   • {paths.get_figure_path('03a_dendrogram.png')}")
    print(f"   • {paths.get_figure_path('03a_correlation_heatmap.png')}")
    print()
    print(f"✨ SUMMARY:")
    print(f"   • Reduced from {len(acoustic_index_cols)} to {len(selected_indices)} indices")
    print(f"   • Removed {len(acoustic_index_cols) - len(selected_indices)} redundant indices")
    print(f"   • Preserved indices span {len(cluster_dict)} unique acoustic features")
    print()
    print("🔄 READY FOR SCRIPT 3b: Feature Importance Analysis")

    return selected_indices, cluster_dict


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_dendrogram(linkage, index_names, threshold, paths):
    """Create hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create dendrogram
    dendro = hierarchy.dendrogram(
        linkage,
        labels=index_names,
        ax=ax,
        color_threshold=1 - threshold,
        above_threshold_color='gray'
    )

    # Add horizontal line at threshold
    ax.axhline(y=1 - threshold, color='red', linestyle='--',
               label=f'Correlation threshold: {threshold}')

    ax.set_title('Hierarchical Clustering of Acoustic Indices\n(Based on Correlation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Acoustic Index', fontsize=12)
    ax.set_ylabel('Distance (1 - |correlation|)', fontsize=12)
    ax.legend()

    # Rotate labels
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()

    figure_path = paths.get_figure_path('03a_dendrogram.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(corr_matrix, selected_indices, paths):
    """Create correlation heatmap highlighting selected indices."""
    fig, ax = plt.subplots(figsize=(16, 14))

    # Reorder matrix to put selected indices first
    other_indices = [idx for idx in corr_matrix.index if idx not in selected_indices]
    ordered_indices = selected_indices + other_indices
    corr_ordered = corr_matrix.loc[ordered_indices, ordered_indices]

    # Create heatmap
    sns.heatmap(corr_ordered, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                square=True, linewidths=0,
                cbar_kws={'label': 'Correlation'},
                ax=ax)

    # Add dividing lines to separate selected vs removed indices
    n_selected = len(selected_indices)
    ax.axhline(y=n_selected, color='black', linewidth=2)
    ax.axvline(x=n_selected, color='black', linewidth=2)

    # Add text annotations
    ax.text(n_selected/2, -1, 'SELECTED', ha='center', fontweight='bold', fontsize=12)
    ax.text(n_selected + len(other_indices)/2, -1, 'REMOVED', ha='center', fontweight='bold', fontsize=12)

    ax.set_title('Acoustic Index Correlation Matrix\n(Reordered: Selected First, Then Removed)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    figure_path = paths.get_figure_path('03a_correlation_heatmap.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    selected_indices, cluster_dict = main()

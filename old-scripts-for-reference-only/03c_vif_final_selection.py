#!/usr/bin/env python3
"""
Script 3c: VIF-Based Final Index Selection
==========================================

Purpose: Ensure final index set has low multicollinearity
Key Question: Do our selected indices provide unique, non-redundant information?

This script performs iterative VIF (Variance Inflation Factor) reduction to ensure
the final acoustic index set has low multicollinearity (VIF < 5). This is critical
for GLMM coefficient interpretability and stability.

Input: vif_candidate_indices.json (top ~15 indices from feature importance)
Output: indices_final_vif_checked.json (final set with VIF < 5)

Design Philosophy:
- Conservative: VIF threshold of 5 (standard in regression analysis)
- Flexible: Can manually include/exclude specific indices
- Flagged indices: Warns if ACTtFraction or other important indices are removed
- Transparent: Reports VIF values and removal decisions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import pipeline utilities
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# CONFIGURATION
# ============================================================================

VIF_THRESHOLD = 5.0  # Standard threshold for low multicollinearity
MIN_INDICES = 5  # Don't reduce below this many indices

# Indices to flag if removed (from previous analyses)
FLAGGED_INDICES = ['ACTtFraction', 'BI', 'BioEnergy', 'nROI']

# Optional: Manually force inclusion of specific indices
# Set to empty list if you want pure VIF-based selection
FORCE_INCLUDE = []  # e.g., ['ACTtFraction'] to always keep it


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 3c: VIF-BASED FINAL INDEX SELECTION")
    print("=" * 60)
    print(f"VIF threshold: {VIF_THRESHOLD}")
    print(f"Minimum indices: {MIN_INDICES}")
    if FORCE_INCLUDE:
        print(f"Force include: {', '.join(FORCE_INCLUDE)}")
    print()

    # Setup paths
    paths = ProjectPaths()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load combined dataset
    combined_path = paths.processed_data / "df_combined.parquet"
    df_combined = pd.read_parquet(combined_path)
    print(f"✓ Loaded combined dataset: {len(df_combined)} records")

    # Load candidate indices from feature importance analysis
    candidates_path = paths.processed_data / "vif_candidate_indices.json"
    with open(candidates_path, 'r') as f:
        candidates_data = json.load(f)
    candidate_indices = candidates_data['candidates']
    print(f"✓ Loaded candidate indices: {len(candidate_indices)} indices")

    # Load feature importance comparison to check flagged indices
    importance_path = paths.processed_data / "feature_importance_comparison.csv"
    importance_comp = pd.read_csv(importance_path)
    print(f"✓ Loaded feature importance data")
    print()

    # ========================================================================
    # PREPARE DATA
    # ========================================================================

    print("🧹 PREPARING DATA")
    print("-" * 40)

    # Extract candidate indices
    df_indices = df_combined[candidate_indices].copy()

    # Clean data
    df_indices_clean = df_indices.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Original rows: {len(df_indices)}")
    print(f"After cleaning: {len(df_indices_clean)} rows")
    print()

    # ========================================================================
    # ITERATIVE VIF REDUCTION
    # ========================================================================

    print("📊 ITERATIVE VIF REDUCTION")
    print("-" * 60)

    # Start with all candidate indices
    remaining_indices = candidate_indices.copy()

    # Separate forced indices from others
    if FORCE_INCLUDE:
        remaining_indices = [idx for idx in remaining_indices if idx not in FORCE_INCLUDE]
        print(f"Force including: {', '.join(FORCE_INCLUDE)}")
        print(f"Testing VIF for remaining: {len(remaining_indices)} indices")
        print()

    iteration = 0
    removal_history = []

    while True:
        iteration += 1

        # Get current index set (forced + remaining)
        current_indices = FORCE_INCLUDE + remaining_indices
        print(f"\nIteration {iteration}: Testing {len(current_indices)} indices")

        # Check minimum threshold
        if len(remaining_indices) <= max(1, MIN_INDICES - len(FORCE_INCLUDE)):
            print(f"⚠️  Reached minimum indices threshold ({MIN_INDICES})")
            break

        # Calculate VIF for all indices
        X_current = df_indices_clean[current_indices].values
        vif_results = calculate_vif_safe(X_current, current_indices)

        if vif_results is None:
            print("⚠️  VIF calculation failed - using current set")
            break

        # Find max VIF (excluding forced indices)
        removable_vifs = {idx: vif for idx, vif in vif_results.items()
                         if idx not in FORCE_INCLUDE}

        if not removable_vifs:
            print("✓ All indices are force-included, stopping")
            break

        max_vif_idx = max(removable_vifs, key=removable_vifs.get)
        max_vif = removable_vifs[max_vif_idx]

        print(f"  Max VIF: {max_vif_idx} = {max_vif:.2f}")

        # Check if all VIFs are below threshold
        if max_vif <= VIF_THRESHOLD:
            print(f"\n✅ All VIFs below {VIF_THRESHOLD}!")
            break

        # Remove the worst offender
        print(f"  ❌ Removing {max_vif_idx}")

        # Check if this is a flagged index
        if max_vif_idx in FLAGGED_INDICES:
            print(f"  ⚠️  WARNING: Removing flagged index {max_vif_idx}")
            importance_row = importance_comp[importance_comp['index'] == max_vif_idx]
            if not importance_row.empty:
                mean_imp = importance_row.iloc[0]['mean_importance']
                print(f"     Mean importance across taxa: {mean_imp:.4f}")

        removal_history.append({
            'iteration': iteration,
            'removed': max_vif_idx,
            'vif': float(max_vif),
            'flagged': max_vif_idx in FLAGGED_INDICES
        })

        remaining_indices.remove(max_vif_idx)

    # Final index set
    final_indices = FORCE_INCLUDE + remaining_indices
    final_indices = sorted(final_indices)  # Sort for consistency

    print()
    print("=" * 60)
    print("✅ FINAL VIF RESULTS")
    print("=" * 60)

    # Calculate final VIFs
    X_final = df_indices_clean[final_indices].values
    final_vif = calculate_vif_safe(X_final, final_indices)

    if final_vif is not None:
        vif_df = pd.DataFrame({
            'Index': final_indices,
            'VIF': [final_vif[idx] for idx in final_indices]
        }).sort_values('VIF', ascending=False)

        print(f"\n✓ Final index set: {len(final_indices)} indices")
        print(f"\nFinal VIF values:")
        print(vif_df.to_string(index=False))
        print(f"\nMax VIF: {vif_df['VIF'].max():.2f}")
        print(f"Mean VIF: {vif_df['VIF'].mean():.2f}")
    else:
        print("⚠️  Could not calculate final VIFs")

    print()

    # ========================================================================
    # CHECK FLAGGED INDICES
    # ========================================================================

    print("🚩 FLAGGED INDEX STATUS")
    print("-" * 60)

    for flagged_idx in FLAGGED_INDICES:
        if flagged_idx in final_indices:
            print(f"✓ {flagged_idx}: INCLUDED")
        else:
            print(f"✗ {flagged_idx}: REMOVED")
            # Check if it was in candidates
            if flagged_idx in candidate_indices:
                removal_record = next((r for r in removal_history if r['removed'] == flagged_idx), None)
                if removal_record:
                    print(f"  Removed in iteration {removal_record['iteration']} (VIF = {removal_record['vif']:.2f})")
            else:
                print(f"  Was not in candidate set from feature importance analysis")

    print()

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("💾 SAVING RESULTS")
    print("-" * 40)

    # Save final indices
    output_data = {
        'final_indices': final_indices,
        'n_final': len(final_indices),
        'n_candidates': len(candidate_indices),
        'vif_threshold': VIF_THRESHOLD,
        'force_include': FORCE_INCLUDE,
        'removal_history': removal_history,
        'flagged_indices_status': {
            idx: idx in final_indices for idx in FLAGGED_INDICES
        },
        'final_vif_values': final_vif if final_vif else {},
        'timestamp': datetime.now().isoformat()
    }

    variant = os.getenv('MBON_INDICES_VARIANT', 'v2')
    output_path = paths.processed_data / "indices_final_vif_checked.json"
    variant_output_path = paths.processed_data / f"indices_final_vif_checked_{variant}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved: {output_path}")

    with open(variant_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved: {variant_output_path}")

    # Save VIF table
    if final_vif is not None:
        vif_table_path = paths.processed_data / "final_vif_values.csv"
        vif_table_variant_path = paths.processed_data / f"final_vif_values_{variant}.csv"
        vif_df.to_csv(vif_table_path, index=False)
        print(f"✓ Saved: {vif_table_path}")
        vif_df.to_csv(vif_table_variant_path, index=False)
        print(f"✓ Saved: {vif_table_variant_path}")

    print()

    # ========================================================================
    # GENERATE COMPARISON REPORT
    # ========================================================================

    print("📋 SELECTION SUMMARY")
    print("-" * 60)

    print(f"\nPipeline progression:")
    print(f"  • Script 01: Data preparation")
    print(f"  • Script 03a: Correlation reduction → {len([i for i in importance_comp['index']])} indices")
    print(f"  • Script 03b: Feature importance → {len(candidate_indices)} top candidates")
    print(f"  • Script 03c: VIF selection → {len(final_indices)} final indices")

    print(f"\nReduction summary:")
    print(f"  • Started with: {len(candidate_indices)} candidates")
    print(f"  • Removed: {len(candidate_indices) - len(final_indices)} indices")
    print(f"  • Final set: {len(final_indices)} indices")
    print(f"  • All VIF < {VIF_THRESHOLD}: {'Yes' if max(final_vif.values()) < VIF_THRESHOLD else 'No'}")

    print(f"\nFinal indices ready for GLMM:")
    for i, idx in enumerate(final_indices, 1):
        flag = " ⭐" if idx in FLAGGED_INDICES else ""
        vif_val = final_vif.get(idx, 'N/A')
        vif_str = f"{vif_val:.2f}" if isinstance(vif_val, (int, float)) else str(vif_val)
        print(f"  {i:2d}. {idx:20s} (VIF: {vif_str}){flag}")

    print()

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print("=" * 60)
    print("🎉 VIF-BASED FINAL SELECTION COMPLETED")
    print("=" * 60)
    print()
    print("📋 KEY OUTPUTS:")
    print(f"   • {output_path}")
    if final_vif is not None:
        print(f"   • {vif_table_path}")
    print()
    print("🔄 READY FOR SCRIPT 4: Prepare Model Data")

    return final_indices, final_vif, removal_history


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_vif_safe(X, index_names):
    """Calculate VIF with error handling."""
    try:
        vif_values = {}
        for i, idx_name in enumerate(index_names):
            vif = variance_inflation_factor(X, i)
            # Handle inf/nan from numerical issues
            if np.isinf(vif) or np.isnan(vif):
                vif = 999.0
            vif_values[idx_name] = vif
        return vif_values
    except Exception as e:
        print(f"⚠️  VIF calculation error: {e}")
        return None


if __name__ == "__main__":
    final_indices, vif_values, history = main()

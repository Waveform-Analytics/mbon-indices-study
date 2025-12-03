"""
Stage 01: Index Reduction

Reduces ~62 acoustic indices to 5-10 final indices through:
1. Correlation pruning (|r| > threshold)
2. VIF analysis (VIF > threshold)
3. Category coverage check
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_interim_parquet, save_summary_json
from mbon_indices.utils.logging import setup_stage_logging


def load_index_metadata(root: Path) -> pd.DataFrame:
    """Load index metadata with categories and descriptions."""
    metadata_path = root / "data" / "raw" / "metadata" / "Updated_Index_Categories_v2.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"✓ Loaded index metadata: {len(df)} indices with categories")
    return df


def extract_index_columns(df: pd.DataFrame) -> list[str]:
    """Extract acoustic index column names (exclude keys and metadata)."""
    exclude = ['station', 'datetime', 'date', 'hour', 'Filename', 'Date']
    # Only keep numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    index_cols = [c for c in numeric_cols if c not in exclude]
    print(f"✓ Identified {len(index_cols)} numeric acoustic index columns")
    return index_cols


def standardize_indices(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Standardize indices (z-score) within each station-year group."""
    df_std = df.copy()
    df_std['year'] = df_std['datetime'].dt.year

    for idx in index_cols:
        # Z-score within station-year
        df_std[idx] = df_std.groupby(['station', 'year'])[idx].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
        )

    print(f"✓ Standardized {len(index_cols)} indices within station-year groups")
    return df_std


def compute_correlations(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for indices."""
    # Use only numeric index data
    index_data = df[index_cols].select_dtypes(include=[np.number])

    # Compute correlation matrix
    corr_matrix = index_data.corr(method='pearson')

    print(f"✓ Computed correlation matrix: {corr_matrix.shape}")
    return corr_matrix


def identify_high_correlations(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Identify index pairs with |correlation| > threshold."""
    # Get upper triangle (avoid duplicates)
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            idx1 = corr_matrix.columns[i]
            idx2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]

            if abs(corr) > threshold:
                pairs.append({
                    'index1': idx1,
                    'index2': idx2,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })

    pairs_df = pd.DataFrame(pairs).sort_values('abs_correlation', ascending=False)
    print(f"✓ Found {len(pairs_df)} pairs with |r| > {threshold}")
    return pairs_df


def prune_correlated_indices(
    high_corr_pairs: pd.DataFrame,
    indices_df: pd.DataFrame,
    index_cols: list[str]
) -> tuple[set[str], list[dict]]:
    """
    Greedy correlation pruning: for each correlated pair, keep one index.

    Decision rules (priority):
    1. Coverage: keep index with fewer missing values
    2. Alphabetical: deterministic tiebreaker

    Returns:
        kept_indices: set of indices to keep
        dropped: list of dicts with drop details
    """
    kept_indices = set(index_cols)
    dropped = []

    for _, pair in high_corr_pairs.iterrows():
        idx1 = str(pair['index1'])
        idx2 = str(pair['index2'])

        # Skip if either index already dropped
        if idx1 not in kept_indices or idx2 not in kept_indices:
            continue

        # Rule 1: Coverage (count non-missing values)
        coverage1 = indices_df[idx1].notna().sum()
        coverage2 = indices_df[idx2].notna().sum()

        if coverage1 > coverage2:
            keep = idx1
            drop = idx2
            reason = f"Lower coverage ({coverage2} vs {coverage1})"
        elif coverage2 > coverage1:
            keep = idx2
            drop = idx1
            reason = f"Lower coverage ({coverage1} vs {coverage2})"
        else:
            # Rule 2: Alphabetical tiebreaker
            keep = idx1 if idx1 < idx2 else idx2
            drop = idx2 if idx1 < idx2 else idx1
            reason = "Equal coverage; alphabetical tiebreaker"

        # Remove from kept set and record
        kept_indices.remove(drop)
        dropped.append({
            'index': drop,
            'reason': reason,
            'correlated_with': keep,
            'correlation': pair['correlation']
        })

    print(f"✓ Pruned {len(dropped)} indices due to correlation")
    print(f"  Remaining: {len(kept_indices)} indices")

    return kept_indices, dropped


def compute_vif(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each index.

    VIF measures multicollinearity: how much variance of a coefficient
    is inflated due to collinearity with other predictors.
    """
    # Get clean data (drop rows with any missing values)
    index_data = df[index_cols].dropna()

    if len(index_data) < len(index_cols) + 1:
        raise ValueError(f"Not enough complete observations ({len(index_data)}) for VIF calculation")

    # Compute VIF for each index
    vif_data = []
    for i, col in enumerate(index_cols):
        try:
            vif = variance_inflation_factor(index_data.values, i)
            vif_data.append({'index': col, 'vif': vif})
        except Exception as e:
            print(f"  Warning: Could not compute VIF for {col}: {e}")
            vif_data.append({'index': col, 'vif': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    print(f"✓ Computed VIF for {len(vif_df)} indices")
    return vif_df


def prune_by_vif(
    df: pd.DataFrame,
    index_cols: list[str],
    metadata_df: pd.DataFrame,
    vif_threshold: float,
    vif_fallback: float
) -> tuple[list[str], list[dict]]:
    """
    Iteratively remove indices with high VIF until all remaining have VIF <= threshold.

    Uses fallback threshold if strict threshold would violate coverage requirements.

    Returns:
        final_indices: list of indices to keep
        vif_history: list of dicts tracking VIF iterations
    """
    current_indices = list(index_cols)
    vif_history = []
    iteration = 0

    while True:
        iteration += 1
        print(f"\n  VIF Iteration {iteration}: {len(current_indices)} indices")

        # Compute VIF for current set
        vif_df = compute_vif(df, current_indices)

        # Check max VIF
        max_vif = vif_df['vif'].max()
        max_idx = vif_df.loc[vif_df['vif'].idxmax(), 'index']

        print(f"    Max VIF: {max_vif:.2f} ({max_idx})")

        # Check if we're done
        if max_vif <= vif_threshold:
            print(f"  ✓ All indices have VIF ≤ {vif_threshold}")
            break

        # Check category coverage before removing
        remaining_after_drop = [idx for idx in current_indices if idx != max_idx]
        categories_after = get_category_coverage(remaining_after_drop, metadata_df)

        # If we'd lose a category, check if we can use fallback
        if len(categories_after) < 3 and max_vif <= vif_fallback:
            print(f"  → Using fallback threshold {vif_fallback} to preserve category coverage")
            vif_history.append({
                'iteration': iteration,
                'removed': None,
                'vif': max_vif,
                'reason': f'Fallback threshold applied (VIF {max_vif:.2f} ≤ {vif_fallback})'
            })
            break

        # If we're getting too small (< 5 indices), use fallback
        if len(remaining_after_drop) < 5 and max_vif <= vif_fallback:
            print(f"  → Using fallback threshold {vif_fallback} to maintain minimum list size")
            vif_history.append({
                'iteration': iteration,
                'removed': None,
                'vif': max_vif,
                'reason': f'Fallback threshold applied (VIF {max_vif:.2f} ≤ {vif_fallback}, preserving list size)'
            })
            break

        # Remove index with highest VIF
        current_indices.remove(max_idx)
        vif_history.append({
            'iteration': iteration,
            'removed': max_idx,
            'vif': max_vif,
            'reason': f'VIF {max_vif:.2f} > {vif_threshold}'
        })
        print(f"    Removed: {max_idx} (VIF = {max_vif:.2f})")

        # Safety check: don't reduce below 5 indices
        if len(current_indices) < 5:
            print(f"  ! Stopping: minimum list size reached ({len(current_indices)} indices)")
            break

    print(f"\n✓ VIF pruning complete: {len(current_indices)} indices remaining")
    return current_indices, vif_history


def get_category_coverage(index_cols: list[str], metadata_df: pd.DataFrame) -> set[str]:
    """Get set of categories covered by given indices."""
    categories = set()
    for idx in index_cols:
        cat_match = metadata_df[metadata_df['Prefix'] == idx]
        if not cat_match.empty:
            categories.add(cat_match.iloc[0]['Category'])
    return categories


def check_category_coverage(
    final_indices: list[str],
    metadata_df: pd.DataFrame,
    required_categories: list[str] = None
) -> dict:
    """
    Verify that final indices cover required categories.

    Returns summary dict with coverage info.
    """
    if required_categories is None:
        required_categories = ['Spectral Indices', 'Temporal Indices', 'Complexity Indices']

    categories = get_category_coverage(final_indices, metadata_df)

    coverage = {
        'total_indices': len(final_indices),
        'categories_covered': len(categories),
        'categories': list(categories),
        'missing': [cat for cat in required_categories if cat not in categories]
    }

    print(f"✓ Category coverage check:")
    print(f"    Categories covered: {len(categories)} / {len(required_categories)}")
    for cat in categories:
        count = sum(1 for idx in final_indices
                   if not metadata_df[metadata_df['Prefix'] == idx].empty
                   and metadata_df[metadata_df['Prefix'] == idx].iloc[0]['Category'] == cat)
        print(f"      {cat}: {count} indices")

    if coverage['missing']:
        print(f"    ⚠ Missing categories: {', '.join(coverage['missing'])}")

    return coverage


def save_outputs(
    root: Path,
    final_indices: list[str],
    dropped_indices: list[dict],
    vif_history: list[dict],
    coverage: dict,
    corr_matrix: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    metadata_df: pd.DataFrame,
    corr_threshold: float
):
    """Save all Stage 01 outputs per spec."""

    # Create output directories
    (root / "results" / "indices").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # 1. Save final index list as JSON
    final_list_path = root / "results" / "indices" / "index_final_list.json"
    final_list_data = []
    for idx in sorted(final_indices):
        cat_match = metadata_df[metadata_df['Prefix'] == idx]
        if not cat_match.empty:
            category = cat_match.iloc[0]['Category']
            description = cat_match.iloc[0].get('Description', '')
        else:
            category = 'Unknown'
            description = ''

        final_list_data.append({
            'index': idx,
            'category': category,
            'description': description
        })

    save_summary_json({
        'final_indices': final_list_data,
        'count': len(final_indices),
        'coverage': coverage,
        'correlation_threshold': corr_threshold
    }, final_list_path)
    print(f"  ✓ Saved final index list: {final_list_path}")

    # 2. Save indices_final.csv
    indices_final_path = root / "data" / "processed" / "indices_final.csv"
    final_df = pd.DataFrame([
        {
            'index_name': item['index'],
            'kept': True,
            'reason': 'Passed correlation and VIF thresholds',
            'category': item['category'],
            'band': 'Full'  # Placeholder - update based on actual band logic
        }
        for item in final_list_data
    ])
    final_df.to_csv(indices_final_path, index=False)
    print(f"  ✓ Saved indices_final.csv: {indices_final_path}")

    # 3. Save reduction report
    report_path = root / "results" / "tables" / "index_reduction_report.csv"
    report_rows = []

    # Add correlation-pruned indices
    for item in dropped_indices:
        report_rows.append({
            'index': item['index'],
            'stage': 'correlation',
            'reason': item['reason'],
            'correlated_with': item['correlated_with'],
            'correlation': item['correlation'],
            'vif': None
        })

    # Add VIF-pruned indices
    for item in vif_history:
        if item['removed']:
            report_rows.append({
                'index': item['removed'],
                'stage': 'vif',
                'reason': item['reason'],
                'correlated_with': None,
                'correlation': None,
                'vif': item['vif']
            })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    print(f"  ✓ Saved reduction report: {report_path}")

    # 4. Generate correlation heatmap for final indices
    heatmap_path = root / "results" / "figures" / "index_correlation_heatmap.png"
    plot_correlation_heatmap(corr_matrix, final_indices, heatmap_path)
    print(f"  ✓ Saved correlation heatmap: {heatmap_path}")

    # 5. Generate sensitivity heatmap at 0.8 threshold
    sensitivity_path = root / "results" / "figures" / "index_correlation_sensitivity_0_8.png"
    plot_correlation_heatmap(corr_matrix, final_indices, sensitivity_path, threshold=0.8)
    print(f"  ✓ Saved sensitivity heatmap: {sensitivity_path}")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    indices: list[str],
    output_path: Path,
    threshold: float = None
):
    """Plot correlation heatmap for selected indices."""
    # Subset correlation matrix
    subset_corr = corr_matrix.loc[indices, indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        subset_corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )

    title = f'Correlation Matrix: Final {len(indices)} Indices'
    if threshold:
        title += f' (threshold={threshold})'
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Set up logging
    logger = setup_stage_logging(root, "stage01_index_reduction")

    try:
        print("=" * 60)
        print("STAGE 01: INDEX REDUCTION")
        print("=" * 60)
        print()

        # Load configuration
        cfg = load_analysis_config(root)
        corr_threshold = cfg['thresholds']['correlation_r']
        print("Configuration:")
        print(f"  Correlation threshold: {corr_threshold}")
        print(f"  VIF threshold: {cfg['thresholds']['vif']}")
        print(f"  VIF fallback: {cfg['thresholds']['vif_fallback']}")
        print()

        # Load data
        print("Step 1: Loading data...")
        indices_df = load_interim_parquet(root, "aligned_indices")
        print(f"✓ Loaded aligned indices: {len(indices_df):,} rows, {len(indices_df.columns)} columns")
        metadata_df = load_index_metadata(root)
        index_cols = extract_index_columns(indices_df)
        print(f"  Starting with: {len(index_cols)} indices")
        print()

        # Standardize indices
        print("Step 2: Standardizing indices...")
        indices_std = standardize_indices(indices_df, index_cols)
        print()

        # Correlation analysis
        print("Step 3: Correlation analysis...")
        corr_matrix = compute_correlations(indices_std, index_cols)
        high_corr_pairs = identify_high_correlations(corr_matrix, corr_threshold)
        print()

        if len(high_corr_pairs) > 0:
            print("High correlation pairs (top 10):")
            print(high_corr_pairs.head(10).to_string(index=False))
            print()

        # Correlation pruning
        print("Step 4: Correlation pruning...")
        kept_indices, dropped_indices = prune_correlated_indices(
            high_corr_pairs, indices_df, index_cols
        )
        print()

        if len(dropped_indices) > 0:
            print("Dropped indices (first 10):")
            dropped_df = pd.DataFrame(dropped_indices)
            print(dropped_df.head(10).to_string(index=False))
            print()

        # Show remaining indices with categories
        print("Remaining indices after correlation pruning:")
        kept_list = sorted(kept_indices)
        for idx in kept_list:
            # Look up category if available
            cat_match = metadata_df[metadata_df['Prefix'] == idx]
            if not cat_match.empty:
                category = cat_match.iloc[0]['Category']
                print(f"  {idx:20s} ({category})")
            else:
                print(f"  {idx:20s} (category unknown)")
        print()

        print("=" * 60)
        print(f"✓ Correlation pruning complete")
        print(f"  Started with: {len(index_cols)} indices")
        print(f"  Dropped: {len(dropped_indices)} indices")
        print(f"  Remaining: {len(kept_indices)} indices")
        print("=" * 60)
        print()

        # VIF analysis
        print("Step 5: VIF analysis...")
        vif_threshold = cfg['thresholds']['vif']
        vif_fallback = cfg['thresholds']['vif_fallback']

        final_indices, vif_history = prune_by_vif(
            indices_std,
            kept_list,
            metadata_df,
            vif_threshold,
            vif_fallback
        )
        print()

        # Category coverage check
        print("Step 6: Category coverage check...")
        coverage = check_category_coverage(final_indices, metadata_df)
        print()

        # Final summary
        print("=" * 60)
        print("FINAL INDEX LIST")
        print("=" * 60)
        print(f"Total indices: {len(final_indices)}")
        print()
        for idx in sorted(final_indices):
            cat_match = metadata_df[metadata_df['Prefix'] == idx]
            if not cat_match.empty:
                category = cat_match.iloc[0]['Category']
                print(f"  {idx:20s} ({category})")
            else:
                print(f"  {idx:20s} (category unknown)")
        print()

        # Save outputs
        print("Step 7: Saving outputs...")
        save_outputs(
            root,
            final_indices,
            dropped_indices,
            vif_history,
            coverage,
            corr_matrix,
            high_corr_pairs,
            metadata_df,
            corr_threshold
        )
        print()

        print("=" * 60)
        print(f"✓ Stage 01 complete")
        print(f"  Started with: {len(index_cols)} indices")
        print(f"  After correlation pruning: {len(kept_indices)} indices")
        print(f"  After VIF pruning: {len(final_indices)} indices")
        print(f"  Final list size: {len(final_indices)} indices")
        print("=" * 60)
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        # Close logger and restore stdout
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()

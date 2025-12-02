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

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.utils.logging import setup_stage_logging


def load_aligned_indices(root: Path) -> pd.DataFrame:
    """Load aligned acoustic indices from Stage 00 output."""
    indices_path = root / "data" / "interim" / "aligned_indices.parquet"

    if not indices_path.exists():
        raise FileNotFoundError(f"Aligned indices not found: {indices_path}")

    df = pd.read_parquet(indices_path)
    print(f"✓ Loaded aligned indices: {len(df):,} rows, {len(df.columns)} columns")
    return df


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
        idx1 = pair['index1']
        idx2 = pair['index2']

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
        indices_df = load_aligned_indices(root)
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
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        # Close logger and restore stdout
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()

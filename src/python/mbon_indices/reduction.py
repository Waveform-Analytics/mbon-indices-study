"""
Index reduction utilities for Stage 01.

Core functions for correlation pruning and VIF analysis, extracted
to support both the main pipeline and sensitivity analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_index_metadata(root: Path) -> pd.DataFrame:
    """Load index metadata with categories and descriptions."""
    metadata_path = root / "data" / "raw" / "metadata" / "Updated_Index_Categories_v2.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"  Loaded index metadata: {len(df)} indices with categories")
    return df


def extract_index_columns(df: pd.DataFrame) -> list[str]:
    """Extract acoustic index column names (exclude keys and metadata)."""
    exclude = ["station", "datetime", "date", "hour", "Filename", "Date"]
    # Only keep numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    index_cols = [c for c in numeric_cols if c not in exclude]
    print(f"  Identified {len(index_cols)} numeric acoustic index columns")
    return index_cols


def standardize_indices(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """
    Standardize indices (z-score) within each station-year group.

    NOTE: This standardization is INTERNAL to Stage 01 only. The standardized
    values are used for correlation/VIF calculations but are NOT saved to the
    pipeline outputs. Downstream stages (e.g., Stage 05 modeling) receive raw
    index values and apply their own standardization for model fitting.

    The within-station-year approach was chosen to account for potential
    magnitude differences across stations/years when computing correlations.
    However, since Pearson correlation is scale-invariant, this step may be
    unnecessary and could be removed in a future simplification.
    """
    df_std = df.copy()
    df_std["year"] = df_std["datetime"].dt.year

    for idx in index_cols:
        # Z-score within station-year
        df_std[idx] = df_std.groupby(["station", "year"])[idx].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
        )

    print(f"  Standardized {len(index_cols)} indices within station-year groups")
    return df_std


def compute_correlations(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for indices."""
    # Use only numeric index data
    index_data = df[index_cols].select_dtypes(include=[np.number])

    # Compute correlation matrix
    corr_matrix = index_data.corr(method="pearson")

    print(f"  Computed correlation matrix: {corr_matrix.shape}")
    return corr_matrix


def identify_high_correlations(
    corr_matrix: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Identify index pairs with |correlation| > threshold."""
    # Get upper triangle (avoid duplicates)
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            idx1 = corr_matrix.columns[i]
            idx2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]

            if abs(corr) > threshold:
                pairs.append(
                    {
                        "index1": idx1,
                        "index2": idx2,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                    }
                )

    pairs_df = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False)
    print(f"  Found {len(pairs_df)} pairs with |r| > {threshold}")
    return pairs_df


def prune_correlated_indices(
    high_corr_pairs: pd.DataFrame,
    indices_df: pd.DataFrame,
    index_cols: list[str],
    tiebreaker: str = "first",
) -> tuple[set[str], list[dict]]:
    """
    Greedy correlation pruning: for each correlated pair, keep one index.

    Decision rules (priority):
    1. Coverage: keep index with fewer missing values
    2. Tiebreaker: when coverage is equal, use alphabetical order

    Args:
        high_corr_pairs: DataFrame of correlated pairs from identify_high_correlations
        indices_df: DataFrame containing index values
        index_cols: list of index column names
        tiebreaker: "first" keeps alphabetically first index (default/primary),
                    "second" keeps alphabetically second index (for sensitivity analysis)

    Returns:
        kept_indices: set of indices to keep
        dropped: list of dicts with drop details
    """
    if tiebreaker not in ("first", "second"):
        raise ValueError(f"tiebreaker must be 'first' or 'second', got: {tiebreaker}")

    kept_indices = set(index_cols)
    dropped = []

    for _, pair in high_corr_pairs.iterrows():
        idx1 = str(pair["index1"])
        idx2 = str(pair["index2"])

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
            # Rule 2: Alphabetical tiebreaker (direction controlled by parameter)
            if tiebreaker == "first":
                keep = idx1 if idx1 < idx2 else idx2
                drop = idx2 if idx1 < idx2 else idx1
                reason = "Equal coverage; alphabetical tiebreaker (keep first)"
            else:  # tiebreaker == "second"
                keep = idx2 if idx1 < idx2 else idx1
                drop = idx1 if idx1 < idx2 else idx2
                reason = "Equal coverage; alphabetical tiebreaker (keep second)"

        # Remove from kept set and record
        kept_indices.remove(drop)
        dropped.append(
            {
                "index": drop,
                "reason": reason,
                "correlated_with": keep,
                "correlation": pair["correlation"],
            }
        )

    print(f"  Pruned {len(dropped)} indices due to correlation")
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
        raise ValueError(
            f"Not enough complete observations ({len(index_data)}) for VIF calculation"
        )

    # Compute VIF for each index
    vif_data = []
    for i, col in enumerate(index_cols):
        try:
            vif = variance_inflation_factor(index_data.values, i)
            vif_data.append({"index": col, "vif": vif})
        except Exception as e:
            print(f"  Warning: Could not compute VIF for {col}: {e}")
            vif_data.append({"index": col, "vif": np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
    print(f"  Computed VIF for {len(vif_df)} indices")
    return vif_df


def get_category_coverage(index_cols: list[str], metadata_df: pd.DataFrame) -> set[str]:
    """Get set of categories covered by given indices."""
    categories = set()
    for idx in index_cols:
        cat_match = metadata_df[metadata_df["Prefix"] == idx]
        if not cat_match.empty:
            categories.add(cat_match.iloc[0]["Category"])
    return categories


def prune_by_vif(
    df: pd.DataFrame,
    index_cols: list[str],
    metadata_df: pd.DataFrame,
    vif_threshold: float,
    vif_fallback: float,
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
        max_vif = vif_df["vif"].max()
        max_idx = vif_df.loc[vif_df["vif"].idxmax(), "index"]

        print(f"    Max VIF: {max_vif:.2f} ({max_idx})")

        # Check if we're done
        if max_vif <= vif_threshold:
            print(f"  All indices have VIF <= {vif_threshold}")
            break

        # Check category coverage before removing
        remaining_after_drop = [idx for idx in current_indices if idx != max_idx]
        categories_after = get_category_coverage(remaining_after_drop, metadata_df)

        # If we'd lose a category, check if we can use fallback
        if len(categories_after) < 3 and max_vif <= vif_fallback:
            print(
                f"  -> Using fallback threshold {vif_fallback} to preserve category coverage"
            )
            vif_history.append(
                {
                    "iteration": iteration,
                    "removed": None,
                    "vif": max_vif,
                    "reason": f"Fallback threshold applied (VIF {max_vif:.2f} <= {vif_fallback})",
                }
            )
            break

        # If we're getting too small (< 5 indices), use fallback
        if len(remaining_after_drop) < 5 and max_vif <= vif_fallback:
            print(
                f"  -> Using fallback threshold {vif_fallback} to maintain minimum list size"
            )
            vif_history.append(
                {
                    "iteration": iteration,
                    "removed": None,
                    "vif": max_vif,
                    "reason": f"Fallback threshold applied (VIF {max_vif:.2f} <= {vif_fallback}, preserving list size)",
                }
            )
            break

        # Remove index with highest VIF
        current_indices.remove(max_idx)
        vif_history.append(
            {
                "iteration": iteration,
                "removed": max_idx,
                "vif": max_vif,
                "reason": f"VIF {max_vif:.2f} > {vif_threshold}",
            }
        )
        print(f"    Removed: {max_idx} (VIF = {max_vif:.2f})")

        # Safety check: don't reduce below 5 indices
        if len(current_indices) < 5:
            print(
                f"  ! Stopping: minimum list size reached ({len(current_indices)} indices)"
            )
            break

    print(f"\n  VIF pruning complete: {len(current_indices)} indices remaining")
    return current_indices, vif_history


def check_category_coverage(
    final_indices: list[str],
    metadata_df: pd.DataFrame,
    required_categories: list[str] | None = None,
) -> dict:
    """
    Verify that final indices cover required categories.

    Args:
        final_indices: list of index names to check
        metadata_df: DataFrame with index metadata (must have 'Prefix' and 'Category' columns)
        required_categories: categories to check coverage for
            - None (default): dynamically get all categories from metadata
            - [] (empty list): skip category requirement, just report what's covered
            - [list]: use the explicit list provided

    Returns summary dict with coverage info.
    """
    # Get actual categories covered by final indices
    categories = get_category_coverage(final_indices, metadata_df)

    # Determine required categories
    if required_categories is None:
        # Dynamic: get all unique categories from metadata
        required_categories = metadata_df["Category"].dropna().unique().tolist()
    elif len(required_categories) == 0:
        # Empty list: no requirement, just report coverage
        required_categories = []

    coverage = {
        "total_indices": len(final_indices),
        "categories_covered": len(categories),
        "categories": list(categories),
        "missing": [cat for cat in required_categories if cat not in categories],
    }

    print(f"  Category coverage check:")
    print(f"    Categories covered: {len(categories)} / {len(required_categories)}")
    for cat in categories:
        count = sum(
            1
            for idx in final_indices
            if not metadata_df[metadata_df["Prefix"] == idx].empty
            and metadata_df[metadata_df["Prefix"] == idx].iloc[0]["Category"] == cat
        )
        print(f"      {cat}: {count} indices")

    if coverage["missing"]:
        print(f"    Warning: Missing categories: {', '.join(coverage['missing'])}")

    return coverage
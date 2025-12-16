"""
Stage 01: Index Reduction

Reduces ~62 acoustic indices to 5-10 final indices through:
1. Correlation pruning (|r| > threshold)
2. VIF analysis (VIF > threshold)
3. Category coverage check

This is the main pipeline script. For sensitivity analysis comparing
tiebreaker choices, see stage01_sensitivity.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_interim_parquet, save_summary_json
from mbon_indices.reduction import (
    check_category_coverage,
    compute_correlations,
    compute_vif,
    extract_index_columns,
    identify_high_correlations,
    load_index_metadata,
    prune_by_vif,
    prune_correlated_indices,
    standardize_indices,
)
from mbon_indices.utils.logging import setup_stage_logging
from mbon_indices.utils.run_history import append_to_run_history


def save_outputs(
    root: Path,
    final_indices: list[str],
    dropped_indices: list[dict],
    vif_history: list[dict],
    coverage: dict,
    corr_matrix: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    metadata_df: pd.DataFrame,
    corr_threshold: float,
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
        cat_match = metadata_df[metadata_df["Prefix"] == idx]
        if not cat_match.empty:
            category = cat_match.iloc[0]["Category"]
            description = cat_match.iloc[0].get("Description", "")
        else:
            category = "Unknown"
            description = ""

        final_list_data.append(
            {"index": idx, "category": category, "description": description}
        )

    save_summary_json(
        {
            "final_indices": final_list_data,
            "count": len(final_indices),
            "coverage": coverage,
            "correlation_threshold": corr_threshold,
        },
        final_list_path,
    )
    print(f"  Saved final index list: {final_list_path}")

    # 2. Save indices_final.csv
    indices_final_path = root / "data" / "processed" / "indices_final.csv"
    final_df = pd.DataFrame(
        [
            {
                "index_name": item["index"],
                "kept": True,
                "reason": "Passed correlation and VIF thresholds",
                "category": item["category"],
                "band": "Full",  # Placeholder - update based on actual band logic
            }
            for item in final_list_data
        ]
    )
    final_df.to_csv(indices_final_path, index=False)
    print(f"  Saved indices_final.csv: {indices_final_path}")

    # 3. Save reduction report
    report_path = root / "results" / "tables" / "index_reduction_report.csv"
    report_rows = []

    # Add correlation-pruned indices
    for item in dropped_indices:
        report_rows.append(
            {
                "index": item["index"],
                "stage": "correlation",
                "reason": item["reason"],
                "correlated_with": item["correlated_with"],
                "correlation": item["correlation"],
                "vif": None,
            }
        )

    # Add VIF-pruned indices
    for item in vif_history:
        if item["removed"]:
            report_rows.append(
                {
                    "index": item["removed"],
                    "stage": "vif",
                    "reason": item["reason"],
                    "correlated_with": None,
                    "correlation": None,
                    "vif": item["vif"],
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    print(f"  Saved reduction report: {report_path}")

    # 4. Generate correlation heatmap for final indices
    heatmap_path = root / "results" / "figures" / "index_correlation_heatmap.png"
    plot_correlation_heatmap(corr_matrix, final_indices, heatmap_path)
    print(f"  Saved correlation heatmap: {heatmap_path}")

    # 5. Generate sensitivity heatmap at 0.8 threshold
    sensitivity_path = root / "results" / "figures" / "index_correlation_sensitivity_0_8.png"
    plot_correlation_heatmap(corr_matrix, final_indices, sensitivity_path, threshold=0.8)
    print(f"  Saved sensitivity heatmap: {sensitivity_path}")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    indices: list[str],
    output_path: Path,
    threshold: float = None,
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
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax,
    )

    title = f"Correlation Matrix: Final {len(indices)} Indices"
    if threshold:
        title += f" (threshold={threshold})"
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        corr_threshold = cfg["thresholds"]["correlation_r"]
        print("Configuration:")
        print(f"  Correlation threshold: {corr_threshold}")
        print(f"  VIF threshold: {cfg['thresholds']['vif']}")
        print(f"  VIF fallback: {cfg['thresholds']['vif_fallback']}")
        print()

        # Load data
        print("Step 1: Loading data...")
        indices_df = load_interim_parquet(root, "aligned_indices")
        print(f"  Loaded aligned indices: {len(indices_df):,} rows, {len(indices_df.columns)} columns")
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

        # Correlation pruning (primary: keep first alphabetically)
        print("Step 4: Correlation pruning...")
        kept_indices, dropped_indices = prune_correlated_indices(
            high_corr_pairs, indices_df, index_cols, tiebreaker="first"
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
            cat_match = metadata_df[metadata_df["Prefix"] == idx]
            if not cat_match.empty:
                category = cat_match.iloc[0]["Category"]
                print(f"  {idx:20s} ({category})")
            else:
                print(f"  {idx:20s} (category unknown)")
        print()

        print("=" * 60)
        print("Correlation pruning complete")
        print(f"  Started with: {len(index_cols)} indices")
        print(f"  Dropped: {len(dropped_indices)} indices")
        print(f"  Remaining: {len(kept_indices)} indices")
        print("=" * 60)
        print()

        # VIF analysis
        print("Step 5: VIF analysis...")
        vif_threshold = cfg["thresholds"]["vif"]
        vif_fallback = cfg["thresholds"]["vif_fallback"]

        final_indices, vif_history = prune_by_vif(
            indices_std, kept_list, metadata_df, vif_threshold, vif_fallback
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
            cat_match = metadata_df[metadata_df["Prefix"] == idx]
            if not cat_match.empty:
                category = cat_match.iloc[0]["Category"]
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
            corr_threshold,
        )

        # Compute final VIF for run history
        final_vif_df = compute_vif(indices_std, final_indices)
        max_vif_row = final_vif_df.loc[final_vif_df["vif"].idxmax()]

        # Append to run history
        append_to_run_history(
            root=root,
            stage="Stage 01: Index Reduction",
            config={
                "correlation_r": cfg["thresholds"]["correlation_r"],
                "vif": cfg["thresholds"]["vif"],
                "vif_fallback": cfg["thresholds"]["vif_fallback"],
            },
            results={
                "n_start": len(index_cols),
                "n_after_corr": len(kept_indices),
                "n_final": len(final_indices),
                "final_indices": ", ".join(sorted(final_indices)),
                "categories": f"{len(coverage['categories'])} ({', '.join(sorted(coverage['categories']))})",
                "max_vif": f"{max_vif_row['vif']:.2f} ({max_vif_row['index']})",
            },
            log_path=str(logger.log_path.relative_to(root)),
        )
        print()

        print("=" * 60)
        print("Stage 01 complete")
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
"""
Stage 01: Index Reduction (VIF-Only)

Reduces ~60 acoustic indices to ~15-20 final indices using Variance Inflation
Factor (VIF) as the sole criterion. VIF measures multicollinearity in a
multivariate sense - how well each index can be predicted by all others.

This approach is fully deterministic with no arbitrary tiebreaker decisions.

Historical note: Previously used correlation + VIF pruning, but sensitivity
analysis showed arbitrary tiebreaker choices materially affected results.
See scripts/sensitivity_analysis_index_reduction.py for that analysis.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_interim_parquet, save_summary_json
from mbon_indices.reduction import (
    check_category_coverage,
    compute_vif,
    extract_index_columns,
    load_index_metadata,
    prune_by_vif,
    standardize_indices,
)
from mbon_indices.utils.logging import setup_stage_logging
from mbon_indices.utils.run_history import append_to_run_history


def save_outputs(
    root: Path,
    final_indices: list[str],
    vif_history: list[dict],
    coverage: dict,
    metadata_df: pd.DataFrame,
    vif_threshold: float,
    vif_enabled: bool = True,
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

    method = "All indices (VIF disabled)" if not vif_enabled else "VIF-only (no correlation pruning)"
    save_summary_json(
        {
            "final_indices": final_list_data,
            "count": len(final_indices),
            "coverage": coverage,
            "vif_threshold": vif_threshold if vif_enabled else None,
            "vif_enabled": vif_enabled,
            "method": method,
        },
        final_list_path,
    )
    print(f"  Saved final index list: {final_list_path}")

    # 2. Save indices_final.csv
    indices_final_path = root / "data" / "processed" / "indices_final.csv"
    reason = "VIF disabled - all indices retained" if not vif_enabled else f"Passed VIF threshold (VIF <= {vif_threshold})"
    final_df = pd.DataFrame(
        [
            {
                "index_name": item["index"],
                "kept": True,
                "reason": reason,
                "category": item["category"],
                "band": "Full",
            }
            for item in final_list_data
        ]
    )
    final_df.to_csv(indices_final_path, index=False)
    print(f"  Saved indices_final.csv: {indices_final_path}")

    # 3. Save reduction report (VIF history)
    report_path = root / "results" / "tables" / "index_reduction_report.csv"
    report_rows = []

    for item in vif_history:
        if item.get("removed"):
            report_rows.append(
                {
                    "index": item["removed"],
                    "iteration": item["iteration"],
                    "vif": item["vif"],
                    "reason": item["reason"],
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    print(f"  Saved reduction report: {report_path}")

    # 4. Generate VIF progression plot
    if report_rows:
        plot_vif_progression(report_rows, root)


def plot_vif_progression(vif_history: list[dict], root: Path):
    """Plot VIF values across iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = [item["iteration"] for item in vif_history]
    vifs = [item["vif"] for item in vif_history]
    labels = [item["index"] for item in vif_history]

    ax.bar(range(len(iterations)), vifs, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.axhline(y=2, color="red", linestyle="--", label="VIF threshold (2)")
    ax.axhline(y=5, color="orange", linestyle="--", label="VIF fallback (5)")

    ax.set_xlabel("Removed Index")
    ax.set_ylabel("VIF at Removal")
    ax.set_title("VIF-Based Index Reduction Progression")
    ax.legend()

    plt.tight_layout()
    output_path = root / "results" / "figures" / "index_vif_progression.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved VIF progression plot: {output_path}")


def main():
    # Set up logging
    logger = setup_stage_logging(root, "stage01_index_reduction")

    try:
        print("=" * 60)
        print("STAGE 01: INDEX REDUCTION (VIF-Only)")
        print("=" * 60)
        print()

        # Load configuration
        cfg = load_analysis_config(root)
        vif_enabled = cfg["thresholds"].get("vif_enabled", True)
        vif_threshold = cfg["thresholds"]["vif"]
        vif_fallback = cfg["thresholds"]["vif_fallback"]

        print("Configuration:")
        print(f"  VIF filtering enabled: {vif_enabled}")
        print(f"  VIF threshold: {vif_threshold}")
        print(f"  VIF fallback: {vif_fallback}")
        print()
        if vif_enabled:
            print("Method: VIF-only reduction (no correlation pruning)")
            print("  - Fully deterministic, no arbitrary tiebreaker decisions")
            print("  - Iteratively remove highest VIF index until all VIF <= threshold")
        else:
            print("Method: VIF filtering DISABLED - retaining all indices")
            print("  - GAMM select=TRUE will handle regularization")
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

        # VIF-based reduction (starting from ALL indices)
        print("Step 3: VIF-based reduction...")

        if not vif_enabled:
            print(f"  VIF filtering DISABLED - keeping all {len(index_cols)} indices")
            final_indices = index_cols
            vif_history = []
        else:
            print(f"  Starting VIF pruning from {len(index_cols)} indices")
            final_indices, vif_history = prune_by_vif(
                indices_std, index_cols, metadata_df, vif_threshold, vif_fallback
            )
        print()

        # Category coverage check
        print("Step 4: Category coverage check...")
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
        print("Step 5: Saving outputs...")
        save_outputs(
            root,
            final_indices,
            vif_history,
            coverage,
            metadata_df,
            vif_threshold,
            vif_enabled,
        )

        # Compute final VIF for run history (only meaningful if VIF filtering occurred)
        if vif_enabled:
            final_vif_df = compute_vif(indices_std, final_indices)
            max_vif_row = final_vif_df.loc[final_vif_df["vif"].idxmax()]
            max_vif_str = f"{max_vif_row['vif']:.2f} ({max_vif_row['index']})"
        else:
            max_vif_str = "N/A (VIF disabled)"

        # Append to run history
        append_to_run_history(
            root=root,
            stage="Stage 01: Index Reduction",
            config={
                "method": "All indices (VIF disabled)" if not vif_enabled else "VIF-only",
                "vif_enabled": vif_enabled,
                "vif": vif_threshold,
                "vif_fallback": vif_fallback,
            },
            results={
                "n_start": len(index_cols),
                "n_final": len(final_indices),
                "n_removed": len(index_cols) - len(final_indices),
                "final_indices": ", ".join(sorted(final_indices)),
                "categories": f"{len(coverage['categories'])} ({', '.join(sorted(coverage['categories']))})",
                "max_vif": max_vif_str,
            },
            log_path=str(logger.log_path.relative_to(root)),
        )
        print()

        print("=" * 60)
        print("Stage 01 complete")
        print(f"  Started with: {len(index_cols)} indices")
        if vif_enabled:
            print(f"  Removed (VIF): {len(index_cols) - len(final_indices)} indices")
        else:
            print("  VIF filtering: DISABLED (all indices retained)")
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
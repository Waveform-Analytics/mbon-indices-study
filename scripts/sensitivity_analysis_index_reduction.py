"""
Sensitivity Analysis: Index Reduction Tiebreaker Robustness

Comprehensive analysis assessing how sensitive MODEL RESULTS are to arbitrary
tiebreaker choices during correlation pruning. This script:

1. Runs index reduction with both tiebreakers (primary vs alternate)
2. Prepares analysis-ready datasets for both index sets
3. Fits GLMM models using both sets
4. Compares model outputs (AIC, significant predictors, coefficients)

IMPORTANT: Run this AFTER the main pipeline completes (after Stage 05).
This is a side analysis that does NOT affect the main pipeline outputs.

Usage:
    python scripts/sensitivity_analysis_index_reduction.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_interim_parquet, load_processed_parquet, save_summary_json
from mbon_indices.reduction import (
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


# =============================================================================
# Feature Engineering (adapted from Stage 03)
# =============================================================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from datetime_local."""
    out = df.copy()
    out["hour_of_day"] = out["datetime_local"].dt.hour
    out["sin_hour"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["day_of_year"] = out["datetime_local"].dt.dayofyear
    out["date"] = out["datetime_local"].dt.date.astype(str)
    return out


def create_grouping_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Create grouping IDs for random effects."""
    out = df.copy()
    out["day_id"] = out["date"].astype(str) + "_" + out["station"]
    out["month_id"] = out["datetime_local"].dt.strftime("%Y-%m")
    return out


def create_ar1_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Create time_within_day sequence for AR1 modeling."""
    out = df.copy()
    out = out.sort_values(["day_id", "datetime"])
    out["time_within_day"] = out.groupby("day_id").cumcount()
    return out


def prepare_analysis_data(
    indices_df: pd.DataFrame,
    env_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    final_indices: list[str],
) -> pd.DataFrame:
    """
    Prepare analysis-ready dataset for a given set of indices.
    Mirrors Stage 03 feature engineering.
    """
    # Filter indices to final list
    keep_cols = ["station", "datetime", "datetime_local"] + [
        c for c in final_indices if c in indices_df.columns
    ]
    indices_filtered = indices_df[keep_cols].copy()

    # Merge with environment
    merged = pd.merge(
        indices_filtered,
        env_df[["station", "datetime", "temperature", "depth"]],
        on=["station", "datetime"],
        how="inner",
    )

    # Merge with community metrics
    merged = pd.merge(
        merged,
        metrics_df,
        on=["station", "datetime"],
        how="inner",
    )

    # Create features
    merged = create_temporal_features(merged)
    merged = create_grouping_factors(merged)
    merged = create_ar1_sequence(merged)

    return merged


# =============================================================================
# Index Reduction
# =============================================================================

def run_reduction(
    indices_df: pd.DataFrame,
    indices_std: pd.DataFrame,
    index_cols: list[str],
    metadata_df: pd.DataFrame,
    corr_threshold: float,
    vif_threshold: float,
    vif_fallback: float,
    tiebreaker: str,
) -> tuple[list[str], list[dict]]:
    """
    Run full reduction pipeline with specified tiebreaker.
    Returns final indices and list of dropped indices.
    """
    print(f"\n  Running reduction with tiebreaker='{tiebreaker}'...")

    # Correlation analysis
    corr_matrix = compute_correlations(indices_std, index_cols)
    high_corr_pairs = identify_high_correlations(corr_matrix, corr_threshold)

    # Correlation pruning
    kept_indices, corr_dropped = prune_correlated_indices(
        high_corr_pairs, indices_df, index_cols, tiebreaker=tiebreaker
    )

    # VIF pruning
    final_indices, vif_history = prune_by_vif(
        indices_std,
        sorted(kept_indices),
        metadata_df,
        vif_threshold,
        vif_fallback,
    )

    print(f"  Result: {len(index_cols)} -> {len(kept_indices)} (corr) -> {len(final_indices)} (VIF)")

    return final_indices, corr_dropped


def compare_index_lists(
    primary_indices: list[str],
    alternate_indices: list[str],
    primary_dropped: list[dict],
    alternate_dropped: list[dict],
) -> dict:
    """Compare primary and alternate index lists."""
    primary_set = set(primary_indices)
    alternate_set = set(alternate_indices)

    common = primary_set & alternate_set
    only_primary = primary_set - alternate_set
    only_alternate = alternate_set - primary_set

    comparison = {
        "primary_count": len(primary_indices),
        "alternate_count": len(alternate_indices),
        "common_count": len(common),
        "common_indices": sorted(common),
        "only_primary": sorted(only_primary),
        "only_alternate": sorted(only_alternate),
    }

    # Add details about why indices differ
    swap_details = []
    for idx in only_primary:
        alt_drop = next((d for d in alternate_dropped if d["index"] == idx), None)
        if alt_drop:
            swap_details.append({
                "index": idx,
                "in_primary": True,
                "dropped_in_alternate_for": alt_drop["correlated_with"],
                "correlation": alt_drop["correlation"],
            })

    for idx in only_alternate:
        pri_drop = next((d for d in primary_dropped if d["index"] == idx), None)
        if pri_drop:
            swap_details.append({
                "index": idx,
                "in_primary": False,
                "dropped_in_primary_for": pri_drop["correlated_with"],
                "correlation": pri_drop["correlation"],
            })

    comparison["swap_details"] = swap_details

    return comparison


# =============================================================================
# R Model Fitting
# =============================================================================

R_MODEL_SCRIPT = '''
# Sensitivity Analysis Model Fitting
# Called by sensitivity_analysis_index_reduction.py

library(arrow)
library(glmmTMB)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
indices_json <- args[2]
output_path <- args[3]
label <- args[4]

# Load data
data <- read_parquet(data_path)
indices <- fromJSON(indices_json)

cat(sprintf("Fitting models for %s (%d indices)...\\n", label, length(indices)))

# Response variables (subset for sensitivity - just fit key ones)
responses <- c("fish_activity", "dolphin_activity", "vessel_presence")

# Build formula
fixed_effects <- paste(c(indices, "temperature", "depth", "sin_hour", "cos_hour"), collapse = " + ")

results <- list()

for (response in responses) {
    cat(sprintf("  Fitting %s...\\n", response))

    # Determine family based on response
    if (grepl("presence", response)) {
        family <- binomial(link = "logit")
    } else {
        family <- nbinom2(link = "log")
    }

    # Build formula
    formula_str <- sprintf("%s ~ %s + (1|station) + (1|month_id)", response, fixed_effects)
    formula <- as.formula(formula_str)

    # Subset data for this response
    model_data <- data[!is.na(data[[response]]), ]

    # Scale predictors
    for (pred in c(indices, "temperature", "depth")) {
        if (pred %in% names(model_data)) {
            model_data[[pred]] <- scale(model_data[[pred]])[,1]
        }
    }

    tryCatch({
        fit <- glmmTMB(formula, data = model_data, family = family)

        # Extract results
        summ <- summary(fit)
        coefs <- as.data.frame(summ$coefficients$cond)
        coefs$predictor <- rownames(coefs)

        results[[response]] <- list(
            converged = fit$fit$convergence == 0,
            AIC = AIC(fit),
            n_obs = nrow(model_data),
            coefficients = coefs,
            significant_predictors = coefs$predictor[coefs$`Pr(>|z|)` < 0.05]
        )
    }, error = function(e) {
        cat(sprintf("    Error: %s\\n", e$message))
        results[[response]] <<- list(
            converged = FALSE,
            error = e$message
        )
    })
}

# Save results
write_json(results, output_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("Results saved to %s\\n", output_path))
'''


def run_r_models(data_path: Path, indices: list[str], output_path: Path, label: str) -> dict:
    """Run R script to fit models and return results."""
    # Write temporary R script
    r_script_path = root / "scripts" / "_temp_sensitivity_model.R"
    r_script_path.write_text(R_MODEL_SCRIPT)

    # Convert indices to JSON string
    indices_json = json.dumps(indices)

    try:
        result = subprocess.run(
            [
                "Rscript",
                str(r_script_path),
                str(data_path),
                indices_json,
                str(output_path),
                label,
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
        )

        if result.returncode != 0:
            print(f"  R script error: {result.stderr}")
            return {"error": result.stderr}

        print(result.stdout)

        # Load results
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
        else:
            return {"error": "Output file not created"}

    finally:
        # Clean up temp script
        if r_script_path.exists():
            r_script_path.unlink()


def compare_model_results(primary_results: dict, alternate_results: dict) -> dict:
    """Compare model results between primary and alternate runs."""
    comparison = {}

    for response in primary_results.keys():
        if response not in alternate_results:
            continue

        pri = primary_results[response]
        alt = alternate_results[response]

        if "error" in pri or "error" in alt:
            comparison[response] = {
                "primary_error": pri.get("error"),
                "alternate_error": alt.get("error"),
            }
            continue

        # AIC comparison (convert from string if needed, handle NA)
        try:
            pri_aic = float(pri["AIC"])
            alt_aic = float(alt["AIC"])
            aic_diff = alt_aic - pri_aic
        except (ValueError, TypeError):
            # AIC might be "NA" or None if model had issues
            comparison[response] = {
                "primary_AIC": pri.get("AIC"),
                "alternate_AIC": alt.get("AIC"),
                "error": "Could not compare AIC (NA or invalid value)",
            }
            continue

        # Significant predictors comparison
        pri_sig = set(pri.get("significant_predictors", []))
        alt_sig = set(alt.get("significant_predictors", []))

        comparison[response] = {
            "primary_AIC": pri_aic,
            "alternate_AIC": alt_aic,
            "AIC_difference": aic_diff,
            "AIC_interpretation": interpret_aic_diff(aic_diff),
            "primary_significant": sorted(pri_sig),
            "alternate_significant": sorted(alt_sig),
            "sig_in_both": sorted(pri_sig & alt_sig),
            "sig_only_primary": sorted(pri_sig - alt_sig),
            "sig_only_alternate": sorted(alt_sig - pri_sig),
        }

    return comparison


def interpret_aic_diff(diff: float) -> str:
    """Interpret AIC difference (alternate - primary)."""
    abs_diff = abs(diff)
    if abs_diff < 2:
        return "equivalent (|diff| < 2)"
    elif abs_diff < 4:
        return "weak difference (2 <= |diff| < 4)"
    elif abs_diff < 10:
        return "moderate difference (4 <= |diff| < 10)"
    else:
        return "strong difference (|diff| >= 10)"


# =============================================================================
# Main
# =============================================================================

def main():
    logger = setup_stage_logging(root, "sensitivity_analysis_index_reduction")

    try:
        print("=" * 70)
        print("SENSITIVITY ANALYSIS: INDEX REDUCTION TIEBREAKER ROBUSTNESS")
        print("=" * 70)
        print()
        print("This analysis tests whether arbitrary tiebreaker choices during")
        print("correlation pruning materially affect model conclusions.")
        print()

        # Load configuration
        cfg = load_analysis_config(root)
        corr_threshold = cfg["thresholds"]["correlation_r"]
        vif_threshold = cfg["thresholds"]["vif"]
        vif_fallback = cfg["thresholds"]["vif_fallback"]

        print("Configuration:")
        print(f"  Correlation threshold: {corr_threshold}")
        print(f"  VIF threshold: {vif_threshold}")
        print(f"  VIF fallback: {vif_fallback}")
        print()

        # =====================================================================
        # Step 1: Load data
        # =====================================================================
        print("Step 1: Loading data...")

        indices_df = load_interim_parquet(root, "aligned_indices")
        print(f"  Loaded aligned indices: {len(indices_df):,} rows")

        env_df = load_interim_parquet(root, "aligned_environment")
        print(f"  Loaded aligned environment: {len(env_df):,} rows")

        metrics_df = load_processed_parquet(root, "community_metrics")
        print(f"  Loaded community metrics: {len(metrics_df):,} rows")

        metadata_df = load_index_metadata(root)
        index_cols = extract_index_columns(indices_df)
        print(f"  Starting indices: {len(index_cols)}")
        print()

        # Standardize once
        print("  Standardizing indices for reduction...")
        indices_std = standardize_indices(indices_df, index_cols)
        print()

        # =====================================================================
        # Step 2: Run index reduction with both tiebreakers
        # =====================================================================
        print("Step 2: Running index reduction...")
        print("=" * 50)

        print("\nPRIMARY (tiebreaker='first'):")
        primary_indices, primary_dropped = run_reduction(
            indices_df, indices_std, index_cols, metadata_df,
            corr_threshold, vif_threshold, vif_fallback,
            tiebreaker="first",
        )

        print("\nALTERNATE (tiebreaker='second'):")
        alternate_indices, alternate_dropped = run_reduction(
            indices_df, indices_std, index_cols, metadata_df,
            corr_threshold, vif_threshold, vif_fallback,
            tiebreaker="second",
        )

        # =====================================================================
        # Step 3: Compare index lists
        # =====================================================================
        print()
        print("Step 3: Comparing index lists...")
        print("=" * 50)

        index_comparison = compare_index_lists(
            primary_indices, alternate_indices,
            primary_dropped, alternate_dropped,
        )

        print(f"\n  Primary: {index_comparison['primary_count']} indices")
        print(f"  Alternate: {index_comparison['alternate_count']} indices")
        print(f"  Common: {index_comparison['common_count']} indices")
        print(f"  Only in primary: {index_comparison['only_primary']}")
        print(f"  Only in alternate: {index_comparison['only_alternate']}")

        if not index_comparison["only_primary"] and not index_comparison["only_alternate"]:
            print("\n  INDEX LISTS ARE IDENTICAL - no model comparison needed")
            robustness = "identical"
        else:
            print("\n  Index lists differ - proceeding to model comparison...")
            robustness = "needs_model_comparison"
        print()

        # =====================================================================
        # Step 4: Prepare analysis-ready datasets
        # =====================================================================
        print("Step 4: Preparing analysis-ready datasets...")

        primary_data = prepare_analysis_data(
            indices_df, env_df, metrics_df, primary_indices
        )
        print(f"  Primary dataset: {len(primary_data):,} rows")

        alternate_data = prepare_analysis_data(
            indices_df, env_df, metrics_df, alternate_indices
        )
        print(f"  Alternate dataset: {len(alternate_data):,} rows")

        # Save temporary datasets for R
        sensitivity_dir = root / "results" / "sensitivity"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)

        primary_data_path = sensitivity_dir / "primary_data.parquet"
        alternate_data_path = sensitivity_dir / "alternate_data.parquet"

        primary_data.to_parquet(primary_data_path)
        alternate_data.to_parquet(alternate_data_path)
        print(f"  Saved datasets to {sensitivity_dir}")
        print()

        # =====================================================================
        # Step 5: Run models (if indices differ)
        # =====================================================================
        model_comparison = {}

        if robustness != "identical":
            print("Step 5: Fitting models...")
            print("=" * 50)

            primary_results_path = sensitivity_dir / "primary_model_results.json"
            alternate_results_path = sensitivity_dir / "alternate_model_results.json"

            print("\nFitting PRIMARY models...")
            primary_results = run_r_models(
                primary_data_path, primary_indices,
                primary_results_path, "primary"
            )

            print("\nFitting ALTERNATE models...")
            alternate_results = run_r_models(
                alternate_data_path, alternate_indices,
                alternate_results_path, "alternate"
            )

            # =====================================================================
            # Step 6: Compare model results
            # =====================================================================
            print()
            print("Step 6: Comparing model results...")
            print("=" * 50)

            if "error" not in primary_results and "error" not in alternate_results:
                model_comparison = compare_model_results(primary_results, alternate_results)

                for response, comp in model_comparison.items():
                    print(f"\n  {response}:")
                    if "error" in str(comp):
                        print(f"    Error in comparison")
                        continue
                    print(f"    Primary AIC: {comp['primary_AIC']:.1f}")
                    print(f"    Alternate AIC: {comp['alternate_AIC']:.1f}")
                    print(f"    Difference: {comp['AIC_difference']:.1f} ({comp['AIC_interpretation']})")
                    if comp["sig_only_primary"]:
                        print(f"    Significant only in primary: {comp['sig_only_primary']}")
                    if comp["sig_only_alternate"]:
                        print(f"    Significant only in alternate: {comp['sig_only_alternate']}")

                # Determine overall robustness
                aic_diffs = [
                    abs(c.get("AIC_difference", 0))
                    for c in model_comparison.values()
                    if isinstance(c, dict) and "AIC_difference" in c
                ]
                if all(d < 2 for d in aic_diffs):
                    robustness = "highly_robust"
                elif all(d < 10 for d in aic_diffs):
                    robustness = "moderately_robust"
                else:
                    robustness = "sensitive"
            else:
                print("  Model fitting encountered errors - comparison incomplete")
                robustness = "error"

        # =====================================================================
        # Step 7: Save summary
        # =====================================================================
        print()
        print("Step 7: Saving results...")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "correlation_threshold": corr_threshold,
                "vif_threshold": vif_threshold,
                "vif_fallback": vif_fallback,
            },
            "index_comparison": index_comparison,
            "model_comparison": model_comparison,
            "robustness": robustness,
            "interpretation": {
                "identical": "Tiebreaker choice has no effect on index selection",
                "highly_robust": "Models are equivalent (all AIC differences < 2)",
                "moderately_robust": "Models show minor differences (AIC differences < 10)",
                "sensitive": "Models show meaningful differences - investigate further",
                "error": "Analysis incomplete due to errors",
            }.get(robustness, "Unknown"),
        }

        summary_path = sensitivity_dir / "sensitivity_summary.json"
        save_summary_json(summary, summary_path)
        print(f"  Saved summary: {summary_path}")

        # Also save comparison CSV
        if index_comparison["swap_details"]:
            swap_df = pd.DataFrame(index_comparison["swap_details"])
            swap_path = sensitivity_dir / "index_swaps.csv"
            swap_df.to_csv(swap_path, index=False)
            print(f"  Saved index swaps: {swap_path}")

        print()
        print("=" * 70)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print(f"Robustness: {robustness.upper()}")
        print(f"Interpretation: {summary['interpretation']}")
        print()
        print(f"Results saved to: {sensitivity_dir}")
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
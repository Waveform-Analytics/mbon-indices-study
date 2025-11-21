#!/usr/bin/env python3
"""
Script 4: Prepare Model Data for GLMMs (v2 - Universal + Taxa-Specific)
=======================================================================

Creates datasets for both universal and taxa-specific GLMM approaches.
"""

import pandas as pd
import json
from pathlib import Path
import os
from mbon_pipeline.core.paths import ProjectPaths

# Target groups and response variables
TARGET_GROUPS = {
    'fish': ['fish_activity', 'fish_richness', 'fish_present'],
    'dolphin': ['dolphin_activity', 'dolphin_whistles', 'dolphin_echolocation',
                'dolphin_burst_pulses', 'dolphin_present'],
    'vessel': ['vessel_present']
}

COVARIATES = ['temp', 'depth', 'month', 'hour', 'station']

def create_model_dataset(df_indices, df_taxa, acoustic_indices, response_vars, paths, filename):
    """
    Create a model-ready dataset with specified indices and responses.

    Args:
        df_indices: DataFrame with standardized acoustic indices
        df_taxa: DataFrame with taxa metrics
        acoustic_indices: List of acoustic index names to include
        response_vars: List of response variable names to include
        paths: ProjectPaths instance
        filename: Output filename

    Returns:
        DataFrame: Model-ready dataset
    """
    # Filter acoustic indices to those present
    available_indices = [idx for idx in acoustic_indices if idx in df_indices.columns]
    missing = [idx for idx in acoustic_indices if idx not in df_indices.columns]
    if missing:
        print(f"⚠️  Skipping {len(missing)} missing indices: {', '.join(missing)}")

    # Merge datasets
    data = pd.merge(
        df_taxa[['Date', 'station'] + response_vars + ['month']],
        df_indices[['Date', 'station'] + available_indices + ['temp', 'depth', 'hour']],
        on=['Date', 'station'],
        how='inner'
    )

    # Column order: Date, responses, indices, covariates
    column_order = (['Date'] + response_vars + available_indices +
                   ['temp', 'depth', 'month', 'hour', 'station'])
    data = data[[c for c in column_order if c in data.columns]]

    # Save
    output_path = paths.processed_data / filename
    data.to_csv(output_path, index=False)

    return data, output_path

def main():
    print("=" * 60)
    print("SCRIPT 4: PREPARE MODEL DATA (UNIVERSAL + TAXA-SPECIFIC)")
    print("=" * 60)
    print()

    # Setup
    paths = ProjectPaths()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load standardized indices
    df_standardized = pd.read_parquet(paths.processed_data / "df_combined_standardized.parquet")
    print(f"✓ Standardized indices: {len(df_standardized)} records")

    # Load taxa metrics
    df_taxa = pd.read_parquet(paths.processed_data / "taxa_metrics.parquet")
    print(f"✓ Taxa metrics: {len(df_taxa)} records")

    # Validation: vessel_count must not be present
    if 'vessel_count' in df_taxa.columns:
        raise ValueError("vessel_count detected in taxa metrics; presence-only policy violated")

    # Add environmental data and hour to df_standardized
    df_raw = pd.read_parquet(paths.processed_data / "df_combined.parquet")
    df_standardized['temp'] = df_raw['Water temp (°C)']
    df_standardized['depth'] = df_raw['Water depth (m)']
    df_standardized['hour'] = pd.to_datetime(df_standardized['Date']).dt.hour

    print()

    # ========================================================================
    # LOAD INDEX SELECTIONS
    # ========================================================================

    print("📋 LOADING INDEX SELECTIONS")
    print("-" * 40)

    # Universal indices (prefer variant-specific if available)
    variant = os.getenv('MBON_INDICES_VARIANT', 'v2')
    vif_universal_variant = paths.processed_data / f"indices_final_vif_checked_{variant}.json"
    vif_universal_path = vif_universal_variant if vif_universal_variant.exists() else paths.processed_data / "indices_final_vif_checked.json"
    with open(vif_universal_path, 'r') as f:
        universal_indices = json.load(f)['final_indices']
    print(f"✓ Universal approach: {len(universal_indices)} indices")
    print(f"  {', '.join(universal_indices)}")

    # Taxa-specific indices
    taxa_indices = {}
    for group in TARGET_GROUPS.keys():
        with open(paths.processed_data / f"indices_final_{group}.json", 'r') as f:
            taxa_indices[group] = json.load(f)['final_indices']
        print(f"✓ {group.capitalize()}: {len(taxa_indices[group])} indices")
        print(f"  {', '.join(taxa_indices[group])}")

    print()

    # ========================================================================
    # CREATE DATASETS
    # ========================================================================

    print("💾 CREATING MODEL DATASETS")
    print("-" * 40)

    datasets_created = []

    # 1. Universal approach (all indices, all responses)
    all_responses = []
    for group_vars in TARGET_GROUPS.values():
        all_responses.extend(group_vars)

    data, path = create_model_dataset(
        df_standardized, df_taxa, universal_indices, all_responses,
        paths, "model_data_universal.csv"
    )
    print(f"✓ Universal: {path.name}")
    print(f"  {len(data)} rows × {len(universal_indices)} indices × {len(all_responses)} responses")
    datasets_created.append(('universal', len(data), len(universal_indices), len(all_responses)))

    # 2-4. Taxa-specific approaches
    for group, response_vars in TARGET_GROUPS.items():
        data, path = create_model_dataset(
            df_standardized, df_taxa, taxa_indices[group], response_vars,
            paths, f"model_data_{group}.csv"
        )
        print(f"✓ {group.capitalize()}: {path.name}")
        print(f"  {len(data)} rows × {len(taxa_indices[group])} indices × {len(response_vars)} responses")
        datasets_created.append((group, len(data), len(taxa_indices[group]), len(response_vars)))

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 60)
    print("✅ MODEL DATA PREPARATION COMPLETE")
    print("=" * 60)
    print()

    print("Datasets created:")
    for approach, n_rows, n_indices, n_responses in datasets_created:
        print(f"  {approach:12s}: {n_rows:5d} obs × {n_indices:2d} indices × {n_responses:2d} responses")

    print()
    print("🔄 Ready for Script 05: GLMM fitting")
    print()

if __name__ == "__main__":
    main()

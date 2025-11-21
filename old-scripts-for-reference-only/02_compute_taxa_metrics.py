#!/usr/bin/env python3
"""
Script 2: Compute Multi-Taxa Metrics
====================================

Purpose: Calculate response variables for all taxa (fish, dolphins, vessels)
Key Question: What biological and anthropogenic activity patterns are present?

This script creates all response variables that will be modeled in GLMMs:
- Fish community metrics (richness, activity, presence)
- Dolphin metrics (activity by call type, total activity, presence)
- Vessel metrics (presence, count)

Key Outputs:
- data/processed/taxa_metrics.parquet - All response variables
- figures/02_taxa_distributions.png - Distribution summaries
- figures/02_temporal_patterns.png - Temporal patterns by taxa

Design Philosophy:
- Metadata-driven: Uses det_column_names.csv to identify taxa groups
- Modular: Creates all response variables in one place
- Flexible: Easy to add new metrics or taxa groups
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

# Import pipeline utilities
from mbon_pipeline.utils.metadata import MetadataManager
from mbon_pipeline.core.paths import ProjectPaths

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("SCRIPT 2: COMPUTE MULTI-TAXA METRICS")
    print("=" * 60)
    print()

    # Setup paths
    paths = ProjectPaths()
    metadata_mgr = MetadataManager(project_root=paths.project_root)

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("📥 LOADING DATA")
    print("-" * 40)

    # Load combined dataset
    combined_path = paths.processed_data / "df_combined.parquet"
    df_combined = pd.read_parquet(combined_path)
    print(f"✓ Loaded combined dataset: {len(df_combined)} records")

    # Load lookup table
    lookup_path = paths.processed_data / "lookup_table.json"
    with open(lookup_path, 'r') as f:
        lookup_table = json.load(f)
    print(f"✓ Loaded lookup table: {len(lookup_table)} columns")
    print()

    # ========================================================================
    # IDENTIFY TAXA GROUPS FROM METADATA
    # ========================================================================

    print("🔍 IDENTIFYING TAXA GROUPS FROM METADATA")
    print("-" * 40)

    # Load detection metadata
    det_metadata = metadata_mgr.load_detection_metadata()

    # Filter to columns marked for keeping
    det_metadata_keep = det_metadata[det_metadata['keep_species'] == 1].copy()

    # Group by taxa type
    taxa_groups = {}
    for group_name in det_metadata_keep['group'].unique():
        group_cols = det_metadata_keep[det_metadata_keep['group'] == group_name]['long_name'].tolist()
        # Filter to columns actually present in data
        available_cols = [col for col in group_cols if col in df_combined.columns]
        if available_cols:
            taxa_groups[group_name] = available_cols

    # Report what we found
    print(f"Found {len(taxa_groups)} taxa groups in data:")
    for group, cols in taxa_groups.items():
        print(f"  • {group}: {len(cols)} columns")
        for col in cols:
            print(f"    - {col}")
    print()

    # ========================================================================
    # COMPUTE FISH METRICS
    # ========================================================================

    print("🐟 COMPUTING FISH METRICS")
    print("-" * 40)

    fish_cols = taxa_groups.get('fish', [])
    if not fish_cols:
        print("⚠️  No fish columns found!")
        fish_metrics = pd.DataFrame()
    else:
        df_fish = df_combined[fish_cols].copy()

        # Fish richness: number of species detected (any value > 0)
        fish_richness = (df_fish > 0).sum(axis=1)

        # Fish activity: sum of ordinal intensity values
        # (0=none, 1=one, 2=several, 3=chorus)
        fish_activity = df_fish.sum(axis=1)

        # Fish presence: binary indicator (any fish detected?)
        fish_present = (fish_activity > 0).astype(int)

        fish_metrics = pd.DataFrame({
            'fish_richness': fish_richness,
            'fish_activity': fish_activity,
            'fish_present': fish_present
        })

        print(f"✓ Fish species included: {len(fish_cols)}")
        print(f"  Detection rate: {(fish_present == 1).sum() / len(fish_present):.1%}")
        print(f"  Mean richness: {fish_richness.mean():.2f} species")
        print(f"  Mean activity: {fish_activity.mean():.2f}")
        print()

    # ========================================================================
    # COMPUTE DOLPHIN METRICS
    # ========================================================================

    print("🐬 COMPUTING DOLPHIN METRICS")
    print("-" * 40)

    dolphin_cols = taxa_groups.get('dolphin', [])
    if not dolphin_cols:
        print("⚠️  No dolphin columns found!")
        dolphin_metrics = pd.DataFrame()
    else:
        df_dolphin = df_combined[dolphin_cols].copy()

        # Individual call type metrics (if available)
        dolphin_whistles = df_dolphin['Bottlenose dolphin whistles'] if 'Bottlenose dolphin whistles' in df_dolphin.columns else pd.Series(0, index=df_dolphin.index)
        dolphin_echolocation = df_dolphin['Bottlenose dolphin echolocation'] if 'Bottlenose dolphin echolocation' in df_dolphin.columns else pd.Series(0, index=df_dolphin.index)
        dolphin_burst_pulses = df_dolphin['Bottlenose dolphin burst pulses'] if 'Bottlenose dolphin burst pulses' in df_dolphin.columns else pd.Series(0, index=df_dolphin.index)

        # Total dolphin activity: sum of all call types
        dolphin_activity = df_dolphin.sum(axis=1)

        # Dolphin presence: binary indicator
        dolphin_present = (dolphin_activity > 0).astype(int)

        dolphin_metrics = pd.DataFrame({
            'dolphin_whistles': dolphin_whistles,
            'dolphin_echolocation': dolphin_echolocation,
            'dolphin_burst_pulses': dolphin_burst_pulses,
            'dolphin_activity': dolphin_activity,
            'dolphin_present': dolphin_present
        })

        print(f"✓ Dolphin call types: {len(dolphin_cols)}")
        print(f"  Detection rate: {(dolphin_present == 1).sum() / len(dolphin_present):.1%}")
        print(f"  Mean activity: {dolphin_activity.mean():.2f}")
        print(f"  Whistles: {dolphin_whistles.sum()} total detections")
        print(f"  Echolocation: {dolphin_echolocation.sum()} total detections")
        print(f"  Burst pulses: {dolphin_burst_pulses.sum()} total detections")
        print()

    # ========================================================================
    # COMPUTE VESSEL METRICS (presence-only)
    # ========================================================================

    print("🚢 COMPUTING VESSEL METRICS (presence-only)")
    print("-" * 40)

    vessel_cols = taxa_groups.get('vessel', [])
    if not vessel_cols:
        print("⚠️  No vessel columns found!")
        vessel_metrics = pd.DataFrame()
    else:
        df_vessel = df_combined[vessel_cols].copy()

        # Vessel presence: binary indicator
        vessel_present = (df_vessel.sum(axis=1) > 0).astype(int)

        vessel_metrics = pd.DataFrame({
            'vessel_present': vessel_present
        })

        print(f"✓ Vessel columns: {len(vessel_cols)}")
        print(f"  Detection rate: {(vessel_present == 1).sum() / len(vessel_present):.1%}")
        print()

    # ========================================================================
    # COMBINE ALL METRICS
    # ========================================================================

    print("🔗 COMBINING ALL METRICS")
    print("-" * 40)

    # Combine all metrics
    taxa_metrics = pd.concat([fish_metrics, dolphin_metrics, vessel_metrics], axis=1)

    # Add metadata for reference
    taxa_metrics['Date'] = df_combined['Date'].values
    taxa_metrics['station'] = df_combined['station'].values

    # Extract temporal features
    taxa_metrics['datetime'] = pd.to_datetime(taxa_metrics['Date'])
    taxa_metrics['month'] = taxa_metrics['datetime'].dt.month
    taxa_metrics['day_of_year'] = taxa_metrics['datetime'].dt.dayofyear

    print(f"✓ Combined metrics dataset: {len(taxa_metrics)} records")
    print(f"✓ Total metrics: {len(taxa_metrics.columns) - 4} response variables")
    print()

    # ========================================================================
    # DATA QUALITY CHECK
    # ========================================================================

    print("📊 DATA QUALITY CHECK")
    print("-" * 40)

    # Check for missing values
    missing_summary = taxa_metrics.isnull().sum()
    if missing_summary.sum() > 0:
        print("⚠️  Missing values detected:")
        for col in missing_summary[missing_summary > 0].index:
            print(f"  {col}: {missing_summary[col]} missing")
    else:
        print("✓ No missing values detected")

    # Summary statistics
    print("\nResponse variable summary:")
    numeric_cols = taxa_metrics.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['month', 'day_of_year']]
    print(taxa_metrics[numeric_cols].describe())
    print()

    # ========================================================================
    # VALIDATION
    # ========================================================================

    # Prevent accidental reintroduction of vessel_count
    if 'vessel_count' in taxa_metrics.columns:
        raise ValueError("vessel_count detected in taxa_metrics; presence-only policy violated")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("💾 SAVING RESULTS")
    print("-" * 40)

    # Save taxa metrics
    output_path = paths.processed_data / "taxa_metrics.parquet"
    taxa_metrics.to_parquet(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    # Save summary report
    report = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_records': len(taxa_metrics),
        'taxa_groups_found': list(taxa_groups.keys()),
        'fish': {
            'columns': fish_cols,
            'detection_rate': float((taxa_metrics['fish_present'] == 1).sum() / len(taxa_metrics)) if 'fish_present' in taxa_metrics else 0,
            'mean_richness': float(taxa_metrics['fish_richness'].mean()) if 'fish_richness' in taxa_metrics else 0,
            'mean_activity': float(taxa_metrics['fish_activity'].mean()) if 'fish_activity' in taxa_metrics else 0
        },
        'dolphins': {
            'columns': dolphin_cols,
            'detection_rate': float((taxa_metrics['dolphin_present'] == 1).sum() / len(taxa_metrics)) if 'dolphin_present' in taxa_metrics else 0,
            'mean_activity': float(taxa_metrics['dolphin_activity'].mean()) if 'dolphin_activity' in taxa_metrics else 0
        },
        'vessels': {
            'columns': vessel_cols,
            'detection_rate': float((taxa_metrics['vessel_present'] == 1).sum() / len(taxa_metrics)) if 'vessel_present' in taxa_metrics else 0
        }
    }

    report_path = paths.processed_data / "taxa_metrics_summary.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: {report_path}")
    print()

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    print("📈 GENERATING VISUALIZATIONS")
    print("-" * 40)

    create_distribution_plots(taxa_metrics, paths)
    create_temporal_patterns(taxa_metrics, paths)

    print("✓ Distribution plots saved")
    print("✓ Temporal pattern plots saved")
    print()

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print("=" * 60)
    print("🎉 TAXA METRICS COMPUTATION COMPLETED")
    print("=" * 60)
    print()
    print("📋 KEY OUTPUTS:")
    print(f"   • {output_path}")
    print(f"   • {report_path}")
    print(f"   • {paths.get_figure_path('02_taxa_distributions.png')}")
    print(f"   • {paths.get_figure_path('02_temporal_patterns.png')}")
    print()
    print("🔄 READY FOR SCRIPT 3: Acoustic Index Selection")

    return taxa_metrics, report


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_distribution_plots(taxa_metrics, paths):
    """Create distribution summary plots for all taxa."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Taxa Response Variable Distributions', fontsize=16, fontweight='bold')

    # Fish metrics
    if 'fish_richness' in taxa_metrics.columns:
        axes[0, 0].hist(taxa_metrics['fish_richness'], bins=20, edgecolor='black')
        axes[0, 0].set_title('Fish Richness')
        axes[0, 0].set_xlabel('Number of Species')
        axes[0, 0].set_ylabel('Frequency')

    if 'fish_activity' in taxa_metrics.columns:
        axes[0, 1].hist(taxa_metrics['fish_activity'], bins=20, edgecolor='black')
        axes[0, 1].set_title('Fish Activity')
        axes[0, 1].set_xlabel('Activity Score')
        axes[0, 1].set_ylabel('Frequency')

    if 'fish_present' in taxa_metrics.columns:
        fish_present_counts = taxa_metrics['fish_present'].value_counts()
        axes[0, 2].bar(fish_present_counts.index, fish_present_counts.values)
        axes[0, 2].set_title('Fish Presence')
        axes[0, 2].set_xlabel('Present (0=No, 1=Yes)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_xticks([0, 1])

    # Dolphin metrics
    if 'dolphin_activity' in taxa_metrics.columns:
        axes[1, 0].hist(taxa_metrics['dolphin_activity'], bins=20, edgecolor='black')
        axes[1, 0].set_title('Dolphin Activity (All Call Types)')
        axes[1, 0].set_xlabel('Activity Score')
        axes[1, 0].set_ylabel('Frequency')

    if 'dolphin_whistles' in taxa_metrics.columns:
        axes[1, 1].hist(taxa_metrics['dolphin_whistles'], bins=20, edgecolor='black')
        axes[1, 1].set_title('Dolphin Whistles')
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_ylabel('Frequency')

    if 'dolphin_present' in taxa_metrics.columns:
        dolphin_present_counts = taxa_metrics['dolphin_present'].value_counts()
        axes[1, 2].bar(dolphin_present_counts.index, dolphin_present_counts.values)
        axes[1, 2].set_title('Dolphin Presence')
        axes[1, 2].set_xlabel('Present (0=No, 1=Yes)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_xticks([0, 1])

    # Vessel metrics
    if 'vessel_present' in taxa_metrics.columns:
        vessel_present_counts = taxa_metrics['vessel_present'].value_counts()
        axes[2, 0].bar(vessel_present_counts.index, vessel_present_counts.values)
        axes[2, 0].set_title('Vessel Presence')
        axes[2, 0].set_xlabel('Present (0=No, 1=Yes)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_xticks([0, 1])

    # Remove vessel count visualization (presence-only policy)

    # Detection rate comparison
    detection_rates = {}
    if 'fish_present' in taxa_metrics.columns:
        detection_rates['Fish'] = (taxa_metrics['fish_present'] == 1).mean()
    if 'dolphin_present' in taxa_metrics.columns:
        detection_rates['Dolphins'] = (taxa_metrics['dolphin_present'] == 1).mean()
    if 'vessel_present' in taxa_metrics.columns:
        detection_rates['Vessels'] = (taxa_metrics['vessel_present'] == 1).mean()

    if detection_rates:
        axes[2, 2].bar(detection_rates.keys(), detection_rates.values())
        axes[2, 2].set_title('Detection Rate Comparison')
        axes[2, 2].set_ylabel('Detection Rate')
        axes[2, 2].set_ylim(0, 1)
        for i, (taxon, rate) in enumerate(detection_rates.items()):
            axes[2, 2].text(i, rate + 0.02, f'{rate:.1%}', ha='center')

    plt.tight_layout()
    figure_path = paths.get_figure_path('02_taxa_distributions.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_temporal_patterns(taxa_metrics, paths):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    if 'month' in taxa_metrics.columns:
        monthly_presence = pd.DataFrame()
        if 'fish_present' in taxa_metrics.columns:
            monthly_presence['Fish'] = taxa_metrics.groupby('month')['fish_present'].mean() * 100
        if 'dolphin_present' in taxa_metrics.columns:
            monthly_presence['Dolphins'] = taxa_metrics.groupby('month')['dolphin_present'].mean() * 100
        if 'vessel_present' in taxa_metrics.columns:
            monthly_presence['Vessels'] = taxa_metrics.groupby('month')['vessel_present'].mean() * 100
        if not monthly_presence.empty:
            monthly_presence.plot(ax=axes[0], marker='o', linewidth=3)
            axes[0].set_title('Percent Presence by Month')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Presence (%)')
            axes[0].set_ylim(0, 100)
            axes[0].legend(frameon=False)

    if 'station' in taxa_metrics.columns:
        station_presence = pd.DataFrame()
        if 'fish_present' in taxa_metrics.columns:
            station_presence['Fish'] = taxa_metrics.groupby('station')['fish_present'].mean() * 100
        if 'dolphin_present' in taxa_metrics.columns:
            station_presence['Dolphins'] = taxa_metrics.groupby('station')['dolphin_present'].mean() * 100
        if 'vessel_present' in taxa_metrics.columns:
            station_presence['Vessels'] = taxa_metrics.groupby('station')['vessel_present'].mean() * 100
        if not station_presence.empty:
            desired_order = [s for s in ['37M', '14M', '9M'] if s in station_presence.index]
            if desired_order:
                station_presence = station_presence.reindex(desired_order)
            station_presence.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Percent Presence by Station')
            axes[1].set_xlabel('Station')
            axes[1].set_ylabel('Presence (%)')
            axes[1].set_ylim(0, 100)
            axes[1].legend(frameon=False)
            axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    figure_path = paths.get_figure_path('02_temporal_patterns.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    taxa_metrics, report = main()

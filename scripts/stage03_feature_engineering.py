"""
Stage 03: Feature Engineering

Creates temporal features, grouping IDs, AR1 sequences, and merges all data sources
into a single analysis-ready dataset for modeling.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import (
    load_interim_parquet,
    load_processed_parquet,
    load_final_indices_list,
    save_parquet,
    save_summary_json,
)
from mbon_indices.utils.logging import setup_stage_logging


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from datetime_local.

    Uses fixed EST (UTC-5) for biological interpretation and consistent 2-hour bins.
    """
    out = df.copy()

    # Extract hour of day from local time (0-23)
    out['hour_of_day'] = out['datetime_local'].dt.hour

    # Cyclic encoding of hour (24-hour cycle)
    out['sin_hour'] = np.sin(2 * np.pi * out['hour_of_day'] / 24)
    out['cos_hour'] = np.cos(2 * np.pi * out['hour_of_day'] / 24)

    # Day of year from local time
    out['day_of_year'] = out['datetime_local'].dt.dayofyear

    # Date for grouping
    out['date'] = out['datetime_local'].dt.date.astype(str)

    print(f"✓ Created temporal features")
    return out


def create_grouping_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grouping IDs for random effects.

    - day_id: date + station for daily random effects
    - month_id: year-month for monthly grouping
    """
    out = df.copy()

    # day_id = date_station (e.g., "2018-01-01_9M")
    out['day_id'] = out['date'].astype(str) + '_' + out['station']

    # month_id = YYYY-MM
    out['month_id'] = out['datetime_local'].dt.strftime('%Y-%m')

    print(f"✓ Created grouping factors")
    print(f"    Unique day_id: {out['day_id'].nunique()}")
    print(f"    Unique month_id: {out['month_id'].nunique()}")
    return out


def create_ar1_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time_within_day sequence for AR1 modeling.

    Resets to 0 at start of each day_id (daily grouping).
    Orders by datetime within each day.
    """
    out = df.copy()

    # Sort by day_id and datetime
    out = out.sort_values(['day_id', 'datetime'])

    # Create 0-based sequence within each day
    out['time_within_day'] = out.groupby('day_id').cumcount()

    print(f"✓ Created AR1 sequence (time_within_day)")
    print(f"    Max sequence per day: {out.groupby('day_id')['time_within_day'].max().max()}")
    return out


def scale_covariates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Z-score scale specified covariate columns.

    Creates new columns with '_z' suffix.
    """
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[f'{col}_z'] = (out[col] - out[col].mean()) / out[col].std()

    print(f"✓ Scaled covariates: {columns}")
    return out


def merge_all_sources(
    indices_df: pd.DataFrame,
    env_df: pd.DataFrame,
    metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all data sources on station + datetime.

    Uses inner join to keep only complete observations.
    """
    # Merge indices with environment
    merged = pd.merge(
        indices_df,
        env_df,
        on=['station', 'datetime'],
        how='inner'
    )
    print(f"  After indices + environment merge: {len(merged):,} rows")

    # Merge with community metrics
    merged = pd.merge(
        merged,
        metrics_df,
        on=['station', 'datetime'],
        how='inner'
    )
    print(f"  After adding community metrics: {len(merged):,} rows")

    return merged


def validate_features(df: pd.DataFrame) -> dict:
    """
    Validate feature engineering outputs.

    Returns dict with validation results.
    """
    issues = []

    # Check cyclic features in range
    for col in ['sin_hour', 'cos_hour']:
        if col in df.columns:
            if df[col].min() < -1.01 or df[col].max() > 1.01:
                issues.append(f"{col}: values outside [-1, 1] range")

    # Check time_within_day is contiguous within day_id
    for day_id in df['day_id'].unique():
        day_data = df[df['day_id'] == day_id].sort_values('datetime')
        expected = list(range(len(day_data)))
        actual = day_data['time_within_day'].tolist()
        if expected != actual:
            issues.append(f"{day_id}: time_within_day not contiguous (expected {expected}, got {actual})")
            break  # Only report first issue

    # Check for required columns
    required = ['datetime', 'datetime_local', 'station', 'date', 'hour_of_day',
                'sin_hour', 'cos_hour', 'day_of_year', 'day_id', 'month_id',
                'time_within_day', 'temperature', 'depth']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def generate_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for feature engineering."""
    summary = {
        'total_rows': len(df),
        'stations': sorted(df['station'].unique().tolist()),
        'date_range': {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat()
        },
        'temporal_features': {
            'sin_hour_range': [float(df['sin_hour'].min()), float(df['sin_hour'].max())],
            'cos_hour_range': [float(df['cos_hour'].min()), float(df['cos_hour'].max())],
            'day_of_year_range': [int(df['day_of_year'].min()), int(df['day_of_year'].max())]
        },
        'grouping': {
            'unique_day_ids': int(df['day_id'].nunique()),
            'unique_month_ids': int(df['month_id'].nunique())
        },
        'by_station': {}
    }

    for station in summary['stations']:
        station_data = df[df['station'] == station]
        summary['by_station'][station] = {
            'rows': len(station_data),
            'unique_days': int(station_data['day_id'].nunique()),
            'mean_temperature': float(station_data['temperature'].mean()),
            'mean_depth': float(station_data['depth'].mean())
        }

    return summary


def plot_temporal_checks(df: pd.DataFrame, output_path: Path):
    """Create visualization to check temporal feature distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Hour of day distribution
    axes[0, 0].hist(df['hour_of_day'], bins=24, edgecolor='black')
    axes[0, 0].set_xlabel('Hour of Day (local time)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Hour of Day Distribution')

    # Cyclic features
    axes[0, 1].scatter(df['sin_hour'], df['cos_hour'], alpha=0.3, s=1)
    axes[0, 1].set_xlabel('sin(hour)')
    axes[0, 1].set_ylabel('cos(hour)')
    axes[0, 1].set_title('Cyclic Hour Encoding')
    axes[0, 1].set_aspect('equal')

    # Day of year
    axes[1, 0].hist(df['day_of_year'], bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Day of Year Distribution')

    # Time within day by station
    for station in df['station'].unique():
        station_data = df[df['station'] == station]
        axes[1, 1].hist(station_data['time_within_day'], bins=20, alpha=0.5, label=station)
    axes[1, 1].set_xlabel('Time Within Day (sequence)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('AR1 Sequence Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_outputs(root: Path, analysis_df: pd.DataFrame, summary: dict):
    """Save all Stage 03 outputs per spec."""

    # Create output directories
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)

    # 1. Save analysis-ready parquet
    analysis_path = root / "data" / "processed" / "analysis_ready.parquet"
    save_parquet(analysis_df, analysis_path)
    print(f"  ✓ Saved analysis-ready dataset: {analysis_path}")

    # 2. Save schema CSV
    schema_path = root / "results" / "tables" / "feature_engineering_schema.csv"
    schema_df = pd.DataFrame({
        'column': analysis_df.columns,
        'dtype': [str(analysis_df[col].dtype) for col in analysis_df.columns]
    })
    schema_df.to_csv(schema_path, index=False)
    print(f"  ✓ Saved schema: {schema_path}")

    # 3. Save temporal feature checks plot
    plot_path = root / "results" / "figures" / "temporal_feature_checks.png"
    plot_temporal_checks(analysis_df, plot_path)
    print(f"  ✓ Saved temporal checks plot: {plot_path}")

    # 4. Save summary JSON
    summary_path = root / "results" / "logs" / "feature_engineering_summary.json"
    save_summary_json(summary, summary_path)
    print(f"  ✓ Saved summary: {summary_path}")


def main():
    # Set up logging
    logger = setup_stage_logging(root, "stage03_feature_engineering")

    try:
        print("=" * 60)
        print("STAGE 03: FEATURE ENGINEERING")
        print("=" * 60)
        print()

        # Load configuration
        cfg = load_analysis_config(root)
        print("Configuration loaded")
        print()

        # Step 1: Load all data sources
        print("Step 1: Loading data sources...")

        # Load final indices list
        final_indices = load_final_indices_list(root)
        print(f"✓ Loaded final indices: {len(final_indices)} indices")

        # Load aligned indices and filter to final list
        indices_df = load_interim_parquet(root, "aligned_indices")
        keep_cols = ['station', 'datetime', 'datetime_local'] + [c for c in final_indices if c in indices_df.columns]
        indices_df = indices_df[keep_cols].copy()
        print(f"✓ Loaded aligned indices: {len(indices_df):,} rows, {len(final_indices)} indices")

        # Load aligned environment
        env_df = load_interim_parquet(root, "aligned_environment")
        env_df = env_df[['station', 'datetime', 'temperature', 'depth']].copy()
        print(f"✓ Loaded aligned environment: {len(env_df):,} rows")

        # Load community metrics
        metrics_df = load_processed_parquet(root, "community_metrics")
        print(f"✓ Loaded community metrics: {len(metrics_df):,} rows, {len(metrics_df.columns)} columns")
        print()

        # Step 2: Merge all sources
        print("Step 2: Merging data sources...")
        merged_df = merge_all_sources(indices_df, env_df, metrics_df)
        print(f"✓ Merged all sources: {len(merged_df):,} rows")
        print()

        # Step 3: Create temporal features
        print("Step 3: Creating temporal features...")
        merged_df = create_temporal_features(merged_df)
        print()

        # Step 4: Create grouping factors
        print("Step 4: Creating grouping factors...")
        merged_df = create_grouping_factors(merged_df)
        print()

        # Step 5: Create AR1 sequence
        print("Step 5: Creating AR1 sequence...")
        merged_df = create_ar1_sequence(merged_df)
        print()

        # Step 6: Scale covariates (if configured)
        print("Step 6: Scaling covariates...")
        scale_covariates_flag = cfg.get('covariates', {}).get('scale', False)
        if scale_covariates_flag:
            to_scale = cfg.get('covariates', {}).get('to_scale', ['temperature', 'depth'])
            merged_df = scale_covariates(merged_df, to_scale)
        else:
            print("  Skipped (not enabled in config)")
        print()

        # Step 7: Validation
        print("Step 7: Validating features...")
        validation = validate_features(merged_df)
        if validation['valid']:
            print("✓ All validation checks passed")
        else:
            print("⚠ Validation issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        print()

        # Step 8: Generate summary
        print("Step 8: Generating summary...")
        summary = generate_summary(merged_df)
        print(f"✓ Summary generated")
        print()

        # Display summary
        print("Feature Engineering Summary:")
        print(f"  Total observations: {summary['total_rows']:,}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Unique day_ids: {summary['grouping']['unique_day_ids']}")
        print(f"  Unique month_ids: {summary['grouping']['unique_month_ids']}")
        print()
        for station in summary['stations']:
            station_sum = summary['by_station'][station]
            print(f"  {station}:")
            print(f"    Rows: {station_sum['rows']:,}")
            print(f"    Unique days: {station_sum['unique_days']}")
            print(f"    Mean temperature: {station_sum['mean_temperature']:.2f}°C")
            print(f"    Mean depth: {station_sum['mean_depth']:.2f}m")
        print()

        # Step 9: Save outputs
        print("Step 9: Saving outputs...")
        save_outputs(root, merged_df, summary)
        print()

        print("=" * 60)
        print("✓ Stage 03 complete")
        print(f"  Final dataset: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
        print("=" * 60)
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        # Close logger and restore stdout
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()

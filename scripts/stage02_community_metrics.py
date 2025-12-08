"""
Stage 02: Community Metrics

Derives biological community metrics from aligned detections:
- Fish: activity, richness, presence
- Dolphin: echolocation, burst pulse, whistle counts + activity + presence
- Vessel: presence
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.data import load_interim_parquet, save_parquet, save_summary_json
from mbon_indices.utils.logging import setup_stage_logging
from mbon_indices.utils.run_history import append_to_run_history


def load_column_metadata(root: Path) -> pd.DataFrame:
    """Load detection column metadata with keep_species filter."""
    metadata_path = root / "data" / "raw" / "metadata" / "det_column_names.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Column metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"✓ Loaded column metadata: {len(df)} columns")
    return df


def get_species_columns(metadata_df: pd.DataFrame, group: str) -> list[str]:
    """Get column names for a group where keep_species=1."""
    filtered = metadata_df[
        (metadata_df['group'] == group) &
        (metadata_df['keep_species'] == 1)
    ]
    columns = filtered['long_name'].tolist()
    print(f"  {group}: {len(columns)} columns")
    for col in columns:
        print(f"    - {col}")
    return columns


def compute_fish_metrics(df: pd.DataFrame, fish_cols: list[str]) -> pd.DataFrame:
    """
    Compute fish community metrics.

    Returns DataFrame with columns: fish_activity, fish_richness, fish_presence
    """
    # Ensure numeric types
    fish_data = df[fish_cols].apply(pd.to_numeric, errors='coerce')

    # Fish activity: sum across species
    fish_activity = fish_data.sum(axis=1)

    # Fish richness: count species with >0 detections
    fish_richness = (fish_data > 0).sum(axis=1)

    # Fish presence: binary (1 if any species present)
    fish_presence = (fish_richness > 0).astype(int)

    return pd.DataFrame({
        'fish_activity': fish_activity,
        'fish_richness': fish_richness,
        'fish_presence': fish_presence
    })


def compute_dolphin_metrics(df: pd.DataFrame, dolphin_cols: list[str]) -> pd.DataFrame:
    """
    Compute dolphin community metrics.

    Returns DataFrame with columns: dolphin_echolocation, dolphin_burst_pulse,
    dolphin_whistle, dolphin_activity, dolphin_presence
    """
    # Convert to numeric (handles string columns)
    dolphin_data = df[dolphin_cols].apply(pd.to_numeric, errors='coerce')

    # Individual call types
    dolphin_echolocation = dolphin_data.iloc[:, 0] if len(dolphin_cols) > 0 else pd.Series(0, index=df.index)
    dolphin_burst_pulse = dolphin_data.iloc[:, 1] if len(dolphin_cols) > 1 else pd.Series(0, index=df.index)
    dolphin_whistle = dolphin_data.iloc[:, 2] if len(dolphin_cols) > 2 else pd.Series(0, index=df.index)

    # Dolphin activity: sum across call types
    dolphin_activity = dolphin_data.sum(axis=1)

    # Dolphin presence: binary (1 if any call type present)
    dolphin_presence = (dolphin_activity > 0).astype(int)

    return pd.DataFrame({
        'dolphin_echolocation': dolphin_echolocation,
        'dolphin_burst_pulse': dolphin_burst_pulse,
        'dolphin_whistle': dolphin_whistle,
        'dolphin_activity': dolphin_activity,
        'dolphin_presence': dolphin_presence
    })


def compute_vessel_metrics(df: pd.DataFrame, vessel_col: str) -> pd.DataFrame:
    """
    Compute vessel metrics.

    Returns DataFrame with column: vessel_presence
    """
    # Convert to numeric and create binary presence
    vessel_data = pd.to_numeric(df[vessel_col], errors='coerce').fillna(0)
    vessel_presence = (vessel_data > 0).astype(int)

    return pd.DataFrame({
        'vessel_presence': vessel_presence
    })


def validate_metrics(metrics_df: pd.DataFrame) -> dict:
    """
    Validate community metrics.

    Returns dict with validation results.
    """
    issues = []

    # Check for negative values in count columns
    count_cols = [c for c in metrics_df.columns if c not in ['datetime', 'station']]
    for col in count_cols:
        if (metrics_df[col] < 0).any():
            n_negative = (metrics_df[col] < 0).sum()
            issues.append(f"{col}: {n_negative} negative values")

    # Check presence flags are binary
    presence_cols = [c for c in metrics_df.columns if 'presence' in c]
    for col in presence_cols:
        unique_vals = metrics_df[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            issues.append(f"{col}: non-binary values found: {unique_vals}")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def generate_summary(metrics_df: pd.DataFrame) -> dict:
    """Generate summary statistics for community metrics."""
    summary = {
        'total_rows': len(metrics_df),
        'stations': sorted(metrics_df['station'].unique().tolist()),
        'date_range': {
            'start': metrics_df['datetime'].min().isoformat(),
            'end': metrics_df['datetime'].max().isoformat()
        },
        'by_station': {}
    }

    # Per-station summaries
    for station in summary['stations']:
        station_data = metrics_df[metrics_df['station'] == station]
        summary['by_station'][station] = {
            'rows': len(station_data),
            'fish_presence_fraction': float(station_data['fish_presence'].mean()),
            'dolphin_presence_fraction': float(station_data['dolphin_presence'].mean()),
            'vessel_presence_fraction': float(station_data['vessel_presence'].mean()),
            'mean_fish_richness': float(station_data['fish_richness'].mean())
        }

    return summary


def save_outputs(root: Path, metrics_df: pd.DataFrame, summary: dict):
    """Save all Stage 02 outputs per spec."""

    # Create output directories
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)

    # 1. Save community metrics parquet
    metrics_path = root / "data" / "processed" / "community_metrics.parquet"
    save_parquet(metrics_df, metrics_path)
    print(f"  ✓ Saved community metrics: {metrics_path}")

    # 2. Save schema CSV
    schema_path = root / "results" / "tables" / "community_metrics_schema.csv"
    schema_df = pd.DataFrame({
        'column': metrics_df.columns,
        'dtype': [str(metrics_df[col].dtype) for col in metrics_df.columns]
    })
    schema_df.to_csv(schema_path, index=False)
    print(f"  ✓ Saved schema: {schema_path}")

    # 3. Save summary JSON
    summary_path = root / "results" / "logs" / "community_metrics_summary.json"
    save_summary_json(summary, summary_path)
    print(f"  ✓ Saved summary: {summary_path}")


def main():
    # Set up logging
    logger = setup_stage_logging(root, "stage02_community_metrics")

    try:
        print("=" * 60)
        print("STAGE 02: COMMUNITY METRICS")
        print("=" * 60)
        print()

        # Load configuration
        cfg = load_analysis_config(root)
        print("Configuration:")
        print(f"  assume_zero_when_missing: {cfg.get('community_metrics', {}).get('assume_zero_when_missing', False)}")
        print()

        # Load data
        print("Step 1: Loading data...")
        detections_df = load_interim_parquet(root, "aligned_detections")
        print(f"✓ Loaded aligned detections: {len(detections_df):,} rows, {len(detections_df.columns)} columns")
        metadata_df = load_column_metadata(root)
        print()

        # Identify species columns
        print("Step 2: Identifying species columns...")
        fish_cols = get_species_columns(metadata_df, 'fish')
        dolphin_cols = get_species_columns(metadata_df, 'dolphin')
        vessel_cols = get_species_columns(metadata_df, 'vessel')
        print()

        # Compute metrics
        print("Step 3: Computing community metrics...")

        # Keep only datetime and station as keys
        keys = detections_df[['datetime', 'station']].copy()

        # Fish metrics
        print("  Computing fish metrics...")
        fish_metrics = compute_fish_metrics(detections_df, fish_cols)

        # Dolphin metrics
        print("  Computing dolphin metrics...")
        dolphin_metrics = compute_dolphin_metrics(detections_df, dolphin_cols)

        # Vessel metrics
        print("  Computing vessel metrics...")
        vessel_metrics = compute_vessel_metrics(detections_df, vessel_cols[0] if vessel_cols else 'Vessel')

        # Combine all metrics
        metrics_df = pd.concat([keys, fish_metrics, dolphin_metrics, vessel_metrics], axis=1)
        print(f"✓ Computed metrics: {len(metrics_df):,} rows, {len(metrics_df.columns)} columns")
        print()

        # Validation
        print("Step 4: Validating metrics...")
        validation = validate_metrics(metrics_df)
        if validation['valid']:
            print("✓ All validation checks passed")
        else:
            print("⚠ Validation issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        print()

        # Generate summary
        print("Step 5: Generating summary...")
        summary = generate_summary(metrics_df)
        print(f"✓ Summary generated for {len(summary['stations'])} stations")
        print()

        # Display summary
        print("Community Metrics Summary:")
        print(f"  Total observations: {summary['total_rows']:,}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print()
        for station in summary['stations']:
            station_sum = summary['by_station'][station]
            print(f"  {station}:")
            print(f"    Rows: {station_sum['rows']:,}")
            print(f"    Fish presence: {station_sum['fish_presence_fraction']:.1%}")
            print(f"    Dolphin presence: {station_sum['dolphin_presence_fraction']:.1%}")
            print(f"    Vessel presence: {station_sum['vessel_presence_fraction']:.1%}")
            print(f"    Mean fish richness: {station_sum['mean_fish_richness']:.2f}")
        print()

        # Save outputs
        print("Step 6: Saving outputs...")
        save_outputs(root, metrics_df, summary)
        print()

        # Append to run history
        append_to_run_history(
            root=root,
            stage="Stage 02: Community Metrics",
            config={
                "fish_species": len(fish_cols),
                "dolphin_cols": len(dolphin_cols),
                "vessel_cols": len(vessel_cols)
            },
            results={
                "rows": len(metrics_df),
                "stations": ", ".join(summary['stations']),
                "fish_presence_mean": f"{metrics_df['fish_presence'].mean():.1%}",
                "dolphin_presence_mean": f"{metrics_df['dolphin_presence'].mean():.1%}",
                "vessel_presence_mean": f"{metrics_df['vessel_presence'].mean():.1%}"
            },
            log_path=str(logger.log_path.relative_to(root))
        )

        print("=" * 60)
        print("✓ Stage 02 complete")
        print("=" * 60)
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        # Close logger and restore stdout
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()

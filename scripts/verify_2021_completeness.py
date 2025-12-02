"""
Verify that all 2021 data has complete 2-hour interval coverage.

For a full year at 2-hour resolution:
- 365 days * 12 intervals per day = 4,380 records per station
- 3 stations * 4,380 = 13,140 total records expected
"""

import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent


def check_temporal_completeness(df: pd.DataFrame, name: str, year: int = 2021):
    """Check if dataframe has complete 2-hour intervals for the year."""
    print(f"\n{'='*60}")
    print(f"{name.upper()} - {year} Completeness Check")
    print(f"{'='*60}")

    # Filter to 2021 only
    df_year = df[df['datetime'].dt.year == year].copy()

    if len(df_year) == 0:
        print(f"⚠️  No data for {year}")
        return

    print(f"Total rows for {year}: {len(df_year)}")

    # Check by station
    if 'station' in df_year.columns:
        print(f"\nBy Station:")
        for station in sorted(df_year['station'].unique()):
            station_df = df_year[df_year['station'] == station]
            print(f"  {station}: {len(station_df):,} rows")

            # Expected: 365 days * 12 (2-hour intervals) = 4,380
            expected = 365 * 12
            if len(station_df) == expected:
                print(f"    ✓ Complete (expected {expected:,})")
            else:
                print(f"    ⚠️  Expected {expected:,}, got {len(station_df):,} (diff: {len(station_df) - expected:+,})")

            # Check for gaps
            station_df = station_df.sort_values('datetime')
            time_diffs = station_df['datetime'].diff()
            expected_diff = pd.Timedelta(hours=2)

            # Find gaps (anything > 2 hours)
            gaps = time_diffs[time_diffs > expected_diff]
            if len(gaps) > 0:
                print(f"    ⚠️  Found {len(gaps)} gaps > 2 hours:")
                for idx, gap in gaps.head(5).items():
                    gap_dt = station_df.loc[idx, 'datetime']
                    print(f"      - {gap} gap at {gap_dt}")
            else:
                print(f"    ✓ No gaps (all consecutive 2-hour intervals)")

            # Check date range
            start = station_df['datetime'].min()
            end = station_df['datetime'].max()
            print(f"    Date range: {start} to {end}")


def main():
    print("VERIFYING 2021 DATA COMPLETENESS")
    print("Expected: 4,380 records per station (365 days * 12 intervals)")
    print("Expected total: 13,140 records (3 stations)")

    interim = root / "data" / "interim"

    # Check each aligned dataset
    artifacts = {
        "Detections": interim / "aligned_detections.parquet",
        "Environment": interim / "aligned_environment.parquet",
        "Indices": interim / "aligned_indices.parquet",
        "SPL": interim / "aligned_spl.parquet",
        "Base": interim / "aligned_base.parquet",
    }

    for name, path in artifacts.items():
        if not path.exists():
            print(f"\n⚠️  {name}: File not found at {path}")
            continue

        df = pd.read_parquet(path)
        check_temporal_completeness(df, name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("All datasets should have:")
    print("  - 4,380 rows per station for 2021")
    print("  - No gaps > 2 hours")
    print("  - Date range: 2021-01-01 to 2021-12-31")


if __name__ == "__main__":
    main()

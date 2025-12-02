"""
Test each loader individually with actual config
"""

import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.loaders.detections import load_detections
from mbon_indices.loaders.environment import load_environment
from mbon_indices.loaders.spl import load_spl
from mbon_indices.loaders.indices import load_indices


def test_detections_loader():
    print("=" * 60)
    print("TESTING DETECTIONS LOADER")
    print("=" * 60)

    cfg = load_analysis_config(root)
    stations = ["9M"]
    years = [2018]

    try:
        df = load_detections(root, stations, years, cfg)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns[:5])}")
        if 'datetime' in df.columns:
            print(f"✓ Datetime column exists")
            print(f"  - Non-null: {(~df['datetime'].isna()).sum()} / {len(df)}")
            print(f"  - Sample values:")
            print(df[['datetime', 'station']].head())
        else:
            print("✗ No datetime column!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_environment_loader():
    print("=" * 60)
    print("TESTING ENVIRONMENT LOADER")
    print("=" * 60)

    cfg = load_analysis_config(root)
    stations = ["9M"]
    years = [2018]

    try:
        df = load_environment(root, stations, years, cfg)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        if 'datetime' in df.columns:
            print(f"✓ Datetime column exists")
            print(f"  - Non-null: {(~df['datetime'].isna()).sum()} / {len(df)}")
            print(f"  - Sample values:")
            print(df[['datetime', 'station', 'temperature', 'depth']].head())
        else:
            print("✗ No datetime column!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_spl_loader():
    print("=" * 60)
    print("TESTING SPL LOADER")
    print("=" * 60)

    cfg = load_analysis_config(root)
    stations = ["9M"]
    years = [2021]

    try:
        df = load_spl(root, stations, years, cfg)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns[:5])}")
        if 'datetime' in df.columns:
            print(f"✓ Datetime column exists")
            print(f"  - Non-null: {(~df['datetime'].isna()).sum()} / {len(df)}")
            print(f"  - Sample values:")
            print(df[['datetime', 'station']].head())
        else:
            print("✗ No datetime column!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_indices_loader():
    print("=" * 60)
    print("TESTING INDICES LOADER")
    print("=" * 60)

    cfg = load_analysis_config(root)
    stations = ["9M"]
    years = [2021]

    try:
        df = load_indices(root, stations, years, cfg)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns[:10])}")
        if 'datetime' in df.columns:
            print(f"✓ Datetime column exists")
            print(f"  - Non-null: {(~df['datetime'].isna()).sum()} / {len(df)}")
            print(f"  - Sample values:")
            print(df[['datetime', 'station']].head())
        else:
            print("✗ No datetime column!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    print()


if __name__ == "__main__":
    test_detections_loader()
    test_environment_loader()
    test_spl_loader()
    test_indices_loader()

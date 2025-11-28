"""
Debug script to test each loader individually and inspect raw data
"""

import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.paths import (
    detections_excel_path,
    environmental_temp_path,
    environmental_depth_path,
    spl_excel_path,
)


def test_detections():
    """Test detection file loading and inspect structure."""
    print("=" * 60)
    print("TESTING DETECTIONS")
    print("=" * 60)

    station = "9M"
    year = 2018
    path = detections_excel_path(root, year, station)

    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")

    if path.exists():
        try:
            # Try reading with different sheet names
            xl = pd.ExcelFile(path)
            print(f"Available sheets: {xl.sheet_names}")

            # Read Data sheet
            df = pd.read_excel(path, sheet_name="Data")
            print(f"\nShape: {df.shape}")
            print(f"\nFirst 5 column names:\n{list(df.columns)[:5]}")
            print(f"\nAll columns containing 'Date' or 'Time':")
            date_time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            print(date_time_cols)

            print(f"\nFirst few rows of potential datetime columns:")
            for col in date_time_cols[:3]:
                print(f"\n{col}:")
                print(df[col].head())
                print(f"  dtype: {df[col].dtype}")

        except Exception as e:
            print(f"Error: {e}")
    print()


def test_environment():
    """Test environment file loading and inspect structure."""
    print("=" * 60)
    print("TESTING ENVIRONMENT")
    print("=" * 60)

    station = "9M"
    year = 2018

    for data_type, get_path in [("Temperature", environmental_temp_path), ("Depth", environmental_depth_path)]:
        print(f"\n{data_type}:")
        print("-" * 40)
        path = get_path(root, year, station)
        print(f"Path: {path}")
        print(f"Exists: {path.exists()}")

        if path.exists():
            try:
                xl = pd.ExcelFile(path)
                print(f"Available sheets: {xl.sheet_names}")

                df = pd.read_excel(path, sheet_name="Data")
                print(f"Shape: {df.shape}")
                print(f"\nFirst 5 column names:\n{list(df.columns)[:5]}")

                print(f"\nColumns containing 'Date' or 'Time':")
                date_time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
                print(date_time_cols)

                print(f"\nFirst few rows:")
                print(df.head(3))

            except Exception as e:
                print(f"Error: {e}")
    print()


def test_spl():
    """Test SPL file loading and inspect structure."""
    print("=" * 60)
    print("TESTING SPL")
    print("=" * 60)

    station = "9M"
    year = 2021
    path = spl_excel_path(root, year, station)

    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")

    if path.exists():
        try:
            xl = pd.ExcelFile(path)
            print(f"Available sheets: {xl.sheet_names}")

            df = pd.read_excel(path, sheet_name="Data")
            print(f"\nShape: {df.shape}")
            print(f"\nColumn names:\n{list(df.columns)}")

            if 'Date' in df.columns and 'Time' in df.columns:
                print(f"\nFirst few Date values:")
                print(df['Date'].head())
                print(f"  dtype: {df['Date'].dtype}")

                print(f"\nFirst few Time values:")
                print(df['Time'].head())
                print(f"  dtype: {df['Time'].dtype}")

        except Exception as e:
            print(f"Error: {e}")
    print()


def test_indices():
    """Test indices file loading and inspect structure."""
    print("=" * 60)
    print("TESTING INDICES")
    print("=" * 60)

    indices_dir = root / "data" / "raw" / "indices"
    print(f"Indices directory: {indices_dir}")
    print(f"Exists: {indices_dir.exists()}")

    if indices_dir.exists():
        # Find a sample file
        csv_files = list(indices_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files")

        if csv_files:
            sample_file = csv_files[0]
            print(f"\nSample file: {sample_file.name}")

            try:
                df = pd.read_csv(sample_file)
                print(f"Shape: {df.shape}")
                print(f"\nFirst 10 column names:\n{list(df.columns)[:10]}")

                print(f"\nColumns containing 'Date' or 'Time' or 'timestamp':")
                date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'timestamp'])]
                print(date_cols)

                if date_cols:
                    col = date_cols[0]
                    print(f"\nFirst few rows of '{col}':")
                    print(df[col].head())
                    print(f"  dtype: {df[col].dtype}")

            except Exception as e:
                print(f"Error: {e}")
    print()


if __name__ == "__main__":
    test_detections()
    test_environment()
    test_spl()
    test_indices()

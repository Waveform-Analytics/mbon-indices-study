"""
Common I/O functions for loading and saving data artifacts.

These functions standardize access to interim and processed data files
across pipeline stages.
"""

from pathlib import Path
import json

import pandas as pd


def load_interim_parquet(root: Path, name: str) -> pd.DataFrame:
    """
    Load a parquet file from data/interim/.

    Parameters
    ----------
    root : Path
        Project root directory.
    name : str
        File name without extension (e.g., "aligned_indices").

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = root / "data" / "interim" / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Interim file not found: {path}")

    df = pd.read_parquet(path)
    return df


def load_processed_parquet(root: Path, name: str) -> pd.DataFrame:
    """
    Load a parquet file from data/processed/.

    Parameters
    ----------
    root : Path
        Project root directory.
    name : str
        File name without extension (e.g., "community_metrics").

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = root / "data" / "processed" / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    df = pd.read_parquet(path)
    return df


def load_final_indices_list(root: Path) -> list[str]:
    """
    Load the list of final index column names from indices_final.csv.

    Parameters
    ----------
    root : Path
        Project root directory.

    Returns
    -------
    list[str]
        List of index column names.

    Raises
    ------
    FileNotFoundError
        If indices_final.csv does not exist.
    """
    path = root / "data" / "processed" / "indices_final.csv"

    if not path.exists():
        raise FileNotFoundError(f"Final indices list not found: {path}")

    df = pd.read_csv(path)
    return df["index_name"].tolist()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to parquet with consistent settings.

    Creates parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_summary_json(summary: dict, path: Path) -> None:
    """
    Save summary dictionary to JSON with consistent formatting.

    Creates parent directories if needed.

    Parameters
    ----------
    summary : dict
        Summary data to save.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

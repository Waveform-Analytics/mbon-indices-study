"""
Detections loader

Reads manual detections Excel files, applies metadata column mapping,
and produces a canonical DataFrame with `datetime` (UTC) and `station`.

Example
-------
>>> from pathlib import Path
>>> from mbon_indices.config import load_analysis_config
>>> root = Path('.')
>>> cfg = load_analysis_config(root)
>>> df = load_detections(root, ["9M"], [2018], cfg)
"""

from pathlib import Path

import pandas as pd
from mbon_indices.config import get_temporal_settings
from mbon_indices.paths import det_metadata_map_path, detections_excel_path


def _read_metadata_map(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_column_map(df: pd.DataFrame, mapping: pd.DataFrame | None) -> pd.DataFrame:
    if mapping is None:
        return df
    m = dict(zip(mapping["original"], mapping["canonical"], strict=False))
    return df.rename(columns=m)


def _parse_datetime(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    return df


def load_detections(root: Path, stations: list[str], years: list[int], analysis_cfg: dict) -> pd.DataFrame:
    temporal = get_temporal_settings(analysis_cfg)
    dfs = []
    meta = _read_metadata_map(det_metadata_map_path(root))
    for year in years:
        for station in stations:
            path = detections_excel_path(root, year, station)
            if not path.exists():
                continue
            df = pd.read_excel(path, engine="openpyxl")
            df = _apply_column_map(df, meta)
            df = _parse_datetime(df, temporal.get("timezone", "UTC"))
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)

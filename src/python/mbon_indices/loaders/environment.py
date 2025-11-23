"""
Environment loader

Reads temperature and depth Excel files per station/year and merges
on `datetime`. Units handling and imputation are applied in alignment
steps, not in this loader.

Example
-------
>>> from pathlib import Path
>>> df = load_environment(Path('.'), ["9M"], [2018])
"""

from pathlib import Path

import pandas as pd
from mbon_indices.paths import environmental_depth_path, environmental_temp_path


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("datetime", "timestamp"):
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], utc=True)
            break
    return df


def load_environment(root: Path, stations: list[str], years: list[int]) -> pd.DataFrame:
    dfs = []
    for year in years:
        for station in stations:
            tpath = environmental_temp_path(root, year, station)
            dpath = environmental_depth_path(root, year, station)
            tdf = pd.read_excel(tpath, engine="openpyxl") if tpath.exists() else pd.DataFrame(columns=["datetime", "temperature"])
            ddf = pd.read_excel(dpath, engine="openpyxl") if dpath.exists() else pd.DataFrame(columns=["datetime", "depth"])
            tdf = _parse_datetime(tdf)
            ddf = _parse_datetime(ddf)
            df = pd.merge(tdf, ddf, on="datetime", how="outer")
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station", "temperature", "depth"])  
    return pd.concat(dfs, ignore_index=True)
"""
SPL loader

Reads 1-hour RMS SPL Excel files and produces a canonical
DataFrame with `datetime` (UTC) and `station`. Aggregation to
2-hour bins is performed in the alignment stage.

Example
-------
>>> from pathlib import Path
>>> df = load_spl(Path('.'), ["9M"], [2018])
"""

from pathlib import Path

import pandas as pd
from mbon_indices.paths import spl_excel_path


def load_spl(root: Path, stations: list[str], years: list[int]) -> pd.DataFrame:
    dfs = []
    for year in years:
        for station in stations:
            path = spl_excel_path(root, year, station)
            if not path.exists():
                continue
            df = pd.read_excel(path, engine="openpyxl")
            for c in ("datetime", "timestamp"):
                if c in df.columns:
                    df["datetime"] = pd.to_datetime(df[c], utc=True)
                    break
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)
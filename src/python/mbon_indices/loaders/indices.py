"""
Indices loader

Loads acoustic indices CSVs from the directory specified by
`predictors.indices_source` and filters by bands in config.

Example
-------
>>> from pathlib import Path
>>> from mbon_indices.config import load_analysis_config
>>> root = Path('.')
>>> cfg = load_analysis_config(root)
>>> df = load_indices(root, ["9M"], [2018], cfg)
"""

from pathlib import Path

import pandas as pd
from mbon_indices.config import get_indices_band_policy, get_indices_source


def load_indices(root: Path, stations: list[str], years: list[int], analysis_cfg: dict) -> pd.DataFrame:
    src = get_indices_source(analysis_cfg, root)
    band_cfg = get_indices_band_policy(analysis_cfg)
    dfs = []
    for year in years:
        for station in stations:
            pattern = src / station / str(year)
            if not pattern.exists():
                continue
            for p in pattern.glob("*.csv"):
                df = pd.read_csv(p)
                if "band" in df.columns and band_cfg["include_bands"]:
                    df = df[df["band"].isin(band_cfg["include_bands"])]
                for c in ("datetime", "timestamp"):
                    if c in df.columns:
                        df["datetime"] = pd.to_datetime(df[c], utc=True)
                        break
                df["station"] = station
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)
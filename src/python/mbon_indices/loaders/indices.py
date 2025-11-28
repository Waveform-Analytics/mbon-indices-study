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
from mbon_indices.config import (
    get_indices_band_policy,
    get_indices_source,
    get_source_settings,
    get_temporal_settings,
)
from mbon_indices.utils.datetime import parse_datetime as parse_dt


def load_indices(root: Path, stations: list[str], years: list[int], analysis_cfg: dict) -> pd.DataFrame:
    src_dir = get_indices_source(analysis_cfg, root)
    band_cfg = get_indices_band_policy(analysis_cfg)
    src_settings = get_source_settings(analysis_cfg, "indices")
    temporal = get_temporal_settings(analysis_cfg)
    tz = temporal.get("timezone")
    if not tz:
        raise ValueError("analysis.yml:temporal.timezone must be defined")
    dfs = []
    for year in years:
        for station in stations:
            # Files are not organized in sub-folders; station and year are in filenames.
            candidates = list(src_dir.glob(f"*{station}*{year}*.csv"))
            if not candidates:
                continue
            for p in candidates:
                name = p.name.lower()
                inc_bands = [b.lower() for b in band_cfg.get("include_bands", [])]
                if inc_bands and not any(b in name for b in inc_bands):
                    continue
                df = pd.read_csv(p)
                df = parse_dt(df, src_settings, tz, source="indices")
                df["station"] = station
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)
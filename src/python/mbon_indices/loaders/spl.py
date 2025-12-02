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
from mbon_indices.config import get_source_settings, get_temporal_settings
from mbon_indices.paths import spl_excel_path
from mbon_indices.utils.datetime import parse_datetime as parse_dt


def load_spl(root: Path, stations: list[str], years: list[int], analysis_cfg: dict | None = None) -> pd.DataFrame:
    settings = get_source_settings(analysis_cfg or {}, "spl") if analysis_cfg else {"sheet_name": None, "timezone": "UTC"}
    temporal = get_temporal_settings(analysis_cfg or {}) if analysis_cfg else {"timezone": "UTC"}
    tz = temporal.get("timezone")
    if not tz:
        raise ValueError("analysis.yml:temporal.timezone must be defined")
    dfs = []
    for year in years:
        for station in stations:
            path = spl_excel_path(root, year, station)
            if not path.exists():
                continue
            sheet = settings.get("sheet_name")
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df = parse_dt(df, settings, tz, source="spl")
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)
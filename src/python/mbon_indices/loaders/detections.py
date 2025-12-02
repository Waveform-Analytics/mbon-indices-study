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
from mbon_indices.config import get_source_settings, get_temporal_settings
from mbon_indices.paths import det_metadata_map_path, detections_excel_path
from mbon_indices.utils.datetime import parse_datetime as parse_dt


def _read_metadata_map(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_column_map(df: pd.DataFrame, mapping: pd.DataFrame | None) -> pd.DataFrame:
    if mapping is None:
        return df
    cols = {c.lower(): c for c in mapping.columns}
    orig = cols.get("original") or cols.get("original_name") or cols.get("source")
    canon = cols.get("canonical") or cols.get("canonical_name") or cols.get("target")
    if not orig or not canon:
        return df
    m = dict(zip(mapping[orig], mapping[canon], strict=False))
    return df.rename(columns=m)


# datetime parsing is unified via mbon_indices.utils.datetime.parse_datetime


def load_detections(root: Path, stations: list[str], years: list[int], analysis_cfg: dict) -> pd.DataFrame:
    src = get_source_settings(analysis_cfg, "detections")
    temporal = get_temporal_settings(analysis_cfg)
    dfs = []
    meta = _read_metadata_map(det_metadata_map_path(root))
    for year in years:
        for station in stations:
            path = detections_excel_path(root, year, station)
            if not path.exists():
                continue
            sheet = src.get("sheet_name")
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df = _apply_column_map(df, meta)
            tz = temporal.get("timezone")
            if not tz:
                raise ValueError("analysis.yml:temporal.timezone must be defined")
            df = parse_dt(df, src, tz, source="detections")
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station"])  
    return pd.concat(dfs, ignore_index=True)

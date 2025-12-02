"""
Environment loader

Reads temperature and depth Excel files per station/year using
configuration for datetime parsing and sheet selection, and
metadata mapping for raw→canonical column names. Produces a
canonical DataFrame with `datetime` (UTC), `station`, `temperature`,
and `depth`.

Example
-------
>>> from pathlib import Path
>>> df = load_environment(Path('.'), ["9M"], [2018])
"""

from pathlib import Path

import pandas as pd
import yaml
from mbon_indices.config import get_source_settings, get_temporal_settings
from mbon_indices.paths import environmental_depth_path, environmental_temp_path
from mbon_indices.utils.datetime import parse_datetime as parse_dt


def _read_env_metadata(root: Path) -> dict:
    meta_path = root / "data" / "raw" / "metadata" / "env_column_names.yml"
    if not meta_path.exists():
        return {}
    with meta_path.open("r") as f:
        return yaml.safe_load(f) or {}


# use unified parse_dt directly in loader


def load_environment(root: Path, stations: list[str], years: list[int], analysis_cfg: dict | None = None) -> pd.DataFrame:
    settings = get_source_settings(analysis_cfg or {}, "environment") if analysis_cfg else {"sheet_name": None, "datetime_column": "datetime", "timezone": "UTC"}
    temporal = get_temporal_settings(analysis_cfg or {}) if analysis_cfg else {"timezone": "UTC"}
    meta = _read_env_metadata(root)
    t_raw = meta.get("temperature_column")
    d_raw = meta.get("depth_column")
    dfs = []
    for year in years:
        for station in stations:
            tpath = environmental_temp_path(root, year, station)
            dpath = environmental_depth_path(root, year, station)
            sheet = settings.get("sheet_name")
            tdf = pd.read_excel(tpath, sheet_name=sheet, engine="openpyxl") if tpath.exists() else pd.DataFrame()
            ddf = pd.read_excel(dpath, sheet_name=sheet, engine="openpyxl") if dpath.exists() else pd.DataFrame()
            tz = temporal.get("timezone")
            if not tz:
                raise ValueError("analysis.yml:temporal.timezone must be defined")
            if not tdf.empty:
                tdf = parse_dt(tdf, settings, tz, source="environment")
            if not ddf.empty:
                ddf = parse_dt(ddf, settings, tz, source="environment")
            if t_raw and t_raw in tdf.columns:
                tdf = tdf.rename(columns={t_raw: "temperature"})
            if d_raw and d_raw in ddf.columns:
                ddf = ddf.rename(columns={d_raw: "depth"})
            if "temperature" in tdf.columns:
                tdf["temperature"] = pd.to_numeric(tdf["temperature"], errors="coerce")
            if "depth" in ddf.columns:
                ddf["depth"] = pd.to_numeric(ddf["depth"], errors="coerce")
            if not tdf.empty:
                tdf = tdf[[c for c in ["datetime", "temperature"] if c in tdf.columns]].copy()
            if not ddf.empty:
                ddf = ddf[[c for c in ["datetime", "depth"] if c in ddf.columns]].copy()
            if not tdf.empty and not ddf.empty:
                df = pd.merge(tdf, ddf, on="datetime", how="outer")
            elif not tdf.empty:
                df = tdf.copy()
            elif not ddf.empty:
                df = ddf.copy()
            else:
                df = pd.DataFrame(columns=["datetime", "temperature", "depth"])  
            df["station"] = station
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime", "station", "temperature", "depth"])  
    return pd.concat(dfs, ignore_index=True)
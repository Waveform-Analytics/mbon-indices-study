from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_analysis_config(root: Path) -> dict[str, Any]:
    """
    Read and return the analysis YAML configuration.

    Example
    -------
    >>> from pathlib import Path
    >>> cfg = load_analysis_config(Path('.'))
    """
    return read_yaml(root / "config" / "analysis.yml")


def load_stations_config(root: Path) -> dict[str, Any]:
    """
    Read and return the stations YAML configuration.
    """
    return read_yaml(root / "config" / "stations.yml")


def get_stations_include(analysis_cfg: dict[str, Any]) -> list:
    """
    Return list of station IDs to include from config.
    """
    return list(analysis_cfg.get("stations", {}).get("include", []))


def get_temporal_settings(analysis_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Return temporal settings used by alignment and parsing.
    """
    return {
        "resolution_hours": analysis_cfg.get("temporal", {}).get("resolution_hours", 2),
        "timezone": analysis_cfg.get("temporal", {}).get("timezone", "UTC"),
    }


def get_alignment_policy(analysis_cfg: dict[str, Any], stations_cfg: dict[str, Any]) -> dict[str, Any]:
    env_gap = (
        analysis_cfg.get("alignment", {}).get("env_max_gap_hours")
        or stations_cfg.get("alignment", {}).get("env_max_gap_hours")
    )
    return {
        "env_max_gap_hours": env_gap,
        "spl_aggregation": analysis_cfg.get("alignment", {}).get("spl_aggregation", "mean"),
        "resolution_hours": analysis_cfg.get("temporal", {}).get("resolution_hours", 2),
    }


def get_indices_source(analysis_cfg: dict[str, Any], root: Path) -> Path:
    """
    Resolve the indices source directory relative to project root.
    """
    src = analysis_cfg.get("predictors", {}).get("indices_source")
    base = Path(src) if src else Path("data/raw/indices/")
    return (root / base).resolve()


def get_indices_band_policy(analysis_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Return indices band selection configuration.
    """
    p = analysis_cfg.get("predictors", {})
    return {
        "include_bands": list(p.get("include_bands", [])),
        "band_policy": p.get("band_policy", "separate"),
        "analysis_band": p.get("analysis_band"),
    }


def get_source_settings(analysis_cfg: dict[str, Any], source: str) -> dict[str, Any]:
    """
    Return normalized loader settings for a given source type.

    Keys returned:
    - required_columns: list[str]
    - datetime_column: str | None
    - compose_datetime: dict | None with keys {date_key, time_key}
    - sheet_name: str | None (default to first sheet when None)
    - timezone: str (default "UTC")
    """
    s = analysis_cfg.get("sources", {}).get(source, {})
    return {
        "required_columns": list(s.get("required_columns", [])),
        "datetime_column": s.get("datetime_column"),
        "compose_datetime": s.get("compose_datetime"),
        "sheet_name": s.get("sheet_name"),
        "timezone": s.get("timezone", analysis_cfg.get("temporal", {}).get("timezone", "UTC")),
    }


def validate_source_settings(settings: dict[str, Any], source: str) -> dict[str, Any]:
    """
    Validate loader settings and return the same dict.

    Rules:
    - Either datetime_column or compose_datetime must be provided to avoid candidate searches.
    - compose_datetime must contain both date_key and time_key when provided.
    - sheet_name may be None (loader defaults to first sheet).
    - timezone defaults to "UTC" when unspecified.
    Raises ValueError with clear messages including config path.
    """
    dt_col = settings.get("datetime_column")
    comp = settings.get("compose_datetime")
    if not dt_col and not comp:
        raise ValueError(f"analysis.yml:sources.{source}: must provide datetime_column or compose_datetime")
    if comp is not None:
        if not isinstance(comp, dict) or "date_key" not in comp or "time_key" not in comp:
            raise ValueError(f"analysis.yml:sources.{source}.compose_datetime: must specify both date_key and time_key")
    req = settings.get("required_columns", [])
    if not isinstance(req, list):
        raise ValueError(f"analysis.yml:sources.{source}.required_columns: must be a list")
    return settings
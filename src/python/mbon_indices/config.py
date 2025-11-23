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
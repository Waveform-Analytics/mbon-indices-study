"""
Stage 00 — Alignment

Orchestrates loading detections/environment/indices/SPL and performs
UTC normalization, 2-hour binning, station filtering, aggregation,
and forward-fill per configuration. Writes aligned parquet artifacts
to `data/interim/` and prints a JSON summary.
"""

import sys
import json
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import (
    load_analysis_config,
    load_stations_config,
    get_stations_include,
    get_temporal_settings,
    get_alignment_policy,
)
from mbon_indices.loaders.detections import load_detections
from mbon_indices.loaders.environment import load_environment
from mbon_indices.loaders.indices import load_indices
from mbon_indices.loaders.spl import load_spl
from mbon_indices.align.temporal import (
    floor_to_resolution,
    add_date_hour,
    aggregate_numeric_mean,
    deduplicate_keep_last,
    forward_fill_with_limit,
)
from mbon_indices.align.stations import keep_allowed_stations


def ensure_dir(p: Path) -> None:
    """Create directory `p` (and parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path, order: list[str] | None = None) -> None:
    """Write DataFrame to parquet with optional column ordering and sort.

    When `order` is provided, restrict output to those columns only.
    """
    if order:
        cols = [c for c in order if c in df.columns]
        df = df[cols].copy()
    df.sort_values(["station", "datetime"], inplace=True)
    df.to_parquet(path)


def main() -> None:
    """Run Stage 00 alignment and emit aligned artifacts and a summary."""
    analysis = load_analysis_config(root)
    stations_cfg = load_stations_config(root)
    stations = get_stations_include(analysis)
    temporal = get_temporal_settings(analysis)
    policy = get_alignment_policy(analysis, stations_cfg)
    years = [2018, 2021]

    out_dir = root / "data" / "interim"
    ensure_dir(out_dir)

    det = load_detections(root, stations, years, analysis)
    env = load_environment(root, stations, years, analysis)
    idx = load_indices(root, stations, years, analysis)
    spl = load_spl(root, stations, years, analysis)

    det = keep_allowed_stations(det, stations)
    env = keep_allowed_stations(env, stations)
    idx = keep_allowed_stations(idx, stations)
    spl = keep_allowed_stations(spl, stations)

    res_hours = temporal.get("resolution_hours")
    if res_hours is None:
        raise ValueError("analysis.yml:temporal.resolution_hours must be defined")
    try:
        res_hours = int(res_hours)
    except Exception as e:
        raise ValueError("analysis.yml:temporal.resolution_hours must be an integer") from e
    det = floor_to_resolution(det, res_hours)
    env = floor_to_resolution(env, res_hours)
    idx = floor_to_resolution(idx, res_hours)
    spl = floor_to_resolution(spl, res_hours)

    det = deduplicate_keep_last(det, ["station", "datetime"]) if not det.empty else det
    env = aggregate_numeric_mean(env, ["station", "datetime"]) if not env.empty else env
    idx = aggregate_numeric_mean(idx, ["station", "datetime"]) if not idx.empty else idx
    agg_cols = [c for c in spl.columns if c not in ["station", "datetime"]]
    if policy.get("spl_aggregation", "mean") == "mean" and agg_cols:
        spl = aggregate_numeric_mean(spl, ["station", "datetime"]) if not spl.empty else spl
    else:
        spl = deduplicate_keep_last(spl, ["station", "datetime"]) if not spl.empty else spl

    limit = 0
    if policy.get("env_max_gap_hours"):
        limit = int(policy["env_max_gap_hours"]) // int(res_hours)
    if not env.empty and limit > 0:
        env = forward_fill_with_limit(env, ["station"], [c for c in ["temperature", "depth"] if c in env.columns], limit)

    det_aligned = add_date_hour(det)
    env_aligned = add_date_hour(env)
    idx_aligned = add_date_hour(idx)
    spl_aligned = add_date_hour(spl)

    base = pd.merge(det_aligned, env_aligned, on=["station", "datetime"], how="inner")
    if not spl_aligned.empty:
        base = pd.merge(base, spl_aligned, on=["station", "datetime"], how="left")

    write_parquet(det_aligned, out_dir / "aligned_detections.parquet", order=["station", "datetime", "date", "hour"])
    write_parquet(env_aligned, out_dir / "aligned_environment.parquet", order=["station", "datetime", "date", "hour", "temperature", "depth"])
    write_parquet(idx_aligned, out_dir / "aligned_indices.parquet", order=["station", "datetime", "date", "hour"])
    write_parquet(spl_aligned, out_dir / "aligned_spl.parquet", order=["station", "datetime", "date", "hour"])
    write_parquet(base, out_dir / "aligned_base.parquet", order=["station", "datetime", "date", "hour"])

    summary = {
        "rows": {
            "detections": int(len(det_aligned)),
            "environment": int(len(env_aligned)),
            "indices": int(len(idx_aligned)),
            "spl": int(len(spl_aligned)),
            "base": int(len(base)),
        },
        "stations": stations,
        "resolution_hours": res_hours,
        "env_max_gap_hours": policy.get("env_max_gap_hours"),
        "spl_aggregation": policy.get("spl_aggregation"),
        "outputs": {
            "aligned_base": str(out_dir / "aligned_base.parquet"),
            "aligned_detections": str(out_dir / "aligned_detections.parquet"),
            "aligned_environment": str(out_dir / "aligned_environment.parquet"),
            "aligned_indices": str(out_dir / "aligned_indices.parquet"),
            "aligned_spl": str(out_dir / "aligned_spl.parquet"),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Stage 00 — Verify Loaders

Reads config and prints resolved stations, temporal/alignment policy,
and example input paths using pretty-printed JSON. Does not perform
IO on raw files; intended for quick setup checks.
"""

import sys
import json
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import (
    load_analysis_config,
    load_stations_config,
    get_stations_include,
    get_temporal_settings,
    get_alignment_policy,
)
from mbon_indices.paths import (
    detections_excel_path,
    environmental_temp_path,
    environmental_depth_path,
    spl_excel_path,
    det_metadata_map_path,
)

def main():
    """Print configuration values and example input paths as JSON."""
    analysis = load_analysis_config(root)
    stations_cfg = load_stations_config(root)
    stations = get_stations_include(analysis)
    temporal = get_temporal_settings(analysis)
    policy = get_alignment_policy(analysis, stations_cfg)
    years = [2018, 2021]

    print(json.dumps({
        "stations": stations,
        "temporal": temporal,
        "alignment_policy": policy,
    }, indent=2))

    if stations:
        s = stations[0]
        y = years[0]
        print(json.dumps({
            "detections": str(detections_excel_path(root, y, s)),
            "env_temp": str(environmental_temp_path(root, y, s)),
            "env_depth": str(environmental_depth_path(root, y, s)),
            "spl": str(spl_excel_path(root, y, s)),
            "det_column_map": str(det_metadata_map_path(root)),
        }, indent=2))

if __name__ == "__main__":
    main()

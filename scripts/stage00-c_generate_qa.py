"""
Stage 00 — Generate QA artifacts

Reads aligned artifacts from `data/interim/` and produces:
- results/tables/alignment_schema.csv
- results/figures/alignment_completeness.png
- results/logs/alignment_summary.json
"""

import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent
sys.path.append(str(root / "src" / "python"))

from mbon_indices.config import load_analysis_config
from mbon_indices.utils.run_history import append_to_run_history
from mbon_indices.qa.alignment import compute_schema, completeness_union, plot_completeness, write_summary_json


def main() -> None:
    analysis = load_analysis_config(root)
    out_tables = root / "results" / "tables"
    out_figs = root / "results" / "figures"
    out_logs = root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    interim = root / "data" / "interim"
    artifacts = {
        "aligned_base": interim / "aligned_base.parquet",
        "aligned_detections": interim / "aligned_detections.parquet",
        "aligned_environment": interim / "aligned_environment.parquet",
        "aligned_indices": interim / "aligned_indices.parquet",
        "aligned_spl": interim / "aligned_spl.parquet",
    }

    schema_df = compute_schema(artifacts)
    schema_df.to_csv(out_tables / "alignment_schema.csv", index=False)

    det = pd.read_parquet(artifacts["aligned_detections"]) if artifacts["aligned_detections"].exists() else pd.DataFrame()
    env = pd.read_parquet(artifacts["aligned_environment"]) if artifacts["aligned_environment"].exists() else pd.DataFrame()
    comp_df = completeness_union(det, env)

    fig_size = analysis.get("exploratory", {}).get("figure_size", [1600, 900])
    dpi = analysis.get("exploratory", {}).get("dpi", 300)
    plot_completeness(comp_df, out_figs / "alignment_completeness.png", fig_size_px=(fig_size[0], fig_size[1]), dpi=dpi)

    rows = {k: int(pd.read_parquet(v).shape[0]) if v.exists() else 0 for k, v in artifacts.items()}
    missing = {}
    if artifacts["aligned_environment"].exists():
        env_df = pd.read_parquet(artifacts["aligned_environment"])  
        for c in ["temperature", "depth"]:
            if c in env_df.columns:
                missing[c] = float(env_df[c].isna().mean())
    extra = {
        "resolution_hours": analysis.get("temporal", {}).get("resolution_hours"),
        "env_max_gap_hours": analysis.get("alignment", {}).get("env_max_gap_hours"),
        "spl_aggregation": analysis.get("alignment", {}).get("spl_aggregation"),
        "unit_conversions": "none_applied",
    }
    write_summary_json(out_logs / "alignment_summary.json", rows, missing, comp_df, extra)

    # Append to run history (no logger in this script, so no log_path)
    append_to_run_history(
        root=root,
        stage="Stage 00-c: QA Artifacts",
        config={
            "artifacts_checked": len(artifacts)
        },
        results={
            "schema_columns": len(schema_df),
            "missing_temp_frac": f"{missing.get('temperature', 0):.2%}",
            "missing_depth_frac": f"{missing.get('depth', 0):.2%}"
        }
    )


if __name__ == "__main__":
    main()
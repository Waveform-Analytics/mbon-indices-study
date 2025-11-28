"""
QA utilities for Stage 00 alignment.

Provides helpers to compute schema summaries, completeness metrics,
and write summary JSON. Rendering uses matplotlib with dimensions
derived from configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def df_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "non_null_fraction"])
    s = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes.values],
        "non_null_fraction": [float(1.0 - df[c].isna().mean()) for c in df.columns],
    })
    return s


def compute_schema(artifacts: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, p in artifacts.items():
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        sch = df_schema(df)
        sch.insert(0, "artifact", name)
        rows.append(sch)
    if not rows:
        return pd.DataFrame(columns=["artifact", "column", "dtype", "non_null_fraction"])
    return pd.concat(rows, ignore_index=True)


def completeness_union(det: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    det = det[["station", "datetime"]].dropna()
    env = env[["station", "datetime"]].dropna()
    det["datetime"] = pd.to_datetime(det["datetime"], utc=True)
    env["datetime"] = pd.to_datetime(env["datetime"], utc=True)
    det["date"] = pd.to_datetime(det["datetime"]).dt.date
    env["date"] = pd.to_datetime(env["datetime"]).dt.date
    det["year"] = pd.to_datetime(det["datetime"]).dt.year
    env["year"] = pd.to_datetime(env["datetime"]).dt.year

    # union expected bins per station-year
    union = pd.concat([det, env], ignore_index=True).drop_duplicates(["station", "datetime"])  
    exp = union.groupby(["station", "year"], as_index=False).size().rename(columns={"size": "expected"})

    # intersection of det and env per datetime
    inter = pd.merge(det, env, on=["station", "datetime"], how="inner")
    inter["year"] = pd.to_datetime(inter["datetime"]).dt.year
    got = inter.groupby(["station", "year"], as_index=False).size().rename(columns={"size": "present"})

    comp = pd.merge(exp, got, on=["station", "year"], how="left").fillna({"present": 0})
    comp["completeness"] = comp.apply(lambda r: float(r["present"]) / float(r["expected"]) if r["expected"] else 0.0, axis=1)
    return comp.sort_values(["station", "year"])  


def plot_completeness(comp: pd.DataFrame, out_png: Path, fig_size_px=(1600, 900), dpi=300) -> None:
    if comp.empty:
        fig_w = fig_size_px[0] / dpi
        fig_h = fig_size_px[1] / dpi
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        plt.text(0.5, 0.5, "No aligned data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png)
        plt.close()
        return
    stations = sorted(comp["station"].unique())
    fig_w = fig_size_px[0] / dpi
    fig_h = fig_size_px[1] / dpi
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    for _i, st in enumerate(stations):
        sub = comp[comp["station"] == st]
        xs = [str(y) for y in sub["year"]]
        ys = sub["completeness"].values
        plt.bar([f"{st}-{x}" for x in xs], ys, label=st)
    plt.ylim(0, 1)
    plt.ylabel("Completeness (fraction)")
    plt.xlabel("Station-Year")
    plt.title("Alignment Completeness by Station-Year")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def write_summary_json(out_path: Path, rows: dict[str, int], missing: dict[str, float], completeness: pd.DataFrame, extra: dict[str, object] | None = None) -> None:
    comp_rows = completeness.to_dict(orient="records") if not completeness.empty else []
    payload = {
        "rows": rows,
        "missingness": missing,
        "completeness": comp_rows,
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
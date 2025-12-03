"""
Temporal alignment utilities.

Functions here provide UTC binning to fixed-hour resolution,
safe aggregation of numeric columns, deduplication behavior,
and forward-fill with limits for environmental covariates.
"""

from __future__ import annotations

import pandas as pd


def floor_to_resolution(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Floor `datetime` to the specified hour resolution (UTC).

    Parameters
    - df: DataFrame with a `datetime` column
    - hours: integer hour size (e.g., 2)

    Returns a copy with `datetime` floored to bins.
    """
    if "datetime" not in df.columns:
        return df
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime"] = df["datetime"].dt.floor(f"{hours}h")
    return df


def add_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived `datetime_local`, `date` and `hour` columns based on `datetime`.

    `datetime` is expected to be in UTC.
    `datetime_local` is converted to America/New_York timezone for biological interpretation.
    Missing `datetime` values are set to NaT before derivation.
    """
    out = df.copy()
    if "datetime" not in out.columns:
        out["datetime"] = pd.NaT

    # Convert UTC to local time (America/New_York)
    out["datetime_local"] = out["datetime"].dt.tz_convert("America/New_York")

    out["date"] = out["datetime"].dt.date.astype("string")
    out["hour"] = out["datetime"].dt.hour
    return out


def aggregate_numeric_mean(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate numeric columns by mean over `group_cols`.

    Non-numeric columns not in `group_cols` are taken as last observed.
    """
    if df.empty:
        return df
    if any(c not in df.columns for c in group_cols):
        return df
    non_numeric = df.select_dtypes(exclude=["number"]).columns
    numeric = [c for c in df.columns if c not in non_numeric]
    if not numeric:
        return df.drop_duplicates(group_cols, keep="last")
    agg = df.groupby(group_cols, as_index=False)[numeric].mean()
    for c in non_numeric:
        if c in group_cols:
            continue
        if c in df.columns:
            agg[c] = df.groupby(group_cols)[c].last().values
    return agg


def deduplicate_keep_last(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Sort by `by` columns and keep last occurrence for duplicates.
    """
    if df.empty:
        return df
    if any(c not in df.columns for c in by):
        return df
    return df.sort_values(by).drop_duplicates(subset=by, keep="last")


def forward_fill_with_limit(
    df: pd.DataFrame, group_cols: list[str], cols: list[str], limit: int
) -> pd.DataFrame:
    """
    Forward-fill specified columns per group with a maximum step limit.

    Parameters
    - group_cols: columns to group by (e.g., station)
    - cols: columns to forward-fill (e.g., temperature, depth)
    - limit: maximum number of consecutive NaNs to fill
    """
    if df.empty:
        return df
    out = df.sort_values(group_cols + ["datetime"]).copy()
    for c in cols:
        if c in out.columns:
            out[c] = out.groupby(group_cols)[c].transform(lambda s: s.ffill(limit=limit))
    return out

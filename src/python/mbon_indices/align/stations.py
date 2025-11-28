"""
Station alignment utilities.

Provide simple helpers to harmonize station codes to a canonical
allow-list configured in `stations.include`.
"""

from __future__ import annotations

import pandas as pd


def keep_allowed_stations(df: pd.DataFrame, allowed: list[str]) -> pd.DataFrame:
    """
    Filter rows to stations present in `allowed`.
    Returns a copy; if `station` column is missing, returns input.
    """
    if "station" not in df.columns:
        return df
    return df[df["station"].isin(allowed)].copy()
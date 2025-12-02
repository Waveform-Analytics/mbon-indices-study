"""
Unified datetime parsing utility for loaders.

Parses timestamps using configuration-driven settings and applies
timezone localization and conversion based on config values.
Raises clear exceptions when configuration or required columns are
missing, avoiding silent legacy fallbacks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def parse_datetime(df: pd.DataFrame, settings: dict[str, Any], target_timezone: str, source: str) -> pd.DataFrame:
    """
    Parse `datetime` column using configuration settings.

    Parameters
    - df: input DataFrame containing raw timestamp columns
    - settings: loader settings dict (from analysis.yml:sources.<source>) with keys:
        - datetime_column: exact column name to parse (preferred)
        - compose_datetime: {date_key, time_key} when timestamps are split
        - timezone: source timezone string
    - target_timezone: desired output timezone (e.g., analysis.temporal.timezone)
    - source: source name for error messages (e.g., "detections", "environment")

    Behavior
    - Uses `datetime_column` directly when present
    - Else uses `compose_datetime` strictly from specified keys
    - Raises ValueError when configuration is insufficient or columns missing
    - Localizes to `settings.timezone` if naive, then converts to `target_timezone`
    """
    if settings is None:
        raise ValueError(f"analysis.yml:sources.{source}: settings must be provided")

    dt_col = settings.get("datetime_column")
    comp = settings.get("compose_datetime")
    src_tz = settings.get("timezone")

    # Resolve column names flexibly (handle trailing spaces/case differences)
    def _resolve_column(name: str) -> str:
        if name in df.columns:
            return name
        # try trimmed exact match
        trimmed = {c.strip(): c for c in df.columns}
        if name.strip() in trimmed:
            return trimmed[name.strip()]
        # try case-insensitive trimmed
        lowered = {c.strip().lower(): c for c in df.columns}
        key = name.strip().lower()
        if key in lowered:
            return lowered[key]
        raise ValueError(
            f"analysis.yml:sources.{source}: column '{name}' not found (checked exact, trimmed, case-insensitive)"
        )

    if dt_col:
        actual = _resolve_column(dt_col)
        dt = pd.to_datetime(df[actual], errors="coerce")
    elif isinstance(comp, dict) and comp.get("date_key") and comp.get("time_key"):
        dkey = _resolve_column(comp["date_key"])
        tkey = _resolve_column(comp["time_key"])

        # Handle numeric time values (e.g., Excel fraction of day)
        if pd.api.types.is_numeric_dtype(df[tkey]):
            base = pd.to_datetime(df[dkey], errors="coerce")
            delta = pd.to_timedelta(df[tkey], unit="D")
            dt = base + delta
        # Handle datetime objects from Excel (SPL case)
        elif pd.api.types.is_datetime64_any_dtype(df[dkey]) or hasattr(df[dkey].iloc[0], 'date'):
            # Extract date from Date column and time from Time column
            combined = []
            for date_val, time_val in zip(df[dkey], df[tkey]):
                if pd.notna(date_val) and pd.notna(time_val):
                    try:
                        # Get date part (handle both datetime and Timestamp)
                        if hasattr(date_val, 'date'):
                            date_part = date_val.date()
                        else:
                            date_part = pd.to_datetime(date_val).date()

                        # Get time part (handle both datetime and Timestamp)
                        if hasattr(time_val, 'time'):
                            time_part = time_val.time()
                        else:
                            time_part = pd.to_datetime(time_val).time()

                        # Combine date and time
                        combined_dt = pd.Timestamp.combine(date_part, time_part)
                        combined.append(combined_dt)
                    except Exception:
                        combined.append(pd.NaT)
                else:
                    combined.append(pd.NaT)
            dt = pd.Series(combined, index=df.index)
        else:
            # Fallback: convert to strings and parse
            dt = pd.to_datetime(df[dkey].astype(str) + " " + df[tkey].astype(str), errors="coerce")
    else:
        raise ValueError(
            f"analysis.yml:sources.{source}: must provide datetime_column or compose_datetime"
        )

    # Localize to source timezone if naive, then convert to target timezone
    if src_tz:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(src_tz, nonexistent="NaT", ambiguous="NaT")
        else:
            dt = dt.dt.tz_convert(src_tz)

    if target_timezone:
        dt = dt.dt.tz_convert(target_timezone)

    df = df.copy()
    df["datetime"] = dt
    return df
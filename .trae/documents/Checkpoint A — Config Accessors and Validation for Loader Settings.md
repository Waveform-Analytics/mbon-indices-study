# Objective
Add configuration-driven loader settings and validators (exact datetime column, timezone, Excel sheet selection) without modifying loader code yet. No state changes beyond config and accessors; pause for review after this checkpoint.

## Files to Create/Edit
- `config/analysis.yml`
  - Add `sources` section with per-type entries:
    - `required_columns`: list of columns that must exist
    - `datetime_column`: exact column name to parse into `datetime` (no candidate search when provided)
    - `compose_datetime`: optional `{ date_key: "...", time_key: "..." }` only if split date/time must be composed (used strictly when present)
    - `sheet_name`: Excel sheet name; **if omitted, loader defaults to the first sheet**
    - `timezone`: e.g., `UTC` for normalization
  - Example:
    ```yaml
    sources:
      detections:
        required_columns: ["Date"]
        datetime_column: "Date"
        compose_datetime: { date_key: "Date", time_key: "Time" }
        sheet_name: Data
        timezone: UTC
      environment:
        required_columns: ["Date"]
        datetime_column: "Date"
        compose_datetime: { date_key: "Date", time_key: "Time" }
        timezone: UTC
      spl:
        required_columns: ["Date"]
        datetime_column: "Date"
        compose_datetime: { date_key: "Date", time_key: "Time" }
        sheet_name: Data
        timezone: UTC
      indices:
        required_columns: []
        datetime_column: "timestamp"
        timezone: UTC
    ```

- `src/python/mbon_indices/config.py`
  - Add accessors and validators:
    - `get_source_settings(analysis_cfg: dict, source: str) -> dict`
      - Reads `sources.<source>`; returns normalized dict with defaults (`timezone=UTC`, `sheet_name=None`).
    - `validate_source_settings(settings: dict, source: str) -> dict`
      - Ensures either `datetime_column` or `compose_datetime` is defined.
      - Validates `compose_datetime` contains both `date_key` and `time_key`.
      - Returns clear, user-friendly error messages referencing the config path (e.g., `analysis.yml:sources.detections.datetime_column`).
  - Docstrings and type hints for both functions.

## Behavior Rules
- When `datetime_column` is present in config: **use it directly**; do not search candidates.
- When `compose_datetime` is present: strictly compose from specified keys; normalize to `timezone`.
- When `sheet_name` is missing: **default to the first sheet** for Excel.
- When neither `datetime_column` nor `compose_datetime` is present: mark as fallback; loaders (in checkpoint B+) will log a warning and use legacy heuristics to preserve continuity.

## Error Handling
- Missing required column: `Required column "Date" not found for detections (analysis.yml:sources.detections.required_columns)`.
- Invalid composition: `compose_datetime must specify both date_key and time_key (analysis.yml:sources.<source>.compose_datetime)`.
- Missing `sheet_name`: no error; default behavior applies.
- Missing `timezone`: default `UTC`.

## Checkpoint A Deliverables
- Updated `config/analysis.yml` with `sources` section (as above).
- New functions in `src/python/mbon_indices/config.py`: `get_source_settings`, `validate_source_settings` with docstrings and type hints.
- A short review summary of files changed: paths + 2–3 sentence rationale each.

## Pause for Review
- After implementing these changes, I will stop and show the diffs/paths for your review before wiring loaders (Checkpoint B).

## Future (Next Checkpoints, not included here)
- B: Wire detections/SPL/environment/indices loaders to use these settings; remove hardcoded sheet; eliminate candidate searches when config present.
- C: Unit tests for config parsing and validation; timezone handling tests.
- D: Spec and checklist updates to reference `sources.*` keys and default-to-first-sheet behavior.

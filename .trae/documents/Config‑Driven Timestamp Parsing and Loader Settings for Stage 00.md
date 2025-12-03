# Checkpoint A — Config Accessors and Validation for Loader Settings

## Goal
Introduce configuration-driven loader settings (exact datetime column, timezone, Excel sheet selection) and validators, without changing any loader code yet.

## Files to Create/Edit
- `config/analysis.yml`
  - Add `sources` section for each raw type (`detections`, `environment`, `spl`, `indices`).
  - Keys:
    - `required_columns`: list (sanity checks)
    - `datetime_column`: exact column name to parse (preferred)
    - `compose_datetime`: optional `{ date_key: "...", time_key: "..." }` when `datetime_column` is not suitable
    - `sheet_name`: Excel sheet name; if omitted, default to first sheet
    - `timezone`: timezone string (e.g., `UTC`) for normalization
  - Rationale: Centralize operational settings; remove hardcoded sheet usage; eliminate candidate searches when config is present.

- `src/python/mbon_indices/config.py`
  - Add:
    - `get_source_settings(analysis_cfg: dict, source: str) -> dict`
      - Reads `sources.<source>`; returns normalized dict with defaults.
    - `validate_source_settings(settings: dict, source: str) -> dict`
      - Validates presence and types; ensures either `datetime_column` or `compose_datetime` are defined; returns clear errors when invalid.
    - Behavior:
      - If `sheet_name` missing ⇒ loader will default to first sheet.
      - If `datetime_column` provided ⇒ use it (no candidate search).
      - If `compose_datetime` provided ⇒ compose strictly from `date_key` + `time_key`.
      - If both missing ⇒ fallback path flagged (legacy behavior allowed).
  - Add concise docstrings and type hints; keep messages friendly and reference the config path (e.g., `analysis.yml:sources.detections.datetime_column`).

## Validation & Error Messages
- Missing required column: `Required column "Date" not found for detections (analysis.yml:sources.detections.required_columns)`
- Invalid compose structure: `compose_datetime must specify both date_key and time_key (analysis.yml:sources.<source>.compose_datetime)`
- Missing sheet: none (default to first sheet);
- Missing timezone: default to `UTC`.

## Checkpoint Pause
After implementing the above:
- I will present the exact `analysis.yml` diff and the new functions in `config.py` (paths and summaries) for your review.
- No loader changes will be made until you approve Checkpoint A.

## Tests (stub to prepare in a later checkpoint)
- `tests/test_sources_config.py` to validate parsing and error messages (added in a subsequent checkpoint).

## Documentation Notes (to update in a later checkpoint)
- Stage spec and checklist will be updated to reference `sources.*` keys and default-to-first-sheet rule after you approve the config and accessors.

## Execution/Tooling
- No runtime changes at this checkpoint; next checkpoint will wire loaders to new accessors.

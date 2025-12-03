# Review of Environment Metadata
- Observed: `data/raw/metadata/env_column_names.yml` currently contains only `temperature_column: "Water temp (°C)"`.
- Recommendation:
  - Add `depth_column: "Water depth (m)"` so both environment streams are covered.
  - Option A (minimal): keep keys as raw header names only, with canonical names defined by contract (spec: `temperature`, `depth`).
  - Option B (explicit): structure as raw→canonical mapping, e.g.
    ```yaml
    temperature:
      raw: "Water temp (°C)"
      canonical: "temperature"
    depth:
      raw: "Water depth (m)"
      canonical: "depth"
    ```
  - I recommend Option A (minimal) to avoid duplication; canonical names live in the spec/code contract.

# Canonical Names Placement
- Canonical names are part of the stage spec contract (e.g., outputs must have `temperature`, `depth`, `datetime`, `station`).
- Metadata files list raw header names per source and map them to canonical via loader logic.
- Config remains operational (datetime, sheet, timezone, policies) and should not duplicate raw header values.

# Implementation Plan
## Files to Edit/Create
- `data/raw/metadata/env_column_names.yml`
  - Add `depth_column: "Water depth (m)"` (Option A minimal format).
- `src/python/mbon_indices/loaders/environment.py`
  - Read `env_column_names.yml`.
  - Use `sources.environment.sheet_name` (default first sheet) and `datetime_column` from config.
  - Map raw columns to canonical: `temperature` and `depth`.
  - Normalize `datetime` to UTC; merge temp/depth on `datetime`.
  - Emit canonical columns only.
- `src/python/mbon_indices/loaders/detections.py`
  - Already reads `det_column_names.csv` for raw→canonical.
  - Ensure loader uses config `sources.detections` for sheet and timezone.
  - Keep emitting canonical columns per mapping.
- `src/python/mbon_indices/loaders/indices.py`
  - If raw index headers need normalization beyond metadata categories, we will keep current behavior and rely on categories file; optional future mapping file if needed.

## Loader Behavior (Environment)
- Read temperature file and depth file for each station/year, selecting sheet `Data` (or first if unspecified) and `datetime_column: "Date and time"` from config.
- Rename raw headers to canonical:
  - `"Water temp (°C)"` → `temperature`
  - `"Water depth (m)"` → `depth`
- Merge on `station, datetime` (UTC). No candidate search; strictly use config + metadata.

## Breakpoints (as requested)
- `src/python/mbon_indices/loaders/environment.py`:
  - After reading each sheet: inspect `df.columns` and sample rows.
  - After renaming: verify `temperature`/`depth` present.
  - Before merge: confirm both have `datetime` parsed as UTC.
- `src/python/mbon_indices/loaders/detections.py`:
  - After applying `det_column_names.csv` mapping.
  - Before return: verify canonical columns present and `datetime` populated.

## Tests (added in subsequent checkpoint)
- `tests/test_env_metadata_mapping.py`:
  - YAML parse; presence of both `temperature_column` and `depth_column`.
  - Loader renames to canonical and merges correctly.
- `tests/test_detections_mapping.py`:
  - CSV mapping applied; canonical outputs produced.

## Spec Updates
- `specs/stages/00-data-prep-alignment-spec.md`:
  - Document canonical environment column names (`temperature`, `depth`) and the use of `env_column_names.yml` for raw header mapping.
  - Note config is operational (datetime, sheet, timezone); metadata handles raw headers.
- `specs/reviews/00-data-prep-checklist.md`:
  - Add pre-condition: `env_column_names.yml` present with both temperature and depth mappings.

## Deliverables for Checkpoint B
- Edits to environment loader to read metadata and emit canonical columns.
- Minor updates to detections loader to ensure config-driven sheet/timezone are applied cleanly.
- Present diffs and paths for your review before proceeding to tests/spec updates.

If you approve, I will implement the above (Option A minimal metadata format), add breakpoints where noted, run `ruff`, and pause with a summary for your review.
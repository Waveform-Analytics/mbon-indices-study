# ADR-02: Config-Driven Data Loaders with Unified Datetime Parsing

**Date:** 2025-11-28
**Status:** Accepted
**Deciders:** Implementation Team
**Related:** Stage 00 Data Prep & Alignment

## Context

Data loaders need to handle four different data sources (detections, environment, SPL, indices) with varying datetime formats:
- Excel files with datetime columns (detections, environment)
- Excel files with separate Date + Time columns (SPL)
- CSV files with string datetime columns (indices)
- Different sheet names, timezones, and column naming conventions

Previous approach (from old scripts) used hardcoded sheet names, heuristic column searches, and source-specific datetime logic scattered across loaders.

## Decision

Implement a **config-driven loader architecture** with:

1. **Config Section** (`config/analysis.yml:sources.*`)
   - Per-source settings: `datetime_column`, `compose_datetime`, `sheet_name`, `timezone`, `required_columns`
   - Single source of truth for operational loader settings
   - Eliminates hardcoded values in loader code

2. **Unified Datetime Parser** (`src/python/mbon_indices/utils/datetime.py`)
   - Handles both single-column datetime (`datetime_column`) and split date+time (`compose_datetime`)
   - Supports Excel datetime objects, numeric time fractions, and string parsing
   - Consistent timezone localization/conversion
   - Column name resolution with flexible matching (exact, trimmed, case-insensitive)

3. **Metadata Mapping Files**
   - `data/raw/metadata/det_column_names.csv`: Detection raw→canonical column names
   - `data/raw/metadata/env_column_names.yml`: Environment raw→canonical column names
   - Separates column mapping from operational config

## Rationale

**Benefits:**
- **Flexibility:** Easy to adapt to new data sources or format changes (edit config, not code)
- **Clarity:** Config explicitly documents datetime structure for each source
- **Maintainability:** Datetime logic centralized in one parser, not scattered
- **Error messages:** Config path references in errors (e.g., "analysis.yml:sources.spl.datetime_column")
- **Testability:** Config validation separate from loader logic

**Trade-offs:**
- Additional config complexity (but well-documented and validated)
- Learning curve for new contributors (mitigated by examples and docstrings)

## Implementation Details

### Config Example
```yaml
sources:
  detections:
    datetime_column: "Date"          # Single column (already datetime)
    sheet_name: Data
    timezone: UTC
  spl:
    compose_datetime:                 # Split Date + Time
      date_key: "Date"
      time_key: "Time"
    sheet_name: Data
    timezone: UTC
```

### Critical Discovery: SPL Date+Time Handling
Excel stores SPL Date column as datetime with microsecond offsets (e.g., `01:00:02.880`) due to recording precision. To get clean 2-hour intervals, **must** combine:
- Date part from Date column (actual recording date)
- Time part from Time column (clean hourly timestamps: 00:00, 01:00, etc.)

Implementation uses `pd.Timestamp.combine(date_part, time_part)` to ensure precise alignment.

### Config Accessors
- `get_source_settings(analysis_cfg, source)`: Returns normalized settings with defaults
- `validate_source_settings(settings, source)`: Validates structure and raises clear errors

## Consequences

**Positive:**
- Stage 00 loaders successfully handle all four data types with consistent datetime handling
- Config changes don't require code changes (e.g., switching sheet names)
- Datetime parsing bugs isolated to single parser function
- Future data sources can be added by extending config

**Negative:**
- Config must be kept in sync with actual file structure (mitigated by validation)
- DateTime parser complexity increased to handle all cases (mitigated by comprehensive docstrings)

**Neutral:**
- Config serves as documentation of data source structure
- Metadata files serve as documentation of column mappings

## Alternatives Considered

1. **Hardcoded sheet names and column searches** (old approach)
   - Rejected: Brittle, requires code changes for format variations

2. **Automatic format detection**
   - Rejected: Fragile heuristics, silent failures, hard to debug

3. **Per-loader config files**
   - Rejected: Config fragmentation, harder to maintain consistency

## Verification

- All loaders tested with actual 2018+2021 data
- 100% datetime parsing success for detections, environment (26,250+ rows)
- 99.6% datetime parsing success for SPL (39 NaT from missing Time values)
- 100% datetime parsing success for indices (17,231 rows)
- Alignment pipeline produces clean 2-hour intervals with no unexpected gaps

## References

- Stage 00 Spec: specs/stages/00-data-prep-alignment-spec.md
- Implementation: src/python/mbon_indices/loaders/*.py
- Datetime Parser: src/python/mbon_indices/utils/datetime.py
- Config: config/analysis.yml:sources
- Verification: specs/reviews/00-data-prep-VERIFICATION.md

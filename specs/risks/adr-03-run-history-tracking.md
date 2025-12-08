# ADR-03: Run History Tracking

## Context

- Pipeline stages produce outputs that depend on configuration parameters
- When parameters change (e.g., VIF threshold from 5 to 2), outputs change
- Need to track what configuration was used for each run and what results were produced
- Specs should document methodology and rationale, not specific run outcomes
- Detailed execution logs exist but are verbose and hard to scan

## Decision

Add a shared `RUN_HISTORY.md` file that all stage scripts append to after successful completion.

### Location
- `results/logs/RUN_HISTORY.md`

### Entry Format
Each entry contains:
- Timestamp and stage name
- Key configuration values used
- Key results (counts, paths, metrics)
- Optional notes field for manual annotations

### Implementation
- Python: Shared utility in `src/python/mbon_indices/utils/run_history.py`
- R: Inline function (only Stage 05 currently uses R)
- Each stage script calls the utility at the end of a successful run

### Stages to Update
| Script | Language | Status |
|--------|----------|--------|
| `stage00-a_verify_loaders.py` | Python | Pending |
| `stage00-b_align.py` | Python | Pending |
| `stage00-c_generate_qa.py` | Python | Pending |
| `stage01_index_reduction.py` | Python | Done |
| `stage02_community_metrics.py` | Python | Pending |
| `stage03_feature_engineering.py` | Python | Pending |
| `stage04_exploratory_viz.py` | Python | Pending |
| `stage05_modeling.R` | R | Done |

### Spec Updates
Add to Outputs section of each stage spec:
```
- Appends summary to `results/logs/RUN_HISTORY.md`
```

## Consequences

### Positive
- Clear audit trail of what was run and when
- Easy to scan history without reading verbose logs
- Separates methodology (specs) from outcomes (run history)
- Supports reproducibility by recording configuration

### Negative
- Another file to maintain
- Potential for file conflicts if multiple runs overlap (unlikely in practice)
- Need to update all stage scripts

### Trade-offs Accepted
- Manual notes field is optional; entries are useful without them
- R utility is inline rather than shared; acceptable since only one R script exists

## Alternatives

1. **Embed results in specs** — Rejected: mixes methodology with outcomes
2. **Rely on git history** — Rejected: commits may batch multiple changes; not scannable
3. **JSON log file** — Rejected: less human-readable; markdown is easier to review
4. **Database** — Rejected: overkill for this project scale

## References

- `specs/stages/01-index-reduction-spec.md` — First spec to reference RUN_HISTORY.md
- `specs/stages/05-modeling-spec.md` — Second spec to reference RUN_HISTORY.md
- `results/logs/RUN_HISTORY.md` — The tracking file itself
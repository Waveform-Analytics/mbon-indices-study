# MBON Acoustic Indices Study Constitution

## Core Principles

### I. Research-First Workflow
This is a data analysis project, not a software product. Priorities:
- Reproducibility over abstraction
- Interpretable outputs over elegant code
- Manual verification over automated tests
- Step-by-step collaboration with user review

### II. Script Naming Convention
Analysis scripts follow staged naming:
- `stage0X_purpose.R` - main pipeline stages
- `stage0Xb_purpose.R`, `stage0Xc_purpose.R` - sub-stages
- `generate_*.R`, `visualize_*.R` - output generation helpers

### III. Output Structure
All outputs go to `results/` with consistent subdirectories:
- `results/models/<metric>/` - fitted model objects (.rds)
- `results/tables/` - CSV outputs for analysis results
- `results/figures/` - PNG visualizations
- `results/logs/` - execution metadata (JSON)

### IV. Configuration-Driven
- All analysis parameters in `config/analysis.yml`
- Response variables, thresholds, model settings defined once
- Scripts read config, don't hardcode parameters

### V. Data Flow
```
data/raw/ → data/interim/ → data/processed/analysis_ready.parquet
                                    ↓
                            results/{models,tables,figures}/
                                    ↓
                            docs/results-viewer.qmd
```

## Quality Gates

- **Convergence check**: All models must converge before downstream analysis
- **Output verification**: Review generated tables/figures before proceeding
- **Documentation sync**: Update `rebuild-docs.sh` when adding new outputs

## Scope

- 9 response variables (3 presence, 6 count)
- 17 VIF-filtered acoustic indices
- 3 stations (9M, 14M, 37M)
- GAMMs with AR1 autocorrelation via mgcv::bam()

**Version**: 1.0 | **Created**: 2026-02-02

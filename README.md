# MBON Acoustic Indices Study

Can acoustic indices — automated summaries of underwater soundscape characteristics — predict biological activity in an estuary?

**Study site:** Three passive acoustic monitoring stations along the May River estuary, SC (9M, 14M, 37M), 2021 data at 2-hour resolution. 13,102 observations, 60 acoustic indices, 9 biological response variables (fish, dolphins, vessels).

## Key Findings

**Presence detection works.** Vessel presence: AUC 0.93 (excellent). Fish and dolphin presence: AUC ~0.77 (moderate, useful for screening).

**Activity counts don't generalize.** Count models (fish activity, dolphin clicks, etc.) fit training data but fail on held-out weeks (negative R²). The data is too zero-inflated and variable for week-to-week prediction.

**60 indices > 17 VIF-filtered indices.** Letting GAMM regularization (`select=TRUE`) handle feature selection outperformed manual VIF pre-filtering (avg ΔAIC = -262).

**Acoustic indices add real value beyond environment/time.** All 9 metrics show ΔAIC > 70 when adding indices to a baseline model with only temperature, depth, and time variables. For dolphins, indices dominate; for fish, indices and environment contribute roughly equally.

**No universal "best index."** Top predictors vary by metric and station. HFC dominates vessel detection; ACTspFract dominates dolphin presence; LFC matters most for fish presence. Station-level models show different indices matter at different locations.

Full results: [`docs/results-viewer.html`](docs/results-viewer.html) (run `./rebuild-docs.sh` to generate, or open the pre-built HTML).

## Project Organization

```
├── config/
│   └── analysis.yml              # All analysis parameters (responses, thresholds, GAMM settings)
├── data/
│   ├── raw/                      # Original data (detections, environment, indices by year/station)
│   ├── interim/                  # Aligned parquet files (stage 00 output)
│   └── processed/                # Analysis-ready dataset, community metrics, final index list
├── scripts/
│   ├── stage00-*.py              # Data loading, alignment, QA
│   ├── stage01_index_reduction.py
│   ├── stage02_community_metrics.py
│   ├── stage03_feature_engineering.py
│   ├── stage04_exploratory_viz.py
│   ├── stage05_modeling.R        # Main GAMM fitting (9 response variables)
│   ├── stage05b_validation.R     # Leave-one-week-out cross-validation
│   ├── stage05c_diagnostics.R    # Effect sizes, zero-inflation, concurvity
│   ├── stage05d_baseline.R       # Baseline comparison (indices vs no-indices)
│   ├── stage05e_*.R              # Acoustic-only models, per-station effects
│   └── stage05f_*.R              # Per-station effect sizes
├── results/
│   ├── figures/                  # All plots (per-metric subdirs + summary figures)
│   ├── tables/                   # CSV summaries (model fits, effect sizes, validation)
│   ├── models/                   # Saved GAMM .rds files
│   └── logs/                     # Processing logs with timestamps
├── specs/stages/                 # Detailed specs for each analysis stage
├── docs/
│   ├── results-viewer.qmd        # Quarto results document (source)
│   ├── results-viewer.html       # Rendered results (rebuild with ./rebuild-docs.sh)
│   └── presentations/            # Slide decks
├── src/python/mbon_indices/      # Python package (data loading/processing utilities)
└── rebuild-docs.sh               # Copies figures/tables to docs/ and renders Quarto
```

## Quickstart

### Prerequisites

- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- R with `mgcv`, `jsonlite`, `dplyr`, `ggplot2`, `pROC`
- [Quarto](https://quarto.org/) (for rendering docs)

### Setup

```bash
uv sync
```

### Run the pipeline

Stages 00–04 are Python, stages 05+ are R:

```bash
# Data prep (Python)
uv run scripts/stage00-a_verify_loaders.py
uv run scripts/stage00-b_align.py
uv run scripts/stage01_index_reduction.py
uv run scripts/stage02_community_metrics.py
uv run scripts/stage03_feature_engineering.py
uv run scripts/stage04_exploratory_viz.py

# Modeling (R) — these take a while
Rscript scripts/stage05_modeling.R
Rscript scripts/stage05b_validation.R
Rscript scripts/stage05c_diagnostics.R
Rscript scripts/stage05d_baseline.R
Rscript scripts/stage05e_acoustic_only.R
Rscript scripts/stage05e_bystation_effects.R
Rscript scripts/stage05f_bystation_effect_sizes.R

# Build results site
./rebuild-docs.sh
open docs/results-viewer.html
```

### Configuration

All analysis parameters live in [`config/analysis.yml`](config/analysis.yml) — response variable families, GAMM settings (`smooth_k`, `select`), scaling options, station list, etc.

## Models

**Generalized Additive Mixed Models (GAMMs)** via `mgcv::bam()` with:

- 60 acoustic indices as smooth predictors (`k=3`)
- Temperature and depth as smooth covariates
- Cyclic smooths for hour-of-day and day-of-year
- Station, month, and day as random effects
- AR1 autocorrelation (data-driven ρ)
- Shrinkage selection (`select=TRUE`) — penalizes unnecessary complexity, no manual pre-filtering needed

Binary responses (presence) use binomial family; count responses use negative binomial.

## License

Research project — not currently licensed for external use.

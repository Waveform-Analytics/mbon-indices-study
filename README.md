# MBON Acoustic Indices Study

## Overview

This project investigates whether **acoustic indices** — summary statistics computed from underwater sound recordings — can predict **biological community metrics** in estuarine environments. We analyze passive acoustic monitoring data from three sites along the May River, South Carolina to understand relationships between soundscape characteristics and the presence/activity of fish, dolphins, and vessels.

## Study Sites

Data were collected at three stations along the May River estuary:

| Station | Location | Description |
|---------|----------|-------------|
| **37M** | River mouth | Closest to open water |
| **14M** | Mid-river | Intermediate position |
| **9M** | Up-river | Furthest inland |

## Data Sources

- **Acoustic indices**: ~60 metrics computed from sound recordings (e.g., acoustic complexity, entropy, bioacoustic index), currently available for 2021
- **Manual detections**: Expert annotations of fish calls, dolphin vocalizations, and vessel noise for 2018 and 2021
- **Environmental data**: Water temperature and depth at each station

## Species and Metrics

### Fish (8 species)
- Silver perch
- Oyster toadfish (boat whistle and grunt calls)
- Black drum
- Spotted seatrout
- Red drum
- Atlantic croaker
- Weakfish

**Derived metrics**: activity (total call intensity), richness (species count per time bin), presence (binary)

### Bottlenose Dolphins
- Echolocation clicks
- Burst pulses
- Whistles

**Derived metrics**: counts by call type, total activity, presence (binary)

### Vessels
**Derived metrics**: presence (binary)

## Research Questions

1. Which acoustic indices best capture variation in biological activity?
2. Can we predict fish and dolphin presence/activity from soundscape metrics?
3. How do these relationships vary across stations and time of day?

## Analysis Approach

The analysis proceeds through a series of stages:

### Stage 0: Data Preparation
Align all data sources to a common 2-hour temporal resolution and standardize formats across stations and years.

### Stage 1: Index Reduction
Reduce ~60 acoustic indices to a smaller, non-redundant set using correlation analysis and variance inflation factor (VIF) screening. This avoids multicollinearity issues in downstream models.

**Current result**: 20 final indices representing spectral, temporal, and complexity dimensions of the soundscape.

### Stage 2: Community Metrics
Derive biological response variables from manual detections:
- **Fish**: activity, richness, presence
- **Dolphins**: counts by call type, total activity, presence
- **Vessels**: presence

### Stage 3: Feature Engineering
Create temporal features (time of day, day of year) and grouping variables needed for mixed-effects modeling.

### Stages 4–6: Statistical Modeling
Fit Generalized Linear Mixed Models (GLMM) and Generalized Additive Mixed Models (GAMM) to assess which indices predict each community metric, accounting for:
- Station-level variation (random effects)
- Temporal autocorrelation
- Non-linear relationships (GAMM smooth terms)

### Stages 7–10: Model Selection & Reporting
Compare models via cross-validation, select best predictors, and generate visualizations and reports.

## Current Progress

| Stage | Description | Status |
|-------|-------------|--------|
| 00 | Data Prep & Alignment | Complete |
| 01 | Index Reduction | Complete (20 indices) |
| 02 | Community Metrics | Complete (9 response variables) |
| 03 | Feature Engineering | Ready to implement |
| 04–10 | Modeling & Reporting | Planned |

## Project Organization

```
mbon-indices-study/
├── data/
│   ├── raw/                 # Original data files
│   ├── interim/             # Cleaned, aligned data
│   └── processed/           # Analysis-ready datasets
├── results/
│   ├── figures/             # Plots and visualizations
│   ├── tables/              # Summary tables
│   └── logs/                # Processing logs
├── specs/                   # Detailed documentation for each stage
│   └── stages/              # Stage-by-stage specifications
├── scripts/                 # Executable analysis scripts
├── src/python/              # Python code for data processing
├── src/r/                   # R code for statistical modeling
└── config/                  # Analysis parameters
```

### Where to Find More Information

- **Stage specifications** (`specs/stages/`): Detailed documentation of inputs, outputs, methods, and decisions for each analysis stage
- **Configuration** (`config/analysis.yml`): Parameter values and thresholds used in the analysis
- **Processing logs** (`results/logs/`): Timestamped records of each analysis run

## Key Outputs

- `data/processed/indices_final.csv` — Final set of acoustic indices for modeling
- `data/processed/community_metrics.parquet` — Biological response variables
- `data/processed/analysis_ready.parquet` — Combined dataset for modeling (after Stage 03)
- `results/figures/` — Correlation heatmaps, diagnostic plots, model visualizations

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and workflow details.

## Acknowledgments

*To be added*

## License

Research project — not currently licensed for external use.

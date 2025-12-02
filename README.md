# MBON Acoustic Indices Analysis

Multistage pipeline for analyzing acoustic indices from passive acoustic monitoring data in relation to environmental covariates and biological detections.

## Project Structure

```
mbon-indices-study/
├── data/                    # Data files (gitignored)
│   ├── raw/                 # Original data files
│   ├── interim/             # Aligned and cleaned data
│   └── processed/           # Final analysis-ready data
├── results/                 # Analysis outputs (gitignored)
│   ├── logs/                # Timestamped execution logs
│   ├── figures/             # Generated plots
│   ├── tables/              # Summary tables
│   └── models/              # Fitted model objects
├── specs/                   # Stage specifications and documentation
│   ├── stages/              # Per-stage specs
│   ├── templates/           # Spec templates
│   └── reviews/             # Stage checklists
├── scripts/                 # Executable stage scripts
├── src/
│   ├── python/
│   │   └── mbon_indices/    # Python package for data processing
│   └── r/                   # R code for modeling (GLMM, GAMM)
├── config/                  # Configuration files
└── .trae/                   # Project rules and documentation

```

## Installation

### Prerequisites
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Install Package in Editable Mode

```bash
# Install the mbon_indices package in editable mode
uv pip install -e .

# This allows:
# - Import from anywhere: `from mbon_indices.config import load_analysis_config`
# - IDEs/linters can resolve package references
# - Changes to source code are immediately available
```

### Development Tools

```bash
# Install development dependencies (ruff linter/formatter)
uv pip install -e ".[dev]"
```

## Usage

### Running Stage Scripts

Each stage has a corresponding script in `scripts/`:

```bash
# Stage 00: Data preparation and temporal alignment
python scripts/stage00_align.py

# Stage 01: Index reduction (correlation + VIF pruning)
python scripts/stage01_index_reduction.py
```

All scripts produce timestamped logs in `results/logs/`.

### Importing the Package

```python
from mbon_indices.config import load_analysis_config
from mbon_indices.utils.logging import setup_stage_logging

# Load configuration
cfg = load_analysis_config(project_root)

# Set up logging for a stage
logger = setup_stage_logging(project_root, "stage01_index_reduction")
```

## Workflow

This project follows a **spec-driven development** approach:

1. **Write specs** - Define stage inputs, outputs, methods, and acceptance criteria
2. **Review specs** - Use checklists in `specs/reviews/`
3. **Implement** - Write code only after spec approval
4. **Verify** - Check outputs against acceptance criteria
5. **Document** - Update spec Change Records

See `specs/_index.md` for stage status and `.trae/rules/project_rules.md` for workflow details.

## Pipeline Stages

- **00: Data Prep & Alignment** - ✅ Complete
- **01: Index Reduction** - 🚧 In Progress
- **02: Community Metrics** - Approved
- **03: Feature Engineering** - Approved
- **04: Exploratory Visualization** - Approved
- **05: GLMM Modeling** - Approved
- **06: GAMM Modeling** - Approved
- **07-10: Cross-Validation, Selection, Visualization, Reporting** - Draft

## Configuration

Analysis parameters are defined in `config/analysis.yml`:
- Correlation and VIF thresholds
- Station lists
- Temporal resolution
- Feature engineering rules

## License

Research project - not currently licensed for external use.

# Contributing

Development guide for the MBON Acoustic Indices Study.

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

## Running Stage Scripts

Each stage has a corresponding script in `scripts/`:

```bash
# Stage 00: Data preparation and temporal alignment
python scripts/stage00-b_align.py

# Stage 01: Index reduction (correlation + VIF pruning)
python scripts/stage01_index_reduction.py

# Stage 02: Community metrics
python scripts/stage02_community_metrics.py
```

All scripts produce timestamped logs in `results/logs/`.

## Using the Python Package

```python
from mbon_indices.config import load_analysis_config
from mbon_indices.utils.logging import setup_stage_logging

# Load configuration
cfg = load_analysis_config(project_root)

# Set up logging for a stage
logger = setup_stage_logging(project_root, "stage01_index_reduction")
```

## Development Workflow

This project uses a **spec-driven approach**: each analysis stage has a written specification (in `specs/stages/`) that defines inputs, outputs, methods, and acceptance criteria before implementation begins.

### Workflow Steps

1. **Write/review spec** — Define what the stage does and how success is measured
2. **Implement** — Write code to fulfill the spec
3. **Verify** — Check outputs against acceptance criteria
4. **Document** — Update the spec's Change Record with implementation notes

### Key Files

- `specs/_index.md` — Overview of all stages and their status
- `specs/stages/*.md` — Individual stage specifications
- `specs/reviews/*.md` — Checklists for stage review
- `.trae/rules/project_rules.md` — Project conventions and workflow details

## Configuration

Analysis parameters are centralized in `config/analysis.yml`:

- **Thresholds**: Correlation (0.7), VIF (5)
- **Stations**: 9M, 14M, 37M
- **Temporal resolution**: 2-hour bins
- **Model settings**: Random effects, smoothing parameters

The config file is the source of truth — specs reference config keys rather than hardcoding values.

## Code Organization

```
src/python/mbon_indices/
├── config.py          # Configuration loading and validation
├── loaders/           # Data loading for each source type
├── align/             # Temporal alignment utilities
├── qa/                # Quality assurance checks
└── utils/             # Logging, datetime parsing
```

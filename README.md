# MBON Acoustic Indices Study

Can acoustic indices — automated summaries of underwater soundscape characteristics — predict biological activity in an estuary?

**Study site:** Three passive acoustic monitoring stations along the May River estuary, SC (9M, 14M, 37M), 2021 data at 2-hour resolution. 13,102 observations, 60 acoustic indices, 9 biological response variables (fish, dolphins, vessels).

## View Results (No Setup Required)

A pre-built results viewer is included in the repository. Just open it in your browser:

- **macOS:** `open docs/results-viewer.html`
- **Windows:** double-click `docs\results-viewer.html` in File Explorer
- **Linux:** `xdg-open docs/results-viewer.html`

If you only want to see the results, you're done — no software installation needed.

---

## Getting Started

Follow these steps to set up the environment and re-run the full analysis pipeline.

### 1. Install Prerequisites

You need **Python** (with uv) and **R**. Quarto is optional (only needed to rebuild the HTML results viewer).

<details>
<summary><strong>macOS</strong></summary>

```bash
# Install Python (if not already installed)
brew install python

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install R — download from https://cran.r-project.org/bin/macosx/
# Or with Homebrew:
brew install r

# Optional: Install Quarto — download from https://quarto.org/docs/get-started/
```

</details>

<details>
<summary><strong>Windows</strong></summary>

1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/). During install, check "Add Python to PATH."
2. **uv**: Open PowerShell and run:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **R 4.x**: Download from [cran.r-project.org](https://cran.r-project.org/bin/windows/base/). During install, check "Add R to PATH" if asked (or add it manually: the default location is `C:\Program Files\R\R-4.x.x\bin`).
4. **Optional — Quarto**: Download from [quarto.org](https://quarto.org/docs/get-started/).

</details>

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
# Install Python and R
sudo apt update
sudo apt install python3 python3-pip r-base

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: Install Quarto — download .deb from https://quarto.org/docs/get-started/
```

</details>

### 2. Clone and Set Up

```bash
git clone <repo-url>
cd mbon-indices-study

# Install Python dependencies
uv sync

# Install R dependencies (exact versions are locked in renv.lock)
Rscript -e "renv::restore()"
```

**RStudio users:** Instead of the command line, you can open `mbon-indices-study.Rproj` in RStudio. It will automatically activate renv. Then run `renv::restore()` in the R console to install packages.

> **What is renv?** It's like a package manager for R — similar to how `uv.lock` locks Python package versions, `renv.lock` locks R package versions. This ensures everyone gets the exact same package versions. You don't need to install renv separately; it bootstraps itself automatically.

### 3. Run the Pipeline

The easiest way — runs all stages in order with progress messages:

```bash
python run_pipeline.py
```

This takes roughly 30–60 minutes total (the R modeling stages are the slowest part).

**Options:**

```bash
python run_pipeline.py --check       # Just check if prerequisites are installed
python run_pipeline.py --from 05a    # Resume from a specific stage
python run_pipeline.py --only 05c    # Run only one stage
```

### 4. Run Stages Individually (Optional)

If you prefer to run stages one at a time, or need to re-run a specific stage:

```bash
# Data prep — Python (stages 00–04, ~2–5 minutes)
uv run scripts/stage00-a_verify_loaders.py
uv run scripts/stage00-b_align.py
uv run scripts/stage00-c_generate_qa.py
uv run scripts/stage01_index_reduction.py
uv run scripts/stage02_community_metrics.py
uv run scripts/stage03_feature_engineering.py
uv run scripts/stage04_exploratory_viz.py

# Modeling — R (stages 05+, ~20–50 minutes)
Rscript scripts/stage05_modeling.R
Rscript scripts/stage05b_validation.R
Rscript scripts/stage05c_diagnostics.R
Rscript scripts/stage05d_baseline.R
Rscript scripts/stage05e_acoustic_only.R
Rscript scripts/stage05e_bystation_effects.R
Rscript scripts/stage05f_bystation_effect_sizes.R
```

### 5. Rebuild the Results Viewer (Optional)

The pre-built HTML is already included. Only run this if you've re-run the pipeline and want to update the results viewer:

```bash
./rebuild-docs.sh                     # macOS/Linux
# Windows: run the commands in rebuild-docs.sh manually, or use Git Bash
```

---

## Key Findings

**Presence detection works.** Vessel presence: AUC 0.93 (excellent). Fish and dolphin presence: AUC ~0.77 (moderate, useful for screening).

**Activity counts don't generalize.** Count models (fish activity, dolphin clicks, etc.) fit training data but fail on held-out weeks (negative R²). The data is too zero-inflated and variable for week-to-week prediction.

**60 indices > 17 VIF-filtered indices.** Letting GAMM regularization (`select=TRUE`) handle feature selection outperformed manual VIF pre-filtering (avg ΔAIC = -262).

**Acoustic indices add real value beyond environment/time.** All 9 metrics show ΔAIC > 70 when adding indices to a baseline model with only temperature, depth, and time variables. For dolphins, indices dominate; for fish, indices and environment contribute roughly equally.

**No universal "best index."** Top predictors vary by metric and station. HFC dominates vessel detection; ACTspFract dominates dolphin presence; LFC matters most for fish presence. Station-level models show different indices matter at different locations.

## Project Organization

```
├── config/
│   └── analysis.yml              # All analysis parameters (responses, thresholds, GAMM settings)
├── data/
│   ├── raw/                      # Original data (committed — detections, environment, indices)
│   ├── interim/                  # Aligned parquet files (generated by stage 00)
│   └── processed/                # Analysis-ready dataset (generated by stages 01–03)
├── scripts/
│   ├── stage00-*.py              # Data loading, alignment, QA
│   ├── stage01–04_*.py           # Index reduction, community metrics, feature engineering, viz
│   ├── stage05_modeling.R        # Main GAMM fitting (9 response variables)
│   ├── stage05b–f_*.R            # Validation, diagnostics, baseline, per-station analyses
├── results/                      # Generated outputs (figures, tables, models, logs)
├── specs/stages/                 # Detailed specs for each analysis stage
├── docs/
│   ├── results-viewer.qmd        # Quarto results document (source)
│   ├── results-viewer.html       # Pre-built results (just open this to view)
│   └── presentations/            # Slide decks
├── src/python/mbon_indices/      # Python package (data loading/processing utilities)
├── run_pipeline.py               # Cross-platform pipeline runner
└── rebuild-docs.sh               # Copies results to docs/ and renders Quarto
```

## Data

Raw data is included in the repository under `data/raw/`:

| Folder | Contents | Size |
|--------|----------|------|
| `data/raw/2018/` | Detections, environment, SPL (Excel) | ~6 MB |
| `data/raw/2021/` | Detections, environment, SPL (Excel) | ~6 MB |
| `data/raw/indices/` | Acoustic index CSVs (3 stations × 2 bands) | ~196 MB |
| `data/raw/metadata/` | Column mappings, index metadata | <1 MB |

Intermediate and processed data (`data/interim/`, `data/processed/`) are regenerated by the pipeline and not committed.

## Models

**Generalized Additive Mixed Models (GAMMs)** via `mgcv::bam()` with:

- 60 acoustic indices as smooth predictors (`k=3`)
- Temperature and depth as smooth covariates
- Cyclic smooths for hour-of-day and day-of-year
- Station, month, and day as random effects
- AR1 autocorrelation (data-driven ρ)
- Shrinkage selection (`select=TRUE`) — penalizes unnecessary complexity, no manual pre-filtering needed

Binary responses (presence) use binomial family; count responses use negative binomial.

## Configuration

All analysis parameters live in [`config/analysis.yml`](config/analysis.yml) — response variable families, GAMM settings (`smooth_k`, `select`), scaling options, station list, etc.

## License

Research project — not currently licensed for external use.

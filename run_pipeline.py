#!/usr/bin/env python3
"""
MBON Acoustic Indices Study — Full Pipeline Runner

Runs all analysis stages in order (Python data prep → R modeling → docs).
Works on Linux, macOS, and Windows.

Usage:
    python run_pipeline.py              # Run everything
    python run_pipeline.py --from 3     # Resume from stage 03
    python run_pipeline.py --only 5a    # Run just stage 05a
    python run_pipeline.py --check      # Check prerequisites only
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Stage definitions ──────────────────────────────────────────────────────────

STAGES = [
    {
        "id": "00a",
        "name": "Verify loaders & config",
        "cmd": ["uv", "run", "scripts/stage00-a_verify_loaders.py"],
    },
    {
        "id": "00b",
        "name": "Align data to 2-hour bins",
        "cmd": ["uv", "run", "scripts/stage00-b_align.py"],
    },
    {
        "id": "00c",
        "name": "Generate QA artifacts",
        "cmd": ["uv", "run", "scripts/stage00-c_generate_qa.py"],
    },
    {
        "id": "01",
        "name": "Index reduction (VIF check)",
        "cmd": ["uv", "run", "scripts/stage01_index_reduction.py"],
    },
    {
        "id": "02",
        "name": "Community metrics",
        "cmd": ["uv", "run", "scripts/stage02_community_metrics.py"],
    },
    {
        "id": "03",
        "name": "Feature engineering",
        "cmd": ["uv", "run", "scripts/stage03_feature_engineering.py"],
    },
    {
        "id": "04",
        "name": "Exploratory visualization",
        "cmd": ["uv", "run", "scripts/stage04_exploratory_viz.py"],
    },
    {
        "id": "05a",
        "name": "GAMM modeling (9 response variables)",
        "cmd": ["Rscript", "scripts/stage05_modeling.R"],
        "slow": True,
    },
    {
        "id": "05b",
        "name": "Leave-one-week-out cross-validation",
        "cmd": ["Rscript", "scripts/stage05b_validation.R"],
        "slow": True,
    },
    {
        "id": "05c",
        "name": "Diagnostics & effect sizes",
        "cmd": ["Rscript", "scripts/stage05c_diagnostics.R"],
    },
    {
        "id": "05d",
        "name": "Baseline comparison",
        "cmd": ["Rscript", "scripts/stage05d_baseline.R"],
        "slow": True,
    },
    {
        "id": "05e",
        "name": "Acoustic-only models",
        "cmd": ["Rscript", "scripts/stage05e_acoustic_only.R"],
        "slow": True,
    },
    {
        "id": "05e-bystation",
        "name": "Per-station effects",
        "cmd": ["Rscript", "scripts/stage05e_bystation_effects.R"],
        "slow": True,
    },
    {
        "id": "05f",
        "name": "Per-station effect sizes",
        "cmd": ["Rscript", "scripts/stage05f_bystation_effect_sizes.R"],
    },
    {
        "id": "docs",
        "name": "Rebuild documentation site",
        "cmd": ["quarto", "render", "docs/results-viewer.qmd"],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def check_command(name):
    """Check if a command is available on PATH."""
    return shutil.which(name) is not None


def check_prerequisites():
    """Verify all required tools are installed."""
    checks = {
        "python": sys.executable is not None,
        "uv": check_command("uv"),
        "Rscript": check_command("Rscript"),
        "quarto": check_command("quarto"),
    }

    print("Checking prerequisites:\n")
    all_ok = True
    for tool, found in checks.items():
        status = "OK" if found else "MISSING"
        print(f"  {tool:12s} {status}")
        if not found:
            all_ok = False

    # Check R packages via renv
    if checks["Rscript"]:
        print()
        renv_lock = Path("renv.lock")
        if renv_lock.exists():
            result = subprocess.run(
                ["Rscript", "-e", "renv::status()"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  R packages:  OK (renv)")
            else:
                print("  R packages:  Run 'Rscript -e \"renv::restore()\"' to install")
                all_ok = False
        else:
            print("  R packages:  renv.lock not found (install packages manually)")

    # Check Python packages
    if checks["uv"]:
        venv_path = Path(".venv")
        if venv_path.exists():
            print("  Python env:  OK (.venv exists)")
        else:
            print("  Python env:  Run 'uv sync' to create")
            all_ok = False

    print()
    if all_ok:
        print("All prerequisites met. Ready to run the pipeline.")
    else:
        print("Some prerequisites are missing. See above for details.")
    return all_ok


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def run_stage(stage, project_root):
    """Run a single pipeline stage."""
    stage_id = stage["id"]
    name = stage["name"]
    slow = stage.get("slow", False)

    slow_note = " (this may take a while)" if slow else ""
    print(f"\n{'='*70}")
    print(f"  Stage {stage_id}: {name}{slow_note}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(stage["cmd"], cwd=project_root)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {format_duration(elapsed)}")
        return False

    print(f"\n  Completed in {format_duration(elapsed)}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MBON Acoustic Indices Study — Pipeline Runner"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites only (don't run anything)",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        metavar="STAGE",
        help="Resume from this stage ID (e.g., --from 03 or --from 05a)",
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        help="Run only this stage ID (e.g., --only 05a)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.check:
        check_prerequisites()
        return

    # Determine which stages to run
    stage_ids = [s["id"] for s in STAGES]

    if args.only:
        if args.only not in stage_ids:
            print(f"Unknown stage: {args.only}")
            print(f"Available stages: {', '.join(stage_ids)}")
            sys.exit(1)
        stages_to_run = [s for s in STAGES if s["id"] == args.only]
    elif args.from_stage:
        if args.from_stage not in stage_ids:
            print(f"Unknown stage: {args.from_stage}")
            print(f"Available stages: {', '.join(stage_ids)}")
            sys.exit(1)
        start_idx = stage_ids.index(args.from_stage)
        stages_to_run = STAGES[start_idx:]
    else:
        stages_to_run = STAGES

    # Check prerequisites
    if not check_prerequisites():
        print("\nFix the issues above before running the pipeline.")
        sys.exit(1)

    # Run stages
    total_start = time.time()
    n = len(stages_to_run)
    print(f"\n{'#'*70}")
    print(f"  Running {n} stage{'s' if n > 1 else ''}")
    print(f"{'#'*70}")

    for i, stage in enumerate(stages_to_run, 1):
        print(f"\n  [{i}/{n}]", end="")
        if not run_stage(stage, project_root):
            print(f"\nPipeline stopped at stage {stage['id']}.")
            print(f"Fix the error above, then resume with: python run_pipeline.py --from {stage['id']}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*70}")
    print(f"  Pipeline complete! Total time: {format_duration(total_elapsed)}")
    print(f"{'#'*70}")
    print(f"\nView results: open docs/results-viewer.html")


if __name__ == "__main__":
    main()

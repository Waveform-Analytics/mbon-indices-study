"""
Run history tracking for stage scripts.

Appends concise summaries to results/logs/RUN_HISTORY.md after each successful run.
This separates methodology (in specs) from outcomes (in run history).

See specs/risks/adr-03-run-history-tracking.md for rationale.
"""

from datetime import datetime
from pathlib import Path


def append_to_run_history(
    root: Path,
    stage: str,
    config: dict[str, any],
    results: dict[str, any],
    log_path: str = "",
    notes: str = ""
) -> None:
    """
    Append a summary entry to RUN_HISTORY.md.

    Args:
        root: Project root directory
        stage: Stage name (e.g., "Stage 01: Index Reduction")
        config: Key configuration values used (dict of key-value pairs)
        results: Key results produced (dict of key-value pairs)
        log_path: Path to detailed log file (relative to project root)
        notes: Optional notes (default empty, user can fill in later)

    Example:
        append_to_run_history(
            root=root,
            stage="Stage 01: Index Reduction",
            config={"correlation_r": 0.6, "vif": 2},
            results={"n_start": 60, "n_final": 14, "categories": 5},
            log_path="results/logs/stage01_index_reduction_20251208_111021.txt",
            notes="Tightened thresholds per Zuur et al. 2010"
        )
    """
    history_path = root / "results" / "logs" / "RUN_HISTORY.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Format config as bullet points
    config_lines = [f"  - {k}: {v}" for k, v in config.items()]
    config_str = "\n".join(config_lines) if config_lines else "  - (none)"

    # Format results as bullet points
    results_lines = [f"  - {k}: {v}" for k, v in results.items()]
    results_str = "\n".join(results_lines) if results_lines else "  - (none)"

    # Format log path
    log_str = log_path if log_path else "(none)"

    entry = f"""## {timestamp} — {stage}

- **Config**:
{config_str}
- **Results**:
{results_str}
- **Log**: {log_str}
- **Notes**: {notes}

---

"""

    # Ensure directory exists
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to file
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"  Appended to run history: {history_path.relative_to(root)}")
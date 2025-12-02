"""
Logging utilities for stage scripts.

Provides timestamped logging with automatic archiving of previous runs.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path


class StageLogger:
    """
    Captures stdout to both terminal and timestamped log file.

    Usage:
        logger = setup_stage_logging(root, "stage01_index_reduction")
        try:
            print("Your output here")
        finally:
            logger.close()
            sys.stdout = logger.terminal
    """

    def __init__(self, log_path: Path):
        """
        Initialize logger that writes to both terminal and file.

        Args:
            log_path: Path to the log file
        """
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message: str):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        """Flush both terminal and log file buffers."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        self.log_file.close()


def setup_stage_logging(root: Path, stage_name: str) -> StageLogger:
    """
    Set up timestamped logging with archiving of previous logs.

    Creates a timestamped log file in results/logs/ and moves any previous
    logs for the same stage to results/logs/archive/.

    Args:
        root: Project root directory
        stage_name: Name of the stage (e.g., "stage01_index_reduction")

    Returns:
        StageLogger instance that captures stdout

    Example:
        logger = setup_stage_logging(root, "stage01_index_reduction")
        sys.stdout = logger
        try:
            print("Processing...")
        finally:
            logger.close()
            sys.stdout = logger.terminal
    """
    logs_dir = root / "results" / "logs"
    archive_dir = logs_dir / "archive"

    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{stage_name}_{timestamp}.txt"
    log_path = logs_dir / log_filename

    # Archive any existing logs for this stage
    for existing_log in logs_dir.glob(f"{stage_name}_*.txt"):
        if existing_log != log_path:
            archive_path = archive_dir / existing_log.name
            shutil.move(str(existing_log), str(archive_path))
            print(f"Archived previous log: {existing_log.name} → archive/")

    # Create and configure logger
    logger = StageLogger(log_path)
    sys.stdout = logger

    print(f"Log file: {log_path.relative_to(root)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return logger

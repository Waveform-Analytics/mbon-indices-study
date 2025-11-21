"""
Core infrastructure components for MBON pipeline.

Provides fundamental building blocks like path management, configuration,
and base pipeline functionality.
"""

from .config import AnalysisConfig
from .paths import ProjectPaths

__all__ = ["AnalysisConfig", "ProjectPaths"]
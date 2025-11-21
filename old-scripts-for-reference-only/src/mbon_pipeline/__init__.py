"""
MBON Data Pipeline Package

A package for standardizing data loading, temporal alignment, and preprocessing 
operations for the MBON USC 2025 marine bioacoustics project.

This package extracts common patterns from the analysis scripts to provide:
- Consistent data loading across different data types (acoustic indices, detections, environmental, SPL)
- Standardized temporal alignment and aggregation operations  
- Unified project path and configuration management
- Extensible framework for analysis-specific operations

Example usage:
    from mbon_pipeline import MBONDataLoader, TemporalAligner, AnalysisConfig, ProjectPaths
    
    config = AnalysisConfig(year=2021, stations=['9M', '14M', '37M'])
    paths = ProjectPaths()
    loader = MBONDataLoader(config, paths)
    
    # Load data
    indices_data = loader.load_acoustic_indices()
    detections_data = loader.load_detections_with_species_filter()
"""

__version__ = "0.1.0"
__author__ = "Michelle Weirathmueller"

# Core imports that users will commonly need
from .core.config import AnalysisConfig
from .core.paths import ProjectPaths
from .data.loaders import MBONDataLoader
from .data.temporal import TemporalAligner
from .analysis.feature_reduction import FeatureReducer, FeatureReductionResult
from .utils.metadata import MetadataManager

__all__ = [
    "AnalysisConfig",
    "ProjectPaths", 
    "MBONDataLoader",
    "TemporalAligner",
    "FeatureReducer",
    "FeatureReductionResult",
    "MetadataManager",
]

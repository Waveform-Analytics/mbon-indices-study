"""
Configuration management for MBON pipeline.

Provides standardized configuration classes for common analysis parameters,
making it easy to manage settings across different scripts and analyses.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class AnalysisConfig:
    """
    Configuration for MBON analysis parameters.
    
    Centralizes common parameters that appear across multiple scripts,
    reducing repetition and making configuration changes easier.
    
    Example:
        # Default configuration
        config = AnalysisConfig()
        
        # Custom configuration  
        config = AnalysisConfig(
            year=2022,
            stations=['9M', '14M'],
            aggregation_hours=1
        )
    """
    
    # Core analysis parameters
    year: int = 2021
    stations: List[str] = field(default_factory=lambda: ['9M', '14M', '37M'])
    
    # Temporal alignment parameters
    aggregation_hours: int = 2  # Aggregate to 2-hour intervals by default
    
    # Data processing parameters  
    correlation_threshold: float = 0.85  # For feature reduction analysis
    target_clusters: int = 18  # Target number of acoustic index clusters
    
    # Quality thresholds
    min_data_coverage: float = 0.7  # Minimum data coverage to include in analysis
    max_missing_fraction: float = 0.3  # Maximum fraction of missing data allowed
    
    # SPL windowing parameters
    spl_window_hours: float = 1.0  # Window size for SPL temporal matching

    # Acoustic indices source configuration
    # Variant selects which indices dataset to use:
    #   - 'v2' uses data/raw/indices/*_v2_Final.csv (default)
    #   - 'culled' uses data/raw/indices/culled/*_Final.csv (reduced set)
    indices_variant: str = "v2"
    # Bandwidth selects file family within indices: 'FullBW' or '8kHz'
    indices_bandwidth: str = "FullBW"
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.year < 2020 or self.year > 2030:
            raise ValueError(f"Year {self.year} seems unrealistic for MBON data")
            
        if not self.stations:
            raise ValueError("At least one station must be specified")
            
        if self.aggregation_hours <= 0:
            raise ValueError("Aggregation hours must be positive")
            
        if not (0 < self.correlation_threshold <= 1):
            raise ValueError("Correlation threshold must be between 0 and 1")
            
        if self.target_clusters <= 0:
            raise ValueError("Target clusters must be positive")

        # Validate indices configuration
        if self.indices_variant not in {"v2", "culled"}:
            raise ValueError("indices_variant must be one of {'v2','culled'}")
        if self.indices_bandwidth not in {"FullBW", "8kHz"}:
            raise ValueError("indices_bandwidth must be one of {'FullBW','8kHz'}")
    
    @classmethod
    def from_script_params(cls, **kwargs) -> 'AnalysisConfig':
        """
        Create configuration from script parameters.
        
        Useful for backward compatibility with existing scripts that
        define parameters directly.
        
        Args:
            **kwargs: Any configuration parameters to override
            
        Returns:
            AnalysisConfig instance with specified parameters
        """
        return cls(**kwargs)
    
    def get_station_year_combinations(self) -> List[tuple]:
        """
        Get all station-year combinations for data loading.
        
        Returns:
            List of (station, year) tuples
        """
        return [(station, self.year) for station in self.stations]
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'year': self.year,
            'stations': self.stations,
            'aggregation_hours': self.aggregation_hours,
            'correlation_threshold': self.correlation_threshold,
            'target_clusters': self.target_clusters,
            'min_data_coverage': self.min_data_coverage,
            'max_missing_fraction': self.max_missing_fraction,
            'spl_window_hours': self.spl_window_hours,
            'indices_variant': self.indices_variant,
            'indices_bandwidth': self.indices_bandwidth
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return (f"AnalysisConfig(year={self.year}, "
                f"stations={self.stations}, "
                f"aggregation_hours={self.aggregation_hours})")


@dataclass
class DataQualityConfig:
    """
    Configuration for data quality assessment parameters.
    
    Optional extension for more detailed quality control settings.
    """
    
    # Temporal coverage requirements
    min_temporal_coverage: float = 0.8
    max_gap_days: int = 7  # Maximum gap in days before flagging
    
    # Value range checks (can be extended as needed)
    temperature_range: tuple = (-5.0, 35.0)  # Reasonable temperature range (°C)
    depth_range: tuple = (0.0, 200.0)  # Reasonable depth range (m)
    
    # Outlier detection
    outlier_z_threshold: float = 3.0  # Z-score threshold for outlier detection
    
    def validate_temperature(self, temp_values) -> bool:
        """Check if temperature values are within reasonable range."""
        if temp_values is None or len(temp_values) == 0:
            return False
        return (temp_values >= self.temperature_range[0]).all() and \
               (temp_values <= self.temperature_range[1]).all()
    
    def validate_depth(self, depth_values) -> bool:
        """Check if depth values are within reasonable range."""
        if depth_values is None or len(depth_values) == 0:
            return False
        return (depth_values >= self.depth_range[0]).all() and \
               (depth_values <= self.depth_range[1]).all()


# Column classification metadata paths
DET_METADATA_FILE = "data/raw/metadata/det_column_names.csv"
INDEX_METADATA_FILE = "data/raw/metadata/Updated_Index_Categories_v2.csv"

# Predefined configurations for common use cases
DEFAULT_CONFIG = AnalysisConfig()

CONFIG_2021_ALL_STATIONS = AnalysisConfig(
    year=2021,
    stations=['9M', '14M', '37M'],
    aggregation_hours=2
)

CONFIG_2021_HOURLY = AnalysisConfig(
    year=2021,
    stations=['9M', '14M', '37M'], 
    aggregation_hours=1
)

# Configuration for feature reduction analysis
CONFIG_FEATURE_REDUCTION = AnalysisConfig(
    year=2021,
    stations=['9M', '14M', '37M'],
    aggregation_hours=2,
    correlation_threshold=0.85,
    target_clusters=18
)
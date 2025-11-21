"""
Project path management for MBON pipeline.

Provides centralized path resolution and file path construction
following the established project structure patterns.
"""

from pathlib import Path
from typing import Optional, Union


class ProjectPaths:
    """
    Centralized path management for the MBON project.
    
    Handles automatic project root detection and provides standardized
    path construction for all data types following established patterns.
    
    Example:
        paths = ProjectPaths()
        indices_file = paths.get_indices_path('9M', 2021)
        detections_file = paths.get_detections_path('14M', 2021)
    """
    
    def __init__(self, custom_root: Optional[Union[str, Path]] = None):
        """
        Initialize project paths.
        
        Args:
            custom_root: Optional custom project root path. If not provided,
                        will automatically detect project root by looking for data/raw.
        """
        if custom_root:
            self.project_root = Path(custom_root)
        else:
            self.project_root = self._find_project_root()
            
        # Core directory structure
        self.data_root = self.project_root / "data"
        self.raw_data = self.data_root / "raw"
        self.processed_data = self.data_root / "processed"
        
        # Ensure critical directories exist (read-only check, don't create)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        if not self.raw_data.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data}")
    
    def _find_project_root(self) -> Path:
        """
        Find project root by looking for the data/raw folder structure.
        
        This matches the pattern used in existing scripts.
        """
        # Start from current file location or current working directory
        try:
            current_dir = Path(__file__).parent
        except NameError:
            current_dir = Path.cwd()
            
        project_root = current_dir
        while not (project_root / "data" / "raw").exists() and project_root != project_root.parent:
            project_root = project_root.parent
            
        if not (project_root / "data" / "raw").exists():
            raise FileNotFoundError(
                "Could not find project root. Looking for directory containing 'data/raw/'"
            )
            
        return project_root
    
    def ensure_output_dirs(self, create_figures_dir: bool = True) -> None:
        """
        Ensure output directories exist, creating them if necessary.
        
        Args:
            create_figures_dir: Whether to create the figures subdirectory
        """
        self.processed_data.mkdir(parents=True, exist_ok=True)
        
        if create_figures_dir:
            self.get_figures_dir().mkdir(parents=True, exist_ok=True)
    
    def get_figures_dir(self, subfolder: str = "fresh_start_figures") -> Path:
        """Get figures output directory path."""
        return self.processed_data / subfolder
    
    # Acoustic Indices paths
    def get_indices_path(self, station: str, year: int, *, variant: str = "v2", bandwidth: str = "FullBW") -> Path:
        """
        Get path to acoustic indices file for a station and year.

        Args:
            station: Station identifier (e.g., '9M', '14M', '37M')
            year: Year (e.g., 2021)
            variant: Which dataset variant to use: 'v2' or 'culled'
            bandwidth: Bandwidth family: 'FullBW' or '8kHz'

        Returns:
            Path to acoustic indices CSV file
        """
        base = self.raw_data / "indices"
        if variant == "culled":
            base = base / "culled"
            suffix = "Final"
            vtag = ""
        else:
            suffix = "v2_Final"
            vtag = "_v2"

        filename = f"Acoustic_Indices_{station}_{year}_{bandwidth}{'_' + suffix if variant=='culled' else '_' + suffix}.csv"
        # Backward compatibility: exact filename string using computed components
        # culled: Acoustic_Indices_{station}_{year}_{bandwidth}_Final.csv
        # v2:     Acoustic_Indices_{station}_{year}_{bandwidth}_v2_Final.csv
        return base / filename
    
    # Detection data paths
    def get_detections_path(self, station: str, year: int) -> Path:
        """
        Get path to manual detections file for a station and year.
        
        Args:
            station: Station identifier (e.g., '9M', '14M', '37M')
            year: Year (e.g., 2021)
            
        Returns:
            Path to detections Excel file
        """
        return self.raw_data / str(year) / "detections" / f"Master_Manual_{station}_2h_{year}.xlsx"
    
    def get_species_metadata_path(self) -> Path:
        """Get path to species metadata file for filtering detections."""
        return self.raw_data / "metadata" / "det_column_names.csv"
    
    # Environmental data paths
    def get_temperature_path(self, station: str, year: int) -> Path:
        """Get path to temperature data file."""
        return self.raw_data / str(year) / "environmental" / f"Master_{station}_Temp_{year}.xlsx"
    
    def get_depth_path(self, station: str, year: int) -> Path:
        """
        Get path to depth/pressure data file.
        
        Note: Some stations may use "Depth" while others use "Press" in filename.
        """
        return self.raw_data / str(year) / "environmental" / f"Master_{station}_Depth_{year}.xlsx"
    
    def get_pressure_path(self, station: str, year: int) -> Path:
        """Get alternate path for pressure data (some stations use this naming)."""
        return self.raw_data / str(year) / "environmental" / f"Master_{station}_Press_{year}.xlsx"
    
    # SPL data paths  
    def get_spl_path(self, station: str, year: int) -> Path:
        """Get path to SPL (Sound Pressure Level) data file."""
        return self.raw_data / str(year) / "rms_spl" / f"Master_rmsSPL_{station}_1h_{year}.xlsx"
    
    # Output paths
    def get_aligned_dataset_path(self, year: int, suffix: str = "") -> Path:
        """
        Get path for saving aligned dataset output.
        
        Args:
            year: Year for the dataset
            suffix: Optional suffix for the filename
            
        Returns:
            Path for aligned dataset parquet file
        """
        filename = f"aligned_dataset_{year}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".parquet"
        return self.processed_data / filename
    
    def get_quality_report_path(self, suffix: str = "") -> Path:
        """Get path for data quality report JSON file."""
        filename = "data_quality_report"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        return self.processed_data / filename
    
    def get_figure_path(self, figure_name: str, subfolder: str = "fresh_start_figures") -> Path:
        """
        Get path for saving figure outputs.
        
        Args:
            figure_name: Name of the figure file (should include extension)
            subfolder: Subdirectory under processed data for figures
            
        Returns:
            Path for figure file
        """
        return self.get_figures_dir(subfolder) / figure_name
    
    def __repr__(self) -> str:
        return f"ProjectPaths(project_root='{self.project_root}')"
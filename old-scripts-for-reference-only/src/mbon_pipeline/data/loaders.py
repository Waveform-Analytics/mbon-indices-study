"""
Data loading utilities for MBON pipeline.

Provides standardized loaders for all MBON data types with consistent
error handling, validation, and data type management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

from ..core.config import AnalysisConfig
from ..core.paths import ProjectPaths

warnings.filterwarnings('ignore')


class MBONDataLoader:
    """
    Unified data loader for all MBON data types.
    
    Provides consistent loading interface and error handling for:
    - Acoustic indices
    - Manual detections (with species filtering)
    - Environmental data (temperature, depth/pressure, SPL)
    
    Example:
        config = AnalysisConfig(year=2021, stations=['9M', '14M'])
        paths = ProjectPaths()
        loader = MBONDataLoader(config, paths)
        
        indices_data = loader.load_acoustic_indices()
        detections_data = loader.load_detections_with_species_filter()
    """
    
    def __init__(self, config: AnalysisConfig, paths: ProjectPaths, verbose: bool = True):
        """
        Initialize data loader.
        
        Args:
            config: Analysis configuration with stations, year, etc.
            paths: Project paths manager
            verbose: Whether to print loading progress messages
        """
        self.config = config
        self.paths = paths
        self.verbose = verbose
        
        # Cache for loaded metadata to avoid reloading
        self._species_metadata = None
        
    def _log(self, message: str, level: str = "info") -> None:
        """Print log message if verbose mode enabled."""
        if self.verbose:
            if level == "error":
                print(f"✗ {message}")
            elif level == "warning":
                print(f"⚠️ {message}")
            else:
                print(f"✓ {message}")
    
    def load_acoustic_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Load acoustic indices data for all configured stations.
        
        Returns:
            Dictionary mapping station names to DataFrames with acoustic indices
        """
        self._log("LOADING ACOUSTIC INDICES", "info")
        self._log("-" * 30, "info")
        
        indices_data = {}
        
        # Determine indices source from configuration (with safe defaults)
        variant = getattr(self.config, "indices_variant", "v2")
        bandwidth = getattr(self.config, "indices_bandwidth", "FullBW")

        loading_summary = {
            'variant': variant,
            'bandwidth': bandwidth,
            'stations': {},
            'timestamp': datetime.now().isoformat()
        }

        for station in self.config.stations:
            file_path = self.paths.get_indices_path(station, self.config.year, variant=variant, bandwidth=bandwidth)
            
            if not file_path.exists():
                self._log(f"{station}: File not found - {file_path}", "warning")
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                # Handle datetime creation - look for common datetime column names
                datetime_col = self._find_datetime_column(df)
                if datetime_col:
                    df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
                    valid_datetimes = (~df['datetime'].isna()).sum()
                    self._log(f"{station}: Created {valid_datetimes} valid datetimes from {len(df)} rows")
                else:
                    self._log(f"{station}: No recognizable datetime column found", "warning")
                
                # Add station identifier
                df['station'] = station
                
                # Get basic info
                file_size_mb = file_path.stat().st_size / (1024*1024)
                date_range = (df.iloc[0]['datetime'].date(), df.iloc[-1]['datetime'].date()) if 'datetime' in df.columns and len(df) > 0 else None
                
                indices_data[station] = df

                # Collect station-level summary for comparison purposes
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                acoustic_cols_estimate = [c for c in numeric_cols if c not in ['Water temp (°C)', 'Water depth (m)']]
                loading_summary['stations'][station] = {
                    'rows': int(len(df)),
                    'columns': int(len(df.columns)),
                    'numeric_columns': int(len(numeric_cols)),
                    'estimated_acoustic_columns': int(len(acoustic_cols_estimate)),
                    'file_size_mb': round(file_size_mb, 2)
                }
                
                self._log(f"{station}: {len(df)} rows, {len(df.columns)} columns ({file_size_mb:.1f} MB)")
                if date_range:
                    self._log(f"   Date range: {date_range[0]} to {date_range[1]}")
                    
            except Exception as e:
                self._log(f"{station}: Error loading - {e}", "error")
        
        self._log(f"Acoustic indices loaded for {len(indices_data)}/{len(self.config.stations)} stations")

        # Save loading summary for reproducible comparison across variants
        try:
            summary_path = self.paths.processed_data / f"indices_loading_summary_{variant}.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(loading_summary, f, indent=2)
            self._log(f"Saved indices loading summary: {summary_path.name}")
        except Exception as e:
            self._log(f"Could not save indices loading summary: {e}", "warning")
        return indices_data
    
    def load_detections_with_species_filter(self) -> Dict[str, pd.DataFrame]:
        """
        Load manual detection data with automatic species filtering.
        
        Uses species metadata file to determine which species columns to keep.
        
        Returns:
            Dictionary mapping station names to filtered detection DataFrames
        """
        self._log("LOADING MANUAL DETECTIONS", "info")
        self._log("-" * 30, "info")
        
        # Load species metadata if not already loaded
        if self._species_metadata is None:
            self._species_metadata = self._load_species_metadata()
        
        if self._species_metadata is None:
            self._log("Cannot load detections without species metadata", "error")
            return {}
        
        keep_columns = self._species_metadata[self._species_metadata['keep_species'] == 1]['long_name'].tolist()
        # Note: We normalize 'Date ' -> 'Date' during loading, so only need 'Date' here
        essential_columns = ['Date', 'Time', 'Deployment ID', 'File']

        self._log(f"Species to keep: {', '.join(keep_columns[:5])}{' ...' if len(keep_columns) > 5 else ''}")
        
        detection_data = {}
        
        for station in self.config.stations:
            file_path = self.paths.get_detections_path(station, self.config.year)
            
            if not file_path.exists():
                self._log(f"{station}: File not found - {file_path}", "warning")
                continue
                
            try:
                # Load the "Data" sheet from Excel file
                df = pd.read_excel(file_path, sheet_name="Data")

                # Normalize Date column name: 'Date ' (with space) -> 'Date' (without space)
                # Some files have trailing space in column name, which causes inconsistency
                if 'Date ' in df.columns and 'Date' not in df.columns:
                    df = df.rename(columns={'Date ': 'Date'})
                    self._log(f"   Normalized 'Date ' column to 'Date' for {station}")

                # Filter columns based on metadata
                available_keep_cols = [col for col in keep_columns if col in df.columns]
                available_essential_cols = [col for col in essential_columns if col in df.columns]
                cols_to_keep = list(set(available_keep_cols + available_essential_cols))

                df_filtered = df[cols_to_keep].copy()
                
                # Handle datetime creation - prioritize Date column, handle variations
                datetime_col = self._find_datetime_column(df_filtered, prefer_date_only=True)
                if datetime_col:
                    df_filtered['datetime'] = pd.to_datetime(df_filtered[datetime_col], errors='coerce')
                    valid_datetimes = (~df_filtered['datetime'].isna()).sum()
                    self._log(f"   Created {valid_datetimes} valid datetimes from column '{datetime_col}'")
                else:
                    self._log(f"   Warning: Could not create datetime column for {station}", "warning")
                
                # Convert Time column to string to avoid Parquet serialization issues
                if 'Time' in df_filtered.columns:
                    df_filtered['Time'] = df_filtered['Time'].astype(str)
                
                # Convert non-numeric object columns to numeric where possible, string otherwise
                object_cols = df_filtered.select_dtypes(include=['object']).columns
                for col in object_cols:
                    if col not in ['Date', 'datetime', 'station', 'Time', 'Deployment ID', 'File']:
                        try:
                            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                        except:
                            df_filtered[col] = df_filtered[col].astype(str)
                
                # Add station identifier
                df_filtered['station'] = station
                
                detection_data[station] = df_filtered
                
                self._log(f"{station}: {len(df_filtered)} rows, {len(df_filtered.columns)} columns (filtered from {len(df.columns)})")
                self._log(f"   Species columns: {len(available_keep_cols)}")
                
            except Exception as e:
                self._log(f"{station}: Error loading - {e}", "error")
        
        self._log(f"Manual detection data loaded for {len(detection_data)}/{len(self.config.stations)} stations")
        return detection_data
    
    def load_environmental_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all environmental data (temperature, depth, SPL).
        
        Returns:
            Dictionary with structure:
            {
                'temperature': {station: DataFrame, ...},
                'depth': {station: DataFrame, ...},
                'spl': {station: DataFrame, ...}
            }
        """
        self._log("LOADING ENVIRONMENTAL DATA", "info") 
        self._log("-" * 30, "info")
        
        env_data = {
            'temperature': {},
            'depth': {},
            'spl': {}
        }
        
        # Load temperature data
        for station in self.config.stations:
            temp_file = self.paths.get_temperature_path(station, self.config.year)
            
            if temp_file.exists():
                try:
                    df_temp = pd.read_excel(temp_file, sheet_name="Data")
                    
                    # Handle datetime
                    if 'Date and time' in df_temp.columns:
                        df_temp['datetime'] = pd.to_datetime(df_temp['Date and time'], errors='coerce')
                        df_temp['station'] = station
                        env_data['temperature'][station] = df_temp
                        self._log(f"{station} temperature: {len(df_temp)} rows")
                    else:
                        self._log(f"{station} temperature: Missing 'Date and time' column", "warning")
                        
                except Exception as e:
                    self._log(f"{station} temperature: Error loading - {e}", "error")
            else:
                self._log(f"{station} temperature: File not found", "warning")
        
        # Load depth/pressure data
        for station in self.config.stations:
            # Try depth first, then pressure naming
            depth_file = self.paths.get_depth_path(station, self.config.year)
            pressure_file = self.paths.get_pressure_path(station, self.config.year)
            
            loaded = False
            for file_path, data_type in [(depth_file, "depth"), (pressure_file, "pressure")]:
                if file_path.exists() and not loaded:
                    try:
                        df_depth = pd.read_excel(file_path, sheet_name="Data")
                        
                        # Handle datetime
                        if 'Date and time' in df_depth.columns:
                            df_depth['datetime'] = pd.to_datetime(df_depth['Date and time'], errors='coerce')
                            df_depth['station'] = station
                            env_data['depth'][station] = df_depth
                            self._log(f"{station} {data_type}: {len(df_depth)} rows")
                            loaded = True
                        else:
                            self._log(f"{station} {data_type}: Missing 'Date and time' column", "warning")
                            
                    except Exception as e:
                        self._log(f"{station} {data_type}: Error loading - {e}", "error")
            
            if not loaded:
                self._log(f"{station} depth/pressure: No valid file found", "warning")
        
        # Load SPL data
        spl_data = self._load_spl_data()
        env_data['spl'] = spl_data
        
        # Summary
        temp_count = len(env_data['temperature'])
        depth_count = len(env_data['depth'])
        spl_count = len(env_data['spl'])
        
        self._log(f"Environmental data - Temperature: {temp_count} stations, "
                 f"Depth: {depth_count} stations, SPL: {spl_count} stations")
        
        return env_data
    
    def _load_spl_data(self) -> Dict[str, pd.DataFrame]:
        """Load SPL data with complex datetime handling."""
        self._log("Loading SPL data...")
        
        spl_data = {}
        
        for station in self.config.stations:
            spl_file = self.paths.get_spl_path(station, self.config.year)
            
            if spl_file.exists():
                try:
                    df_spl = pd.read_excel(spl_file, sheet_name="Data")
                    
                    # Handle complex SPL datetime combination
                    if 'Date' in df_spl.columns and 'Time' in df_spl.columns:
                        combined_datetimes = []
                        for date_val, time_val in zip(df_spl['Date'], df_spl['Time']):
                            if pd.notna(date_val) and pd.notna(time_val):
                                try:
                                    # Handle different time formats
                                    if hasattr(time_val, 'time'):
                                        time_part = time_val.time()
                                    else:
                                        time_part = time_val
                                    
                                    # Combine with actual date
                                    combined_dt = pd.to_datetime(
                                        date_val.date().strftime('%Y-%m-%d') + ' ' + 
                                        time_part.strftime('%H:%M:%S')
                                    )
                                    combined_datetimes.append(combined_dt)
                                except:
                                    combined_datetimes.append(pd.NaT)
                            else:
                                combined_datetimes.append(pd.NaT)
                        
                        df_spl['datetime'] = pd.Series(combined_datetimes)
                        df_spl['station'] = station
                        
                        valid_datetimes = (~df_spl['datetime'].isna()).sum()
                        spl_data[station] = df_spl
                        
                        self._log(f"{station} SPL: {len(df_spl)} rows ({valid_datetimes} valid datetimes)")
                    else:
                        self._log(f"{station} SPL: Missing Date/Time columns", "warning")
                        
                except Exception as e:
                    self._log(f"{station} SPL: Error loading - {e}", "error")
            else:
                self._log(f"{station} SPL: File not found", "warning")
        
        return spl_data
    
    def _load_species_metadata(self) -> Optional[pd.DataFrame]:
        """Load species metadata for filtering detections."""
        metadata_path = self.paths.get_species_metadata_path()
        
        if not metadata_path.exists():
            self._log(f"Species metadata file not found: {metadata_path}", "error")
            return None
        
        try:
            metadata_df = pd.read_csv(metadata_path)
            self._log(f"Loaded species metadata: {len(metadata_df)} species")
            return metadata_df
        except Exception as e:
            self._log(f"Error loading species metadata: {e}", "error")
            return None
    
    def _find_datetime_column(self, df: pd.DataFrame, prefer_date_only: bool = False) -> Optional[str]:
        """
        Find the most likely datetime column in a DataFrame.
        
        Args:
            df: DataFrame to search
            prefer_date_only: If True, prefer 'Date' over other datetime columns
            
        Returns:
            Column name that likely contains datetime data, or None
        """
        # Priority order for datetime columns
        if prefer_date_only:
            datetime_candidates = ['Date', 'Date ', 'datetime', 'DateTime', 'time', 'Time']
        else:
            datetime_candidates = ['datetime', 'DateTime', 'Date', 'Date ', 'time', 'Time']
        
        for col in datetime_candidates:
            if col in df.columns:
                return col
        
        return None
    
    def get_acoustic_index_columns(self, indices_df: pd.DataFrame) -> List[str]:
        """
        Extract acoustic index column names from DataFrame.
        
        Args:
            indices_df: DataFrame with acoustic indices
            
        Returns:
            List of column names that contain acoustic index data
        """
        exclude_cols = ['datetime', 'station', 'year', 'Date', 'Time', 'Filename', 
                       'Deployment ID', 'File']
        
        acoustic_cols = []
        for col in indices_df.columns:
            if col not in exclude_cols and indices_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                acoustic_cols.append(col)
        
        return acoustic_cols
    
    def get_species_columns(self, detections_df: pd.DataFrame) -> List[str]:
        """
        Extract species/detection column names from DataFrame.

        Args:
            detections_df: DataFrame with detection data

        Returns:
            List of column names that contain species detection data
        """
        exclude_cols = ['datetime', 'station', 'Date', 'Time', 'Deployment ID', 'File']

        species_cols = []
        for col in detections_df.columns:
            if col not in exclude_cols and detections_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                species_cols.append(col)

        return species_cols
    
    def get_loading_summary(self, *data_dicts) -> Dict[str, Any]:
        """
        Generate a summary of loaded data.
        
        Args:
            *data_dicts: Variable number of data dictionaries to summarize
            
        Returns:
            Dictionary with loading summary statistics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'data_sources': {}
        }
        
        # Basic summary of what was requested vs loaded
        expected_stations = len(self.config.stations)
        
        for i, data_dict in enumerate(data_dicts):
            data_type = f"data_source_{i}"
            if isinstance(data_dict, dict):
                loaded_stations = len(data_dict)
                total_rows = sum(len(df) for df in data_dict.values())
                
                summary['data_sources'][data_type] = {
                    'expected_stations': expected_stations,
                    'loaded_stations': loaded_stations,
                    'success_rate': loaded_stations / expected_stations,
                    'total_rows': total_rows,
                    'stations': list(data_dict.keys())
                }
        
        return summary
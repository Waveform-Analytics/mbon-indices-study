"""
Temporal alignment and aggregation utilities for MBON pipeline.

Provides standardized temporal processing operations including:
- Alignment to regular time grids (2-hour intervals)
- Multi-source data temporal merging
- SPL windowed matching
- Datetime handling and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

from ..core.config import AnalysisConfig


class TemporalAligner:
    """
    Handles temporal alignment and aggregation operations.
    
    Provides methods to align multiple data sources to a common temporal grid,
    typically based on the detection data intervals (2-hour by default).
    
    Example:
        config = AnalysisConfig(aggregation_hours=2)
        aligner = TemporalAligner(config)
        
        aligned_df = aligner.align_to_detection_grid(
            detection_data, indices_data, env_data
        )
    """
    
    def __init__(self, config: AnalysisConfig, verbose: bool = True):
        """
        Initialize temporal aligner.
        
        Args:
            config: Analysis configuration with temporal parameters
            verbose: Whether to print progress messages
        """
        self.config = config
        self.verbose = verbose
        self.target_resolution = config.aggregation_hours
        
    def _log(self, message: str, level: str = "info") -> None:
        """Print log message if verbose mode enabled."""
        if self.verbose:
            if level == "error":
                print(f"✗ {message}")
            elif level == "warning":
                print(f"⚠️ {message}")
            else:
                print(f"✓ {message}")
    
    def align_to_detection_grid(
        self, 
        detection_data: Dict[str, pd.DataFrame],
        indices_data: Optional[Dict[str, pd.DataFrame]] = None,
        env_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
    ) -> pd.DataFrame:
        """
        Align all data sources to the detection temporal grid.
        
        Uses detection data as the reference time grid and aligns all other
        data sources to match those time intervals.
        
        Args:
            detection_data: Detection data by station (reference grid)
            indices_data: Optional acoustic indices data by station
            env_data: Optional environmental data (temperature, depth, SPL)
            
        Returns:
            Combined DataFrame with all data aligned to detection time grid
        """
        self._log("TEMPORAL ALIGNMENT TO DETECTION GRID")
        self._log("-" * 40)
        
        aligned_stations = {}
        
        for station in self.config.stations:
            if station not in detection_data:
                self._log(f"{station}: No detection data available", "warning")
                continue
                
            self._log(f"Aligning data for station {station}...")
            
            # Start with detection data as the base (already at target resolution)
            base_df = detection_data[station].copy()
            
            # Ensure datetime column exists and is properly formatted
            base_df = self._ensure_datetime_column(base_df)
            if 'datetime' not in base_df.columns:
                self._log(f"   Skipping {station}: No valid datetime column", "warning")
                continue
            
            # Floor datetime to exact intervals to ensure clean alignment
            base_df['datetime'] = base_df['datetime'].dt.floor(f'{self.target_resolution}h')
            
            # Clean up data types for Parquet compatibility
            base_df = self._clean_data_types(base_df)
            
            # Add station identifier if not present
            if 'station' not in base_df.columns:
                base_df['station'] = station
            
            station_df = base_df.copy()
            
            # Add acoustic indices if available
            if indices_data and station in indices_data:
                station_df = self._merge_acoustic_indices(station_df, indices_data[station])
            
            # Add environmental data if available
            if env_data:
                station_df = self._merge_environmental_data(station_df, env_data, station)
            
            aligned_stations[station] = station_df
            self._log(f"   Station {station}: {len(station_df)} aligned records")
        
        # Combine all stations
        if aligned_stations:
            combined_df = pd.concat(aligned_stations.values(), ignore_index=True)
            self._log(f"Combined dataset: {len(combined_df)} total records across {len(aligned_stations)} stations")
            return combined_df
        else:
            self._log("No data could be aligned", "warning")
            return pd.DataFrame()
    
    def _ensure_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has a properly formatted datetime column.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with standardized datetime column
        """
        df = df.copy()
        
        # If datetime already exists and is valid, return as-is
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                if not df['datetime'].isna().all():
                    return df
            except:
                pass
        
        # Try to create datetime from Date column
        # Note: 'Date ' (with space) is normalized to 'Date' upstream in loaders
        if 'Date' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['Date'], errors='coerce')
                valid_count = (~df['datetime'].isna()).sum()
                if valid_count > 0:
                    self._log(f"      Created {valid_count} valid datetimes from 'Date'")
                    return df
            except:
                pass
        
        return df
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data types for Parquet compatibility and consistency.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with cleaned data types
        """
        df = df.copy()
        
        # Convert Time column to string to avoid Parquet issues
        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(str)
        
        # Handle object columns - convert to numeric if possible, string otherwise
        object_cols = df.select_dtypes(include=['object']).columns
        exclude_from_conversion = ['Date', 'datetime', 'station', 'Time', 'Deployment ID', 'File']

        for col in object_cols:
            if col not in exclude_from_conversion:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = df[col].astype(str)
        
        return df
    
    def _merge_acoustic_indices(self, base_df: pd.DataFrame, indices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge acoustic indices data with base DataFrame.
        
        Args:
            base_df: Base DataFrame with detection time grid
            indices_df: Acoustic indices DataFrame
            
        Returns:
            Merged DataFrame
        """
        self._log("   - Adding acoustic indices...")
        
        try:
            indices_df = indices_df.copy()
            
            # Ensure datetime column
            indices_df = self._ensure_datetime_column(indices_df)
            if 'datetime' not in indices_df.columns:
                self._log("     No valid datetime in indices data", "warning")
                return base_df
            
            # Get numeric columns (the actual indices)
            numeric_cols = indices_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['station']]
            
            if not numeric_cols:
                self._log("     No numeric columns found in indices data", "warning")
                return base_df
            
            # Aggregate to target resolution using mean
            indices_df['datetime_grouped'] = indices_df['datetime'].dt.floor(f'{self.target_resolution}h')
            
            agg_dict = {col: 'mean' for col in numeric_cols}
            indices_aggregated = indices_df.groupby('datetime_grouped').agg(agg_dict).reset_index()
            indices_aggregated.rename(columns={'datetime_grouped': 'datetime'}, inplace=True)
            
            # Merge with base DataFrame
            merged_df = base_df.merge(indices_aggregated, on='datetime', how='left')
            
            self._log(f"     Added {len(numeric_cols)} acoustic indices")
            return merged_df
            
        except Exception as e:
            self._log(f"     Error processing acoustic indices: {e}", "warning")
            return base_df
    
    def _merge_environmental_data(
        self, 
        base_df: pd.DataFrame, 
        env_data: Dict[str, Dict[str, pd.DataFrame]], 
        station: str
    ) -> pd.DataFrame:
        """
        Merge environmental data (temperature, depth, SPL) with base DataFrame.
        
        Args:
            base_df: Base DataFrame with detection time grid
            env_data: Environmental data dictionary
            station: Station identifier
            
        Returns:
            Merged DataFrame
        """
        merged_df = base_df.copy()
        
        # Add temperature data
        if 'temperature' in env_data and station in env_data['temperature']:
            merged_df = self._merge_temperature_data(merged_df, env_data['temperature'][station])
        
        # Add depth data
        if 'depth' in env_data and station in env_data['depth']:
            merged_df = self._merge_depth_data(merged_df, env_data['depth'][station])
        
        # Add SPL data (requires special windowed matching)
        if 'spl' in env_data and station in env_data['spl']:
            merged_df = self._merge_spl_data_windowed(merged_df, env_data['spl'][station])
        
        return merged_df
    
    def _merge_temperature_data(self, base_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
        """Merge temperature data using standard aggregation."""
        self._log("   - Adding temperature data...")
        
        try:
            temp_df = temp_df.copy()
            
            if 'Date and time' not in temp_df.columns or 'Water temp (°C)' not in temp_df.columns:
                self._log("     Missing required temperature columns", "warning")
                return base_df
            
            temp_df['datetime'] = pd.to_datetime(temp_df['Date and time'], errors='coerce')
            temp_df['datetime_grouped'] = temp_df['datetime'].dt.floor(f'{self.target_resolution}h')
            
            temp_aggregated = temp_df.groupby('datetime_grouped').agg({
                'Water temp (°C)': 'mean'
            }).reset_index()
            temp_aggregated.rename(columns={'datetime_grouped': 'datetime'}, inplace=True)
            
            merged_df = base_df.merge(temp_aggregated, on='datetime', how='left')
            
            self._log("     Added temperature data")
            return merged_df
            
        except Exception as e:
            self._log(f"     Error processing temperature data: {e}", "warning")
            return base_df
    
    def _merge_depth_data(self, base_df: pd.DataFrame, depth_df: pd.DataFrame) -> pd.DataFrame:
        """Merge depth/pressure data using standard aggregation."""
        self._log("   - Adding depth data...")
        
        try:
            depth_df = depth_df.copy()
            
            if 'Date and time' not in depth_df.columns or 'Water depth (m)' not in depth_df.columns:
                self._log("     Missing required depth columns", "warning")
                return base_df
            
            depth_df['datetime'] = pd.to_datetime(depth_df['Date and time'], errors='coerce')
            depth_df['datetime_grouped'] = depth_df['datetime'].dt.floor(f'{self.target_resolution}h')
            
            depth_aggregated = depth_df.groupby('datetime_grouped').agg({
                'Water depth (m)': 'mean'
            }).reset_index()
            depth_aggregated.rename(columns={'datetime_grouped': 'datetime'}, inplace=True)
            
            merged_df = base_df.merge(depth_aggregated, on='datetime', how='left')
            
            self._log("     Added depth data")
            return merged_df
            
        except Exception as e:
            self._log(f"     Error processing depth data: {e}", "warning")
            return base_df
    
    def _merge_spl_data_windowed(self, base_df: pd.DataFrame, spl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SPL data using windowed matching approach.
        
        SPL data requires special handling due to timing misalignments.
        Uses a time window around each detection period to find matching SPL data.
        """
        self._log("   - Adding SPL data...")
        
        try:
            # Define SPL columns to process
            spl_columns = ['Broadband (1-40000 Hz)', 'Low (50-1200 Hz)', 'High (7000-40000 Hz)']
            available_spl_cols = [col for col in spl_columns if col in spl_df.columns]
            
            if not available_spl_cols:
                self._log("     No recognizable SPL columns found", "warning")
                return base_df
            
            if 'datetime' not in spl_df.columns:
                self._log("     No datetime column in SPL data", "warning")
                return base_df
            
            # Perform windowed matching for each detection time point
            spl_aligned_rows = []
            time_window = pd.Timedelta(hours=self.config.spl_window_hours)
            
            for _, det_row in base_df.iterrows():
                det_datetime = det_row['datetime']
                
                # Find SPL data within time window
                mask = (spl_df['datetime'] >= det_datetime - time_window) & \
                       (spl_df['datetime'] <= det_datetime + time_window)
                
                spl_window = spl_df.loc[mask, available_spl_cols]
                
                if len(spl_window) > 0:
                    # Calculate mean SPL values for this time window
                    spl_means = spl_window.mean()
                    spl_row = {
                        self._clean_spl_column_name(col): spl_means[col] 
                        for col in available_spl_cols
                    }
                else:
                    # No SPL data for this time window
                    spl_row = {
                        self._clean_spl_column_name(col): np.nan 
                        for col in available_spl_cols
                    }
                
                spl_aligned_rows.append(spl_row)
            
            # Add SPL columns to base DataFrame
            if spl_aligned_rows:
                spl_df_aligned = pd.DataFrame(spl_aligned_rows)
                merged_df = pd.concat([base_df.reset_index(drop=True), spl_df_aligned.reset_index(drop=True)], axis=1)
                
                self._log(f"     Added {len(available_spl_cols)} SPL columns")
                return merged_df
            else:
                self._log("     No SPL data could be aligned", "warning")
                return base_df
                
        except Exception as e:
            self._log(f"     Error processing SPL data: {e}", "warning")
            return base_df
    
    def _clean_spl_column_name(self, col_name: str) -> str:
        """Convert SPL column name to a clean format for output."""
        return f"spl_{col_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
    
    def aggregate_to_intervals(
        self, 
        df: pd.DataFrame, 
        target_hours: Optional[int] = None,
        datetime_col: str = 'datetime',
        method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate data to specified time intervals.
        
        Args:
            df: DataFrame to aggregate
            target_hours: Target interval in hours (defaults to config value)
            datetime_col: Name of datetime column
            method: Aggregation method ('mean', 'sum', 'median', etc.)
            
        Returns:
            Aggregated DataFrame
        """
        if target_hours is None:
            target_hours = self.target_resolution
            
        df_agg = df.copy()
        
        # Ensure datetime column exists
        if datetime_col not in df_agg.columns:
            self._log(f"Datetime column '{datetime_col}' not found", "error")
            return df_agg
        
        # Create time bins
        df_agg[datetime_col] = pd.to_datetime(df_agg[datetime_col])
        df_agg['datetime_bin'] = df_agg[datetime_col].dt.floor(f'{target_hours}h')
        
        # Get numeric columns for aggregation
        numeric_cols = df_agg.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['station']]  # Keep station as grouping var
        
        if not numeric_cols:
            self._log("No numeric columns found for aggregation", "warning")
            return df_agg
        
        # Group by station and time bins
        group_cols = ['datetime_bin']
        if 'station' in df_agg.columns:
            group_cols.append('station')
        
        agg_dict = {col: method for col in numeric_cols}
        aggregated = df_agg.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename datetime column back
        aggregated = aggregated.rename(columns={'datetime_bin': datetime_col})
        
        self._log(f"Aggregated to {target_hours}-hour intervals: {df.shape} → {aggregated.shape}")
        return aggregated
    
    def validate_temporal_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that temporal alignment was successful.
        
        Args:
            df: Aligned DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        if 'datetime' not in df.columns:
            validation['is_valid'] = False
            validation['issues'].append("No datetime column found")
            return validation
        
        # Check for regular intervals
        if 'station' in df.columns:
            for station in df['station'].unique():
                station_data = df[df['station'] == station].sort_values('datetime')
                if len(station_data) > 1:
                    time_diffs = station_data['datetime'].diff().dropna()
                    expected_diff = pd.Timedelta(hours=self.target_resolution)
                    
                    # Allow some tolerance for irregular intervals
                    tolerance = pd.Timedelta(minutes=30)
                    irregular_count = sum(abs(time_diffs - expected_diff) > tolerance)
                    
                    if irregular_count > len(time_diffs) * 0.1:  # More than 10% irregular
                        validation['issues'].append(f"Station {station}: {irregular_count} irregular intervals")
        
        # Basic statistics
        validation['statistics'] = {
            'total_records': len(df),
            'date_range': (df['datetime'].min(), df['datetime'].max()) if len(df) > 0 else None,
            'stations': df['station'].nunique() if 'station' in df.columns else 1,
            'missing_datetime': df['datetime'].isna().sum()
        }
        
        return validation
"""
Metadata management utilities for MBON pipeline.

Handles loading and parsing of metadata files that define column classifications,
acoustic index categories, and species/detection column specifications.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import warnings

from ..core.config import DET_METADATA_FILE, INDEX_METADATA_FILE


class MetadataManager:
    """
    Manages metadata for column classification and acoustic index categorization.
    
    Loads and provides access to:
    - Detection/species column specifications
    - Acoustic index categories and descriptions  
    - Column classification for proper data handling
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize metadata manager.
        
        Args:
            project_root: Root directory of the project. If None, will search upward
                         from current directory to find project root.
        """
        self.project_root = project_root or self._find_project_root()
        self._det_metadata = None
        self._index_metadata = None
        
        # Cache for classification results
        self._cached_classifications = {}
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for characteristic directories."""
        current = Path.cwd()
        
        # Look for characteristic project directories
        for parent in [current] + list(current.parents):
            if (parent / "data" / "raw" / "metadata").exists():
                return parent
        
        # Fallback to current directory
        warnings.warn("Could not find project root, using current directory")
        return current
    
    def load_detection_metadata(self) -> pd.DataFrame:
        """Load detection/species column metadata."""
        if self._det_metadata is None:
            metadata_path = self.project_root / DET_METADATA_FILE
            if not metadata_path.exists():
                raise FileNotFoundError(f"Detection metadata not found: {metadata_path}")
            self._det_metadata = pd.read_csv(metadata_path)
        return self._det_metadata
    
    def load_index_metadata(self) -> pd.DataFrame:
        """Load acoustic index metadata."""
        if self._index_metadata is None:
            metadata_path = self.project_root / INDEX_METADATA_FILE
            if not metadata_path.exists():
                raise FileNotFoundError(f"Index metadata not found: {metadata_path}")
            self._index_metadata = pd.read_csv(metadata_path)
        return self._index_metadata
    
    def get_species_columns(self, keep_only: bool = True) -> List[str]:
        """
        Get list of species/detection columns.
        
        Args:
            keep_only: If True, only return columns marked for keeping
                      
        Returns:
            List of species/detection column names
        """
        det_meta = self.load_detection_metadata()
        
        if keep_only:
            return det_meta[det_meta['keep_species'] == 1]['long_name'].tolist()
        else:
            return det_meta['long_name'].tolist()
    
    def get_acoustic_index_columns(self) -> List[str]:
        """Get list of expected acoustic index column names."""
        index_meta = self.load_index_metadata()
        return index_meta['Prefix'].tolist()
    
    def get_index_categories(self) -> Dict[str, Dict[str, str]]:
        """
        Get acoustic index categories and descriptions.
        
        Returns:
            Dictionary mapping index names to category information
        """
        index_meta = self.load_index_metadata()
        
        categories = {}
        for _, row in index_meta.iterrows():
            categories[row['Prefix']] = {
                'category': row['Category'],
                'subcategory': row['Subcategory'], 
                'description': row['Description']
            }
        
        return categories
    
    def classify_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Classify a list of columns into different types.
        
        Args:
            columns: List of column names to classify
            
        Returns:
            Dictionary with keys: 'metadata', 'spl', 'species', 'acoustic_indices', 'unknown'
        """
        # Cache key for this column set
        cache_key = tuple(sorted(columns))
        if cache_key in self._cached_classifications:
            return self._cached_classifications[cache_key]
        
        # Load metadata
        expected_species = set(self.get_species_columns())
        expected_indices = set(self.get_acoustic_index_columns())
        
        # Define metadata columns
        metadata_cols = {
            'datetime', 'station', 'Date', 'Date ', 'Time', 'Deployment ID', 'File',
            'Water temp (°C)', 'Water depth (m)'
        }
        
        # Define technical columns that should NOT be treated as acoustic indices
        # even if they might match naming patterns
        acoustic_exclusions = {
            'FrequencyResolution',  # Technical parameter, not an acoustic index
            'SampleRate',          # Technical parameter
            'Duration',            # Technical parameter
            'WindowSize',          # Technical parameter
        }
        
        classification = {
            'metadata': [],
            'spl': [],
            'species': [],
            'acoustic_indices': [],
            'unknown': []
        }
        
        for col in columns:
            if col in metadata_cols:
                classification['metadata'].append(col)
            elif col.startswith('spl_'):
                classification['spl'].append(col) 
            elif col in expected_species:
                classification['species'].append(col)
            elif col in acoustic_exclusions:
                # Technical parameters should be classified as metadata, not acoustic indices
                classification['metadata'].append(col)
            elif col in expected_indices:
                classification['acoustic_indices'].append(col)
            else:
                # Check if it might be an acoustic index variant
                is_likely_acoustic = False
                
                # Check for ROI variants (nROI, aROI are likely variants of ROItotal/ROIcover)
                if 'roi' in col.lower():
                    is_likely_acoustic = True
                # Check if it starts like a known index
                elif any(col.lower().startswith(idx.lower()[:4]) for idx in expected_indices):
                    is_likely_acoustic = True
                
                if is_likely_acoustic:
                    classification['acoustic_indices'].append(col)
                else:
                    classification['unknown'].append(col)
        
        # Cache result
        self._cached_classifications[cache_key] = classification
        
        return classification
    
    def validate_dataset_columns(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Validate that a dataset has expected columns.
        
        Args:
            df: DataFrame to validate
            verbose: Whether to print validation messages
            
        Returns:
            Dictionary with validation results
        """
        classification = self.classify_columns(df.columns.tolist())
        
        expected_species = set(self.get_species_columns())
        expected_indices = set(self.get_acoustic_index_columns())
        
        # Check for missing expected columns
        missing_species = expected_species - set(classification['species'])
        missing_indices = expected_indices - set(classification['acoustic_indices'])
        
        validation = {
            'total_columns': len(df.columns),
            'classification': classification,
            'missing_species': list(missing_species),
            'missing_indices': list(missing_indices),
            'unexpected_columns': classification['unknown'],
            'is_valid': len(missing_species) == 0 and len(missing_indices) <= 2  # Allow some missing
        }
        
        if verbose:
            print(f"📊 Dataset Validation Results:")
            print(f"   • Total columns: {validation['total_columns']}")
            print(f"   • Metadata: {len(classification['metadata'])}")
            print(f"   • SPL: {len(classification['spl'])}")
            print(f"   • Species/detections: {len(classification['species'])}")
            print(f"   • Acoustic indices: {len(classification['acoustic_indices'])}")
            print(f"   • Unknown/unexpected: {len(classification['unknown'])}")
            
            if missing_species:
                print(f"   ⚠️ Missing species columns: {missing_species}")
            if missing_indices:
                print(f"   ⚠️ Missing acoustic indices: {missing_indices}")
            if classification['unknown']:
                print(f"   ❓ Unexpected columns: {classification['unknown']}")
        
        return validation
    
    def get_acoustic_indices_by_category(self, include_unknown: bool = True) -> Dict[str, List[str]]:
        """
        Get acoustic indices grouped by category.
        
        Args:
            include_unknown: Whether to include indices not in metadata
            
        Returns:
            Dictionary mapping category names to lists of indices
        """
        index_meta = self.load_index_metadata()
        
        # Group by category
        by_category = {}
        for _, row in index_meta.iterrows():
            category = row['Category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(row['Prefix'])
        
        # Add unknown category if requested
        if include_unknown:
            by_category['Unknown/Variant'] = []
        
        return by_category
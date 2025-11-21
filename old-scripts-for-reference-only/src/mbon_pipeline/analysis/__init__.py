"""
Analysis modules for MBON pipeline.

This subpackage contains reusable analysis components, such as
feature reduction for acoustic indices.
"""

from .feature_reduction import FeatureReducer, FeatureReductionResult

__all__ = [
    "FeatureReducer",
    "FeatureReductionResult",
]
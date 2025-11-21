"""
Feature reduction utilities for acoustic indices.

Implements correlation analysis, dynamic clustering (no fixed target), and
representative selection to reduce redundant acoustic indices.

Designed to be reusable and configurable within the mbon_pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ..core.config import AnalysisConfig
from ..utils.metadata import MetadataManager


@dataclass
class FeatureReductionResult:
    selected_indices: List[str]
    clean_indices: List[str]
    optimal_clusters: int
    cluster_labels: List[int]
    cluster_sizes: Dict[int, int]
    silhouette_scores: Dict[int, float]
    best_silhouette: float
    high_corr_pairs: List[Dict[str, float]]
    corr_threshold: float
    generation_timestamp: str


class FeatureReducer:
    """
    Reduce acoustic indices to a non-redundant set using dynamic clustering.

    Workflow:
    1) Identify acoustic index columns from aligned dataset
    2) Correlation analysis and cleaning (NaNs/constants)
    3) Dynamic cluster optimization via silhouette score
    4) Representative selection per cluster (default: highest variance)
    """

    def __init__(self, config: AnalysisConfig, verbose: bool = True, metadata_manager: Optional[MetadataManager] = None):
        self.config = config
        self.verbose = verbose
        self.metadata_manager = metadata_manager or MetadataManager()

    # -------- Public API --------
    def reduce(self, df_aligned: pd.DataFrame,
               index_selector: str = "variance",
               exclude_cols: Optional[List[str]] = None) -> FeatureReductionResult:
        """
        Run full feature reduction on aligned dataset.

        Args:
            df_aligned: DataFrame produced by Script 1 (aligned data)
            index_selector: Strategy for representative selection. Options:
                - "variance": choose member with highest variance (default)
            exclude_cols: Optional extra columns to exclude from index detection
        """
        if self.verbose:
            print("[FeatureReducer] Identifying acoustic indices from aligned dataset...")
        acoustic_cols = self._identify_acoustic_indices(df_aligned, exclude_cols)

        if self.verbose:
            print(f"[FeatureReducer] Found {len(acoustic_cols)} candidate indices")
            print("[FeatureReducer] Running correlation analysis and cleaning...")
        corr_matrix, high_corr_pairs, clean_indices = self._analyze_correlations(df_aligned, acoustic_cols)

        if self.verbose:
            print("[FeatureReducer] Optimizing cluster count dynamically...")
        optimal_clusters, cluster_labels, sil_scores, best_sil = self._find_optimal_clusters(corr_matrix, clean_indices)

        if self.verbose:
            print(f"[FeatureReducer] Selecting representatives using '{index_selector}' criterion...")
        selected, cluster_sizes = self._select_representatives(df_aligned, clean_indices, cluster_labels, optimal_clusters, index_selector)

        return FeatureReductionResult(
            selected_indices=selected,
            clean_indices=clean_indices,
            optimal_clusters=optimal_clusters,
            cluster_labels=cluster_labels.tolist(),
            cluster_sizes=cluster_sizes,
            silhouette_scores=sil_scores,
            best_silhouette=best_sil,
            high_corr_pairs=high_corr_pairs,
            corr_threshold=self.config.correlation_threshold,
            generation_timestamp=datetime.now().isoformat()
        )

    # -------- Internals --------
    def _identify_acoustic_indices(self, df: pd.DataFrame, extra_exclude: Optional[List[str]]) -> List[str]:
        # Use metadata manager for proper column classification
        classification = self.metadata_manager.classify_columns(df.columns.tolist())
        
        if self.verbose:
            validation = self.metadata_manager.validate_dataset_columns(df, verbose=True)
            print(f"[FeatureReducer] Using metadata-based classification")
            print(f"[FeatureReducer] Identified {len(classification['acoustic_indices'])} acoustic indices")
            print(f"[FeatureReducer] Excluded {len(classification['species'])} species/detection columns")
            if classification['unknown']:
                print(f"[FeatureReducer] Found {len(classification['unknown'])} unknown columns: {classification['unknown']}")
        
        acoustic_cols = classification['acoustic_indices']
        
        if extra_exclude:
            acoustic_cols = [col for col in acoustic_cols if col not in extra_exclude]
        
        return acoustic_cols

    def _analyze_correlations(self, df: pd.DataFrame, cols: List[str]):
        df_idx = df[cols].copy()
        # Drop all-NaN and constant columns
        df_idx = df_idx.dropna(axis=1, how='all')
        df_idx = df_idx.loc[:, df_idx.var() > 0]
        clean_cols = df_idx.columns.tolist()
        corr = df_idx.corr()
        # high corr pairs
        high_pairs = []
        th = self.config.correlation_threshold
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > th:
                    high_pairs.append({
                        'index1': corr.columns[i],
                        'index2': corr.columns[j],
                        'correlation': float(val)
                    })
        return corr, high_pairs, clean_cols

    def _find_optimal_clusters(self, corr: pd.DataFrame, clean_cols: List[str]):
        # Distance from absolute correlation
        distance = 1 - corr.abs()
        min_k = 2
        max_k = min(20, max(2, len(clean_cols) // 2))
        rng = range(min_k, max_k + 1)
        sil_scores: List[float] = []
        X = distance.values
        for k in rng:
            model = AgglomerativeClustering(n_clusters=k, linkage='ward', metric='euclidean')
            labels = model.fit_predict(X)
            if len(set(labels)) > 1:
                sil = silhouette_score(X, labels)
            else:
                sil = -1.0
            sil_scores.append(float(sil))
        best_idx = int(np.argmax(sil_scores))
        best_k = list(rng)[best_idx]
        best_sil = sil_scores[best_idx]
        final = AgglomerativeClustering(n_clusters=best_k, linkage='ward', metric='euclidean')
        final_labels = final.fit_predict(X)
        unique, counts = np.unique(final_labels, return_counts=True)
        sil_map = {k: s for k, s in zip(rng, sil_scores)}
        return best_k, final_labels, sil_map, float(best_sil)

    def _select_representatives(self, df: pd.DataFrame, clean_cols: List[str], labels: np.ndarray,
                                n_clusters: int, strategy: str) -> Tuple[List[str], Dict[int, int]]:
        df_idx = df[clean_cols].copy()
        selected: List[str] = []
        sizes: Dict[int, int] = {}
        for cid in range(n_clusters):
            members = [clean_cols[i] for i in range(len(clean_cols)) if labels[i] == cid]
            sizes[cid] = len(members)
            if not members:
                continue
            if strategy == "variance":
                var = df_idx[members].var()
                rep = var.idxmax()
            else:
                var = df_idx[members].var()
                rep = var.idxmax()
            selected.append(rep)
        return selected, sizes
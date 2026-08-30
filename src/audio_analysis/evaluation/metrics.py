"""
Metric calculations and aggregation for contour evaluation.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
from .comparison import RawComparisonResult
from ..config import MetricsConfig, CoverageMode, FreqDiffMode, FragMode, RecallMode

def _select_metrics(raw: RawComparisonResult, cfg: MetricsConfig) -> List[float]:
    """Map config choices to [coverage, false_pos, freq_diff, frag, recall]"""
    cov = getattr(raw, f"coverage_{cfg.coverage_mode.value}")
    fp = raw.false_pos
    fd = getattr(raw, f"freq_diff_{cfg.freq_diff_mode.value}")
    frag = getattr(raw, f"frag_{cfg.frag_mode.value}")
    recall = getattr(raw, f"recall_{cfg.recall_mode.value}")
        
    return [cov, fp, fd, frag, recall]


def aggregate_metrics(results: List[List[float]]) -> Dict[str, Any]:
    if not results:
        return {k: np.nan for k in ['coverage', 'false_pos', 'freq_diff', 'fragmentation', 'recall',
                                    'coverage_std', 'false_pos_std', 'freq_diff_std', 'fragmentation_std', 'recall_std',
                                    'coverage_5_95', 'false_pos_5_95', 'freq_diff_5_95', 'fragmentation_5_95', 'recall_5_95']}
    
    arr = np.array(results).T  # Shape: (5, N_files)

    def _compute_stats(col: np.ndarray):
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            return np.nan, np.nan, [np.nan, np.nan]
        return float(np.mean(valid)), float(np.std(valid)), [float(np.percentile(valid, 5)), float(np.percentile(valid, 95))]

    cov_mean, cov_std, cov_pct = _compute_stats(arr[0])
    fp_mean, fp_std, fp_pct = _compute_stats(arr[1])
    fd_mean, fd_std, fd_pct = _compute_stats(arr[2])
    frag_mean, frag_std, frag_pct = _compute_stats(arr[3])
    recall_mean, recall_std, recall_pct = _compute_stats(arr[4])

    return {
        'coverage': cov_mean, 'false_pos': fp_mean, 'freq_diff': fd_mean, 'fragmentation': frag_mean, 'recall': recall_mean,
        'coverage_std': cov_std, 'false_pos_std': fp_std, 'freq_diff_std': fd_std, 'fragmentation_std': frag_std, 'recall_std': recall_std,
        'coverage_5_95': cov_pct, 'false_pos_5_95': fp_pct, 'freq_diff_5_95': fd_pct, 'fragmentation_5_95': frag_pct, 'recall_5_95': recall_pct,
    }

def compute_all_mode_metrics(
    raw_results: List[RawComparisonResult], 
    cfg: MetricsConfig
) -> Dict[str, Dict[str, Any]]:
    """Compute aggregated metrics for all coverage, freq_diff, frag, and recall modes."""
    all_metrics = {}
    
    # Iterate over all available modes dynamically
    for cov_mode in CoverageMode:
        for fd_mode in FreqDiffMode:
            for frag_mode in FragMode:
                for recall_mode in RecallMode:
                    # Extract values for this specific mode combination
                    selected = []
                    for r in raw_results:
                        cov = getattr(r, f"coverage_{cov_mode.value}")
                        fd = getattr(r, f"freq_diff_{fd_mode.value}")
                        frag = getattr(r, f"frag_{frag_mode.value}")
                        recall = getattr(r, f"recall_{recall_mode.value}")
                            
                        selected.append([cov, r.false_pos, fd, frag, recall])
                    
                    agg = aggregate_metrics(selected)
                    # Store with a clear key: e.g., "total_total_total_total"
                    key = f"{cov_mode.value}_{fd_mode.value}_{frag_mode.value}_{recall_mode.value}"
                    all_metrics[key] = agg
                    
    return all_metrics

def compute_fb_metrics(
    raw_results: Dict[str, List[RawComparisonResult]],
    fb_ids: List[str],
    cfg: MetricsConfig
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-FBID metrics using the selected config strategy."""
    fb_metrics = {}
    for fb in fb_ids:
        if fb in raw_results:
            selected = [_select_metrics(r, cfg) for r in raw_results[fb]]
            fb_metrics[fb] = aggregate_metrics(selected)
        else:
            fb_metrics[fb] = {k: np.nan for k in aggregate_metrics([]).keys()}
    return fb_metrics


def load_metrics(path: Union[str, Path]) -> Dict[str, Any]:
    """Load aggregated metrics from a JSON file, restoring expected structures."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle new structure with embedded config
    if isinstance(data, dict) and "metrics" in data:
        return data["metrics"]
        
    # Fallback for legacy files (flat structure)
    return data

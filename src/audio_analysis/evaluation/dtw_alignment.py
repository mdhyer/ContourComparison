"""
DTW alignment, distance calculation, and statistical testing utilities.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from dtw import dtw as dtw_align
from typing import Dict, List, Any, Tuple

def calculate_dtw_distance(cont1: np.ndarray, cont2: np.ndarray, **dtw_args) -> float:
    """Calculate DTW distance between two contours after interpolation and normalization."""
    if cont1.shape[0] < 2 or cont2.shape[0] < 2:
        return np.nan
    x1 = np.linspace(cont2[0, 0], cont2[-1, 0], 100)
    x2 = np.linspace(cont1[0, 0], cont1[-1, 0], 100)
    c1 = np.interp(x1, cont2[:, 0], cont2[:, 1])
    c2 = np.interp(x2, cont1[:, 0], cont1[:, 1])
    c1 = c1 - np.median(c1)
    c2 = c2 - np.median(c2)
    alignment = dtw_align(c1.astype(np.float32), c2.astype(np.float32), **dtw_args)
    return alignment.distance

def compute_z_scores(results: Dict[str, List[float]], ground_key: str = 'Ground') -> Dict[str, np.ndarray]:
    """Compute Z-scores relative to ground truth results."""
    z_scores = {}
    g_mean = np.nanmean(results[ground_key])
    g_std = np.nanstd(results[ground_key])
    if g_std == 0:
        g_std = 1e-8
    for algo, res in results.items():
        z = (np.array(res) - g_mean) / g_std
        z_scores[algo] = z[~np.isnan(z)]
    return z_scores

def run_statistical_tests(
    z_scores: Dict[str, np.ndarray],
    rand_z_scores: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Run KS and MWU tests between within-individual and between-individual distances."""
    stats = {}
    for algo in z_scores:
        mwu_result = mannwhitneyu(z_scores[algo], rand_z_scores[algo], alternative='two-sided')
        pscore = ks_2samp(z_scores[algo], rand_z_scores[algo])
        stats[algo] = {
            'ks_p': pscore.pvalue,
            'mwu_p': mwu_result.pvalue,
            'mwu_stat': mwu_result.statistic
        }
    return stats

def nearest_neighbor_test(
    dtw_scores: List[Tuple[str, float]],
    target_fb: str,
    top_k: int = 5
) -> bool:
    """Check if target FBID is in top-k nearest neighbors."""
    sorter = np.argsort([d for _, d in dtw_scores])
    sorted_arr = np.array(dtw_scores, dtype=object)[sorter]
    sorted_fbs = [fb for fb, _ in sorted_arr]
    return target_fb in sorted_fbs[:top_k]

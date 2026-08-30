"""
Core contour comparison and metric calculation utilities.
Migrated from legacy silbido_test.py for modernized evaluation pipeline.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class RawComparisonResult:
    coverage_total: float
    coverage_ffc: float
    coverage_total_per_loop: float
    coverage_ffc_per_loop: float
    false_pos: float
    freq_diff_total: float
    freq_diff_ffc: float
    frag_total: float
    frag_ffc: float
    frag_total_per_loop: float
    frag_ffc_per_loop: float
    recall_total: float
    recall_ffc: float
    num_gt_loops: int
    num_pred_contours: int

def compare_dicts(
    ground_dict: Dict[float, float],
    contour_dicts: List[Dict[float, float]],
    ffc_threshold: float = 0.25
) -> RawComparisonResult:
    """Compute all metric variants in a single pass."""
    cov_total = 0
    cov_ffc = 0
    false_pos = 0
    fd_total = 0.0
    fd_ffc = 0.0
    overlap_pts = 0
    ffc_overlap_pts = 0
    
    frag_total = [False] * len(contour_dicts)
    frag_ffc = [False] * len(contour_dicts)

    for g_x in ground_dict:
        hit_total = False
        hit_ffc = False
        for di, c_dict in enumerate(contour_dicts):
            if g_x in c_dict:
                frag_total[di] = True
                hit_total = True
                overlap_pts += 1

                gt_freq = ground_dict[g_x]
                if gt_freq == 0:
                    continue
                    
                rel_fd = abs(c_dict[g_x] - gt_freq) / gt_freq
                fd_total += rel_fd

                if rel_fd <= ffc_threshold:
                    fd_ffc += rel_fd
                    hit_ffc = True
                    ffc_overlap_pts += 1
                    frag_ffc[di] = True

        if hit_total: cov_total += 1
        if hit_ffc: cov_ffc += 1

    n_gt = len(ground_dict)
    n_contours = len(contour_dicts)
    n_gt_segments = 1  # Each call to compare_dicts processes one ground truth segment
    
    cov_total = cov_total / n_gt if n_gt else 0.0
    cov_ffc = cov_ffc / n_gt if n_gt else 0.0

    fd_total = (fd_total / overlap_pts) * 100 if overlap_pts else np.nan
    fd_ffc = (fd_ffc / ffc_overlap_pts) * 100 if ffc_overlap_pts else np.nan

    frag_t = sum(frag_total)
    frag_f = sum(frag_ffc)
    frag_total_val = np.nan if frag_t == 0 else float(frag_t)
    frag_ffc_val = np.nan if frag_f == 0 else float(frag_f)
    
    # Per-loop fragmentation: valid predicted contours per GT loop
    frag_total_per_loop_val = frag_t / n_gt_segments if n_gt_segments > 0 else 0.0
    frag_ffc_per_loop_val = frag_f / n_gt_segments if n_gt_segments > 0 else 0.0

    for ci, cov in enumerate(contour_dicts):
        false_pos += len(contour_dicts[ci]) - len([g_x for g_x in ground_dict if g_x in cov])

    # Recall for single segment (1.0 if any coverage, else 0.0)
    recall_t = 1.0 if cov_total > 0 else 0.0
    recall_f = 1.0 if cov_ffc > 0 else 0.0

    return RawComparisonResult(
        coverage_total=cov_total,
        coverage_ffc=cov_ffc,
        coverage_total_per_loop=cov_total,
        coverage_ffc_per_loop=cov_ffc,
        false_pos=float(false_pos),
        freq_diff_total=fd_total,
        freq_diff_ffc=fd_ffc,
        frag_total=frag_total_val,
        frag_ffc=frag_ffc_val,
        frag_total_per_loop=frag_total_per_loop_val,
        frag_ffc_per_loop=frag_ffc_per_loop_val,
        recall_total=recall_t,
        recall_ffc=recall_f,
        num_gt_loops=n_gt_segments,
        num_pred_contours=n_contours
    )

def compare_contour(
    start: float, end: float,
    ground: np.ndarray,
    mat_contour: List[np.ndarray],
    ffc_threshold: float = 0.25
) -> RawComparisonResult:
    """Single segment comparison."""
    ground_dict = {np.round(float(g[0]), 3): float(g[1]) for g in ground if start <= g[0] <= end}
    if len(ground_dict) < 2:
        return RawComparisonResult(0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    contour_dicts = []
    for mc in mat_contour:
        d = {np.round(float(x), 3): float(c) for x, c in mc if float(c) < 24000}
        if d: contour_dicts.append(d)

    if not contour_dicts:
        return RawComparisonResult(0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    return compare_dicts(ground_dict, contour_dicts, ffc_threshold)


def _split_ground_truth(ground: np.ndarray, discont: Optional[List] = None, gap_threshold: float = 0.1) -> List[
    np.ndarray]:
    """Split ground truth into continuous segments/loops based on discontinuities or time gaps."""
    if len(ground) < 2:
        return [ground] if len(ground) == 1 else []

    segments = []
    start_idx = 0
    for i in range(1, len(ground)):
        is_break = False
        if discont:
            for d in discont:
                # Handle both [start, end] pairs and raw floats
                split_times = [float(d[0]), float(d[1])] if isinstance(d, (list, tuple)) else [float(d)]
                
                for d_val in split_times:
                    if ground[i - 1, 0] < d_val <= ground[i, 0]:
                        is_break = True
                        break
                if is_break:
                    break

        if is_break:
            segments.append(ground[start_idx:i])
            start_idx = i
    segments.append(ground[start_idx:])
    return [seg for seg in segments if len(seg) >= 2]


def compare_contours(
    ground: np.ndarray,
    mat_contour: List[np.ndarray],
    discont: Optional[List] = None,
    ffc_threshold: float = 0.25
) -> RawComparisonResult:
    """Full contour comparison with proper loop segmentation and aggregation."""
    if isinstance(mat_contour, np.ndarray):
        mat_contour = list(mat_contour)
    if not mat_contour:
        return RawComparisonResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    segments = _split_ground_truth(ground, discont)
    if not segments:
        return RawComparisonResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    num_gt_loops = len(segments)
    
    # Raw accumulators
    raw_cov_total = 0; raw_cov_ffc = 0; raw_fp = 0
    raw_fd_total = 0.0; raw_fd_ffc = 0.0; raw_overlap = 0; raw_overlap_ffc = 0
    raw_frag_total = 0; raw_frag_ffc = 0
    total_gt_points = 0
    loop_cov_totals = []
    loop_cov_ffcs = []
    loops_with_hit_total = 0
    loops_with_hit_ffc = 0
    
    # Track unique matched predictions globally to avoid double-counting across loops
    matched_pred_indices_total = set()
    matched_pred_indices_ffc = set()

    for seg in segments:
        g_dict = {np.round(float(g[0]), 3): float(g[1]) for g in seg}
        if len(g_dict) < 2:
            continue
            
        c_dicts = []
        c_dict_indices = []
        for idx, mc in enumerate(mat_contour):
            d = {np.round(float(x), 3): float(c) for x, c in mc if float(c) < 24000}
            if d: 
                c_dicts.append(d)
                c_dict_indices.append(idx)
        if not c_dicts:
            continue

        # Segment-level comparison (mirrors compare_dicts logic)
        seg_cov_t = 0; seg_cov_f = 0; seg_fp = 0
        seg_fd_t = 0.0; seg_fd_f = 0.0; seg_ov = 0; seg_ov_f = 0
        seg_cov_per = [[] for _ in c_dicts]
        seg_cov_ffc_per = [[] for _ in c_dicts]

        for g_x in g_dict:
            hit_t = False; hit_f = False
            for di, c_dict in enumerate(c_dicts):
                if g_x in c_dict:
                    matched_pred_indices_total.add(c_dict_indices[di]); hit_t = True
                    seg_cov_per[di].append(g_x); seg_ov += 1
                    gt_freq = g_dict[g_x]
                    if gt_freq == 0: continue
                    rel_fd = abs(c_dict[g_x] - gt_freq) / gt_freq
                    seg_fd_t += rel_fd
                    if rel_fd <= ffc_threshold:
                        seg_fd_f += rel_fd; hit_f = True
                        seg_cov_ffc_per[di].append(g_x); seg_ov_f += 1
                        matched_pred_indices_ffc.add(c_dict_indices[di])
            if hit_t: seg_cov_t += 1
            if hit_f: seg_cov_f += 1

        for di, cov in enumerate(seg_cov_per):
            seg_fp += len(c_dicts[di]) - len(cov)

        # Accumulate
        raw_cov_total += seg_cov_t
        raw_cov_ffc += seg_cov_f
        raw_fp += seg_fp
        raw_fd_total += seg_fd_t
        raw_fd_ffc += seg_fd_f
        raw_overlap += seg_ov
        raw_overlap_ffc += seg_ov_f
        total_gt_points += len(g_dict)
            
        # Record per-loop coverage fraction
        loop_cov_totals.append(seg_cov_t / len(g_dict))
        loop_cov_ffcs.append(seg_cov_f / len(g_dict))
        
        # Track loops with at least one valid hit
        if seg_cov_t > 0: loops_with_hit_total += 1
        if seg_cov_f > 0: loops_with_hit_ffc += 1

    # Count unique matched predictions across all loops
    raw_frag_total = len(matched_pred_indices_total)
    raw_frag_ffc = len(matched_pred_indices_ffc)

    # Final normalization
    cov_total = raw_cov_total / total_gt_points if total_gt_points else 0.0
    cov_ffc = raw_cov_ffc / total_gt_points if total_gt_points else 0.0
    
    # Per-loop coverage (unweighted mean across loops)
    cov_total_per_loop = np.mean(loop_cov_totals) if loop_cov_totals else 0.0
    cov_ffc_per_loop = np.mean(loop_cov_ffcs) if loop_cov_ffcs else 0.0
    
    fd_total = (raw_fd_total / raw_overlap) * 100 if raw_overlap else np.nan
    fd_ffc = (raw_fd_ffc / raw_overlap_ffc) * 100 if raw_overlap_ffc else np.nan
    
    frag_t = float(raw_frag_total) if raw_frag_total > 0 else np.nan
    frag_f = float(raw_frag_ffc) if raw_frag_ffc > 0 else np.nan
    
    # True per-loop normalization
    frag_t_per_loop = np.nan if raw_frag_total == 0 else raw_frag_total / num_gt_loops
    frag_f_per_loop = np.nan if raw_frag_ffc == 0 else raw_frag_ffc / num_gt_loops
    
    # Loop-level recall
    recall_total = loops_with_hit_total / num_gt_loops if num_gt_loops > 0 else 0.0
    recall_ffc = loops_with_hit_ffc / num_gt_loops if num_gt_loops > 0 else 0.0

    return RawComparisonResult(
        coverage_total=cov_total, coverage_ffc=cov_ffc,
        coverage_total_per_loop=cov_total_per_loop, coverage_ffc_per_loop=cov_ffc_per_loop,
        false_pos=float(raw_fp),
        freq_diff_total=fd_total, freq_diff_ffc=fd_ffc,
        frag_total=frag_t, frag_ffc=frag_f,
        frag_total_per_loop=frag_t_per_loop, frag_ffc_per_loop=frag_f_per_loop,
        recall_total=recall_total, recall_ffc=recall_ffc,
        num_gt_loops=num_gt_loops, num_pred_contours=len(mat_contour)
    )

def compare_loops(
    ground: np.ndarray,
    contour: np.ndarray
) -> Tuple[float, int, float]:
    """Compares ground and predicted contours for coverage, false positives, and mean freq diff."""
    coverage = 0
    freq_diff = []

    for g_x, g_c in ground:
        for c_x, c_c in contour:
            if g_x == c_x:
                coverage += 1
                freq_diff.append(np.abs(g_c - c_c))

    false_pos = len(contour) - coverage
    coverage = coverage / len(ground) if len(ground) > 0 else 0.0
    mean_fd = np.mean(freq_diff) if freq_diff else 0.0
    return coverage, false_pos, mean_fd

"""
Evaluation Subpackage
Handles DTW alignment, metric calculations, and dataset generation.
"""
from .dtw_alignment import calculate_dtw_distance, compute_z_scores, run_statistical_tests, nearest_neighbor_test
from .comparison import compare_contours
from .metrics import aggregate_metrics, compute_fb_metrics
from .dataset_gen import make_white_noise, load_param_worker, generate_dtw_dataset

__all__ = [
    "calculate_dtw_distance",
    "compute_z_scores",
    "run_statistical_tests",
    "nearest_neighbor_test",
    "compare_contours",
    "aggregate_metrics",
    "compute_fb_metrics",
    "make_white_noise",
    "load_param_worker",
    "generate_dtw_dataset",
]

"""
Contour Extraction Subpackage
Contains algorithms and evaluation metrics for whistle contour extraction.
"""
from .crepe import CrepePredictor
from .precompute import run_precompute
from .silbido import run_silbido, run_smcphd, run_sam, _get_matlab_engine

__all__ = [
    "CrepePredictor",
    "run_precompute",
    "run_silbido",
    "run_smcphd",
    "run_sam",
    "_get_matlab_engine",
]

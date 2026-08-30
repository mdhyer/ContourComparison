import pytest
import numpy as np

try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

from audio_analysis.evaluation.dtw_alignment import calculate_dtw_distance
from audio_analysis.evaluation.comparison import compare_contours

@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestPerformance:
    """
    Benchmarks critical pipeline functions to detect regressions after refactoring.
    Run with: pytest tests/test_performance.py --benchmark-only
    """
    @pytest.fixture
    def synthetic_contours(self):
        np.random.seed(42)
        t = np.linspace(0, 5.0, 500)
        freq = 2000 + 500 * np.sin(2 * np.pi * 0.5 * t)
        return np.column_stack([t, freq])

    def test_dtw_alignment_benchmark(self, benchmark, synthetic_contours):
        c1 = synthetic_contours
        c2 = synthetic_contours + np.random.normal(0, 20, synthetic_contours.shape)
        benchmark(calculate_dtw_distance, c1, c2)

    def test_contour_comparison_benchmark(self, benchmark, synthetic_contours):
        ground = synthetic_contours
        contour = [synthetic_contours + np.random.normal(0, 20, synthetic_contours.shape)]
        benchmark(compare_contours, ground, contour)
        
    def test_large_scale_dtw_benchmark(self, benchmark):
        """Stress test DTW with larger, more realistic contour lengths."""
        np.random.seed(123)
        t = np.linspace(0, 10.0, 1000)
        freq = 1500 + 300 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 10, len(t))
        c1 = np.column_stack([t, freq])
        c2 = np.column_stack([t, freq + np.random.normal(0, 15, len(t))])
        
        benchmark(calculate_dtw_distance, c1, c2)

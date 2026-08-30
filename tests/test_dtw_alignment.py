import numpy as np
import pytest
from audio_analysis.evaluation.dtw_alignment import (
    calculate_dtw_distance, compute_z_scores, run_statistical_tests, nearest_neighbor_test
)

class TestDTWAlignment:
    def test_calculate_dtw_distance_identical(self):
        """Verify DTW distance is zero for identical contours."""
        c1 = np.array([[0, 100], [1, 200], [2, 300]])
        c2 = np.array([[0, 100], [1, 200], [2, 300]])
        assert calculate_dtw_distance(c1, c2) == pytest.approx(0.0)

    def test_calculate_dtw_distance_short_contour(self):
        """Verify NaN is returned for contours with fewer than 2 points."""
        c1 = np.array([[0, 100]])
        c2 = np.array([[0, 100], [1, 200]])
        assert np.isnan(calculate_dtw_distance(c1, c2))

    def test_compute_z_scores(self):
        """Verify Z-score calculation relative to ground truth."""
        results = {'Ground': [1.0, 2.0, 3.0], 'Algo1': [1.1, 2.1, 3.1]}
        z = compute_z_scores(results)
        assert 'Algo1' in z
        assert len(z['Algo1']) == 3

    def test_compute_z_scores_zero_std(self):
        """Verify fallback to small std when ground truth variance is zero."""
        results = {'Ground': [1.0, 1.0, 1.0], 'Algo1': [1.0, 1.0, 1.0]}
        z = compute_z_scores(results)
        assert np.all(z['Algo1'] == 0.0)

    def test_run_statistical_tests(self):
        """Verify KS and MWU tests return expected keys."""
        z1 = {'A': np.array([1, 2, 3])}
        z2 = {'A': np.array([1, 2, 3])}
        stats = run_statistical_tests(z1, z2)
        assert 'ks_p' in stats['A']
        assert 'mwu_p' in stats['A']
        assert 'mwu_stat' in stats['A']

    def test_nearest_neighbor_test(self):
        """Verify nearest neighbor ranking logic."""
        scores = [('fb1', 1.0), ('fb2', 2.0), ('fb3', 3.0)]
        assert nearest_neighbor_test(scores, 'fb1', top_k=1)
        assert not nearest_neighbor_test(scores, 'fb3', top_k=1)
        assert nearest_neighbor_test(scores, 'fb3', top_k=3)

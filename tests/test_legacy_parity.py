import pytest
import numpy as np
import sys
import os

# Add repository root to path to allow importing legacy scripts for comparison
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from silbido_test import compare_contours as legacy_compare_contours
    from silbido_test import compare_dicts as legacy_compare_dicts
    from silbido_test import compare_loops as legacy_compare_loops
    from legacy_utils import identify_harmonics as legacy_identify_harmonics
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

from audio_analysis.evaluation.comparison import compare_contours as modern_compare_contours
from audio_analysis.evaluation.comparison import compare_dicts as modern_compare_dicts
from audio_analysis.evaluation.comparison import compare_loops as modern_compare_loops
from audio_analysis.evaluation.dtw_alignment import calculate_dtw_distance as modern_dtw_distance
from audio_analysis.utils.contour_utils import identify_harmonics as modern_identify_harmonics

@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy modules not found in repository root")
class TestLegacyParity:
    """
    Verifies numerical parity between legacy scripts and the modernized package.
    Ensures refactoring did not alter core algorithmic outputs.
    """
    def test_compare_contours_parity(self):
        ground = np.array([[0.0, 100.0], [0.1, 110.0], [0.2, 120.0]])
        contour = [np.array([[0.0, 105.0], [0.1, 115.0], [0.2, 125.0]])]
        discont = []
        
        legacy_res = legacy_compare_contours(ground, contour, discont)
        modern_res = modern_compare_contours(ground, contour, discont)
        
        # Legacy returns (COVERAGE_FFC, FREQ_DIFF_FFC, FREQ_DIFF, FRAG_FFC) due to internal unpacking
        # Modern returns RawComparisonResult with explicit field names
        assert np.isclose(legacy_res[0], modern_res.coverage_ffc, atol=1e-5), "Coverage FFC mismatch"
        assert np.isclose(legacy_res[1], modern_res.freq_diff_ffc, atol=1e-5), "Freq Diff FFC mismatch"
        assert np.isclose(legacy_res[2], modern_res.freq_diff_total, atol=1e-5), "Freq Diff Total mismatch"

    def test_compare_dicts_parity(self):
        ground_dict = {0.0: 100.0, 0.1: 110.0, 0.2: 120.0}
        contour_dicts = [{0.0: 105.0, 0.1: 115.0, 0.2: 125.0}]
        
        legacy_res = legacy_compare_dicts(ground_dict, contour_dicts)
        modern_res = modern_compare_dicts(ground_dict, contour_dicts)
        
        # Legacy returns (COVERAGE_FFC, FREQ_DIFF_FFC, FREQ_DIFF, FRAG_FFC)
        assert np.isclose(legacy_res[0], modern_res.coverage_ffc, atol=1e-5)
        assert np.isclose(legacy_res[1], modern_res.freq_diff_ffc, atol=1e-5)
        assert np.isclose(legacy_res[2], modern_res.freq_diff_total, atol=1e-5)

    def test_compare_loops_parity(self):
        ground = np.array([[0.0, 100.0], [0.1, 110.0], [0.2, 120.0]])
        contour = np.array([[0.0, 105.0], [0.1, 115.0], [0.2, 125.0]])
        
        legacy_res = legacy_compare_loops(ground, contour)
        modern_res = modern_compare_loops(ground, contour)
        
        assert np.isclose(legacy_res[0], modern_res[0], atol=1e-5)
        assert np.isclose(legacy_res[1], modern_res[1], atol=1e-5)
        assert np.isclose(legacy_res[2], modern_res[2], atol=1e-5)

    def test_dtw_distance_known_values(self):
        """Verify DTW distance calculation against known mathematical properties."""
        c1 = np.array([[0, 100], [1, 200], [2, 300]])
        c2 = np.array([[0, 100], [1, 200], [2, 300]])
        
        dist = modern_dtw_distance(c1, c2)
        assert np.isclose(dist, 0.0, atol=1e-5), "Identical contours should yield zero distance"
        
    def test_dtw_distance_symmetry(self):
        """Verify DTW distance is symmetric."""
        np.random.seed(42)
        c1 = np.column_stack([np.linspace(0, 1, 50), np.random.rand(50) * 1000])
        c2 = np.column_stack([np.linspace(0, 1, 50), np.random.rand(50) * 1000])
        
        dist_12 = modern_dtw_distance(c1, c2)
        dist_21 = modern_dtw_distance(c2, c1)
        
        assert np.isclose(dist_12, dist_21, atol=1e-5), "DTW distance should be symmetric"

    def test_identify_harmonics_parity(self):
        """Verify that modern identify_harmonics matches legacy behavior."""
        # Create a fundamental contour (100Hz)
        fundamental = np.array([
            [0.0, 100.0], [0.1, 100.0], [0.2, 100.0], [0.3, 100.0], [0.4, 100.0]
        ])

        # Create a harmonic contour (200Hz) - should be removed
        harmonic = np.array([
            [0.0, 200.0], [0.1, 200.0], [0.2, 200.0], [0.3, 200.0], [0.4, 200.0]
        ])

        # Create a non-harmonic contour (150Hz) - should be kept
        non_harmonic = np.array([
            [0.0, 150.0], [0.1, 150.0], [0.2, 150.0], [0.3, 150.0], [0.4, 150.0]
        ])

        input_contours = [fundamental, harmonic, non_harmonic]

        # Run both versions
        legacy_res = legacy_identify_harmonics(input_contours, tolerance=0.7, freqdiff=0.2)
        modern_res = modern_identify_harmonics(input_contours, tolerance=0.7, freqdiff=0.2)

        # Both should return 2 contours (Fundamental + Non-harmonic)
        assert len(legacy_res) == len(modern_res) == 2, \
            f"Length mismatch: Legacy returned {len(legacy_res)}, Modern returned {len(modern_res)}"

        # Verify the harmonic (200Hz) was removed in both
        # We check if any returned contour has a mean frequency close to 200
        legacy_has_harmonic = any(np.mean(c[:, 1]) > 190 for c in legacy_res)
        modern_has_harmonic = any(np.mean(c[:, 1]) > 190 for c in modern_res)

        assert not legacy_has_harmonic, "Legacy failed to remove harmonic"
        assert not modern_has_harmonic, "Modern failed to remove harmonic"

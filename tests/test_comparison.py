import numpy as np
import pytest
from audio_analysis.evaluation.comparison import compare_dicts, compare_contour, compare_contours, compare_loops, RawComparisonResult

class TestCompareDicts:
    def test_perfect_match(self):
        """Verify metrics when predictions perfectly match ground truth."""
        ground = {0.1: 1000.0, 0.2: 1000.0, 0.3: 1000.0}
        preds = [{0.1: 1000.0, 0.2: 1000.0, 0.3: 1000.0}]
        res = compare_dicts(ground, preds)
        
        assert res.coverage_total == 1.0
        assert res.coverage_ffc == 1.0
        assert res.freq_diff_total == 0.0
        assert res.freq_diff_ffc == 0.0
        assert res.false_pos == 0.0
        assert res.frag_total == 1.0
        assert res.frag_ffc == 1.0
        assert res.num_pred_contours == 1

    def test_partial_match_with_ffc(self):
        """Verify FFC metrics filter out predictions exceeding the 25% threshold."""
        ground = {0.1: 1000.0, 0.2: 1000.0, 0.3: 1000.0}
        # 0.1 matches perfectly, 0.2 matches within 25%, 0.3 misses
        preds = [{0.1: 1000.0, 0.2: 1200.0}]
        res = compare_dicts(ground, preds)
        
        assert res.coverage_total == pytest.approx(2/3)
        assert res.coverage_ffc == pytest.approx(2/3)
        assert res.freq_diff_total > 0
        assert res.freq_diff_ffc > 0
        assert res.false_pos == 0.0

    def test_false_positives(self):
        """Verify false positive calculation when predictions exceed ground truth points."""
        ground = {0.1: 1000.0}
        preds = [{0.1: 1000.0, 0.2: 1000.0, 0.3: 1000.0}]
        res = compare_dicts(ground, preds)
        
        assert res.false_pos == 2.0
        assert res.coverage_total == 1.0

    def test_empty_ground(self):
        """Verify graceful handling of empty ground truth dictionary."""
        ground = {}
        preds = [{0.1: 1000.0}]
        res = compare_dicts(ground, preds)
        
        assert res.coverage_total == 0.0
        assert np.isnan(res.freq_diff_total)
        assert res.false_pos == 1.0

    def test_multiple_contours_fragmentation(self):
        """Verify fragmentation counts across multiple prediction contours."""
        ground = {0.1: 1000.0, 0.2: 1000.0}
        preds = [
            {0.1: 1000.0},
            {0.2: 1000.0}
        ]
        res = compare_dicts(ground, preds)
        
        assert res.frag_total == 2.0
        assert res.num_pred_contours == 2

    def test_zero_ground_freq(self):
        """Verify graceful handling of zero frequency in ground truth (division-by-zero guard)."""
        ground = {0.1: 0.0, 0.2: 1000.0}
        preds = [{0.1: 0.0, 0.2: 1000.0}]
        res = compare_dicts(ground, preds)
        
        # Should not crash, and should handle the zero freq point gracefully
        assert res.coverage_total == 1.0
        assert not np.isnan(res.freq_diff_total)

class TestCompareContour:
    def test_basic_segment(self):
        """Verify single segment comparison with perfect match."""
        ground = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        pred = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        res = compare_contour(0.0, 0.2, ground, [pred])
        
        assert res.coverage_total == 1.0
        assert res.freq_diff_total == 0.0

    def test_out_of_range_freq(self):
        """Verify predictions > 24kHz are filtered out before comparison."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        pred = np.array([[0.0, 25000], [0.1, 25000]])
        res = compare_contour(0.0, 0.1, ground, [pred])
        
        assert res.coverage_total == 0.0
        assert np.isnan(res.freq_diff_total)

    def test_insufficient_ground_points(self):
        """Verify early return when ground truth has fewer than 2 points."""
        ground = np.array([[0.0, 1000]])
        pred = np.array([[0.0, 1000]])
        res = compare_contour(0.0, 0.0, ground, [pred])
        
        assert res.coverage_total == 0.0
        assert np.isnan(res.freq_diff_total)

class TestCompareContours:
    def test_full_comparison(self):
        """Verify full contour comparison delegates correctly."""
        ground = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        pred = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        res = compare_contours(ground, [pred])
        
        assert res.coverage_total == 1.0
        assert res.num_pred_contours == 1

    def test_empty_predictions(self):
        """Verify empty prediction list returns zero metrics."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        res = compare_contours(ground, [])
        
        assert res.coverage_total == 0.0
        assert res.num_pred_contours == 0
        assert res.false_pos == 0.0

class TestCompareContoursWithDiscontinuities:
    def test_explicit_discont_splits_loops(self):
        """Verify that explicit discontinuities correctly split ground truth into loops."""
        # Ground truth spans 0.0 to 1.0
        ground = np.array([[t/10.0, 1000.0] for t in range(11)])
        # Discontinuity at 0.5 splits into 2 loops
        discont = [[0.5, 0.5]]
        pred = [ground]
        
        res = compare_contours(ground, pred, discont=discont)
        assert res.num_gt_loops >= 2
        # 1 predicted contour covers 2 loops -> frag = 0.5 per loop (1/2)
        assert res.frag_total_per_loop == pytest.approx(0.5)
        assert res.frag_ffc_per_loop == pytest.approx(0.5)

    def test_discont_pair_format(self):
        """Verify handling of legacy [end, start] pair format for discontinuities."""
        ground = np.array([[t/10.0, 1000.0] for t in range(11)])
        # Legacy format: [end_of_prev, start_of_next]
        discont = [[0.3, 0.4], [0.7, 0.8]]
        pred = [ground]
        
        res = compare_contours(ground, pred, discont=discont)
        assert res.num_gt_loops == 3
        # 1 predicted contour covers 3 loops -> frag ≈ 0.333 per loop (1/3)
        assert res.frag_total_per_loop == pytest.approx(1/3)

    def test_ffc_fragmentation_per_loop(self):
        """Verify FFC fragmentation only counts contours within frequency threshold."""
        ground = np.array([[t/10.0, 1000.0] for t in range(11)])
        discont = [[0.5, 0.5]]
        # Pred 1: perfect match
        pred1 = ground
        # Pred 2: >25% freq error everywhere
        pred2 = np.array([[t/10.0, 1500.0] for t in range(11)])
        
        res = compare_contours(ground, [pred1, pred2], discont=discont)
        assert res.num_gt_loops >= 2
        # Total frag: 2 unique contours match -> 1.0 per loop (2/2)
        assert res.frag_total_per_loop == pytest.approx(1.0)
        # FFC frag: 1 unique contour within threshold -> 0.5 per loop (1/2)
        assert res.frag_ffc_per_loop == pytest.approx(0.5)

class TestCompareLoops:
    def test_perfect_overlap(self):
        """Verify compare_loops with identical contours."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        contour = np.array([[0.0, 1000], [0.1, 1000]])
        cov, fp, fd = compare_loops(ground, contour)
        assert cov == 1.0
        assert fp == 0
        assert fd == 0.0

    def test_partial_overlap(self):
        """Verify compare_loops with partial time overlap."""
        ground = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        contour = np.array([[0.0, 1050], [0.1, 1000]])
        cov, fp, fd = compare_loops(ground, contour)
        assert cov == pytest.approx(2/3)
        assert fp == 0
        assert fd > 0

    def test_empty_contour(self):
        """Verify compare_loops with empty prediction contour."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        contour = np.array([]).reshape(0, 2)
        cov, fp, fd = compare_loops(ground, contour)
        assert cov == 0.0
        assert fp == 0
        assert fd == 0.0


class TestCompareContoursEdgeCases:
    def test_empty_ground_truth(self):
        """Verify compare_contours handles empty ground truth (covers line 220)."""
        ground = np.array([]).reshape(0, 2)
        pred = [np.array([[0.0, 1000]])]
        res = compare_contours(ground, pred)
        assert res.coverage_total == 0.0
        assert res.num_gt_loops == 0

    def test_all_predictions_filtered_out(self):
        """Verify compare_contours handles predictions filtered out by >24kHz threshold (covers line 182)."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        pred = [np.array([[0.0, 25000], [0.1, 25000]])]
        res = compare_contours(ground, pred)
        assert res.coverage_total == 0.0
        assert res.num_pred_contours == 1

    def test_segment_with_single_point(self):
        """Verify compare_contours handles segments split into < 2 points (filtered out by _split_ground_truth)."""
        ground = np.array([[0.0, 1000], [0.1, 1000]])
        # Discontinuity at 0.05 splits into [0.0] and [0.1], both have 1 point
        discont = [[0.05, 0.05]]
        pred = [ground]
        res = compare_contours(ground, pred, discont=discont)
        # Both segments are filtered out by _split_ground_truth, resulting in 0 loops and zero coverage
        assert res.coverage_total == 0.0
        assert res.num_gt_loops == 0

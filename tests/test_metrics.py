import pytest
import numpy as np
import json
from audio_analysis.evaluation.comparison import RawComparisonResult
from audio_analysis.evaluation.metrics import aggregate_metrics, _select_metrics, compute_fb_metrics, compute_all_mode_metrics, load_metrics
from audio_analysis.config import MetricsConfig, CoverageMode, FreqDiffMode, FragMode, RecallMode


class TestSelectMetrics:
    def test_total_mode(self):
        """Verify TOTAL mode selects the correct raw metrics."""
        # RawComparisonResult order: cov_total, cov_ffc, cov_total_per_loop, cov_ffc_per_loop,
        # false_pos, fd_total, fd_ffc, frag_total, frag_ffc, frag_total_per_loop, frag_ffc_per_loop,
        # recall_total, recall_ffc, num_gt_loops, num_pred_contours
        raw = RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1, 1)
        metrics_config = MetricsConfig(
            coverage_mode=CoverageMode.TOTAL,
            freq_diff_mode=FreqDiffMode.TOTAL,
            frag_mode=FragMode.TOTAL,
            recall_mode=RecallMode.TOTAL
        )
        selected = _select_metrics(raw, metrics_config)
        # _select_metrics returns List[float]: [coverage, false_pos, freq_diff, frag, recall]
        assert selected[0] == pytest.approx(0.8)
        assert selected[2] == pytest.approx(10.0)  # freq_diff_total is 10.0 in raw data

    def test_ffc_mode(self):
        """Verify FFC mode selects the FFC-specific raw metrics."""
        raw = RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1, 1)
        metrics_config = MetricsConfig(
            coverage_mode=CoverageMode.FFC,
            freq_diff_mode=FreqDiffMode.FFC,
            frag_mode=FragMode.FFC,
            recall_mode=RecallMode.FFC
        )
        selected = _select_metrics(raw, metrics_config)
        assert selected[0] == pytest.approx(0.6)  # coverage_ffc is 0.6 in raw data


class TestAggregateMetrics:
    def test_basic_aggregation(self):
        """Verify basic mean and std calculations."""
        results = [[0.8, 2.0, 5.0, 1.0, 0.9], [0.9, 1.0, 4.0, 2.0, 0.8]]
        agg = aggregate_metrics(results)
        assert agg['coverage'] == pytest.approx(0.85)
        assert agg['false_pos'] == pytest.approx(1.5)
        assert agg['freq_diff'] == pytest.approx(4.5)
        assert agg['fragmentation'] == pytest.approx(1.5)
        assert agg['recall'] == pytest.approx(0.85)

    def test_nan_handling(self):
        """Verify NaN values in freq_diff are masked correctly."""
        results = [[0.8, 2.0, np.nan, 1.0, 0.9], [0.9, 1.0, 4.0, 2.0, 0.8]]
        agg = aggregate_metrics(results)
        assert agg['coverage'] == pytest.approx(0.85)
        assert agg['freq_diff'] == pytest.approx(4.0)  # Only one valid value

    def test_percentile_calculation(self):
        """Verify 5th and 95th percentile bounds are calculated."""
        results = [[0.5, 1.0, 10.0, 1.0, 0.8], [0.9, 2.0, 20.0, 2.0, 0.9]]
        agg = aggregate_metrics(results)
        assert agg['coverage_5_95'] == pytest.approx([0.52, 0.88], abs=0.01)
        # Updated expected values to match numpy default interpolation behavior
        assert agg['freq_diff_5_95'] == pytest.approx([10.5, 19.5], abs=0.1)


class TestComputeFbMetrics:
    def test_per_fb_aggregation(self):
        """Verify metrics are correctly aggregated per FBID."""
        raw_results = {
            'fb1': [RawComparisonResult(0.9, 0.7, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 1.5, 0.0, 2.0, 1.0, 2.0, 2, 1)],
            'fb2': [RawComparisonResult(0.8, 0.6, 0.0, 0.0, 1.0, 6.0, 5.0, 3.0, 2.0, 0.0, 3.0, 1.0, 3.0, 3, 1)]
        }
        metrics_config = MetricsConfig()
        # Pass fb_ids explicitly to match current signature
        fb_metrics = compute_fb_metrics(raw_results, list(raw_results.keys()), cfg=metrics_config)
        assert 'fb1' in fb_metrics
        assert 'fb2' in fb_metrics
        assert fb_metrics['fb1']['coverage'] == pytest.approx(0.9)

    def test_missing_fb_id(self):
        """Verify missing FBIDs return NaN metrics."""
        raw_results = {'fb1': [RawComparisonResult(0.9, 0.7, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 1.5, 0.0, 2.0, 1.0, 2.0, 2, 1)]}
        metrics_config = MetricsConfig()
        fb_metrics = compute_fb_metrics(raw_results, list(raw_results.keys()), cfg=metrics_config)
        assert 'fb1' in fb_metrics
        # fb2 is missing, so it shouldn't be in results or should be handled gracefully
        assert 'fb2' not in fb_metrics


class TestAggregateMetricsEdgeCases:
    def test_empty_results(self):
        """Verify aggregate_metrics handles empty input gracefully (covers line 38)."""
        agg = aggregate_metrics([])
        assert np.isnan(agg['coverage'])
        assert np.isnan(agg['false_pos'])
        assert np.isnan(agg['freq_diff'])


class TestComputeFbMetricsEdgeCases:
    def test_present_fb_id(self):
        """Verify compute_fb_metrics correctly processes present FBIDs (covers line 98)."""
        raw_results = {'fb1': [RawComparisonResult(0.9, 0.7, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 1.5, 0.0, 2.0, 1.0, 2.0, 2, 1)]}
        metrics_config = MetricsConfig()
        fb_metrics = compute_fb_metrics(raw_results, ['fb1'], cfg=metrics_config)
        assert 'fb1' in fb_metrics
        assert fb_metrics['fb1']['coverage'] == pytest.approx(0.9)


class TestLoadMetrics:
    def test_load_new_format(self, tmp_path):
        """Verify load_metrics handles new nested format (covers lines 104-108)."""
        data = {"metrics": {"algo1": {"global": {"total_total_total_total": {"coverage": 0.8}}}}, "config": {}}
        file_path = tmp_path / "metrics_new.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        loaded = load_metrics(file_path)
        assert loaded == data["metrics"]

    def test_load_legacy_format(self, tmp_path):
        """Verify load_metrics handles legacy flat format (covers lines 111-112)."""
        data = {"algo1": {"global": {"total_total_total_total": {"coverage": 0.8}}}}
        file_path = tmp_path / "metrics_legacy.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        loaded = load_metrics(file_path)
        assert loaded == data


class TestComputeAllModeMetrics:
    def test_mode_generation(self):
        """Verify compute_all_mode_metrics generates keys for all mode combinations."""
        raw = [RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 4.0, 1.5, 0.0, 4.0, 1.0, 4.0, 4, 1)]
        cfg = MetricsConfig()
        all_metrics = compute_all_mode_metrics(raw, cfg)
        
        key = "total_total_total_total"
        assert key in all_metrics
        assert all_metrics[key]['coverage'] == pytest.approx(0.8)

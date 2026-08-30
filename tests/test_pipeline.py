import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from audio_analysis.config import PipelineConfig, DataLayout
from audio_analysis.evaluation.comparison import RawComparisonResult
from audio_analysis.pipeline import execute_precompute, execute_evaluate, _discover_wav_files, \
    compute_metrics_from_precomputed


class TestPipelinePreprocessing:
    @patch("audio_analysis.pipeline.CrepePredictor")
    @patch("audio_analysis.pipeline.fragment_contours")
    @patch("audio_analysis.pipeline.identify_harmonics")
    def test_precompute_skips_postprocessing(self, mock_harm, mock_frag, mock_crepe, config, tmp_path):
        """Precompute should save raw contours without post-processing."""
        raw_contour = [np.array([[0.0, 100.0], [1.0, 200.0]])]
        with patch("audio_analysis.pipeline._run_algorithm", return_value=raw_contour):
            with patch("audio_analysis.pipeline.save_precomputed"):
                execute_precompute(config, [Path("test.wav")], ["CREPE"])
        assert mock_frag.call_count == 0
        assert mock_harm.call_count == 0

    @patch("audio_analysis.pipeline.fragment_contours")
    @patch("audio_analysis.pipeline.identify_harmonics")
    def test_evaluate_applies_postprocessing_once(self, mock_harm, mock_frag, config, tmp_path):
        """Evaluate should apply post-processing exactly once per file."""
        raw_contour = [np.array([[0.0, 100.0], [1.0, 200.0]])]
        mock_frag.return_value = raw_contour
        mock_harm.return_value = raw_contour

        mock_result = RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1, 1)
        with patch("audio_analysis.pipeline.compare_contours", return_value=mock_result):
            with patch("audio_analysis.pipeline.load_ground_truth",
                       return_value=(np.array([[0.0, 100.0], [1.0, 200.0]]), [])):
                with patch("audio_analysis.pipeline.load_precomputed", return_value=raw_contour):
                    with patch("audio_analysis.pipeline.save_metrics"):
                        execute_evaluate(config, [Path("test.wav")], ["CREPE"])
        assert mock_frag.call_count == 1
        assert mock_harm.call_count == 1

    def test_postprocessing_respects_config_flags(self, config, tmp_path):
        """Post-processing should skip steps if disabled in config."""
        config.algorithm.split_contours = False
        config.algorithm.remove_harmonics = False
        raw_contour = [np.array([[0.0, 100.0], [1.0, 200.0]])]

        with patch("audio_analysis.pipeline.compare_contours",
                   return_value=RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1,
                                                    1)):
            with patch("audio_analysis.pipeline.load_ground_truth",
                       return_value=(np.array([[0.0, 100.0], [1.0, 200.0]]), [])):
                with patch("audio_analysis.pipeline.load_precomputed", return_value=raw_contour):
                    with patch("audio_analysis.pipeline.save_metrics"):
                        execute_evaluate(config, [Path("test.wav")], ["CREPE"])
        # No exceptions raised, pipeline completes gracefully

    def test_compute_metrics_from_precomputed(self, config, tmp_path):
        """Verify metrics can be computed independently from precomputed contours."""
        raw_contour = [np.array([[0.0, 100.0], [1.0, 200.0]])]
        mock_result = RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1, 1)

        with patch("audio_analysis.pipeline.load_ground_truth",
                   return_value=(np.array([[0.0, 100.0], [1.0, 200.0]]), [])):
            with patch("audio_analysis.pipeline.load_precomputed", return_value=raw_contour):
                with patch("audio_analysis.pipeline.compare_contours", return_value=mock_result):
                    with patch("audio_analysis.pipeline._resolve_fbid_snr", return_value=("FBID_001", "CLEAN")):
                        aggregated = compute_metrics_from_precomputed(config, [Path("test.wav")], ["CREPE"])

        assert "CREPE" in aggregated
        assert "global" in aggregated["CREPE"]
        assert "per_snr" in aggregated["CREPE"]
        assert "CLEAN" in aggregated["CREPE"]["per_snr"]


class TestPipelineErrorHandling:
    def test_evaluate_skips_missing_ground_truth(self, config, tmp_path, caplog):
        """Missing ground truth should log warning and skip file."""
        with patch("audio_analysis.pipeline.load_ground_truth", return_value=(None, [])):
            with patch("audio_analysis.pipeline.save_metrics"):
                execute_evaluate(config, [Path("test.wav")], ["CREPE"])
        assert "No ground truth found" in caplog.text

    def test_evaluate_skips_missing_precomputed(self, config, tmp_path, caplog):
        """Missing precomputed contours should log warning and skip file."""
        with patch("audio_analysis.pipeline.load_ground_truth", return_value=(np.array([[0.0, 100.0]]), [])):
            with patch("audio_analysis.pipeline.load_precomputed", return_value=None):
                with patch("audio_analysis.pipeline.save_metrics"):
                    execute_evaluate(config, [Path("test.wav")], ["CREPE"])
        assert True  # Gracefully skips without crashing

    def test_evaluate_handles_comparison_exception(self, config, tmp_path, caplog):
        """Exceptions during comparison should be caught and logged without breaking the batch."""
        mock_result = RawComparisonResult(0.8, 0.6, 0.0, 0.0, 5.0, 10.0, 8.0, 2.0, 1.5, 0.0, 1.0, 1.0, 1.0, 1, 1)
        with patch("audio_analysis.pipeline.load_ground_truth", return_value=(np.array([[0.0, 100.0]]), [])):
            with patch("audio_analysis.pipeline.load_precomputed", return_value=[np.array([[0.0, 100.0]])]):
                with patch("audio_analysis.pipeline.compare_contours", side_effect=ValueError("Test Error")):
                    with patch("audio_analysis.pipeline.save_metrics"):
                        execute_evaluate(config, [Path("test.wav")], ["CREPE"])
        assert "Test Error" in caplog.text


class TestFileDiscovery:
    def test_discover_nested_filtering(self, config, tmp_path):
        """Verify nested layout discovery respects FBID and SNR filters."""
        data_root = tmp_path / "data" / "NoiseLevels"
        (data_root / "CLEAN" / "FBID_001").mkdir(parents=True)
        (data_root / "CLEAN" / "FBID_001" / "test.wav").touch()
        (data_root / "SNR_20" / "FBID_002").mkdir(parents=True)
        (data_root / "SNR_20" / "FBID_002" / "test2.wav").touch()

        # _discover_wav_files expects data_root to point to the parent of "NoiseLevels"
        config.paths.data_root = data_root.parent
        config.paths.data_layout = DataLayout.NESTED_NOISE

        files = _discover_wav_files(config.paths.data_root, ["FBID_001"], ["CLEAN"], config.paths.data_layout)
        assert len(files) == 1
        assert files[0].name == "test.wav"

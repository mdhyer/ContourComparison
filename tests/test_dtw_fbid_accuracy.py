"""
Tests for DTW FBID Accuracy pipeline and visualization utilities.
Mocks file discovery, contour loading, and post-processing to isolate core logic.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from audio_analysis.evaluation.dtw_fbid_accuracy import (
    run_fbid_accuracy_pipeline,
    plot_violin,
    plot_boxplot
)
from audio_analysis.config import PipelineConfig, PathsConfig


def _make_synthetic_contour(n_points: int = 20) -> np.ndarray:
    """Generate a valid (time, freq) contour array for testing."""
    t = np.linspace(0, 1.0, n_points)
    f = 1000 + 200 * np.sin(2 * np.pi * t)
    return np.column_stack([t, f])


def _create_mock_dirs(fbids: list[str]) -> list[MagicMock]:
    """Create mock directory objects with proper .name and .is_dir() behavior."""
    mock_dirs = []
    for fb in fbids:
        m = MagicMock()
        m.name = fb
        m.is_dir.return_value = True
        mock_dirs.append(m)
    return mock_dirs


class TestDtwFbidAccuracy:
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a temporary directory structure and PipelineConfig."""
        data_root = tmp_path / "data"
        data_root.mkdir()
        precompute_dir = tmp_path / "precompute"
        precompute_dir.mkdir()
        params_dir = tmp_path / "params"
        params_dir.mkdir()
        return PipelineConfig(paths=PathsConfig(
            data_root=data_root,
            precompute_dir=precompute_dir,
            params_dir=params_dir
        ))

    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.glob.glob')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.load_precomputed')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.run_ground')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.fragment_contours')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.identify_harmonics')
    def test_run_pipeline_basic(self, mock_id_harm, mock_frag, mock_run_ground, mock_load, mock_glob, mock_config):
        """Verify pipeline structure, metric aggregation, and progress callback."""
        fbids = ["FB1", "FB2"]
        mock_dirs = _create_mock_dirs(fbids)
        
        wav_paths = [str(mock_config.paths.data_root / fb / f"test_{i}.wav") for fb in fbids for i in range(5)]
        mock_glob.return_value = wav_paths

        contour = _make_synthetic_contour(20)
        mock_load.return_value = [contour]
        mock_run_ground.return_value = ([contour], None)
        mock_frag.return_value = [contour]
        mock_id_harm.return_value = [contour]

        progress_calls = []
        def progress_cb(curr, total, msg):
            progress_calls.append((curr, total, msg))

        with patch.object(Path, 'iterdir', return_value=mock_dirs):
            result = run_fbid_accuracy_pipeline(
                config=mock_config,
                algorithms=["Algo1"],
                n_ground=2,
                n_comparisons=2,
                progress_callback=progress_cb
            )

        assert 'results' in result
        assert 'metrics' in result
        assert 'acc_plot_data' in result
        assert result['fbids'] == fbids
        assert 'Algo1' in result['metrics']
        assert all(k in result['metrics']['Algo1'] for k in ['top1', 'top5', 'top20', 'fb_metrics'])
        assert len(progress_calls) > 0
        plt.close('all')

    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.glob.glob')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.load_precomputed')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.run_ground')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.fragment_contours')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.identify_harmonics')
    def test_run_pipeline_empty_contours(self, mock_id_harm, mock_frag, mock_run_ground, mock_load, mock_glob, mock_config):
        """Verify graceful handling when post-processing yields empty contours."""
        fbids = ["FB1"]
        mock_dirs = _create_mock_dirs(fbids)
        wav_paths = [str(mock_config.paths.data_root / fb / f"test_{i}.wav") for fb in fbids for i in range(3)]
        mock_glob.return_value = wav_paths

        mock_load.return_value = []
        mock_run_ground.return_value = ([_make_synthetic_contour(20)], None)
        mock_frag.return_value = []
        mock_id_harm.return_value = []

        with patch.object(Path, 'iterdir', return_value=mock_dirs):
            result = run_fbid_accuracy_pipeline(
                config=mock_config,
                algorithms=["Algo1"],
                n_ground=1,
                n_comparisons=2
            )

        assert result['metrics']['Algo1']['top1'] == 0
        assert result['metrics']['Algo1']['top5'] == 0
        plt.close('all')

    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.glob.glob')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.load_precomputed')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.run_ground')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.fragment_contours')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.identify_harmonics')
    def test_run_pipeline_invalid_contour_shapes(self, mock_id_harm, mock_frag, mock_run_ground, mock_load, mock_glob, mock_config):
        """Verify pipeline skips invalid contours (1D arrays, single points) without crashing."""
        fbids = ["FB1"]
        mock_dirs = _create_mock_dirs(fbids)
        # FIX: Provide enough WAVs for n_ground + n_comparisons
        wav_paths = [str(mock_config.paths.data_root / fb / f"test_{i}.wav") for fb in fbids for i in range(3)]
        mock_glob.return_value = wav_paths

        invalid_contours = [np.array([1.0, 2.0]), np.array([[0.0, 1000.0]])]
        mock_load.return_value = invalid_contours
        mock_run_ground.return_value = ([_make_synthetic_contour(20)], None)
        mock_frag.return_value = invalid_contours
        mock_id_harm.return_value = invalid_contours

        with patch.object(Path, 'iterdir', return_value=mock_dirs):
            result = run_fbid_accuracy_pipeline(
                config=mock_config,
                algorithms=["Algo1"],
                n_ground=1,
                n_comparisons=1
            )

        assert 'Algo1' in result['metrics']
        plt.close('all')

    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.glob.glob')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.load_precomputed')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.run_ground')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.fragment_contours')
    @patch('audio_analysis.evaluation.dtw_fbid_accuracy.identify_harmonics')
    def test_run_pipeline_ground_algorithm(self, mock_id_harm, mock_frag, mock_run_ground, mock_load, mock_glob, mock_config):
        """Verify 'Ground' algorithm uses run_ground instead of load_precomputed."""
        fbids = ["FB1"]
        mock_dirs = _create_mock_dirs(fbids)
        # FIX: Provide enough WAVs for n_ground + n_comparisons
        wav_paths = [str(mock_config.paths.data_root / fb / f"test_{i}.wav") for fb in fbids for i in range(3)]
        mock_glob.return_value = wav_paths

        contour = _make_synthetic_contour(20)
        mock_run_ground.return_value = ([contour], None)
        mock_frag.return_value = [contour]
        mock_id_harm.return_value = [contour]

        with patch.object(Path, 'iterdir', return_value=mock_dirs):
            result = run_fbid_accuracy_pipeline(
                config=mock_config,
                algorithms=["Ground"],
                n_ground=1,
                n_comparisons=1
            )

        # load_precomputed should NOT be called for Ground algorithm
        mock_load.assert_not_called()
        assert 'Ground' in result['metrics']
        plt.close('all')

    def test_plot_violin(self):
        """Verify violin plot generation and axis labels."""
        data = {'Algo1': [0.8, 0.9, 0.7], 'Algo2': [0.6, 0.5, 0.8]}
        algos = ['Algo1', 'Algo2']
        fig = plot_violin(data, algos)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == 'Algorithm'
        assert fig.axes[0].get_ylabel() == 'FBID Accuracy'
        plt.close(fig)

    def test_plot_boxplot(self):
        """Verify box plot generation and axis labels."""
        data = {'Algo1': [0.8, 0.9, 0.7], 'Algo2': [0.6, 0.5, 0.8]}
        algos = ['Algo1', 'Algo2']
        fig = plot_boxplot(data, algos)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == 'Algorithm'
        assert fig.axes[0].get_ylabel() == 'FBID Accuracy'
        plt.close(fig)

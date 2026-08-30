import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pathlib import Path

from audio_analysis.plotting.plotting import (
    _ensure_contour_list, _plot_spectrogram, _apply_limits_to_axes,
    plot_results, fbid_plot, plot_fbid_trends, visualize_together,
    visualize_prediction, plot_metrics_mosaic, plot_preprocessing,
    plot_metrics_mosaic_v2, plot_metric_violin, plot_fragmentation_verification
)
from audio_analysis.config import PipelineConfig, MetricsConfig, CoverageMode, FragMode, FreqDiffMode


@pytest.fixture(autouse=True)
def close_figures():
    """Prevent figure accumulation during test runs."""
    yield
    plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# MOCKS (Accept *args, **kwargs to match actual function signatures)
# ─────────────────────────────────────────────────────────────────────────────
def _mock_audio(*args, **kwargs):
    return np.random.randn(96000), 96000


def _mock_stft(*args, **kwargs):
    f = np.linspace(0, 48000, 513)
    t = np.linspace(0, 1, 100)
    Zxx = np.random.randn(513, 100) + 1j * np.random.randn(513, 100)
    return f, t, Zxx


def _mock_ground(*args, **kwargs):
    gt = np.column_stack([np.linspace(0, 1, 50), np.full(50, 10000)])
    return [gt], []


def _mock_precomputed(*args, **kwargs):
    pred = np.column_stack([np.linspace(0, 1, 50), np.full(50, 10000)])
    return [pred]


def _mock_fragment(*args, **kwargs):
    return args[0] if args else []


def _mock_harmonics(*args, **kwargs):
    return args[0] if args else []


def _mock_butter(*args, **kwargs):
    return np.ones(5), np.ones(5)


def _mock_filtfilt(*args, **kwargs):
    return kwargs.get('x', args[2] if len(args) > 2 else np.zeros(100))


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class TestHelpers:
    def test_ensure_contour_list_none(self):
        assert _ensure_contour_list(None) == []

    def test_ensure_contour_list_empty(self):
        assert _ensure_contour_list([]) == []

    def test_ensure_contour_list_single_array(self):
        arr = np.column_stack([np.linspace(0, 1, 10), np.full(10, 1000)])
        result = _ensure_contour_list(arr)
        assert len(result) == 1 and np.array_equal(result[0], arr)

    def test_ensure_contour_list_list_of_arrays(self):
        c1 = np.column_stack([np.linspace(0, 0.5, 10), np.full(10, 1000)])
        c2 = np.column_stack([np.linspace(0.5, 1, 10), np.full(10, 1200)])
        result = _ensure_contour_list([c1, c2])
        assert len(result) == 2

    def test_ensure_contour_list_invalid_shapes(self):
        assert _ensure_contour_list([np.array([1, 2, 3])]) == []

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    def test_plot_spectrogram_basic(self, mock_stft, mock_sf):
        fig, ax = plt.subplots()
        _plot_spectrogram(ax, "dummy.wav", "CLEAN", show_title=True)
        assert ax.get_title() == "SNR: 0 dB"

    def test_apply_limits_to_axes(self):
        fig, axs = plt.subplots(2, 2)
        limits = {'Coverage': {'x': (0, 1), 'y': (0, 10)}}
        _apply_limits_to_axes({'Coverage': axs[0, 0]}, limits)
        assert axs[0, 0].get_xlim() == (0, 1)
        assert axs[0, 0].get_ylim() == (0, 10)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class TestPlottingFunctions:
    @pytest.mark.parametrize("layout", ["v1", "v2"])
    def test_plot_results_layouts(self, layout):
        noise = np.array([0, 10, 20, 30])
        cov = np.array([0.8, 0.7, 0.6, 0.5])
        fp = np.array([0.1, 0.2, 0.3, 0.4])
        fd = np.array([0.05, 0.1, 0.15, 0.2])
        frag = np.array([1, 2, 3, 4])

        fig, axs = plot_results(noise, cov, fp, fd, frag, "CREPE", layout=layout)
        assert fig is not None
        assert 'E' in axs and 'F' in axs
        if layout == "v1":
            assert 'G' in axs and 'H' in axs
        else:
            assert 'H' in axs

    def test_plot_results_with_uncertainty(self):
        noise = np.array([0, 10, 20])
        cov = np.array([0.8, 0.7, 0.6])
        fp = np.array([0.1, 0.2, 0.3])
        fd = np.array([0.05, 0.1, 0.15])
        frag = np.array([1, 2, 3])
        unc = np.array([[0.75, 0.85], [0.65, 0.75], [0.55, 0.65]])

        fig, axs = plot_results(noise, cov, fp, fd, frag, "CREPE", coverage_unc=unc)
        assert len(axs['E'].collections) > 0  # fill_between creates a PolyCollection

    def test_plot_results_with_limits(self):
        noise = np.array([0, 10, 20])
        cov = np.array([0.8, 0.7, 0.6])
        fp = np.array([0.1, 0.2, 0.3])
        fd = np.array([0.05, 0.1, 0.15])
        frag = np.array([1, 2, 3])
        limits = {'Coverage': {'x': (0, 20), 'y': (0, 1)}}

        fig, axs = plot_results(noise, cov, fp, fd, frag, "CREPE", limits=limits)
        assert axs['E'].get_xlim() == (0, 20)

    @pytest.mark.parametrize("layout", ["v1", "v2"])
    def test_fbid_plot_layouts(self, layout):
        fbs = np.array([1, 2, 3])
        cov = np.array([0.9, 0.8, 0.7])
        frag = np.array([1, 2, 3])
        fd = np.array([0.05, 0.1, 0.15])

        fig, axs = fbid_plot(fbs, cov, frag, fd, layout=layout)
        assert fig is not None

    def test_plot_fbid_trends(self):
        metrics = {
            "CREPE": {
                "per_fbid": {
                    1: {"total_total_total_total": {"coverage": 0.9, "fragmentation": 1, "freq_diff": 0.05,
                                                    "coverage_5_95": [0.8, 1.0]}},
                    2: {"total_total_total_total": {"coverage": 0.8, "fragmentation": 2, "freq_diff": 0.1,
                                                    "coverage_5_95": [0.7, 0.9]}}
                }
            }
        }
        fig, axs = plot_fbid_trends(metrics, ["CREPE"])
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.run_ground', side_effect=_mock_ground)
    @patch('audio_analysis.plotting.plotting._run_algorithm_directly', side_effect=_mock_precomputed)
    def test_visualize_together_direct(self, mock_run, mock_ground, mock_stft, mock_sf):
        fig = visualize_together("dummy.wav", "params", algorithms=["CREPE"], use_precomputed=False)
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.run_ground', side_effect=_mock_ground)
    @patch('audio_analysis.plotting.plotting.load_precomputed', side_effect=_mock_precomputed)
    def test_visualize_together_precomputed(self, mock_load, mock_ground, mock_stft, mock_sf):
        fig = visualize_together("dummy.wav", "params", algorithms=["CREPE"], use_precomputed=True)
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.run_ground', side_effect=_mock_ground)
    @patch('audio_analysis.plotting.plotting.load_precomputed', side_effect=_mock_precomputed)
    @patch('audio_analysis.plotting.plotting.fragment_contours', side_effect=_mock_fragment)
    @patch('audio_analysis.plotting.plotting.identify_harmonics', side_effect=_mock_harmonics)
    def test_visualize_prediction(self, mock_harm, mock_frag, mock_load, mock_ground, mock_stft, mock_sf):
        fig = visualize_prediction("dummy.wav", "CREPE", "../src", "params")
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.run_ground', side_effect=_mock_ground)
    @patch('audio_analysis.plotting.plotting.load_precomputed', side_effect=_mock_precomputed)
    @patch('audio_analysis.plotting.plotting.fragment_contours', side_effect=_mock_fragment)
    @patch('audio_analysis.plotting.plotting.identify_harmonics', side_effect=_mock_harmonics)
    @patch('audio_analysis.plotting.plotting.butter', side_effect=_mock_butter)
    @patch('audio_analysis.plotting.plotting.filtfilt', side_effect=_mock_filtfilt)
    def test_plot_metrics_mosaic(self, mock_f, mock_b, mock_harm, mock_frag, mock_load, mock_ground, mock_stft,
                                 mock_sf):
        fig = plot_metrics_mosaic("dummy.wav", "params", "../src", "load", "CREPE")
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.load_precomputed', side_effect=_mock_precomputed)
    @patch('audio_analysis.plotting.plotting.CrepePredictor')
    @patch('audio_analysis.plotting.plotting.fragment_contours', side_effect=_mock_fragment)
    @patch('audio_analysis.plotting.plotting.identify_harmonics', side_effect=_mock_harmonics)
    def test_plot_preprocessing(self, mock_harm, mock_frag, mock_crepe, mock_load, mock_stft, mock_sf):
        mock_crepe.return_value.predict_crepe.return_value = _mock_precomputed("dummy.wav", "CREPE", None)
        fig = plot_preprocessing("dummy.wav", use_precomputed=False)
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    @patch('audio_analysis.plotting.plotting.run_ground', side_effect=_mock_ground)
    @patch('audio_analysis.plotting.plotting.load_precomputed', side_effect=_mock_precomputed)
    @patch('audio_analysis.plotting.plotting.fragment_contours', side_effect=_mock_fragment)
    @patch('audio_analysis.plotting.plotting.identify_harmonics', side_effect=_mock_harmonics)
    @patch('audio_analysis.plotting.plotting.butter', side_effect=_mock_butter)
    @patch('audio_analysis.plotting.plotting.filtfilt', side_effect=_mock_filtfilt)
    def test_plot_metrics_mosaic_v2(self, mock_f, mock_b, mock_harm, mock_frag, mock_load, mock_ground, mock_stft,
                                    mock_sf):
        fig = plot_metrics_mosaic_v2("dummy.wav", "params", "CREPE")
        assert fig is not None

    def test_plot_metric_violin(self):
        metrics = {
            "CREPE": {"per_fbid": {1: {"total_total_total_total": {"coverage": 0.9}}}},
            "SAM": {"per_fbid": {1: {"total_total_total_total": {"coverage": 0.85}}}}
        }
        fig = plot_metric_violin(metrics, ["CREPE", "SAM"], "Coverage")
        assert fig is not None

    @patch('audio_analysis.plotting.plotting.sf.read', side_effect=_mock_audio)
    @patch('audio_analysis.plotting.plotting.stft', side_effect=_mock_stft)
    def test_plot_fragmentation_verification(self, mock_stft, mock_sf):
        gt = [np.column_stack([np.linspace(0, 1, 50), np.full(50, 10000)])]
        pred = [np.column_stack([np.linspace(0, 1, 50), np.full(50, 10000)])]
        fig = plot_fragmentation_verification("dummy.wav", gt, pred, show_spectrogram=True)
        assert fig is not None
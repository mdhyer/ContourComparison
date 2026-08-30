import pytest
import numpy as np
import tempfile
import os
import soundfile as sf
import matplotlib.pyplot as plt

from audio_analysis.augmentation.snr_scaling import calculate_rms_db, calculate_snr, scale_by_snr
from audio_analysis.augmentation.noise_augmentation import generate_white_noise
from audio_analysis.utils.validation import validate_audio_file, validate_array
from audio_analysis.utils.contour_utils import fragment_contours, identify_harmonics
from audio_analysis.evaluation.metrics import aggregate_metrics
from audio_analysis.evaluation.dtw_alignment import calculate_dtw_distance
from audio_analysis.plotting.plotting import plot_results, plot_fbid_trends
from audio_analysis.config import MetricsConfig
from pathlib import Path
from unittest.mock import patch
from audio_analysis.utils.contour_utils import _resolve_fbid_snr, load_ground_truth

class TestSNRMath:
    def test_calculate_rms_db(self):
        """Verify RMS calculation for a known sine wave."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Sine wave amplitude 1.0 -> RMS = 1/sqrt(2) ~ 0.707 -> dB ~ -3.01
        signal = np.sin(2 * np.pi * 440 * t)
        rms_db = calculate_rms_db(signal)
        assert -3.5 < rms_db < -2.5, f"Expected ~-3dB, got {rms_db}"

    def test_scale_by_snr(self):
        """Verify scaling achieves target SNR."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        noise = np.random.randn(len(t)) * 0.1
        
        target_snr = 20.0
        scaled_signal = scale_by_snr(signal, noise, sr, band=(200, 5000), snr=target_snr)
        
        actual_snr = calculate_snr(scaled_signal, noise, sr, band=(200, 5000))
        assert abs(actual_snr - target_snr) < 1.0, f"Expected ~{target_snr}dB, got {actual_snr}"

    def test_generate_white_noise(self):
        """Verify noise generation properties."""
        sr = 44100
        duration = 1.0
        # Pass std=1.0 to test normalization behavior
        noise = generate_white_noise(duration=duration, sr=sr, std=1.0)
        
        assert len(noise) == int(sr * duration)
        # Standard deviation should be close to 1.0 after normalization in function
        assert 0.9 < np.std(noise) < 1.1

class TestValidation:
    def test_validate_audio_file_invalid(self):
        """Ensure FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            validate_audio_file("/path/to/nonexistent.wav")

    def test_validate_audio_file_silent(self):
        """Ensure ValueError is raised for silent audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Write silent audio
            sr = 44100
            silent = np.zeros(sr)
            sf.write(temp_path, silent, sr)
            
            with pytest.raises(ValueError, match="silent"):
                validate_audio_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_array_nan(self):
        """Ensure ValueError is raised for arrays with NaN."""
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            validate_array(arr)

class TestContourUtils:
    def test_fragment_contours(self):
        """Verify contour fragmentation based on time gaps."""
        # Create a contour with a large time gap
        contour1 = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        contour2 = np.array([[1.0, 1000], [1.1, 1000]]) # Gap of 0.8s
        
        combined = np.vstack((contour1, contour2))
        # Wrap in list as expected by function
        contours = [combined]
        
        fragments = fragment_contours(contours, dur=0.5)
        
        assert len(fragments) == 2
        assert len(fragments[0]) == 3
        assert len(fragments[1]) == 2

    def test_identify_harmonics(self):
        """Verify harmonic removal logic."""
        # Base contour
        c1 = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        # Harmonic contour (2x frequency)
        c2 = np.array([[0.0, 2000], [0.1, 2000], [0.2, 2000]])
        
        contours = [c1, c2]
        result = identify_harmonics(contours, tolerance=0.5, freqdiff=100)
        
        # Should remove the higher harmonic (c2)
        assert len(result) == 1
        assert np.array_equal(result[0], c1)

class TestMetrics:
    def test_aggregate_metrics(self):
        """Verify metric aggregation."""
        # Mock results: [coverage, false_pos, freq_diff, frag, recall]
        results = [
            [0.9, 0.1, 5.0, 2.0, 0.8],
            [0.8, 0.2, 6.0, 3.0, 0.7]
        ]
        
        agg = aggregate_metrics(results)
        
        assert agg['coverage'] == pytest.approx(0.85)
        assert agg['false_pos'] == pytest.approx(0.15)
        assert agg['freq_diff'] == pytest.approx(5.5)
        assert agg['fragmentation'] == pytest.approx(2.5)

    def test_dtw_distance_identical(self):
        """Verify DTW distance is 0 for identical contours."""
        cont = np.array([[0.0, 1000], [0.1, 1000], [0.2, 1000]])
        dist = calculate_dtw_distance(cont, cont)
        assert dist == 0.0

class TestPlotting:
    def test_plot_results(self):
        """Verify plotting function returns a figure without crashing."""
        noise_floats = np.array([10, 20, 30])
        coverage = np.array([0.9, 0.8, 0.7])
        false_pos = np.array([0.1, 0.2, 0.3])
        freq_diff = np.array([5.0, 6.0, 7.0])
        frag = np.array([2.0, 3.0, 4.0])
        
        fig, axs = plot_results(
            noise_floats=noise_floats,
            coverage=coverage,
            false_pos=false_pos,
            freq_diff=freq_diff,
            frag=frag,
            algorithm="TestAlgo"
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fbid_trends_handles_inhomogeneous_uncertainty(self):
        """
        Verifies that plot_fbid_trends gracefully handles inconsistent/malformed 
        uncertainty data across FBIDs without raising a ValueError.
        """
        metrics_config = MetricsConfig()
        mode_key = f"{metrics_config.coverage_mode.value}_{metrics_config.freq_diff_mode.value}_{metrics_config.frag_mode.value}"
        
        # Simulate aggregated metrics with inconsistent uncertainty structures
        aggregated_metrics = {
            "CREPE": {
                "per_fbid": {
                    "FBID_001": {
                        mode_key: {
                            "coverage": 0.9, "fragmentation": 2.0, "freq_diff": 0.1,
                            "coverage_5_95": [0.8, 0.95], "fragmentation_5_95": [1.5, 2.5], "freq_diff_5_95": [0.05, 0.15]
                        }
                    },
                    "FBID_002": {
                        mode_key: {
                            "coverage": 0.85, "fragmentation": 3.0, "freq_diff": 0.12,
                            "coverage_5_95": 0.85,           # Invalid: scalar instead of list
                            "fragmentation_5_95": [2.0],    # Invalid: length 1
                            "freq_diff_5_95": "invalid"     # Invalid: wrong type
                        }
                    },
                    "FBID_003": {
                        mode_key: {
                            "coverage": 0.95, "fragmentation": 1.5, "freq_diff": 0.08,
                            "coverage_5_95": [0.9, 0.99], "fragmentation_5_95": [1.0, 2.0], "freq_diff_5_95": [0.02, 0.14]
                        }
                    }
                }
            }
        }
        
        # Should not raise ValueError despite malformed data
        fig, axs = plot_fbid_trends(
            aggregated_metrics=aggregated_metrics,
            algorithms=["CREPE"],
            metrics_config=metrics_config
        )
        
        assert fig is not None, "Figure should be generated even with malformed uncertainty data."
        assert axs is not None
        
        # Clean up matplotlib figure to prevent display issues in headless tests
        plt.close(fig)


class TestPathResolution:
    def test_resolve_nested_structure(self):
        # Structure: .../NoiseLevels/CLEAN/FBID_001/file.wav
        path = Path("data/NoiseLevels/CLEAN/FBID_001/whistle.wav")
        label, intermediate = _resolve_fbid_snr(path)
        assert label == "FBID_001"
        assert intermediate == "CLEAN"

    def test_resolve_flat_structure(self):
        # Structure: .../SomeIntermediate/FBID_002/file.wav
        path = Path("data/SomeIntermediate/FBID_002/whistle.wav")
        label, intermediate = _resolve_fbid_snr(path)
        assert label == "FBID_002"
        assert intermediate == "SomeIntermediate"


class TestGroundTruthLoading:
    @patch("audio_analysis.utils.contour_utils.loadmat")
    def test_load_mat_ground_truth(self, mock_loadmat, tmp_path):
        # FIXED: contour shape (1, 1, N, 2) ensures [0, 0] returns (N, 2) array
        mock_data = {
            'W': {
                'contour': np.array([[[[0, 100], [1, 200]]]]),
                'discont': np.array([])
            }
        }
        mock_loadmat.return_value = mock_data

        params_dir = tmp_path / "Params" / "FBID_001"
        params_dir.mkdir(parents=True)
        mat_file = params_dir / "whistle_params.mat"
        mat_file.touch()

        wav_path = tmp_path / "FBID_001" / "whistle.wav"
        wav_path.parent.mkdir()
        wav_path.touch()

        result = load_ground_truth(wav_path, params_dir.parent)

        assert result is not None
        contour, discont = result
        assert isinstance(contour, np.ndarray)
        assert len(discont) == 0

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from audio_analysis.utils.validation import (
    validate_audio_file, validate_array, validate_model_weights,
    validate_audio_properties, validate_audio_snr
)
import soundfile as sf

class TestValidation:
    def test_validate_audio_file_invalid(self):
        """Ensure FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            validate_audio_file("nonexistent.wav")

    def test_validate_audio_file_silent(self):
        """Ensure ValueError is raised for silent audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(44100), 44100)
        try:
            with pytest.raises(ValueError, match="silent"):
                validate_audio_file(f.name)
        finally:
            os.unlink(f.name)

    def test_validate_array_nan(self):
        """Ensure ValueError is raised for arrays containing NaN."""
        with pytest.raises(ValueError, match="non-finite"):
            validate_array(np.array([1.0, np.nan]))

    def test_validate_array_empty(self):
        """Ensure ValueError is raised for empty arrays."""
        with pytest.raises(ValueError, match="empty"):
            validate_array(np.array([]))

    def test_validate_model_weights_missing(self):
        """Ensure FileNotFoundError is raised for missing model weights."""
        with pytest.raises(FileNotFoundError):
            validate_model_weights("missing.pth")

    def test_validate_audio_properties_mismatch(self):
        """Ensure ValueError is raised for mismatched sample rate."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(44100), 44100)
        try:
            with pytest.raises(ValueError, match="sample rate"):
                validate_audio_properties(f.name, expected_sr=22050, expected_duration=1.0)
        finally:
            os.unlink(f.name)

    def test_validate_audio_snr_tolerance(self):
        """Verify SNR validation passes within tolerance."""
        np.random.seed(42)  # Ensure reproducibility for random noise generation
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        # Use 5000 Hz to fall within the default bandpass filter (2000, 22000)
        signal = np.sin(2 * np.pi * 5000 * t)
        noise = np.random.randn(sr) * 0.1
        mixed = signal + noise
        # SNR should be roughly 20 dB
        validate_audio_snr(mixed, noise, sr, expected_snr=20.0, tolerance=5.0)

    def test_validate_audio_snr_out_of_tolerance(self):
        """Verify SNR validation fails when outside tolerance."""
        np.random.seed(42)
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 5000 * t)
        noise = np.random.randn(sr) * 0.1
        mixed = signal + noise
        with pytest.raises(ValueError, match="Expected SNR"):
            validate_audio_snr(mixed, noise, sr, expected_snr=50.0, tolerance=2.0)

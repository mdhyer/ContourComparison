import numpy as np
import pytest
from audio_analysis.augmentation.snr_scaling import (
    calculate_rms_db, calculate_snr, scale_by_snr, scale_to_target_rms_db
)

class TestSNRScaling:
    def test_calculate_rms_db(self):
        """Verify RMS calculation for a known sine wave."""
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        rms_db = calculate_rms_db(signal)
        # Sine wave amplitude 1.0 -> RMS = 1/sqrt(2) ~ 0.707 -> dB ~ -3.01
        assert rms_db == pytest.approx(-3.01, abs=0.5)

    def test_scale_by_snr(self):
        """Verify SNR scaling produces correct length and finite values."""
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        noise = np.random.randn(sr)
        scaled = scale_by_snr(signal, noise, sr, snr=20.0)
        assert len(scaled) == len(signal)
        assert np.all(np.isfinite(scaled))

    def test_scale_to_target_rms_db(self):
        """Verify scaling to a specific RMS dB level."""
        signal = np.random.randn(44100)
        scaled = scale_to_target_rms_db(signal, target_db=-20.0)
        rms_db = calculate_rms_db(scaled)
        assert rms_db == pytest.approx(-20.0, abs=0.5)

    def test_calculate_snr_bandpass(self):
        """Verify SNR calculation respects frequency band."""
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        noise = np.random.randn(sr)
        snr = calculate_snr(signal, noise, sr, band=(200, 1000))
        assert np.isfinite(snr)

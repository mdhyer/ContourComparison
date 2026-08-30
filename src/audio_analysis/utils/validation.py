"""
Input validation utilities for audio files, arrays, and model weights.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import soundfile as sf

def validate_audio_file(path: Union[str, Path], min_duration: float = 0.1) -> None:
    """
    Validate that an audio file exists, is readable, and meets minimum duration requirements.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    
    try:
        with sf.SoundFile(p, 'r') as f:
            duration = f.frames / f.samplerate
            if duration < min_duration:
                raise ValueError(f"Audio duration {duration:.2f}s is below minimum threshold {min_duration}s")
            if np.all(np.abs(f.read()) < 1e-6):
                raise ValueError(f"Audio file appears to be silent: {p}")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {p}: {e}")

def validate_array(arr: np.ndarray, name: str = "array") -> None:
    """
    Validate that a numpy array is finite, non-empty, and has correct dimensions.
    """
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
    if arr.ndim < 1:
        raise ValueError(f"{name} must be at least 1-dimensional.")

def validate_model_weights(path: Union[str, Path]) -> None:
    """
    Validate that a model weight file exists and is non-empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model weights not found: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"Model weights file is empty: {p}")

def validate_audio_properties(
    path: Union[str, Path],
    expected_sr: int,
    expected_duration: float,
    expected_channels: int = 1,
    duration_tolerance: float = 0.1
) -> None:
    """Validate sample rate, duration, and channel count of an audio file."""
    p = Path(path)
    with sf.SoundFile(p, 'r') as f:
        if f.samplerate != expected_sr:
            raise ValueError(f"Expected sample rate {expected_sr}, got {f.samplerate}")
        actual_duration = f.frames / f.samplerate
        if abs(actual_duration - expected_duration) > duration_tolerance:
            raise ValueError(f"Expected duration ~{expected_duration}s, got {actual_duration:.2f}s")
        if f.channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channel(s), got {f.channels}")

def validate_audio_snr(
    mixed_audio: np.ndarray,
    original_noise: np.ndarray,
    sr: int,
    expected_snr: float,
    tolerance: float = 2.0,
    band: Tuple[int, int] = (2000, 22000)
) -> None:
    """
    Validate that the actual SNR of a mixed audio file matches the expected SNR within tolerance.
    Requires the original noise component for accurate post-hoc verification.
    """
    from audio_analysis.augmentation.snr_scaling import calculate_snr
    
    min_len = min(len(mixed_audio), len(original_noise))
    mixed = mixed_audio[:min_len]
    noise = original_noise[:min_len]
    
    signal = mixed - noise
    actual_snr = calculate_snr(signal, noise, sr, band=band)
    if abs(actual_snr - expected_snr) > tolerance:
        raise ValueError(f"Expected SNR ~{expected_snr} dB, got {actual_snr:.2f} dB (tolerance: {tolerance} dB)")

"""
Functions to scale audio signals by a specified Signal-to-Noise Ratio (SNR).
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Tuple

def calculate_rms_db(audio: np.ndarray) -> float:
    """
    Calculates Root Mean Square Sound Pressure Level in decibels from raw waveform.
    """
    rms = np.sqrt(np.sum(np.square(audio)) / len(audio))
    return 20 * np.log10(rms + 1e-10)

def calculate_snr(positive: np.ndarray, negative: np.ndarray, sr: int, band: Tuple[int, int] = (50, 225)) -> float:
    """
    Calculates the SNR between a positive signal and a negative signal after bandpass filtering.
    """
    b, a = signal.butter(4, band, btype='bandpass', fs=sr)
    fpositive = signal.filtfilt(b, a, positive)
    spositive = calculate_rms_db(fpositive)

    fnegative = signal.filtfilt(b, a, negative)
    snegative = calculate_rms_db(fnegative)

    return spositive - snegative

def scale_by_snr(positive: np.ndarray, negative: np.ndarray, sr: int, band: Tuple[int, int] = (50, 225), snr: float = 6.0) -> np.ndarray:
    """
    Calculates scale factor to multiply positive signal by to achieve desired SNR when mixed with negative signal.
    """
    _snr = calculate_snr(positive, negative, sr, band=band)
    scale = 10**((snr - _snr) / 20)
    return scale * positive

def scale_to_target_rms_db(audio: np.ndarray, target_db: float) -> np.ndarray:
    """
    Scales an audio signal to a target RMS level in dB.
    """
    current_db = calculate_rms_db(audio)
    scale = 10**((target_db - current_db) / 20)
    return scale * audio

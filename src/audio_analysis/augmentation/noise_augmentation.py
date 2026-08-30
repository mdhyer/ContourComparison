"""
Audio augmentation utilities: white noise generation, pitch shifting, and mixing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import resampy
from librosa.effects import time_stretch
from librosa.core import resample
from librosa.util import fix_length

from .snr_scaling import scale_by_snr


@dataclass
class AugmentationConfig:
    directory: Path
    destination: Path
    audio_extension: str = '.wav'
    noise_duration: float = 3.0
    snr: int = 20
    num_augmentations: int = 1
    sample_rate: int = 96000
    band: tuple = (2000, 22000)
    # Replaces hardcoded SLSNR_20 lookup with a configurable whitelist dict
    whitelist: Optional[Dict[str, List[str]]] = None


def generate_white_noise(std: float = 0.00023220679723766314, duration: float = 3.0, sr: int = 96000) -> np.ndarray:
    """Create white noise for augmentation."""
    num_samples = int(duration * sr)
    whitenoise = np.random.randn(num_samples)
    whitenoise = whitenoise / np.std(whitenoise)
    whitenoise = whitenoise * std
    return whitenoise


def pitch_shift(y: np.ndarray, sr: int, rate: float, res_type: str = "soxr_hq", n_fft: int = 32, **kwargs) -> np.ndarray:
    """Pitch shift audio by a given rate."""
    y_shift = resample(
        time_stretch(y, rate=rate, n_fft=n_fft, **kwargs),
        orig_sr=float(sr)/rate,
        target_sr=sr,
        res_type=res_type,
    )
    return fix_length(y_shift, size=y.shape[-1])


def to_mono(y: np.ndarray) -> np.ndarray:
    """Convert an audio signal to mono by averaging samples across channels."""
    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))
    return y


def augment(wav: str, config: AugmentationConfig, noise: Optional[np.ndarray] = None) -> None:
    """Perform augmentation of a single audio file."""
    _block, sr = sf.read(wav)
    noise_duration = len(_block) / sr
    _block = to_mono(_block)

    if sr != config.sample_rate:
        _block = resampy.resample(_block, sr, config.sample_rate, filter='kaiser_best', parallel=True)
        sr = config.sample_rate

    for n in range(config.num_augmentations):
        block = _block
        path = Path(wav)
        fname = path.stem
        dest = config.destination / f"{fname}_snr_{config.snr}{config.audio_extension}"

        scale_factor = config.snr
        if noise is None:
            aug = generate_white_noise(std=0.00023220679723766314, duration=noise_duration, sr=config.sample_rate)
            start_point = round(random.random() * (len(aug) - len(block)))
        else:
            aug = noise.copy()
            start_point = 0
            
        block = scale_by_snr(block, aug, config.sample_rate, band=config.band, snr=scale_factor)
        aug[start_point:start_point + len(block)] += block

        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), aug, config.sample_rate)


def read_random_noise(noise_dict: dict, noise: str, sample_rate: int, noise_duration: float) -> np.ndarray:
    """Selects random noise file from available directories, loads specified duration of noise, resamples."""
    noise_file = random.choice(noise_dict[random.choice(noise)])
    out_blocks = int(sample_rate * noise_duration)

    with sf.SoundFile(noise_file, 'r') as noise:
        frames = noise.frames
        sr = noise.samplerate
        noise_frames = round((out_blocks / sample_rate) * sr)
        start_point = round(random.random() * (frames - noise_frames))
        noise.seek(start_point)
        block = noise.read(noise_frames)
        block = resampy.resample(to_mono(block), sr, sample_rate, filter='kaiser_fast', parallel=True)
    assert len(block) == out_blocks
    return block


def find_noise(filename: str, depth: str, filetypes: List[str]) -> List[str]:
    """Searches directory for files of supported filetypes"""
    outfiles = []
    base_path = Path(filename)
    for type_ in filetypes:
        pattern = f"{depth}{type_}"
        outfiles.extend([str(p) for p in base_path.glob(pattern)])
    return outfiles

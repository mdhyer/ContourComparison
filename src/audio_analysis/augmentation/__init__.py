"""
Augmentation Subpackage
Handles SNR scaling, noise generation, and audio mixing utilities.
"""
from .snr_scaling import calculate_rms_db, calculate_snr, scale_by_snr, scale_to_target_rms_db
from .noise_augmentation import (
    AugmentationConfig,
    generate_white_noise,
    pitch_shift,
    to_mono,
    augment,
    read_random_noise,
    find_noise,
)

__all__ = [
    "calculate_rms_db",
    "calculate_snr",
    "scale_by_snr",
    "scale_to_target_rms_db",
    "AugmentationConfig",
    "generate_white_noise",
    "pitch_shift",
    "to_mono",
    "augment",
    "read_random_noise",
    "find_noise",
]

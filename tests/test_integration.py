import sys
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from audio_analysis.cli import main
from audio_analysis.augmentation.noise_augmentation import augment, AugmentationConfig, generate_white_noise
from audio_analysis.utils.validation import validate_audio_file, validate_audio_properties, validate_audio_snr

class TestCLIIntegration:
    def test_cli_config_override(self, capsys):
        """Verify CLI arguments correctly override PipelineConfig defaults."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "audio-analysis",
                "precompute",
                "--data-root", "./test_data",
                "--output-dir", "./test_out",
                "--sample-rate", "48000"
            ]
            main()
            captured = capsys.readouterr()
            
            assert "Data Root: test_data" in captured.out
            assert "Output Dir: test_out" in captured.out
            assert "Sample Rate: 48000" in captured.out
        finally:
            sys.argv = original_argv

class TestAugmentationPipeline:
    def test_augmentation_output_properties(self, synthetic_wav, temp_audio_dir):
        """Run augmentation and validate output file properties."""
        wav_path, sr, duration = synthetic_wav
        dest_dir = temp_audio_dir / "augmented"
        
        config = AugmentationConfig(
            directory=temp_audio_dir,
            destination=dest_dir,
            sample_rate=sr,
            snr=20,
            num_augmentations=1,
            band=(2000, 22000)
        )
        
        augment(str(wav_path), config)
        
        output_file = dest_dir / f"{wav_path.stem}_snr_20.wav"
        assert output_file.exists(), "Augmented file was not created."
        
        # Validate basic properties
        validate_audio_file(output_file, min_duration=duration - 0.1)
        validate_audio_properties(output_file, expected_sr=sr, expected_duration=duration, expected_channels=1)

    def test_augmentation_snr_validation(self, synthetic_wav, temp_audio_dir):
        """Validate that generated audio meets expected SNR thresholds."""
        wav_path, sr, duration = synthetic_wav
        dest_dir = temp_audio_dir / "augmented_snr"
        
        # Read original signal
        original_signal, _ = sf.read(str(wav_path))
        original_signal = original_signal.flatten() if original_signal.ndim > 1 else original_signal
        
        # Generate matching noise
        original_noise = generate_white_noise(duration=duration, sr=sr)
        
        config = AugmentationConfig(
            directory=temp_audio_dir,
            destination=dest_dir,
            sample_rate=sr,
            snr=20,
            num_augmentations=1,
            band=(2000, 22000)
        )
        
        augment(str(wav_path), config, noise=original_noise)
        output_file = dest_dir / f"{wav_path.stem}_snr_20.wav"
        
        mixed_audio, _ = sf.read(str(output_file))
        mixed_audio = mixed_audio.flatten() if mixed_audio.ndim > 1 else mixed_audio
        
        # Validate SNR against expected 20 dB (±2 dB tolerance)
        validate_audio_snr(
            mixed_audio=mixed_audio,
            original_noise=original_noise,
            sr=sr,
            expected_snr=20.0,
            tolerance=2.0,
            band=(2000, 22000)
        )

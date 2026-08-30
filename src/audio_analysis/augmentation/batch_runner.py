"""
Batch augmentation runner. Replaces legacy os.system SNR loops with native Python.
"""
import json
import random
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .noise_augmentation import AugmentationConfig, augment, is_in_whitelist

def run_snr_augmentation_batch(
    data_dir: str,
    base_output_dir: str,
    snr_values: List[int],
    noise_type: str = "white",
    whitelist_path: Optional[str] = None,
    limit_per_folder: int = 20,
    sample_rate: int = 96000,
    band: tuple = (2000, 22000)
) -> None:
    """
    Iterates over SNR levels, applies whitelist filtering, and runs augmentation.
    """
    whitelist = None
    if whitelist_path:
        with open(whitelist_path, 'r') as f:
            whitelist = json.load(f)

    for snr in snr_values:
        print(f"\n{'='*40}\nProcessing SNR: {snr} dB\n{'='*40}")
        output_dir = Path(base_output_dir) / f"{snr}db"

        config = AugmentationConfig(
            directory=Path(data_dir),
            destination=output_dir,
            snr=snr,
            noise_type=noise_type,
            sample_rate=sample_rate,
            band=band,
            whitelist=whitelist
        )

        fbid_dirs = sorted(config.directory.iterdir())
        for fbid_dir in fbid_dirs:
            if not fbid_dir.is_dir():
                continue

            wavs = sorted(fbid_dir.glob("*.wav"))
            if not wavs:
                continue

            random.shuffle(wavs)
            count = 0

            for wav_path in tqdm(wavs, desc=f"FBID: {fbid_dir.name}", unit="file", leave=False):
                if is_in_whitelist(wav_path, config):
                    try:
                        augment(str(wav_path), config)
                        count += 1
                    except Exception as e:
                        print(f"  [WARN] Failed {wav_path.name}: {e}")

                if count >= limit_per_folder:
                    break

if __name__ == "__main__":
    # Matches your legacy loop exactly
    SNR_VALUES = [32]  # or range(-20, 24, 4)
    
    run_snr_augmentation_batch(
        data_dir=r"C:\Users\matth\Documents\Updated200x74SignatureWhistles\AudioFolder",
        base_output_dir=r"D:\ContourExtraction\WhiteNoise_SNR_Test\NoiseLevels",
        snr_values=SNR_VALUES,
        noise_type="white",  # Change to "pink" as needed
        whitelist_path=r"src\audio_analysis\utils\SLSNR_20.json",
        limit_per_folder=20
    )

"""
Dataset generation for DTW analysis.
Loads ground truth, crops audio, injects noise, and saves updated params.
"""
from __future__ import annotations
import numpy as np
from scipy.io import loadmat
from scipy import signal
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..augmentation.snr_scaling import scale_by_snr
from .. import config

def make_white_noise(duration: float = 3.0, sample_rate: int = 48000, std: float = 0.00020985778493418198) -> np.ndarray:
    """Generate white noise with specified duration and standard deviation."""
    noise = np.random.rand(int(duration * sample_rate))
    noise -= np.mean(noise)
    noise *= std / np.std(noise)
    return noise

def load_param_worker(param_path: str) -> Tuple[str, np.ndarray, list]:
    """Worker function to load parameters from a .mat file."""
    param_mat = loadmat(param_path)
    this_contour = param_mat['W']['contour'][0, 0]
    try:
        this_discont = param_mat['W']['discont'][0, 0]
    except ValueError:
        this_discont = []
    this_fname = Path(param_path).stem.replace('params', '')
    if this_fname.endswith('_'):
        this_fname = this_fname[:-1]
    return this_fname, this_contour, this_discont

def generate_dtw_dataset(
    wav_dir: str,
    params_dir: str,
    dest_dir: str,
    n_subsample: int = 20,
    snr: float = 20.0,
    band: Tuple[int, int] = None,
    max_workers: Optional[int] = None
) -> None:
    """
    Generate DTW dataset by cropping whistles, adding noise, and saving.
    Uses ProcessPoolExecutor for safe parallel parameter loading.
    """
    if band is None:
        band = config.audio.filter_band

    wav_path = Path(wav_dir)
    params_path = Path(params_dir)
    dest_path = Path(dest_dir)

    fb_dirs = list(wav_path.iterdir())
    ground_contours: Dict[str, np.ndarray] = {}
    ground_disconts: Dict[str, list] = {}

    # Load parameters in parallel
    param_files = list(params_path.rglob("*_params.mat"))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_param_worker, [str(p) for p in param_files])
        for fname, contour, discont in results:
            ground_contours[fname] = contour
            ground_disconts[fname] = discont

    for fb_dir in fb_dirs:
        if not fb_dir.is_dir():
            continue
        fb_name = fb_dir.name
        wavs = list(fb_dir.rglob("*.wav"))
        sortinds = np.argsort([p.stem.split('IND')[-1].split('.')[0].split('_')[0] for p in wavs])
        wavs = [wavs[s] for s in sortinds]

        for wav in wavs[10::n_subsample]:
            wavname = wav.stem
            if '_snr_' in wavname:
                wavname = wavname.split('_snr_')[0]
            if '_CLEAN_' in wavname:
                wavname = wavname.split('_CLEAN_')[0]

            wav_dest = dest_path / wav.relative_to(wav_path)
            params_dest = wav_dest.with_suffix('_params.npy')
            params_dest.parent.mkdir(parents=True, exist_ok=True)
            if wav_dest.exists():
                continue

            try:
                ground = ground_contours[wavname]
                discont = ground_disconts[wavname]
            except KeyError:
                continue

            start = ground[0, 0]
            if len(discont) > 0:
                dis_idx = len(discont) // 2
                dis = discont[dis_idx]
                end = dis[0]
                if dis_idx >= 1:
                    start = discont[dis_idx - 1][1]
                    if end - start < 0.005:
                        dis_idx += 1
                        dis = discont[dis_idx]
                        end = dis[0]
                        if dis_idx >= 1:
                            start = discont[dis_idx - 1][1]
            else:
                end = ground[-1, 0]

            with sf.SoundFile(wav) as wavf:
                sr = wavf.samplerate
                wavf.seek(round(start * sr))
                audio = wavf.read(round((end - start) * sr))

            noise = make_white_noise(duration=5.0, sample_rate=sr)
            audio = scale_by_snr(audio, noise, sr, band=band, snr=snr)

            b, a = signal.butter(4, band, btype='bandpass', fs=sr)
            audio = signal.filtfilt(b, a, audio)

            mix_start = round(round((len(noise) / 2 - len(audio) / 2) / sr, 2) * sr)

            contour = []
            for gt, gf in ground:
                if start <= gt and end >= gt:
                    contour.append([round(gt - start + (mix_start / sr), 3), gf])
            contour = np.array(contour)

            try:
                noise[mix_start : mix_start + len(audio)] += audio
            except ValueError:
                continue

            sf.write(str(wav_dest), noise, sr)
            np.save(str(params_dest), contour, allow_pickle=True)

"""
Utility functions for contour processing, ground truth loading, and path management.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List, Optional, Any, Union, Tuple, Dict

import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def _resolve_fbid_snr(wav_path: Path) -> Tuple[str, str]:
    """Extract label and intermediate directory name from a WAV file path."""
    parts = wav_path.parts
    # Detect nested structure automatically
    if "NoiseLevels" in parts:
        idx = parts.index("NoiseLevels")
        intermediate = parts[idx + 1]  # SNR level
        label = parts[idx + 2]
    else:
        # Flat structure: intermediate is the parent of the label directory
        label = wav_path.parent.name
        intermediate = wav_path.parent.parent.name
    return label, intermediate


def get_data_paths(
        precompute_dir: Optional[Path] = None,
        noise_levels_dir: Optional[Path] = None,
        params_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Resolve and validate data directory paths.
    """
    return {
        "precompute": precompute_dir or Path("PrecomputeContours"),
        "noise_levels": noise_levels_dir or Path("data/NoiseLevels"),
        "params": params_dir or Path("Params"),
    }


def fragment_contours(contours: List[np.ndarray], dur: float = 0.025) -> List[np.ndarray]:
    """
    Break up contours if they are not continuous. Typically used after contour prediction (e.g., CREPE).
    """
    final_contour: List[np.ndarray] = []

    for contour in contours:
        contour = np.asarray(contour)
        if len(contour) == 0:
            continue

        chunk_start = 0
        for ci in range(len(contour)):
            is_last = ci == len(contour) - 1
            time_gap = float(contour[ci + 1, 0]) - float(contour[ci, 0]) if not is_last else 0.0

            if is_last or time_gap > dur:
                chunk = contour[chunk_start: ci + 1]
                if len(chunk) >= 2:
                    final_contour.append(chunk)
                chunk_start = ci + 1

    return final_contour


def identify_harmonics(contours: List[np.ndarray], tolerance: float = 0.7, freqdiff: float = 0.2) -> List[np.ndarray]:
    """
    Remove harmonic contours.
    Restored to match legacy_utils.py logic to prevent over-removal.
    """
    if len(contours) < 2:
        return contours

    rm = set()
    time_tol = 0.0001

    # We must evaluate all pairs before removing to match legacy behavior
    for i in range(len(contours)):
        c_i = np.asarray(contours[i])
        for j in range(len(contours)):
            if i == j:
                continue

            c_j = np.asarray(contours[j])
            matches = []

            # Restore nested loop matching to count ALL alignments within tolerance
            for t_i, f_i in c_i:
                for t_j, f_j in c_j:
                    if abs(t_j - t_i) < time_tol:
                        # Avoid division by zero
                        if f_i == 0 or f_j == 0:
                            continue

                        ratio = max(f_i, f_j) / min(f_i, f_j)
                        rounded_ratio = round(ratio)

                        # CRITICAL FIX: Legacy appends 0 for non-harmonic overlaps
                        # This keeps the denominator high, preventing false positives
                        if abs(ratio - rounded_ratio) < freqdiff:
                            matches.append(0 if rounded_ratio == 1 else 1)
                        else:
                            matches.append(0)

            if len(matches) == 0:
                continue

            harmonic_proportion = sum(matches) / len(matches)

            if harmonic_proportion >= tolerance:
                # Remove the one with higher mean frequency
                if np.mean(c_i[:, 1]) > np.mean(c_j[:, 1]):
                    rm.add(i)
                else:
                    rm.add(j)

    return [c for idx, c in enumerate(contours) if idx not in rm]


def _parse_mat_params(param_mat: Path) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Helper to load and process a .mat params file into a contour array and discontinuity list."""
    try:
        param_data = loadmat(param_mat)
        this_contour = param_data['W']['contour'][0, 0]
        final_contour = []

        this_discont = None
        try:
            this_discont = param_data['W']['discont'][0, 0]
        except Exception:
            this_discont = None

        discont_list = []
        if this_discont is not None and hasattr(this_discont, 'size') and this_discont.size > 0:
            if this_discont.ndim == 1:
                discont_list = this_discont.flatten().tolist()
            else:
                discont_list = [float(d[0]) for d in this_discont]

        start = this_contour[0, 0]
        # Safely iterate over discontinuities only if they exist and are 2D (start, end pairs)
        if this_discont is not None and getattr(this_discont, 'size', 0) > 0 and this_discont.ndim == 2:
            for dis in this_discont:
                end = float(dis[0])
                l_x, l_c = [], []
                for gro in this_contour:
                    if gro[0] >= start and gro[0] <= end:
                        l_x.append(gro[0])
                        l_c.append(gro[1])
                final_contour.append(np.array([l_x, l_c]).T)
                start = float(dis[1])

        end = this_contour[-1, 0]
        l_x, l_c = [], []
        for gro in this_contour:
            if gro[0] >= start and gro[0] <= end:
                l_x.append(gro[0])
                l_c.append(gro[1])
        final_contour.append(np.array([l_x, l_c]).T)

        if len(final_contour) == 1:
            return final_contour[0], discont_list
        return np.vstack(final_contour), discont_list
    except Exception:
        return None, None


def _find_matching_param(wav_stem: str, param_lookup: Dict[str, Path]) -> Optional[Path]:
    """Find the best matching parameter file for a given WAV stem using exact or partial matching."""
    wav_stem_lower = wav_stem.lower()

    # Exact match first
    if wav_stem_lower in param_lookup:
        return param_lookup[wav_stem_lower]

    # Partial match fallback: find param base that is a substring of wav stem
    matches = [p for k, p in param_lookup.items() if k in wav_stem_lower]
    if len(matches) == 1:
        return matches[0]
    # If multiple partial matches, pick the longest base name (most specific) to avoid duplicates
    if matches:
        return max(matches, key=lambda p: len(p.stem))
    return None


def load_ground_truth(
        wav_path: Path,
        params_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """
    Load ground truth contour for a given WAV file from Params/.
    Returns a tuple of (contour_array, discontinuity_times).
    """
    label, _ = _resolve_fbid_snr(wav_path)
    stem = wav_path.stem
    if '_snr_' in stem: stem = stem.split('_snr_')[0]
    if '_CLEAN_' in stem: stem = stem.split('_CLEAN_')[0]

    param_path = params_dir / label / f"{stem}_params.mat"
    if not param_path.exists():
        param_path = params_dir / label / f"{stem}_params.npy"
    if not param_path.exists():
        return None, None

    if param_path.suffix == '.npy':
        loaded = np.load(param_path, allow_pickle=True)
        if isinstance(loaded, dict):
            return loaded["contour"], loaded.get("discont")
        return loaded, None  # backward compat: plain array
    return _parse_mat_params(param_path)


def save_precomputed(
        wav: str,
        contours: List[np.ndarray],
        algorithm: str,
        src: Union[str, Path],
        top_dir: Union[str, Path],
) -> None:
    """Save precomputed contours to a numpy file."""
    wav_path = Path(wav)
    top_path = Path(top_dir)

    label, intermediate = _resolve_fbid_snr(wav_path)
    # Structure: ALGORITHM / INTERMEDIATE / LABEL
    save_dir = top_path / algorithm / intermediate / label
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"{wav_path.stem}_contour.npy"
    np.save(save_file, np.array(contours, dtype=object), allow_pickle=True)


def load_precomputed(
        wav: str,
        ALGORITHM: str = 'Silbido Profundo',
        top_dir: Union[str, Path] = "PrecomputeContours"
) -> Any:
    """Load precomputed contours from numpy files."""
    wav_path = Path(wav)
    label, intermediate = _resolve_fbid_snr(wav_path)
    # Structure: ALGORITHM / INTERMEDIATE / LABEL
    target_dir = Path(top_dir) / ALGORITHM / intermediate / label
    cload = target_dir / f"{wav_path.stem}_contour.npy"

    if cload.exists():
        return np.load(cload, allow_pickle=True)

    # Fallback: search for any .npy file in the directory
    if target_dir.exists():
        cload = target_dir / f"{wav_path.stem}.npy"
        if cload.exists():
            return np.load(cload, allow_pickle=True)

        # Secondary fallback: scan directory for any matching .npy file
        for f in target_dir.glob("*.npy"):
            if wav_path.stem in f.stem:
                return np.load(f, allow_pickle=True)

    raise FileNotFoundError(f"Precomputed contour not found for {wav_path.stem} in {target_dir}")


def run_ground(
        wav: str,
        params_dir: str = "Params",
) -> Tuple[List[np.ndarray], List]:
    """Load and process ground truth contours from .npy or .mat files in Params/."""
    wav_path = Path(wav)
    label, _ = _resolve_fbid_snr(wav_path)
    wavname = wav_path.stem
    if '_snr_' in wavname: wavname = wavname.split('_snr_')[0]
    if '_CLEAN_' in wavname: wavname = wavname.split('_CLEAN_')[0]

    param_path = Path(params_dir) / label / f"{wavname}_params.mat"
    if not param_path.exists():
        param_path = Path(params_dir) / label / f"{wavname}_params.npy"
    if not param_path.exists():
        raise FileNotFoundError(f'Missing params for file {param_path}')

    if param_path.suffix == '.npy':
        loaded = np.load(param_path, allow_pickle=True)
        if isinstance(loaded, dict):
            return [loaded["contour"]], loaded.get("discont") or []
        return [loaded], []

    param_data = loadmat(param_path)
    this_contour = param_data['W']['contour'][0, 0]
    final_contour = []

    try:
        this_discont = param_data['W']['discont'][0, 0]
    except ValueError:
        this_discont = []

    start = this_contour[0, 0]
    for dis in this_discont:
        end = dis[0]
        l_x, l_c = [], []
        for gro in this_contour:
            if gro[0] >= start and gro[0] <= end:
                l_x.append(gro[0])
                l_c.append(gro[1])
        final_contour.append(np.array([l_x, l_c]).T)
        start = dis[1]

    end = this_contour[-1, 0]
    l_x, l_c = [], []
    for gro in this_contour:
        if gro[0] >= start and gro[0] <= end:
            l_x.append(gro[0])
            l_c.append(gro[1])
    final_contour.append(np.array([l_x, l_c]).T)
    return final_contour, this_discont

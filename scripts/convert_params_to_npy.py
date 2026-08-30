"""Convert .mat params files to .npy dict format for sharing without MATLAB.

Usage:
    python scripts/convert_params_to_npy.py <params_dir>
    python scripts/convert_params_to_npy.py --contours <precompute_dir>

Converts all *_params.mat files under <params_dir> to *_params.npy (dict format).
The resulting .npy files contain a dict with keys:
    - "contour": 2D numpy array of shape (N, 2) [time, freq]
    - "discont": flat list of float split time points, or None if absent

With --contours, converts all *_contour.npy files from MATLAB types to pure numpy.
Must be run in an environment with the `matlab` package installed.

Requires scipy (for loadmat). Only needed in a dev environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def convert_mat_to_npy(mat_path: Path, npy_path: Path) -> bool:
    """Convert a single .mat params file to .npy dict format.

    Returns True on success, False on failure.
    """
    try:
        mat_data = loadmat(mat_path)
    except Exception as e:
        print(f"  ❌ Failed to load {mat_path.name}: {e}")
        return False

    # Extract contour
    try:
        contour = mat_data['W']['contour'][0, 0]
    except (KeyError, IndexError, ValueError) as e:
        print(f"  ❌ Could not extract contour from {mat_path.name}: {e}")
        return False

    # Extract and normalize discont
    discont_list = None
    try:
        discont = mat_data['W']['discont'][0, 0]
        if discont.size > 0:
            if discont.ndim == 1:
                discont_list = discont.flatten().tolist()
            else:
                discont_list = [float(d[0]) for d in discont]
    except (KeyError, IndexError, ValueError):
        discont_list = None

    # Save as .npy dict
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), {"contour": contour, "discont": discont_list}, allow_pickle=True)

    discont_info = f"{len(discont_list)} splits" if discont_list else "no splits"
    print(f"  ✅ {mat_path.name} → {npy_path.name} ({discont_info})")
    return True


def _convert_matlab_objects(obj):
    """Recursively convert MATLAB types to native NumPy/Python types."""
    try:
        import matlab
        if isinstance(obj, matlab.double):
            return np.array(obj)
    except (ImportError, NameError):
        pass

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array([_convert_matlab_objects(item) for item in obj], dtype=object)
    if isinstance(obj, (list, tuple)):
        return [_convert_matlab_objects(item) for item in obj]
    return obj


def convert_contour_npy(npy_path: Path) -> bool:
    """Convert a precomputed contour .npy file from MATLAB types to pure numpy.

    Must be run in an environment with the `matlab` package installed.
    Returns True on success, False on failure.
    """
    try:
        data = np.load(str(npy_path), allow_pickle=True)
    except Exception as e:
        print(f"  ❌ Failed to load {npy_path.name}: {e}")
        return False

    converted = _convert_matlab_objects(data)
    np.save(str(npy_path), converted, allow_pickle=True)
    print(f"  ✅ {npy_path.name} → pure numpy")
    return True


def convert_contours_dir(precompute_dir: Path) -> None:
    """Convert all *_contour.npy files under a precompute directory."""
    npy_files = sorted(precompute_dir.rglob("*.npy"))
    if not npy_files:
        print(f"No contour .npy files found in {precompute_dir}")
        return

    print(f"Converting {len(npy_files)} contour file(s) in {precompute_dir}...")
    print()

    success = 0
    failed = 0
    for npy_path in npy_files:
        if convert_contour_npy(npy_path):
            success += 1
        else:
            failed += 1

    print()
    print(f"✅ Done. {success} converted, {failed} failed.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/convert_params_to_npy.py <params_dir>")
        print("    Converts all *_params.mat files to *_params.npy (dict format)")
        print()
        print("  python scripts/convert_params_to_npy.py --contours <precompute_dir>")
        print("    Converts all *_contour.npy files from MATLAB types to pure numpy")
        print()
        print("  Examples:")
        print("    python scripts/convert_params_to_npy.py Projects/Benchmark/Params/")
        print("    python scripts/convert_params_to_npy.py --contours Projects/Benchmark/PrecomputeContours/")
        sys.exit(1)

    if sys.argv[1] == "--contours":
        if len(sys.argv) < 3:
            print("Error: --contours requires a directory argument.")
            sys.exit(1)
        precompute_dir = Path(sys.argv[2])
        if not precompute_dir.exists():
            print(f"❌ Directory not found: {precompute_dir}")
            sys.exit(1)
        convert_contours_dir(precompute_dir)
        return

    # Original params conversion mode
    params_dir = Path(sys.argv[1])
    if not params_dir.exists():
        print(f"❌ Directory not found: {params_dir}")
        sys.exit(1)

    mat_files = sorted(params_dir.rglob("*_params.mat"))
    if not mat_files:
        print(f"No .mat files found in {params_dir}")
        return

    print(f"Converting {len(mat_files)} .mat file(s) in {params_dir}...")
    print()

    success = 0
    failed = 0
    for mat_path in mat_files:
        npy_path = mat_path.with_suffix(".npy")
        if convert_mat_to_npy(mat_path, npy_path):
            success += 1
        else:
            failed += 1

    print()
    print(f"✅ Done. {success} converted, {failed} failed.")
    if success:
        print("   You can now delete the .mat files if desired.")


if __name__ == "__main__":
    main()

"""Convert .mat params files to .npy dict format for sharing without MATLAB.

Usage:
    python scripts/convert_params_to_npy.py <params_dir>
    python scripts/convert_params_to_npy.py --contours <precompute_dir>
    python scripts/convert_params_to_npy.py --params <params_dir>

Converts all *_params.mat files under <params_dir> to *_params.npy (dict format).
The resulting .npy files contain a dict with keys:
    - "contour": 2D numpy array of shape (N, 2) [time, freq]
    - "discont": flat list of float split time points, or None if absent

With --contours, converts all *_contour.npy files from MATLAB types to pure numpy.
With --params, converts all *_params.npy files that may contain embedded
matlab.double objects (from a previous conversion done on a MATLAB machine).

Must be run in an environment with the `matlab` package installed for .mat
conversion. The --params and --contours modes work without MATLAB by using
a shim for unpickling.

Requires scipy (for loadmat). Only needed for .mat → .npy conversion.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shim: allow unpickling .npy files that contain matlab.double objects
# even when the `matlab` package is not installed.
# ---------------------------------------------------------------------------
try:
    import matlab  # noqa: F401
    _MATLAB_INSTALLED = True
except ImportError:
    _MATLAB_INSTALLED = False

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------

def _is_matlab_double(obj) -> bool:
    """Check if an object is a matlab.double (real or shim)."""
    try:
        import matlab
        return isinstance(obj, matlab.double)
    except (ImportError, AttributeError):
        return False


def _convert_matlab_objects(obj):
    """Recursively convert MATLAB types to native NumPy/Python types.

    Handles:
        - matlab.double → np.ndarray (float64)
        - np.ndarray with dtype=object → recursive element conversion
        - list / tuple → recursive element conversion
        - dict → recursive value conversion (for params .npy structure)
        - Everything else → passthrough
    """
    # matlab.double (real or shim)
    if _is_matlab_double(obj):
        return np.asarray(obj, dtype=np.float64)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            # 0-d array is a scalar; unwrap and convert
            return _convert_matlab_objects(obj.item())
        if obj.dtype == object:
            converted = [_convert_matlab_objects(item) for item in obj]
            # Try to collapse to a numeric array if all elements are numeric
            try:
                return np.array(converted, dtype=np.float64)
            except (ValueError, TypeError):
                return np.array(converted, dtype=object)
        return obj

    # dict (params .npy structure: {"contour": ..., "discont": ...})
    if isinstance(obj, dict):
        return {k: _convert_matlab_objects(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        converted = [_convert_matlab_objects(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(converted)
        return converted

    return obj


def _verify_pure_numpy(obj, path: Path) -> bool:
    """Verify that no MATLAB types remain after conversion."""
    if _is_matlab_double(obj):
        print(f"  ⚠️  {path.name}: matlab.double still present after conversion!")
        return False
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj:
            if not _verify_pure_numpy(item, path):
                return False
    if isinstance(obj, dict):
        for v in obj.values():
            if not _verify_pure_numpy(v, path):
                return False
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if not _verify_pure_numpy(item, path):
                return False
    return True


# ---------------------------------------------------------------------------
# .mat → .npy conversion (requires scipy + matlab)
# ---------------------------------------------------------------------------

def convert_mat_to_npy(mat_path: Path, npy_path: Path) -> bool:
    """Convert a single .mat params file to .npy dict format.

    Returns True on success, False on failure.
    """
    if loadmat is None:
        print(f"  ❌ scipy not installed. Cannot load .mat files.")
        return False

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

    # Convert contour to pure numpy immediately
    contour = _convert_matlab_objects(contour)
    if not isinstance(contour, np.ndarray):
        contour = np.asarray(contour, dtype=np.float64)

    # Extract and normalize discont
    discont_list = None
    try:
        discont = mat_data['W']['discont'][0, 0]
        if discont.size > 0:
            discont = _convert_matlab_objects(discont)
            if isinstance(discont, np.ndarray):
                discont_list = discont.flatten().tolist()
            else:
                discont_list = [float(d) for d in discont]
    except (KeyError, IndexError, ValueError):
        discont_list = None

    # Save as .npy dict
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), {"contour": contour, "discont": discont_list}, allow_pickle=True)

    discont_info = f"{len(discont_list)} splits" if discont_list else "no splits"
    print(f"  ✅ {mat_path.name} → {npy_path.name} ({discont_info})")
    return True


# ---------------------------------------------------------------------------
# .npy cleanup: remove embedded matlab.double from precomputed contours
# ---------------------------------------------------------------------------

def convert_contour_npy(npy_path: Path) -> bool:
    """Convert a precomputed contour .npy file from MATLAB types to pure numpy.

    Works with or without the `matlab` package installed (uses shim if needed).
    Returns True on success, False on failure.
    """
    try:
        data = np.load(str(npy_path), allow_pickle=True)
    except Exception as e:
        print(f"  ❌ Failed to load {npy_path.name}: {e}")
        return False

    converted = _convert_matlab_objects(data)

    if not _verify_pure_numpy(converted, npy_path):
        print(f"  ⚠️  {npy_path.name}: conversion incomplete, saving anyway")

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


# ---------------------------------------------------------------------------
# .npy cleanup: remove embedded matlab.double from params files
# ---------------------------------------------------------------------------

def convert_params_npy(npy_path: Path) -> bool:
    """Clean a *_params.npy file that may contain embedded matlab.double objects.

    The file should contain a dict: {"contour": ndarray, "discont": list|None}
    Works with or without the `matlab` package installed (uses shim if needed).
    Returns True on success, False on failure.
    """
    try:
        data = np.load(str(npy_path), allow_pickle=True)
    except Exception as e:
        print(f"  ❌ Failed to load {npy_path.name}: {e}")
        return False

    # Convert the entire structure (dict, arrays, lists)
    converted = _convert_matlab_objects(data)

    # Ensure contour is a proper 2D float64 array
    if isinstance(converted, dict):
        contour = converted.get("contour")
        if contour is not None:
            if _is_matlab_double(contour):
                contour = np.asarray(contour, dtype=np.float64)
            elif isinstance(contour, np.ndarray) and contour.dtype == object:
                contour = np.asarray(contour, dtype=np.float64)
            converted["contour"] = contour

        # Ensure discont is a plain list of Python floats or None
        discont = converted.get("discont")
        if discont is not None:
            if isinstance(discont, np.ndarray):
                converted["discont"] = discont.flatten().tolist()
            elif isinstance(discont, (list, tuple)):
                converted["discont"] = [float(d) for d in discont]

    if not _verify_pure_numpy(converted, npy_path):
        print(f"  ⚠️  {npy_path.name}: conversion incomplete, saving anyway")

    np.save(str(npy_path), converted, allow_pickle=True)
    print(f"  ✅ {npy_path.name} → pure numpy")
    return True


def convert_params_dir(params_dir: Path) -> None:
    """Clean all *_params.npy files under a params directory."""
    npy_files = sorted(params_dir.rglob("*_params.npy"))
    if not npy_files:
        print(f"No params .npy files found in {params_dir}")
        return

    print(f"Cleaning {len(npy_files)} params file(s) in {params_dir}...")
    print()

    success = 0
    failed = 0
    for npy_path in npy_files:
        if convert_params_npy(npy_path):
            success += 1
        else:
            failed += 1

    print()
    print(f"✅ Done. {success} cleaned, {failed} failed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/convert_params_to_npy.py <params_dir>")
        print("    Converts all *_params.mat files to *_params.npy (dict format)")
        print()
        print("  python scripts/convert_params_to_npy.py --contours <precompute_dir>")
        print("    Converts all *_contour.npy files from MATLAB types to pure numpy")
        print()
        print("  python scripts/convert_params_to_npy.py --params <params_dir>")
        print("    Cleans all *_params.npy files of embedded matlab.double objects")
        print()
        print("  Examples:")
        print("    python scripts/convert_params_to_npy.py Projects/Benchmark/Params/")
        print("    python scripts/convert_params_to_npy.py --contours Projects/Benchmark/PrecomputeContours/")
        print("    python scripts/convert_params_to_npy.py --params Projects/Benchmark/Params/")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "--contours":
        if len(sys.argv) < 3:
            print("Error: --contours requires a directory argument.")
            sys.exit(1)
        precompute_dir = Path(sys.argv[2])
        if not precompute_dir.exists():
            print(f"❌ Directory not found: {precompute_dir}")
            sys.exit(1)
        convert_contours_dir(precompute_dir)
        return

    if mode == "--params":
        if len(sys.argv) < 3:
            print("Error: --params requires a directory argument.")
            sys.exit(1)
        params_dir = Path(sys.argv[2])
        if not params_dir.exists():
            print(f"❌ Directory not found: {params_dir}")
            sys.exit(1)
        convert_params_dir(params_dir)
        return

    # Default: .mat → .npy conversion
    params_dir = Path(mode)
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

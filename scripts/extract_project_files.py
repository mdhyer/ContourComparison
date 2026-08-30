"""Extract a subset of files from a full dataset into a streamlined project directory.

Usage:
    python scripts/extract_project_files.py \
        --input /path/to/full/dataset \
        --output /path/to/streamlined/project \
        --algorithms "CREPE" "Silbido Profundo" "Silbido Profundo Default" \
        --files "IND31349" "IND22694" "IND19003"

    # Optionally filter to specific SNR levels:
    python scripts/extract_project_files.py \
        --input /path/to/full/dataset \
        --output Projects/FixedSNR \
        --algorithms "CREPE" "Silbido Profundo" \
        --files "IND19003" \
        --snr-levels 0 5 10

Searches the input directory (standard project layout) for WAV files matching
the given IND identifiers, then copies:
  - WAV files        → output/data/...
  - Params files     → output/Params/...       (once per unique file)
  - Precomputed      → output/PrecomputeContours/<ALGO>/...  (per algorithm)

Supports both flat (data/FBID/file.wav) and nested (data/NoiseLevels/SNR/FBID/file.wav)
layouts. For nested data, all SNR levels are included by default unless --snr-levels
is specified.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_wavs(data_dir: Path, ind_patterns: list[str]) -> list[Path]:
    """Find all WAV files under data_dir whose name contains any of the IND patterns."""
    matches = []
    for wav in sorted(data_dir.rglob("*.wav")):
        name = wav.name
        for pattern in ind_patterns:
            if pattern in name:
                matches.append(wav)
                break
    return matches


def detect_layout(wavs: list[Path], data_dir: Path) -> str:
    """Detect whether the data uses flat or nested (NoiseLevels) layout."""
    for wav in wavs:
        rel = wav.relative_to(data_dir)
        if "NoiseLevels" in rel.parts:
            return "nested"
    return "flat"


def get_snr_level(wav: Path, data_dir: Path) -> str:
    """Extract the SNR level from a WAV path. Returns 'data' for flat layout."""
    rel = wav.relative_to(data_dir)
    parts = rel.parts
    if "NoiseLevels" in parts:
        idx = parts.index("NoiseLevels")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "data"


def get_base_stem(stem: str) -> str:
    """Strip _snr_N or _CLEAN_N suffix to get the base name for params lookup."""
    if "_snr_" in stem:
        return stem.split("_snr_")[0]
    if "_CLEAN_" in stem:
        return stem.split("_CLEAN_")[0]
    return stem


def copy_wav(wav: Path, input_root: Path, output_root: Path) -> Path:
    """Copy a WAV file preserving its relative path under data/."""
    rel = wav.relative_to(input_root / "data")
    dest = output_root / "data" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav, dest)
    return dest


def copy_params(wav: Path, input_root: Path, output_root: Path) -> bool:
    """Copy the corresponding params file (once). Returns True if found and copied."""
    label = wav.parent.name  # FBID directory name
    base_stem = get_base_stem(wav.stem)

    # Try .npy first, then .mat
    for suffix in (".npy", ".mat"):
        src = input_root / "Params" / label / f"{base_stem}_params{suffix}"
        if src.exists():
            dest = output_root / "Params" / label / f"{base_stem}_params{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)
            return True

    return False


def copy_precomputed(wav: Path, algorithm: str, input_root: Path, output_root: Path) -> bool:
    """Copy the precomputed contour for a specific algorithm. Returns True if found."""
    label = wav.parent.name
    intermediate = wav.parent.parent.name  # SNR level (nested) or "data" (flat)

    # Primary: full stem with _snr suffix
    src = (
        input_root
        / "PrecomputeContours"
        / algorithm
        / intermediate
        / label
        / f"{wav.stem}.npy"
    )
    if not src.exists():
        # Fallback: base stem without _snr suffix
        base_stem = get_base_stem(wav.stem)
        src = (
            input_root
            / "PrecomputeContours"
            / algorithm
            / intermediate
            / label
            / f"{base_stem}.npy"
        )

    if not src.exists():
        return False

    dest = (
        output_root
        / "PrecomputeContours"
        / algorithm
        / intermediate
        / label
        / f"{wav.stem}.npy"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract a subset of files from a full dataset into a streamlined project."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the full dataset project root (contains data/, Params/, PrecomputeContours/)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to the output streamlined project directory")
    parser.add_argument("--algorithms", required=True, nargs="+",
                        help='Algorithm names, e.g. "CREPE" "Silbido Profundo" "Silbido Profundo Default"')
    parser.add_argument("--files", required=True, nargs="+",
                        help='IND identifiers to match, e.g. "IND31349" "IND22694"')
    parser.add_argument("--snr-levels", nargs="*", default=None,
                        help='Optional: restrict to specific SNR levels (e.g. 0 5 10). '
                             'If omitted, all discovered SNR levels are included.')
    args = parser.parse_args()

    input_root: Path = args.input
    output_root: Path = args.output
    algorithms: list[str] = args.algorithms
    ind_patterns: list[str] = args.files
    snr_filter: list[str] | None = [str(s) for s in args.snr_levels] if args.snr_levels else None

    # Validate input
    if not input_root.exists():
        print(f"❌ Input directory not found: {input_root}")
        sys.exit(1)

    data_dir = input_root / "data"
    if not data_dir.exists():
        print(f"❌ No data/ directory found in {input_root}")
        sys.exit(1)

    # Find matching WAVs
    all_wavs = find_wavs(data_dir, ind_patterns)
    if not all_wavs:
        print(f"❌ No WAV files found matching, first pass: {ind_patterns}")
        sys.exit(1)

    # Detect layout and discover SNR levels
    layout = detect_layout(all_wavs, data_dir)
    all_snr_levels = sorted(set(get_snr_level(w, data_dir) for w in all_wavs),
                            key=lambda x: (len(x), x))

    print(f"📂 Layout: {layout}")
    print(f"📂 Discovered SNR levels: {', '.join(all_snr_levels)}")
    print(f"📂 Found {len(all_wavs)} WAV file(s) matching {ind_patterns}:")
    for w in all_wavs:
        snr = get_snr_level(w, data_dir)
        print(f"   [{snr}] {w.relative_to(input_root)}")
    print()

    # Apply SNR filter if specified
    if snr_filter:
        # Validate requested levels exist
        missing = [s for s in snr_filter if s not in all_snr_levels]
        if missing:
            print(f"⚠ Requested SNR level(s) not found in data: {', '.join(missing)}")
            print(f"  Available: {', '.join(all_snr_levels)}")
            sys.exit(1)

        wavs = [w for w in all_wavs if get_snr_level(w, data_dir) in snr_filter]
        print(f"🔍 Filtered to SNR levels {snr_filter}: {len(wavs)} WAV file(s) remain")
        print()
    else:
        wavs = all_wavs

    if not wavs:
        print("❌ No WAV files remain after filtering.")
        sys.exit(1)

    # Copy WAVs (once each)
    print(f"📁 Copying WAV files → {output_root / 'data'}/")
    for wav in wavs:
        dest = copy_wav(wav, input_root, output_root)
        print(f"   ✅ {dest.relative_to(output_root)}")
    print()

    # Copy Params (once per unique file)
    print(f"📁 Copying Params → {output_root / 'Params'}/")
    copied_params = set()
    for wav in wavs:
        label = wav.parent.name
        base_stem = get_base_stem(wav.stem)
        key = (label, base_stem)
        if key in copied_params:
            continue
        if copy_params(wav, input_root, output_root):
            copied_params.add(key)
            print(f"   ✅ {label}/{base_stem}_params.*")
        else:
            print(f"   ⚠ No params found for {label}/{base_stem}")
    print()

    # Copy Precomputed Contours (per algorithm, per SNR level)
    for algo in algorithms:
        print(f"📁 Copying Precomputed Contours [{algo}] → {output_root / 'PrecomputeContours' / algo}/")
        found = 0
        for wav in wavs:
            snr = get_snr_level(wav, data_dir)
            if copy_precomputed(wav, algo, input_root, output_root):
                found += 1
                print(f"   ✅ {wav.parent.name}/{wav.stem}.npy")
            else:
                print(f"   ⚠ [{snr}] No contour for {wav.name}")
        print(f"   ({found}/{len(wavs)} files)")
        print()

    # Summary
    snr_count = len(set(get_snr_level(w, data_dir) for w in wavs))
    print(f"✅ Done. {len(wavs)} WAVs across {snr_count} SNR level(s), "
          f"{len(copied_params)} params, {len(algorithms)} algorithm(s) → {output_root}")


if __name__ == "__main__":
    main()
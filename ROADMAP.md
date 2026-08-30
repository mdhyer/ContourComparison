# Roadmap: Data-Free GUI Demo Mode (Slim Dependencies)

## Goal

Enable the GUI to be shared and demonstrated **without** requiring:
- MATLAB installation
- PyTorch / CUDA
- Large audio datasets

The user should be able to:
- `pip install -e .` (no torch, no MATLAB)
- Launch the GUI
- Click "📂 Load Example Project" → picks from bundled `Projects/` subdirectories
- View the **Metrics Table** fully populated
- Render **all plot types** (SNR Trends, FBID Trends, Violin, Spectrogram Overlay,
  Single Prediction, Preprocessing, Mosaic, Fragmentation Verification)
- No real proprietary data, params, or contours in the repo

---

## Strategy

### A. Ship a minimal `Projects/` folder (repo root)

Follows the **exact** structure documented in README.md. Each project is a
self-contained directory with a few example files. Total size: < 5 MB.

```
Projects/
├── Benchmark/                          # Flat layout, single SNR
│   ├── data/
│   │   ├── FB001/
│   │   │   └── FB001_snr_0.wav         # real example audio (1–3s)
│   │   └── FB002/
│   │       └── FB002_snr_0.wav
│   ├── Params/
│   │   ├── FB001/
│   │   │   └── FB001_params.npy        # dict: {"contour": ..., "discont": ...}
│   │   └── FB002/
│   │       └── FB002_params.npy
│   ├── PrecomputeContours/
│   │   ├── Silbido Profundo/
│   │   │   └── data/
│   │   │       ├── FB001/FB001_snr_0_contour.npy
│   │   │       └── FB002/FB002_snr_0_contour.npy
│   │   ├── CREPE/
│   │   │   └── data/
│   │   │       ├── FB001/FB001_snr_0_contour.npy
│   │   │       └── FB002/FB002_snr_0_contour.npy
│   │   └── SAM/
│   │       └── data/
│   │           ├── FB001/FB001_snr_0_contour.npy
│   │           └── FB002/FB002_snr_0_contour.npy
│   └── results/
│       └── evaluation_metrics.json     # pre-computed, all mode combos
│
└── FixedSNR/                           # Nested layout, multiple SNR levels
    ├── data/
    │   └── NoiseLevels/
    │       ├── 0/
    │       │   └── FB001/
    │       │       └── FB001_snr_0.wav
    │       ├── 5/
    │       │   └── FB001/
    │       │       └── FB001_snr_5.wav
    │       └── 10/
    │           └── FB001/
    │               └── FB001_snr_10.wav
    ├── Params/
    │   └── FB001/
    │       └── FB001_params.npy        # dict: {"contour": ..., "discont": ...}
    ├── PrecomputeContours/
    │   ├── Silbido Profundo/
    │   │   ├── 0/FB001/FB001_snr_0_contour.npy
    │   │   ├── 5/FB001/FB001_snr_5_contour.npy
    │   │   └── 10/FB001/FB001_snr_10_contour.npy
    │   ├── CREPE/
    │   │   ├── 0/FB001/FB001_snr_0_contour.npy
    │   │   ├── 5/FB001/FB001_snr_5_contour.npy
    │   │   └── 10/FB001/FB001_snr_10_contour.npy
    │   └── SAM/
    │       ├── 0/FB001/FB001_snr_0_contour.npy
    │       ├── 5/FB001/FB001_snr_5_contour.npy
    │       └── 10/FB001/FB001_snr_10_contour.npy
    └── results/
        └── evaluation_metrics.json
```

**Key decisions:**
- Uses the **same** directory structure as the README documents
- **Real WAV files** (short clips, 1–3s) supplied by the team — no synthetic audio
- Ground truth as `.npy` **dict** with `contour` and `discont` fields → no `scipy.io.loadmat` at runtime
- Contours as `.npy` (object array of 2D arrays) → matches `load_precomputed()`
- `evaluation_metrics.json` in `results/` → matches where `save_metrics()` writes
- Two projects demonstrate both `DataLayout.FLAT` and `DataLayout.NESTED_NOISE`
- Lives at repo root (not inside the package) → no `package-data` needed

### B. Convert `.mat` params to `.npy` (one-time)

The `.mat` files store a struct `W` with fields `contour` and `discont`.
We convert them to a single `.npy` dict:

```python
# Conversion (one-time, in a dev env with scipy):
from scipy.io import loadmat
import numpy as np

mat_data = loadmat("FB001_params.mat")
contour = mat_data['W']['contour'][0, 0]       # 2D array [time, freq]
discont = mat_data['W']['discont'][0, 0]       # 1D or 2D array, or empty

# Normalize discont to a plain list or None
if discont.size == 0:
    discont_list = None
elif discont.ndim == 1:
    discont_list = discont.flatten().tolist()
else:
    discont_list = [float(d[0]) for d in discont]

np.save("FB001_params.npy", {"contour": contour, "discont": discont_list}, allow_pickle=True)
```

**Code changes to support the dict format:**

| Location | Change |
|----------|--------|
| `contour_utils.py` → `load_ground_truth()` | Update the `.npy` branch: if loaded object is a `dict`, unpack `contour` and `discont`. Otherwise (plain array), return `(loaded, None)` for backward compat. |
| `contour_utils.py` → `run_ground()` | Same dict-unpacking logic in the `.npy` branch. |
| `contour_utils.py` line 10 | Move `from scipy.io import loadmat` **inside** `_parse_mat_params()` and `run_ground()`. Only triggered if a `.mat` file is actually encountered. |

Updated `load_ground_truth()` `.npy` branch:

```python
if param_path.suffix == '.npy':
    loaded = np.load(param_path, allow_pickle=True)
    if isinstance(loaded, dict):
        return loaded["contour"], loaded.get("discont")
    return loaded, None  # backward compat: plain array
```

Updated `run_ground()` `.npy` branch:

```python
if param_path.suffix == '.npy':
    loaded = np.load(param_path, allow_pickle=True)
    if isinstance(loaded, dict):
        return [loaded["contour"]], loaded.get("discont") or []
    return [loaded], []
```

**Net effect:** No MATLAB runtime needed. `scipy` stays (for `wavfile`), but `loadmat` is deferred.

### C. Make PyTorch optional (the critical fix)

**The problem:** `crepe.py` has `import torch` and `import torchcrepe` at module level.
`pipeline.py` line 17 does `from .contour_extraction.crepe import CrepePredictor` at
module level. Without torch, importing `pipeline.py` crashes → the entire GUI fails.

| Location | Change |
|----------|--------|
| `crepe.py` | Wrap `import torch` and `import torchcrepe` in a module-level try/except. Set `torch = None` / `torchcrepe = None` on failure. In `CrepePredictor.__init__()`, raise `ImportError("CREPE requires PyTorch. Install with: pip install audio-analysis[models]")` if `torch is None`. |
| `pipeline.py` line 17 | Remove the module-level `from .contour_extraction.crepe import CrepePredictor`. Move it **inside** `_run_algorithm()` under the `if algorithm == "CREPE":` branch. This way `pipeline.py` imports cleanly without torch. |
| `pyproject.toml` | Move `torch`, `torchcrepe`, and `librosa` to `[project.optional-dependencies] models = [...]`. Keep `scipy`, `numpy`, `matplotlib`, `PyQt6`, etc. in base deps. |

**Net effect:** The GUI, pipeline (evaluate with precomputed data), plotting, and
metrics all work without PyTorch. Only *running* CREPE from scratch requires it.

### D. "Load Example Project" button in the GUI

**File:** `src/audio_analysis/gui/ui_components.py` → `ExecutionTab._setup_ui()`

Add a combo + button row above the existing `btn_layout`:

```python
example_row = QHBoxLayout()
example_row.addWidget(QLabel("Example:"))
self.ui_example_project = QComboBox()
self.ui_example_project.addItems(["— Select —"])  # populated at runtime
self.btn_load_example = QPushButton("📂 Load Example Project")
self.btn_load_example.setEnabled(False)
example_row.addWidget(self.ui_example_project)
example_row.addWidget(self.btn_load_example)
example_row.addStretch()
layout.addLayout(example_row)
```

**File:** `src/audio_analysis/gui/main_window.py`

#### 1. Discover available example projects at startup

In `__init__()` (after `_load_defaults()`):

```python
self._discover_example_projects()
```

```python
def _discover_example_projects(self):
    """Find bundled Projects/ directory and populate the example dropdown."""
    candidates = [
        Path.cwd() / "Projects",
        Path(__file__).parent.parent.parent.parent / "Projects",  # repo root
    ]
    projects_dir = None
    for c in candidates:
        if c.is_dir():
            projects_dir = c
            break

    if projects_dir is None:
        return

    subdirs = sorted([d.name for d in projects_dir.iterdir() if d.is_dir()])
    if not subdirs:
        return

    self._example_projects_dir = projects_dir
    self.execution_tab.ui_example_project.addItems(subdirs)
    self.execution_tab.btn_load_example.setEnabled(True)
    self.log(f"📂 Found {len(subdirs)} example project(s) in {projects_dir}")
```

#### 2. Handler

```python
def _load_example_project(self):
    """Load a bundled example project using the existing path resolution."""
    project_name = self.execution_tab.ui_example_project.currentText()
    if not project_name or project_name == "— Select —":
        return

    project_root = self._example_projects_dir / project_name
    if not project_root.exists():
        self.log(f"❌ Example project not found: {project_root}")
        return

    # Use the EXISTING _apply_project_root() logic
    self.config_tab.ui_project_root.setText(str(project_root))
    self._apply_project_root()

    # Discover files (auto-detects flat vs nested layout)
    self._discover_and_populate()

    # Load pre-computed metrics from results/
    metrics_path = project_root / "results" / "evaluation_metrics.json"
    if metrics_path.exists():
        self._load_metrics_from_path(metrics_path)
    else:
        self.log("⚠ No evaluation_metrics.json found in example project.")

    self.log(f"✅ Example project '{project_name}' loaded.")
```

#### 3. Helper

```python
def _load_metrics_from_path(self, path: Path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and "metrics" in data:
            self.aggregated_metrics = data["metrics"]
        else:
            self.aggregated_metrics = data
        self._log_global_metrics()
        self.log(f"✅ Loaded metrics from {path.name}")
    except Exception as e:
        self.log(f"❌ Failed to load metrics: {e}")
```

#### 4. Signal wiring in `_connect_signals()`

```python
self.execution_tab.btn_load_example.clicked.connect(self._load_example_project)
```

**No special "demo mode" flag needed.** The example project IS a valid project.
All existing code paths (discovery, plotting, metrics) work unchanged.

### E. Guard: "Single Prediction" in example mode

"Single Prediction" (`visualize_prediction`) runs the algorithm **live** on the WAV.
Without PyTorch/MATLAB, this will fail.

**Fix in `main_window.py` → `_run_plot()`:**

```python
if viz_type == "Single Prediction":
    self.log("ℹ️ Single Prediction falling back to precomputed contours (no model runtime).")
    viz_type = "Spectrogram Overlay (Together)"
```

This reuses `visualize_together(use_precomputed=True)` which loads the `.npy`
contour and overlays it on the spectrogram — no torch/MATLAB needed.

---

## Files Summary

| File | Action |
|------|--------|
| `Projects/` (new, repo root) | **NEW** — 2 example projects with real WAVs, `.npy` contours, `.npy` params (dict), metrics JSON |
| `src/audio_analysis/utils/contour_utils.py` | Lazy `loadmat` import; update `.npy` branches in `load_ground_truth()` and `run_ground()` to handle dict format |
| `src/audio_analysis/contour_extraction/crepe.py` | Lazy `torch`/`torchcrepe` import (try/except at module level) |
| `src/audio_analysis/pipeline.py` | Lazy import of `CrepePredictor` (move inside `_run_algorithm`) |
| `src/audio_analysis/gui/ui_components.py` | Add example project combo + button to `ExecutionTab` |
| `src/audio_analysis/gui/main_window.py` | Add `_discover_example_projects()`, `_load_example_project()`, `_load_metrics_from_path()`, signal wiring, Single Prediction fallback |
| `pyproject.toml` | Move `torch`, `torchcrepe`, `librosa` to `[project.optional-dependencies] models` |
| `scripts/convert_params_to_npy.py` (new) | **NEW** — one-time conversion script: `.mat` → `.npy` dict |

**No changes to:** `config.py`, `workers.py`, `metrics.py`, `plotting.py`, `silbido.py`
(already handles MATLAB absence gracefully).

---

## `pyproject.toml` Changes

```toml
dependencies = [
    "numpy",
    "scipy",
    "soundfile",
    "resampy",
    "dtw-python",
    "matplotlib",
    "tqdm",
    "pydantic>=2.0",
    "python-dotenv",
    "PyQt6"
]

[project.optional-dependencies]
models = [
    "torch",
    "torchcrepe",
    "librosa"
]
dev = [
    "pytest",
    "pytest-cov",
    "pytest-benchmark",
    "black",
    "flake8",
    "isort",
    "mypy",
    "pdoc3",
]
```

No `package-data` needed — `Projects/` is at repo root, not inside the package.

---

## Conversion Script: `scripts/convert_params_to_npy.py`

One-time dev tool to convert `.mat` params to `.npy` dicts:

```python
"""Convert .mat params files to .npy dict format for sharing without MATLAB."""
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import sys

def convert_mat_to_npy(mat_path: Path, npy_path: Path):
    """Convert a single .mat params file to .npy dict format."""
    mat_data = loadmat(mat_path)
    contour = mat_data['W']['contour'][0, 0]

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

    np.save(str(npy_path), {"contour": contour, "discont": discont_list}, allow_pickle=True)
    print(f"  ✅ {mat_path.name} → {npy_path.name} (discont: {'yes' if discont_list else 'no'})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_params_to_npy.py <params_dir>")
        print("  Converts all *_params.mat files to *_params.npy (dict format)")
        sys.exit(1)

    params_dir = Path(sys.argv[1])
    mat_files = sorted(params_dir.rglob("*_params.mat"))
    if not mat_files:
        print(f"No .mat files found in {params_dir}")
        return

    print(f"Converting {len(mat_files)} .mat files to .npy...")
    for mat_path in mat_files:
        npy_path = mat_path.with_suffix(".npy")
        convert_mat_to_npy(mat_path, npy_path)

    print(f"\n✅ Done. {len(mat_files)} files converted.")
    print("   You can now delete the .mat files if desired.")

if __name__ == "__main__":
    main()
```

---

## Starting State

This ROADMAP was written against the codebase at commit `da9b884`
("docs: add roadmap for data-free GUI demo mode"). Key assumptions about
the current code:

- `contour_utils.py` has `from scipy.io import loadmat` at module level (line 10)
- `crepe.py` has `import torch` and `import torchcrepe` at module level
- `pipeline.py` has `from .contour_extraction.crepe import CrepePredictor` at module level (line 17)
- `silbido.py` already wraps `import matlab.engine` in try/except and returns `[]` gracefully
- `load_ground_truth()` `.npy` branch currently returns `(np.load(...), None)` — ignores discont
- `run_ground()` `.npy` branch currently returns `([np.load(...)], [])` — ignores discont
- `pyproject.toml` has `torch`, `torchcrepe`, `librosa` in base `dependencies`
- No `Projects/` directory exists yet

If the base commit has changed, re-verify these assumptions before starting.

---

## Prerequisites

| Item | Needed For | Notes |
|------|-----------|-------|
| Real WAV files (1–3s clips) | `Projects/*/data/` | Supplied by the team. No generation script needed. |
| `.mat` params files | One-time conversion to `.npy` | Requires `scipy` (for `loadmat`). Only in dev env. |
| Precomputed contour `.npy` files | `Projects/*/PrecomputeContours/` | Already in `.npy` format. No conversion needed. Copy from existing precompute output. |
| Python 3.9+ with `scipy`, `numpy` | Running the conversion script + generating metrics JSON | No torch, no MATLAB needed. |

---

## Setup Workflow (Step-by-Step)

### Step 1: Create the project directory structure

```bash
mkdir -p Projects/Benchmark/{data,Params,PrecomputeContours,results}
mkdir -p Projects/FixedSNR/{data,Params,PrecomputeContours,results}
# ... create subdirectories per the structure in Section A
```

### Step 2: Place real WAV files

Copy the team-supplied WAV clips into the appropriate `data/` subdirectories.
Ensure filenames follow the convention: `<FBID>_snr_<LEVEL>.wav` (e.g., `FB001_snr_0.wav`).

### Step 3: Convert `.mat` params to `.npy` dicts

```bash
# In a dev environment with scipy installed:
python scripts/convert_params_to_npy.py Projects/Benchmark/Params/
python scripts/convert_params_to_npy.py Projects/FixedSNR/Params/
```

**Caveat:** The conversion script assumes the MATLAB struct is named `W`
(i.e., `mat_data['W']['contour']` and `mat_data['W']['discont']`). If any
params files use a different struct name, the script will need adjustment.
Verify by running `scipy.io.loadmat('file.mat')` and inspecting the keys.

### Step 4: Copy precomputed contours

The precomputed contour `.npy` files are already in the correct format
(object array of 2D arrays). Copy them from an existing precompute output
into the `PrecomputeContours/` subdirectories, preserving the
`ALGORITHM/INTERMEDIATE/LABEL/` structure.

For **flat layout** (Benchmark): `PrecomputeContours/ALGO/data/FBID/file_contour.npy`
For **nested layout** (FixedSNR): `PrecomputeContours/ALGO/SNR/FBID/file_contour.npy`

### Step 5: Generate `evaluation_metrics.json`

**This step does NOT require torch or MATLAB.** It only reads the `.npy`
precomputed contours and `.npy` ground truth params, then runs
`compare_contours` (pure numpy).

```bash
# From the repo root, with the package installed:
audio-analysis evaluate --project-root ./Projects/Benchmark
audio-analysis evaluate --project-root ./Projects/FixedSNR
```

This writes `results/evaluation_metrics.json` in each project directory.

**Alternative (if CLI isn't set up yet):** Run the code changes from
Sections B–C first, then use the GUI:
1. Launch GUI
2. Set Project Root to `Projects/Benchmark` → Apply
3. Click Evaluate
4. The metrics JSON is auto-saved to `results/`

### Step 6: Verify the `discont` format

The converted `.npy` params store `discont` as a flat list of floats
(e.g., `[0.35, 1.12]`). Verify this is the format expected by:
- `compare_contours()` in `evaluation/comparison.py`
- `plot_fragmentation_verification()` in `plotting/plotting.py`

**Test:** After loading an example project in the GUI, render
"Fragmentation Verification" and confirm discontinuities are visually
respected (contour segments are split at the correct time points).

If the format is wrong, the conversion script's `discont_list` normalization
may need adjustment (e.g., storing as a 2D array of `[start, end]` pairs
instead of a flat list of start times).

### Step 7: Commit

```bash
git add Projects/ scripts/convert_params_to_npy.py
git add src/audio_analysis/utils/contour_utils.py
git add src/audio_analysis/contour_extraction/crepe.py
git add src/audio_analysis/pipeline.py
git add src/audio_analysis/gui/ui_components.py
git add src/audio_analysis/gui/main_window.py
git add pyproject.toml
git commit -m "feat: add example projects and make torch/MATLAB optional"
```

---

## Open Questions / Risks

| # | Question | Mitigation |
|---|----------|-----------|
| 1 | Does `compare_contours()` expect `discont` as a flat list of floats, or as `[start, end]` pairs? | Verify by running Fragmentation Verification on a converted file. Check `comparison.py` source. |
| 2 | Do all `.mat` params files use the struct name `W`? | Inspect a few files with `loadmat()` before bulk conversion. |
| 3 | Will the precomputed contour `.npy` files from the team's existing pipeline be compatible with `load_precomputed()`? | They should be (same format), but verify by loading one in the GUI. |
| 4 | Does `visualize_together()` / `plot_metrics_mosaic()` need any additional files beyond WAV + precomputed contours + params? | Check `plotting.py` for any additional file I/O (e.g., spectrogram caches). |
| 5 | What is the total size of the example WAVs? | Keep each clip to 1–3s at 48kHz mono (~100–300 KB each). Target < 5 MB total for `Projects/`. |

## Resolved Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | `discont` format in `compare_contours()`? | **Flat list of floats** (split time points). `_split_ground_truth()` also accepts `[start, end]` pairs, but the existing `_parse_mat_params()` produces flat lists. The conversion script matches this. |

---

## What Does NOT Need to Change

- `config.py` — `AlgorithmConfig` already has `Optional` MATLAB paths
- `silbido.py` — already handles MATLAB absence gracefully (returns `[]`)
- `workers.py` — no changes needed; it calls the same pipeline/plot functions
- `metrics.py` — pure numpy, no heavy deps
- `plotting.py` — reads from `aggregated_metrics` dict and file paths; no changes
- `comparison.py` — pure numpy; no changes (unless `discont` format needs adjustment)

---

## Verification Checklist

- [ ] `pip install -e .` succeeds **without** torch or MATLAB
- [ ] `python -c "import audio_analysis.pipeline"` works without torch
- [ ] `python -m audio_analysis.gui` launches (no import errors)
- [ ] Log shows "Found 2 example project(s)"
- [ ] Select "FixedSNR" → Click "📂 Load Example Project"
- [ ] Log confirms paths set, 3 WAVs discovered, metrics loaded
- [ ] Metrics Table populates (3 algorithms × selected mode)
- [ ] SNR Trends → Render → 3-panel figure (one per algorithm)
- [ ] FBID Trends → Render → figure
- [ ] Metric Violin → Render → violin
- [ ] Spectrogram Overlay → Render → spectrogram + contours
- [ ] Single Prediction → Render → falls back to precomputed overlay
- [ ] Preprocessing → Render → figure
- [ ] Metrics Mosaic / v2 → Render → figure
- [ ] Fragmentation Verification → Render → figure (discontinuities respected)
- [ ] Select "Benchmark" → Click "📂 Load Example Project" → flat layout works
- [ ] `git status` shows only `Projects/` + the code changes above
- [ ] `pip install -e .[models]` → torch installs → CREPE can run live
- [ ] `python scripts/convert_params_to_npy.py Projects/Benchmark/Params/` works
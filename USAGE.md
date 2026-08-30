# Audio Analysis Pipeline Usage Guide

This guide provides side-by-side instructions for using the Audio Analysis Pipeline via the Command Line Interface (CLI) and the Graphical User Interface (GUI). Both interfaces share the exact same underlying configuration model, execution logic, metric calculations, and plotting engines.

## Configuration Management
The CLI and GUI use a unified `PipelineConfig` (Pydantic-backed). You can export a configuration from one interface and load it into the other seamlessly.

### Export Configuration
**CLI:**
```bash
audio-analysis --export-config my_workflow.json
```

**GUI:**
1. Navigate to the **Configuration** tab.
2. Click **Save Config** (top of the tab).
3. Choose a location and filename (e.g., `my_workflow.json`).

### Load Configuration
**CLI:**
```bash
audio-analysis --load-config my_workflow.json evaluate
```
*Note: CLI arguments provided after `--load-config` will override the loaded values.*

**GUI:**
1. Navigate to the **Configuration** tab.
2. Click **Load Config**.
3. Select your JSON file. All fields will automatically populate.

### Project Root Auto-Resolution
Both interfaces support a single `Project Root` that auto-resolves canonical subdirectories (`data/`, `Params/`, `PrecomputeContours/`, `results/`).

**CLI:**
```bash
audio-analysis --project-root /path/to/project evaluate
```

**GUI:**
1. Enter the root path in **Project Root**.
2. Click **Apply**. Subdirectory fields will auto-populate.
3. Click **Discover Data** to auto-fill FBIDs and SNR levels.

---

## Workflow Examples

### 1. Precompute Contours
Extract frequency contours using selected algorithms and save them for fast downstream evaluation.

**CLI:**
```bash
audio-analysis precompute \
  --data-root ./data \
  --algorithms "Silbido Profundo" CREPE SAM \
  --data-layout nested \
  --overwrite \
  --subsample 5
```

**GUI:**
1. **Configuration Tab**: Set `Data Root`, select `nested` layout, check desired algorithms.
2. **Preprocessing Controls**: Set `Subsample (every N)` to `5` (optional).
3. **Execution Tab**: Check `Overwrite Existing Contours`.
4. Click **Precompute**. Progress and logs appear in the console/log panel.

---

### 2. Evaluate Metrics
Compare extracted contours against ground truth, apply optional SNR filtering, and aggregate metrics across all mode combinations.

**CLI:**
```bash
audio-analysis evaluate \
  --data-root ./data \
  --output-dir ./results \
  --coverage-mode ffc \
  --freq-diff-threshold 0.25 \
  --filter-snr-20db
```

**GUI:**
1. **Configuration Tab**: Set `Data Root` & `Output Dir`.
2. **Metrics Configuration**: Select `ffc` for Coverage Mode, set `Freq Diff Threshold` to `0.25`.
3. **SNR Filtering**: Check `Filter SNR > 20dB` and browse to `SLSNR_20.json` if not using the package default.
4. **Execution Tab**: Click **Evaluate**.
5. Results are automatically saved to `evaluation_metrics.json` in the output directory and displayed in the **Metrics Table** tab.

---

### 3. Generate Visualizations
Visualize SNR trends, FBID rankings, spectrogram overlays, or metric distributions.

**CLI:**
```bash
audio-analysis plot \
  --data-root ./data \
  --plot-snr-trends \
  --plot-fbids \
  --plot-metric-violin "False Positives" \
  --target-wav ./data/NoiseLevels/CLEAN/FBID_001/sample.wav
```

**GUI:**
1. **Visualization Tab**: Select plot type from the `Viz Type` dropdown.
2. **Metric Violin**: When selected, a `Metric` dropdown appears. Choose your metric (Coverage, False Positives, etc.).
3. Set `Target WAV` (auto-selects first discovered file if empty).
4. Click **Render Plot**. Use **Plot Controls** to adjust axis limits or algorithm selectors.
5. Click **Save Plot** to export as PNG/PDF.

---

### 4. DTW FBID Accuracy Pipeline
Run a nearest-neighbor classification test using Dynamic Time Warping to measure FBID identification accuracy.

**CLI:**
```bash
audio-analysis dtw-fbid \
  --data-root ./data \
  --n-ground 10 \
  --n-comparisons 10 \
  --plot-type both
```

**GUI:**
1. Navigate to the **DTW FBID Accuracy** tab.
2. Set `N Ground Truth` and `N Comparisons` (default: 10 each).
3. Click **Run DTW FBID Accuracy**.
4. Results render in the embedded canvas. Toggle between **Violin Plot** and **Box Plot** using the `Display` dropdown.
5. *Note: "Ground" truth is automatically prepended to the algorithm list as a baseline.*

---

## Advanced & Preprocessing Controls
Both interfaces support fine-tuning contour extraction and metric calculation parameters:

| Parameter | CLI Flag | GUI Control |
|-----------|----------|-------------|
| Sample Rate | `--sample-rate 44100` | Audio Configuration > Sample Rate |
| Data Layout | `--data-layout nested` | Data Layout > Layout dropdown |
| Harmonic Removal | `--no-harmonic-removal` | Preprocessing Controls > Remove Harmonics |
| Harmonic Tolerance | `--harmonic-tolerance 0.7` | Preprocessing Controls > Harmonic Tolerance |
| Contour Splitting | `--no-contour-splitting` | Preprocessing Controls > Split Contours |
| Split Duration Threshold | `--contour-dur-threshold 0.05` | Preprocessing Controls > Contour Dur Threshold |
| Subsampling | `--subsample 5` | Preprocessing Controls > Subsample (every N) |
| Silbido Threshold 1 | `--silbido-threshold1 0.005` | Silbido Profundo Parameters > Threshold 1 |
| Silbido Threshold 2 | `--silbido-threshold2 0.4` | Silbido Profundo Parameters > Threshold 2 |
| Coverage Mode | `--coverage-mode ffc` | Metrics Configuration > Coverage Mode |
| Freq Diff Mode | `--freq-diff-mode total` | Metrics Configuration > Freq Diff Mode |
| Frag Mode | `--frag-mode total_per_loop` | Metrics Configuration > Frag Mode |
| Normalize Frag | `--normalize-frag` | Metrics Configuration > Normalize Frag by Loops |
| SNR > 20dB Filter | `--filter-snr-20db` | Metrics Configuration > Filter SNR > 20dB |

---

## Expected Directory Structure
Ensure your data follows the canonical layout for automatic discovery and path resolution:
```text
project-root/
├── data/
│   └── NoiseLevels/
│       ├── CLEAN/
│       │   └── FBID_001/
│       │       └── IND123_001.wav
│       └── SNR_20/
├── Params/
│   └── FBID_001/
│       └── IND123_001_params.mat
├── PrecomputeContours/
│   └── Silbido Profundo/
│       └── CLEAN/
│           └── FBID_001/
│               └── IND123_001_contour.npy
└── results/
    └── evaluation_metrics.json
```

## Tips & Best Practices
- **Fast Iteration**: Run `precompute` once, then iterate on `evaluate` and `plot` commands without re-running heavy contour extraction.
- **Metric Modes**: The pipeline computes all coverage/freq-diff/frag/recall mode combinations automatically. Use the GUI's **Metrics Table** dropdowns or CLI flags to view specific aggregations.
- **Cross-Platform Paths**: The SNR filter path (`SLSNR_20.json`) resolves relative to the package by default. Override it via CLI or GUI if your JSON lives elsewhere.
- **Logging**: All CLI runs and GUI executions stream progress and warnings to the console/log panel. Check for warnings regarding missing params or filtered files.

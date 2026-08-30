from __future__ import annotations

import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QFormLayout
)
from PyQt6.QtCore import Qt

from audio_analysis.config import PipelineConfig, CoverageMode, FreqDiffMode, FragMode, RecallMode, DataLayout
from audio_analysis.pipeline import _discover_wav_files, save_metrics
from audio_analysis.evaluation.metrics import load_metrics
from audio_analysis.utils.contour_utils import load_precomputed

from .workers import PlotWorker, DtwFbidWorker
from .ui_components import ConfigTab, ProjectTab, VisualizationTab, DtwFbidTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis Pipeline GUI")
        self.setMinimumSize(1200, 800)
        self.resize(1500, 1000)

        self.config = PipelineConfig()
        self.aggregated_metrics: Dict[str, Any] = {}
        self.wav_files: List[Path] = []
        self.plot_worker: Optional[PlotWorker] = None
        self.dtw_worker: Optional[DtwFbidWorker] = None
        self._example_projects_dir: Optional[Path] = None

        self.config_tab = ConfigTab()
        self.project_tab = ProjectTab()
        self.viz_tab = VisualizationTab()
        self.dtw_tab = DtwFbidTab()

        # Metrics Table Tab Container
        self.metrics_tab_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab_widget)

        # Dropdowns for metric modes
        self.metrics_cov_mode = QComboBox()
        self.metrics_cov_mode.addItems([m.value for m in CoverageMode])
        self.metrics_fd_mode = QComboBox()
        self.metrics_fd_mode.addItems([m.value for m in FreqDiffMode])
        self.metrics_frag_mode = QComboBox()
        self.metrics_frag_mode.addItems([m.value for m in FragMode])
        self.metrics_recall_mode = QComboBox()
        self.metrics_recall_mode.addItems([m.value for m in RecallMode])

        mode_layout = QFormLayout()
        mode_layout.addRow("Coverage Mode:", self.metrics_cov_mode)
        mode_layout.addRow("Freq Diff Mode:", self.metrics_fd_mode)
        mode_layout.addRow("Frag Mode:", self.metrics_frag_mode)
        mode_layout.addRow("Recall Mode:", self.metrics_recall_mode)

        metrics_layout.addLayout(mode_layout)

        # Metrics Table Widget
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Algorithm", "Coverage", "False Pos", "Freq Diff", "Fragmentation", "Recall"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setSortingEnabled(True)

        metrics_layout.addWidget(self.metrics_table)

        self._setup_ui()
        self._connect_signals()
        self._load_defaults()
        self._setup_fullscreen_toggle()
        self._discover_example_projects()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()

        # Combined Configuration & Data tab (two-column layout)
        combined_tab = QWidget()
        combined_layout = QHBoxLayout(combined_tab)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.setSpacing(0)

        # Left: Configuration (wider, scrollable)
        combined_layout.addWidget(self.config_tab, stretch=3)

        # Right: Project & Data (narrower, log fills vertical space)
        combined_layout.addWidget(self.project_tab, stretch=2)

        self.tabs.addTab(combined_tab, "Configuration & Data")
        self.tabs.addTab(self.viz_tab, "Visualization")
        self.tabs.addTab(self.metrics_tab_widget, "Metrics Table")
        self.tabs.addTab(self.dtw_tab, "DTW FBID Accuracy")
        main_layout.addWidget(self.tabs)

    def _connect_signals(self):
        # Project & Data tab
        self.project_tab.btn_load_metrics.clicked.connect(self._load_metrics)
        self.project_tab.btn_save_metrics.clicked.connect(self._save_results)
        self.project_tab.btn_load_example.clicked.connect(self._load_example_project)

        # Config tab
        self.config_tab.btn_save_config.clicked.connect(self._save_config)
        self.config_tab.btn_load_config.clicked.connect(self._load_config)
        self.config_tab.btn_discover.clicked.connect(self._discover_and_populate)
        self.config_tab.ui_algorithms.editingFinished.connect(self._sync_algo_dropdown)

        # Visualization tab
        self.viz_tab.btn_render_plot.clicked.connect(lambda: self._run_plot(self.viz_tab.ui_viz_type.currentText()))
        self.viz_tab.btn_save_plot.clicked.connect(self._save_plot)
        self.viz_tab.btn_browse_wav.clicked.connect(self._browse_wav)
        self.viz_tab.ui_viz_type.currentTextChanged.connect(self._on_viz_type_changed)

        # Browse buttons
        self.config_tab.btn_browse_data_root.clicked.connect(lambda: self._browse_dir(self.config_tab.ui_data_root))
        self.config_tab.btn_browse_params.clicked.connect(lambda: self._browse_dir(self.config_tab.ui_params_dir))
        self.config_tab.btn_browse_output.clicked.connect(lambda: self._browse_dir(self.config_tab.ui_output_dir))
        self.config_tab.btn_browse_precompute.clicked.connect(
            lambda: self._browse_dir(self.config_tab.ui_precompute_dir))
        self.config_tab.btn_browse_slsnr_filter.clicked.connect(
            lambda: self._browse_file(self.config_tab.ui_slsnr_filter_path, "JSON Files (*.json)")
        )
        self.config_tab.ui_filter_snr_20db.toggled.connect(self.config_tab.ui_slsnr_filter_path.setEnabled)
        self.config_tab.ui_filter_snr_20db.toggled.connect(self.config_tab.btn_browse_slsnr_filter.setEnabled)

        # Metric mode changes
        self.config_tab.ui_cov_mode.currentTextChanged.connect(self._on_metric_mode_changed)
        self.config_tab.ui_fd_mode.currentTextChanged.connect(self._on_metric_mode_changed)
        self.config_tab.ui_frag_mode.currentTextChanged.connect(self._on_metric_mode_changed)
        self.config_tab.ui_recall_mode.currentTextChanged.connect(self._on_metric_mode_changed)

        # Metrics Table tab dropdowns
        self.metrics_cov_mode.currentTextChanged.connect(self._on_metrics_tab_mode_changed)
        self.metrics_fd_mode.currentTextChanged.connect(self._on_metrics_tab_mode_changed)
        self.metrics_frag_mode.currentTextChanged.connect(self._on_metrics_tab_mode_changed)
        self.metrics_recall_mode.currentTextChanged.connect(self._on_metrics_tab_mode_changed)

        # Project Root selector
        if hasattr(self.config_tab, 'btn_apply_project_root'):
            self.config_tab.btn_apply_project_root.clicked.connect(self._apply_project_root)
        if hasattr(self.config_tab, 'ui_project_root'):
            self.config_tab.btn_browse_project_root.clicked.connect(
                lambda: self._browse_dir(self.config_tab.ui_project_root)
            )

        # DTW FBID Accuracy
        self.dtw_tab.btn_run.clicked.connect(self._run_dtw_fbid)
        self.dtw_tab.btn_save_plot.clicked.connect(self._save_dtw_plot)

        # Initialize UI visibility based on default selection
        self._on_viz_type_changed(self.viz_tab.ui_viz_type.currentText())

    def _load_defaults(self):
        # Block signals to prevent cross-talk during initialization
        widgets_to_block = [
            self.config_tab.ui_cov_mode, self.config_tab.ui_fd_mode,
            self.config_tab.ui_frag_mode, self.config_tab.ui_recall_mode,
            self.metrics_cov_mode, self.metrics_fd_mode,
            self.metrics_frag_mode, self.metrics_recall_mode
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        if hasattr(self.config_tab, 'ui_project_root'):
            self.config_tab.ui_project_root.setText(str(getattr(self.config.paths, 'project_root', Path('.'))))

        self.config_tab.ui_data_root.setText(str(self.config.paths.data_root))
        self.config_tab.ui_params_dir.setText(str(self.config.paths.params_dir))
        self.config_tab.ui_output_dir.setText(str(self.config.paths.output_dir))
        self.config_tab.ui_precompute_dir.setText(str(self.config.paths.precompute_dir))
        self.config_tab.ui_create_dirs.setChecked(self.config.paths.create_dirs)

        self.config_tab.ui_cov_mode.setCurrentText(self.config.metrics.coverage_mode.value)
        self.config_tab.ui_fd_mode.setCurrentText(self.config.metrics.freq_diff_mode.value)
        self.config_tab.ui_frag_mode.setCurrentText(self.config.metrics.frag_mode.value)
        self.config_tab.ui_recall_mode.setCurrentText(self.config.metrics.recall_mode.value)
        self.config_tab.ui_freq_diff_thresh.setValue(self.config.metrics.freq_diff_threshold)
        self.config_tab.ui_filter_snr_20db.setChecked(self.config.filter_snr_20db)
        self.config_tab.ui_slsnr_filter_path.setText(str(self.config.paths.slsnr_20_filter_path) if self.config.paths.slsnr_20_filter_path else "")
        self.config_tab.ui_slsnr_filter_path.setEnabled(self.config.filter_snr_20db)
        self.config_tab.btn_browse_slsnr_filter.setEnabled(self.config.filter_snr_20db)

        self.config_tab.ui_data_layout.setCurrentText(self.config.paths.data_layout.value)
        self.config_tab.ui_sample_rate.setValue(self.config.audio.sample_rate)

        self.config_tab.ui_rm_harm.setChecked(self.config.algorithm.remove_harmonics)
        self.config_tab.ui_harm_tol.setValue(self.config.algorithm.harmonic_tolerance)
        self.config_tab.ui_harm_freq.setValue(self.config.algorithm.harmonic_freqdiff)
        self.config_tab.ui_split_cont.setChecked(self.config.algorithm.split_contours)
        self.config_tab.ui_cont_dur.setValue(self.config.algorithm.contour_dur_threshold)
        self.config_tab.ui_subsample.setValue(1)

        self.config_tab.set_algorithms(["SAM", "Silbido Profundo", "SMC-PHD", "CREPE"])
        self.viz_tab.set_algorithm_options(["SAM", "Silbido Profundo", "SMC-PHD", "CREPE"])

        # Sync Metrics Table tab dropdowns
        self.metrics_cov_mode.setCurrentText(self.config.metrics.coverage_mode.value)
        self.metrics_fd_mode.setCurrentText(self.config.metrics.freq_diff_mode.value)
        self.metrics_frag_mode.setCurrentText(self.config.metrics.frag_mode.value)
        self.metrics_recall_mode.setCurrentText(self.config.metrics.recall_mode.value)

        # Unblock signals
        for w in widgets_to_block:
            w.blockSignals(False)

        # Ensure config is fully synced after UI updates
        self._sync_config()

    def _sync_config(self):
        if hasattr(self.config_tab, 'ui_project_root'):
            self.config.paths.project_root = Path(self.config_tab.ui_project_root.text())

        self.config.paths.data_root = Path(self.config_tab.ui_data_root.text())
        self.config.paths.params_dir = Path(self.config_tab.ui_params_dir.text())
        self.config.paths.output_dir = Path(self.config_tab.ui_output_dir.text())
        self.config.paths.precompute_dir = Path(self.config_tab.ui_precompute_dir.text())
        self.config.paths.create_dirs = self.config_tab.ui_create_dirs.isChecked()

        self.config.metrics.coverage_mode = CoverageMode(self.config_tab.ui_cov_mode.currentText())
        self.config.metrics.freq_diff_mode = FreqDiffMode(self.config_tab.ui_fd_mode.currentText())
        self.config.metrics.frag_mode = FragMode(self.config_tab.ui_frag_mode.currentText())
        self.config.metrics.recall_mode = RecallMode(self.config_tab.ui_recall_mode.currentText())
        self.config.metrics.freq_diff_threshold = self.config_tab.ui_freq_diff_thresh.value()
        self.config.filter_snr_20db = self.config_tab.ui_filter_snr_20db.isChecked()
        filter_path_text = self.config_tab.ui_slsnr_filter_path.text().strip()
        self.config.paths.slsnr_20_filter_path = Path(filter_path_text) if filter_path_text else None

        self.config.paths.data_layout = DataLayout(self.config_tab.ui_data_layout.currentText())
        self.config.audio.sample_rate = self.config_tab.ui_sample_rate.value()

        self.config.algorithm.remove_harmonics = self.config_tab.ui_rm_harm.isChecked()
        self.config.algorithm.harmonic_tolerance = self.config_tab.ui_harm_tol.value()
        self.config.algorithm.harmonic_freqdiff = self.config_tab.ui_harm_freq.value()
        self.config.algorithm.split_contours = self.config_tab.ui_split_cont.isChecked()
        self.config.algorithm.contour_dur_threshold = self.config_tab.ui_cont_dur.value()

    def _sync_algo_dropdown(self):
        """Re-sync the viz algorithm dropdown when the user edits the text field."""
        algos = self.config_tab.get_selected_algorithms()
        if algos:
            self.viz_tab.set_algorithm_options(algos)

    def _on_metric_mode_changed(self):
        self._sync_config()
        # Sync metrics table tab dropdowns
        self.metrics_cov_mode.setCurrentText(self.config.metrics.coverage_mode.value)
        self.metrics_fd_mode.setCurrentText(self.config.metrics.freq_diff_mode.value)
        self.metrics_frag_mode.setCurrentText(self.config.metrics.frag_mode.value)
        self.metrics_recall_mode.setCurrentText(self.config.metrics.recall_mode.value)
        self._log_global_metrics()

    def _on_metrics_tab_mode_changed(self):
        self.config.metrics.coverage_mode = CoverageMode(self.metrics_cov_mode.currentText())
        self.config.metrics.freq_diff_mode = FreqDiffMode(self.metrics_fd_mode.currentText())
        self.config.metrics.frag_mode = FragMode(self.metrics_frag_mode.currentText())
        self.config.metrics.recall_mode = RecallMode(self.metrics_recall_mode.currentText())
        # Sync config tab dropdowns
        self.config_tab.ui_cov_mode.setCurrentText(self.config.metrics.coverage_mode.value)
        self.config_tab.ui_fd_mode.setCurrentText(self.config.metrics.freq_diff_mode.value)
        self.config_tab.ui_frag_mode.setCurrentText(self.config.metrics.frag_mode.value)
        self.config_tab.ui_recall_mode.setCurrentText(self.config.metrics.recall_mode.value)
        self._log_global_metrics()

    def _log_global_metrics(self):
        if not self.aggregated_metrics:
            self.metrics_table.setRowCount(0)
            return
        mode_key = f"{self.config.metrics.coverage_mode.value}_{self.config.metrics.freq_diff_mode.value}_{self.config.metrics.frag_mode.value}_{self.config.metrics.recall_mode.value}"
        self.log(f"📊 Global Metrics ({mode_key}):")

        self.metrics_table.setRowCount(0)
        row = 0
        for algo, m in self.aggregated_metrics.items():
            global_m = m.get('global', {})
            mode_metrics = global_m.get(mode_key, {})
            if not mode_metrics: continue

            cov = mode_metrics.get('coverage', 0)
            cov_std = mode_metrics.get('coverage_std', 0)
            fp = mode_metrics.get('false_pos', 0)
            fp_std = mode_metrics.get('false_pos_std', 0)
            fd = mode_metrics.get('freq_diff', 0)
            fd_std = mode_metrics.get('freq_diff_std', 0)
            frag = mode_metrics.get('fragmentation', 0)
            frag_std = mode_metrics.get('fragmentation_std', 0)
            recall = mode_metrics.get('recall', 0)
            recall_std = mode_metrics.get('recall_std', 0)

            # Helper to safely format std dev (handles NaN gracefully)
            def _std_fmt(v, prec=3):
                return f"{v:.{prec}f}" if isinstance(v, (int, float)) and v == v else "N/A"

            self.log(
                f"  {algo}: Coverage={cov:.3f}±{_std_fmt(cov_std)}, FP={fp:.1f}±{_std_fmt(fp_std, 1)}, FreqDiff={fd:.2f}±{_std_fmt(fd_std, 2)}, Frag={frag:.2f}±{_std_fmt(frag_std, 2)}, Recall={recall:.3f}±{_std_fmt(recall_std)}")

            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(algo))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{cov:.3f} ± {_std_fmt(cov_std)}"))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{fp:.1f} ± {_std_fmt(fp_std, 1)}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{fd:.2f} ± {_std_fmt(fd_std, 2)}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{frag:.2f} ± {_std_fmt(frag_std, 2)}"))
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{recall:.3f} ± {_std_fmt(recall_std)}"))
            row += 1

    def _discover_and_populate(self):
        self._sync_config()
        data_root = self.config.paths.data_root
        params_dir = self.config.paths.params_dir

        fbids = []
        snrs = []
        if data_root.exists():
            snrs = [d.name for d in (data_root / "NoiseLevels").iterdir() if d.is_dir()] if (
                    data_root / "NoiseLevels").exists() else ["CLEAN"]
            for snr in snrs:
                snr_path = data_root / "NoiseLevels" / snr
                if snr_path.exists():
                    fbids.extend([d.name for d in snr_path.iterdir() if d.is_dir()])
        if params_dir.exists():
            fbids.extend([d.name for d in params_dir.iterdir() if d.is_dir()])

        fbids = sorted(set(fbids))
        snrs = sorted(set(snrs))

        self.config_tab.ui_fbids.setText(", ".join(fbids))
        self.config_tab.ui_snr_levels.setText(", ".join(snrs))
        self.log(f"🔍 Discovered {len(fbids)} FBIDs and {len(snrs)} SNR levels.")

        # Discover algorithms from precompute directory
        algo_dirs = []
        if self.config.paths.precompute_dir.exists():
            algo_dirs = [d.name for d in self.config.paths.precompute_dir.iterdir() if d.is_dir()]
        if algo_dirs:
            self.config_tab.set_algorithms(sorted(set(algo_dirs)))
            self.viz_tab.set_algorithm_options(sorted(set(algo_dirs)))
            self.log(f"🔍 Discovered {len(algo_dirs)} algorithms in precompute dir.")

        # Auto-update target WAV field when discovering data
        wav_files = _discover_wav_files(data_root, None, None, self.config.paths.data_layout)
        if wav_files:
            self.wav_files = wav_files
            self.viz_tab.ui_target_wav.setText(str(wav_files[0]))
            self.log(f"🎯 Target WAV updated to: {wav_files[0].name}")

    def _discover_files(self):
        self._sync_config()
        fbids_text = self.config_tab.ui_fbids.text().strip()
        snrs_text = self.config_tab.ui_snr_levels.text().strip()

        fbids = [t.strip() for t in fbids_text.split(",") if t.strip()] if fbids_text else None
        snrs = [t.strip() for t in snrs_text.split(",") if t.strip()] if snrs_text else None

        self.wav_files = _discover_wav_files(
            self.config.paths.data_root, fbids, snrs, self.config.paths.data_layout
        )
        if not self.wav_files:
            self.log("⚠ No WAV files found in Data Root.")
            return False
        self.log(f"📂 Discovered {len(self.wav_files)} WAV files.")

        if not self.viz_tab.ui_target_wav.text():
            self.viz_tab.ui_target_wav.setText(str(self.wav_files[0]))
        return True

    # ─── Example Project Discovery & Loading ───────────────────────────────────

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
        self.project_tab.ui_example_project.addItems(subdirs)
        self.project_tab.btn_load_example.setEnabled(True)
        self.log(f"📂 Found {len(subdirs)} example project(s) in {projects_dir}")

    def _load_example_project(self):
        """Load a bundled example project using the existing path resolution."""
        project_name = self.project_tab.ui_example_project.currentText()
        if not project_name or project_name == "— Select —":
            return

        project_root = self._example_projects_dir / project_name
        if not project_root.exists():
            self.log(f"❌ Example project not found: {project_root}")
            return

        # Use the existing _apply_project_root() logic
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

    def _load_metrics_from_path(self, path: Path):
        """Load metrics JSON from a known path (no file dialog)."""
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

    # ─── Plotting ──────────────────────────────────────────────────────────────

    def _on_viz_type_changed(self, text: str):
        # Show limits for all plot types
        self.viz_tab.show_limits(True)

        # Reset toggle state when switching plot types
        self.viz_tab.controls_toggle.setChecked(False)
        self.viz_tab.controls_frame.setVisible(False)

        # Hide algorithm selector for plots that automatically use all checked algorithms
        needs_algo_selector = text in ["Single Prediction", "Metrics Mosaic", "Metrics Mosaic v2", "Fragmentation Verification"]
        self.viz_tab.set_algo_selector_visible(needs_algo_selector)

        # Switch axis controls based on plot type
        if text in ["SNR Trends", "FBID Trends"]:
            self.viz_tab.set_axis_mode("metrics")
        else:
            self.viz_tab.set_axis_mode("spectrogram")

        # Show metric selector only when Metric Violin is selected
        self.viz_tab.ui_metric_selector.setVisible(text == "Metric Violin")

        # Show spectrogram checkbox only for Fragmentation Verification
        self.viz_tab.ui_show_spectrogram.setVisible(text == "Fragmentation Verification")

    def _run_plot(self, viz_type: str):
        self._sync_config()

        # Ensure wav_files are discovered if not already populated
        if not self.wav_files:
            self._discover_files()

        selected_algo = self.viz_tab.ui_plot_algo.currentText()
        checked_algos = self.config_tab.get_selected_algorithms()

        # Prioritize dropdown selection for plots that use it
        if viz_type in ["Single Prediction", "Metrics Mosaic", "Metrics Mosaic v2", "Fragmentation Verification"]:
            checked_algos = [selected_algo]

        target_wav = self.viz_tab.ui_target_wav.text() or (str(self.wav_files[0]) if self.wav_files else None)

        if viz_type in ["SNR Trends", "FBID Trends"] and not any(a in self.aggregated_metrics for a in checked_algos):
            self.log(f"⚠ No metrics found for selected algorithms. Load Metrics first.")
            return

        if viz_type == "Metric Violin" and not self.aggregated_metrics:
            self.log("⚠ No metrics found. Load Metrics first.")
            return

        # Guard: check for precomputed contours before spawning worker
        needs_precomputed = viz_type in [
            "Spectrogram Overlay (Together)", "Single Prediction", "Preprocessing",
            "Metrics Mosaic", "Metrics Mosaic v2", "Fragmentation Verification"
        ]
        if needs_precomputed and target_wav:
            try:
                load_precomputed(target_wav, selected_algo, top_dir=str(self.config.paths.precompute_dir))
            except FileNotFoundError:
                self.log(f"⚠ No precomputed contours found for '{selected_algo}' on {Path(target_wav).name}. "
                         f"This plot type requires contour data in the Precompute directory.")
                return

        limits = self.viz_tab.get_limits()
        snr_min, snr_max = self.viz_tab.get_snr_plot_range()
        show_spectrogram = self.viz_tab.ui_show_spectrogram.isChecked()

        self.viz_tab.btn_render_plot.setEnabled(False)
        self.project_tab.progress_bar.setValue(0)

        # Resolve actual plot type string for the worker
        actual_plot_type = self.viz_tab.ui_metric_selector.currentText() if viz_type == "Metric Violin" else viz_type

        self.plot_worker = PlotWorker(self.config, self.wav_files, self.aggregated_metrics, selected_algo, actual_plot_type,
                                      target_wav, checked_algos, limits=limits, snr_plot_min=snr_min, snr_plot_max=snr_max,
                                      show_spectrogram=show_spectrogram)
        self.plot_worker.log_signal.connect(self.log)
        self.plot_worker.progress_signal.connect(self.project_tab.progress_bar.setValue)
        self.plot_worker.figure_signal.connect(self._update_canvas)
        self.plot_worker.limits_signal.connect(self.viz_tab.update_limits_from_plot)
        self.plot_worker.error_signal.connect(self._on_error)
        self.plot_worker.finished.connect(self._on_plot_worker_finished)
        self.plot_worker.start()

    def _on_plot_worker_finished(self):
        self.viz_tab.btn_render_plot.setEnabled(True)
        self.plot_worker = None

    # ─── Metrics I/O ───────────────────────────────────────────────────────────

    def _save_results(self):
        self._sync_config()
        if not self.aggregated_metrics:
            self.log("⚠ No results to save. Load Metrics first.")
            return

        default_path = str(self.config.paths.output_dir / "evaluation_metrics.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save Evaluation Results", default_path,
                                              "JSON Files (*.json);;All Files (*)")
        if path:
            try:
                save_metrics(self.aggregated_metrics, Path(path), config=self.config)
                self.log(f"✅ Results saved to {path}")
            except Exception as e:
                self.log(f"❌ Failed to save results: {e}")

    def _load_metrics(self):
        default_dir = str(getattr(self.config.paths, 'project_root', Path('.')))
        path, _ = QFileDialog.getOpenFileName(self, "Load Aggregated Metrics", default_dir, "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                # Extract metrics (handles both new nested and legacy flat formats)
                if isinstance(data, dict) and "metrics" in data:
                    self.aggregated_metrics = data["metrics"]
                else:
                    self.aggregated_metrics = data

                # Extract and apply embedded config if present
                if isinstance(data, dict) and "config" in data:
                    try:
                        self.config = PipelineConfig.model_validate(data["config"])
                        self._load_defaults()
                        self.log(f"✅ Loaded metrics and restored configuration from {path}")
                    except Exception as e:
                        self.log(f"⚠ Metrics loaded, but config restoration failed: {e}")
                else:
                    self.log(f"✅ Loaded metrics from {path}")

                self._log_global_metrics()
            except Exception as e:
                self.log(f"❌ Failed to load metrics: {e}")

    # ─── Plot I/O ──────────────────────────────────────────────────────────────

    def _save_plot(self):
        self._sync_config()
        if self.viz_tab.canvas.figure is None:
            self.log("⚠ No plot to save.")
            return

        viz_type = self.viz_tab.ui_viz_type.currentText().replace(' ', '_')
        default_path = str(self.config.paths.output_dir / f"plot_{viz_type}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", default_path,
                                              "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if path:
            try:
                self.viz_tab.canvas.figure.savefig(path, dpi=600, bbox_inches="tight")
                self.log(f"✅ Plot saved to {path}")
            except Exception as e:
                self.log(f"❌ Failed to save plot: {e}")

    # ─── Config I/O ────────────────────────────────────────────────────────────

    def _save_config(self):
        self._sync_config()
        default_path = str(self.config.paths.output_dir / "pipeline_config.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", default_path, "JSON Files (*.json)")
        if path:
            from audio_analysis.utils.config_io import save_config
            save_config(self.config, Path(path))
            self.log(f"✅ Configuration saved to {path}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if path:
            from audio_analysis.utils.config_io import load_config
            try:
                self.config = load_config(Path(path))
                self._load_defaults()
                self.log(f"✅ Configuration loaded from {path}")
            except Exception as e:
                self.log(f"❌ Failed to load config: {e}")

    # ─── Browse helpers ────────────────────────────────────────────────────────

    def _browse_dir(self, line_edit):
        default_dir = str(getattr(self.config.paths, 'project_root', self.config.paths.data_root))
        path = QFileDialog.getExistingDirectory(self, "Select Directory", default_dir)
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit, filter_str="All Files (*)"):
        default_dir = str(getattr(self.config.paths, 'project_root', self.config.paths.data_root))
        path, _ = QFileDialog.getOpenFileName(self, "Select File", default_dir, filter_str)
        if path:
            line_edit.setText(path)

    def _browse_wav(self):
        default_dir = str(self.config.paths.data_root)
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV File", default_dir, "WAV Files (*.wav)")
        if path:
            self.viz_tab.ui_target_wav.setText(path)

    # ─── Project Root ──────────────────────────────────────────────────────────

    def _apply_project_root(self):
        """Apply project root and resolve canonical subdirectories."""
        new_root = self.config_tab.ui_project_root.text().strip()
        if not new_root or not Path(new_root).exists():
            self.log("⚠ Invalid project root path. Please check the directory.")
            return

        root = Path(new_root)
        self.config.paths.project_root = root
        self.config.paths.data_root = root / "data"
        self.config.paths.params_dir = root / "Params"
        self.config.paths.precompute_dir = root / "PrecomputeContours"
        self.config.paths.output_dir = root / "results"

        # Sync UI fields
        if hasattr(self.config_tab, 'ui_project_root'):
            self.config_tab.ui_project_root.setText(str(self.config.paths.project_root))
        self.config_tab.ui_data_root.setText(str(self.config.paths.data_root))
        self.config_tab.ui_params_dir.setText(str(self.config.paths.params_dir))
        self.config_tab.ui_precompute_dir.setText(str(self.config.paths.precompute_dir))
        self.config_tab.ui_output_dir.setText(str(self.config.paths.output_dir))
        self.log(f"📁 Project root applied. Subdirectories auto-configured.")

    # ─── Canvas & Logging ──────────────────────────────────────────────────────

    def _update_canvas(self, fig):
        if self.viz_tab.canvas.figure is not None:
            plt.close(self.viz_tab.canvas.figure)
        self.viz_tab.canvas.set_figure(fig)
        self.viz_tab.canvas.draw()

    def log(self, msg: str):
        self.project_tab.log_output.append(msg)
        self.project_tab.log_output.verticalScrollBar().setValue(
            self.project_tab.log_output.verticalScrollBar().maximum())

    # ─── Error handling ────────────────────────────────────────────────────────

    def _on_error(self, msg: str):
        self.log(msg)
        QMessageBox.critical(self, "Pipeline Error", msg)
        self._on_plot_worker_finished()

    # ─── Fullscreen ────────────────────────────────────────────────────────────

    def _setup_fullscreen_toggle(self):
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("View")
        self.fullscreen_action = view_menu.addAction("⛶ Fullscreen")
        self.fullscreen_action.setShortcut("F11")
        self.fullscreen_action.triggered.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ─── DTW FBID Accuracy ─────────────────────────────────────────────────────

    def _run_dtw_fbid(self):
        self._sync_config()
        algos = self.config_tab.get_selected_algorithms()
        if not algos:
            self.log("⚠ No algorithms selected.")
            return

        # Always include Ground truth for DTW baseline comparison, placed first
        if "Ground" not in algos:
            algos.insert(0, "Ground")

        self.dtw_tab.btn_run.setEnabled(False)
        self.project_tab.progress_bar.setValue(0)

        self.dtw_worker = DtwFbidWorker(
            self.config,
            algos,
            self.dtw_tab.ui_n_ground.value(),
            self.dtw_tab.ui_n_comp.value()
        )
        self.dtw_worker.log_signal.connect(self.log)
        self.dtw_worker.progress_signal.connect(self.project_tab.progress_bar.setValue)
        self.dtw_worker.finished_signal.connect(self._on_dtw_finished)
        self.dtw_worker.error_signal.connect(self._on_error)
        self.dtw_worker.finished.connect(self._on_dtw_worker_finished)
        self.dtw_worker.start()

    def _on_dtw_finished(self, data: dict):
        self.log("✅ DTW FBID Accuracy complete.")
        self.dtw_tab.update_plots(data['fig_violin'], data['fig_box'])

        # Log metrics
        for algo, m in data['metrics'].items():
            self.log(f"  {algo}: Top-1={m['top1']:.2%}, Top-5={m['top5']:.2%}")

    def _on_dtw_worker_finished(self):
        self.dtw_tab.btn_run.setEnabled(True)
        self.dtw_worker = None

    def _save_dtw_plot(self):
        if self.dtw_tab.canvas.figure is None:
            self.log("⚠ No DTW plot to save.")
            return

        default_path = str(self.config.paths.output_dir / "dtw_fbid_accuracy.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save DTW Plot", default_path,
                                              "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if path:
            try:
                self.dtw_tab.canvas.figure.savefig(path, dpi=600, bbox_inches="tight")
                self.log(f"✅ DTW plot saved to {path}")
            except Exception as e:
                self.log(f"❌ Failed to save plot: {e}")

    # ─── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Ensure all background workers are stopped and the application quits cleanly."""
        if self.plot_worker and self.plot_worker.isRunning():
            self.plot_worker.terminate()
            self.plot_worker.wait()
        if self.dtw_worker and self.dtw_worker.isRunning():
            self.dtw_worker.terminate()
            self.dtw_worker.wait()
        event.accept()
        QApplication.quit()

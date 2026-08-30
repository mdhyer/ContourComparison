from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QFileDialog,
    QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLabel,
    QTextEdit, QProgressBar, QGroupBox, QScrollArea, QToolButton, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
import matplotlib

try:
    matplotlib.use('QtAgg')
except RuntimeError:
    pass

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

from audio_analysis.config import CoverageMode, FreqDiffMode, FragMode, RecallMode


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in the GUI."""

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()

    def resizeEvent(self, event):
        """Handle window resizing to prevent stretching and ensure proper redraw."""
        super().resizeEvent(event)
        self.draw()

    def set_figure(self, fig):
        """Replace the current figure and prepare for redraw."""
        self.figure = fig
        self.draw()


class ConfigTab(QWidget):
    """Configuration tab with directory paths, algorithm selection, and preprocessing controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        config_io_grp = QGroupBox("Configuration I/O")
        config_io_layout = QVBoxLayout()

        config_row = QHBoxLayout()
        self.btn_save_config = QPushButton("💾 Save Config")
        self.btn_load_config = QPushButton("📂 Load Config")
        config_row.addWidget(self.btn_save_config)
        config_row.addWidget(self.btn_load_config)
        config_row.addStretch()

        config_io_layout.addLayout(config_row)
        config_io_grp.setLayout(config_io_layout)
        scroll_layout.addWidget(config_io_grp)

        paths_grp = QGroupBox("Directory Paths")
        paths_layout = QFormLayout()
        self.ui_project_root = QLineEdit()
        self.ui_data_root = QLineEdit()
        self.ui_params_dir = QLineEdit()
        self.ui_output_dir = QLineEdit()
        self.ui_precompute_dir = QLineEdit()

        self.btn_browse_project_root = QPushButton("Browse")
        self.btn_apply_project_root = QPushButton("Apply")
        self.btn_browse_data_root = QPushButton("Browse")
        self.btn_browse_params = QPushButton("Browse")
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_precompute = QPushButton("Browse")

        # Project Root row with Apply button
        pr_row = QHBoxLayout()
        pr_row.addWidget(self.ui_project_root)
        pr_row.addWidget(self.btn_browse_project_root)
        pr_row.addWidget(self.btn_apply_project_root)
        paths_layout.addRow("Project Root:", pr_row)

        for name, line, btn in [
            ("Data Root", self.ui_data_root, self.btn_browse_data_root),
            ("Params Dir", self.ui_params_dir, self.btn_browse_params),
            ("Output Dir", self.ui_output_dir, self.btn_browse_output),
            ("Precompute Dir", self.ui_precompute_dir, self.btn_browse_precompute)
        ]:
            row = QHBoxLayout()
            row.addWidget(line)
            row.addWidget(btn)
            paths_layout.addRow(name, row)

        self.ui_create_dirs = QCheckBox("Create Missing Directories")
        paths_layout.addRow("Auto-Create Dirs:", self.ui_create_dirs)

        paths_grp.setLayout(paths_layout)
        scroll_layout.addWidget(paths_grp)

        disc_grp = QGroupBox("Auto-Discovery")
        disc_layout = QHBoxLayout()
        self.btn_discover = QPushButton("🔍 Discover Data")
        disc_layout.addWidget(self.btn_discover)
        disc_layout.addWidget(QLabel("FBIDs:"))
        self.ui_fbids = QLineEdit()
        self.ui_fbids.setPlaceholderText("Comma-separated FBIDs (leave empty for all)")
        disc_layout.addWidget(self.ui_fbids)
        disc_layout.addWidget(QLabel("SNR Levels:"))
        self.ui_snr_levels = QLineEdit()
        self.ui_snr_levels.setPlaceholderText("Comma-separated SNRs (leave empty for all)")
        disc_layout.addWidget(self.ui_snr_levels)
        disc_grp.setLayout(disc_layout)
        scroll_layout.addWidget(disc_grp)

        algo_grp = QGroupBox("Algorithms")
        algo_layout = QVBoxLayout()

        algo_layout.addWidget(QLabel("Algorithms (comma-separated):"))
        self.ui_algorithms = QLineEdit()
        self.ui_algorithms.setPlaceholderText("e.g., Silbido Profundo, SAM, CREPE")
        algo_layout.addWidget(self.ui_algorithms)

        algo_grp.setLayout(algo_layout)
        scroll_layout.addWidget(algo_grp)

        metrics_grp = QGroupBox("Metrics Configuration")
        metrics_layout = QFormLayout()
        self.ui_cov_mode = QComboBox()
        self.ui_cov_mode.addItems([m.value for m in CoverageMode])
        self.ui_fd_mode = QComboBox()
        self.ui_fd_mode.addItems([m.value for m in FreqDiffMode])
        self.ui_frag_mode = QComboBox()
        self.ui_frag_mode.addItems([m.value for m in FragMode])
        self.ui_recall_mode = QComboBox()
        self.ui_recall_mode.addItems([m.value for m in RecallMode])
        self.ui_freq_diff_thresh = QDoubleSpinBox()
        self.ui_freq_diff_thresh.setRange(0.0, 1.0)
        self.ui_freq_diff_thresh.setDecimals(2)
        self.ui_freq_diff_thresh.setSingleStep(0.05)
        self.ui_freq_diff_thresh.setValue(0.25)
        self.ui_filter_snr_20db = QCheckBox("Filter SNR > 20dB")
        self.ui_slsnr_filter_path = QLineEdit()
        self.ui_slsnr_filter_path.setPlaceholderText("Path to SLSNR_20.json")
        self.btn_browse_slsnr_filter = QPushButton("Browse")

        filter_path_row = QHBoxLayout()
        filter_path_row.addWidget(self.ui_slsnr_filter_path)
        filter_path_row.addWidget(self.btn_browse_slsnr_filter)

        metrics_layout.addRow("Coverage Mode", self.ui_cov_mode)
        metrics_layout.addRow("Freq Diff Mode", self.ui_fd_mode)
        metrics_layout.addRow("Frag Mode", self.ui_frag_mode)
        metrics_layout.addRow("Recall Mode", self.ui_recall_mode)
        metrics_layout.addRow("Freq Diff Threshold", self.ui_freq_diff_thresh)
        metrics_layout.addRow("Filter SNR > 20dB", self.ui_filter_snr_20db)
        metrics_layout.addRow("Filter Path", filter_path_row)
        metrics_grp.setLayout(metrics_layout)
        scroll_layout.addWidget(metrics_grp)

        layout_grp = QGroupBox("Data Layout")
        layout_layout = QFormLayout()
        self.ui_data_layout = QComboBox()
        self.ui_data_layout.addItems(["flat", "nested"])
        layout_layout.addRow("Layout", self.ui_data_layout)
        layout_grp.setLayout(layout_layout)
        scroll_layout.addWidget(layout_grp)

        audio_grp = QGroupBox("Audio Configuration")
        audio_layout = QFormLayout()
        self.ui_sample_rate = QSpinBox()
        self.ui_sample_rate.setRange(1000, 192000)
        self.ui_sample_rate.setValue(48000)
        audio_layout.addRow("Sample Rate", self.ui_sample_rate)

        audio_grp.setLayout(audio_layout)
        scroll_layout.addWidget(audio_grp)

        proc_grp = QGroupBox("Preprocessing Controls")
        proc_layout = QFormLayout()
        self.ui_rm_harm = QCheckBox("Remove Harmonics")
        self.ui_split_cont = QCheckBox("Split Contours")
        self.ui_harm_tol = QDoubleSpinBox()
        self.ui_harm_tol.setRange(0.0, 1.0)
        self.ui_harm_tol.setDecimals(2)
        self.ui_harm_tol.setSingleStep(0.05)
        self.ui_harm_freq = QDoubleSpinBox()
        self.ui_harm_freq.setRange(0.0, 1.0)
        self.ui_harm_freq.setDecimals(2)
        self.ui_harm_freq.setSingleStep(0.05)
        self.ui_cont_dur = QDoubleSpinBox()
        self.ui_cont_dur.setRange(0.0, 1.0)
        self.ui_cont_dur.setDecimals(3)
        self.ui_cont_dur.setSingleStep(0.005)
        self.ui_subsample = QSpinBox()
        self.ui_subsample.setRange(1, 100)
        self.ui_subsample.setValue(1)

        proc_layout.addRow("Remove Harmonics", self.ui_rm_harm)
        proc_layout.addRow("Harmonic Tolerance", self.ui_harm_tol)
        proc_layout.addRow("Harmonic FreqDiff", self.ui_harm_freq)
        proc_layout.addRow("Split Contours", self.ui_split_cont)
        proc_layout.addRow("Contour Dur Threshold", self.ui_cont_dur)
        proc_layout.addRow("Subsample (every N)", self.ui_subsample)
        proc_grp.setLayout(proc_layout)
        scroll_layout.addWidget(proc_grp)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def set_algorithms(self, algo_names: list, checked: bool = True):
        # Auto-populate text box with discovered/sorted names
        self.ui_algorithms.setText(", ".join(algo_names))

    def get_selected_algorithms(self) -> list:
        text = self.ui_algorithms.text().strip()
        if not text:
            return []
        return [name.strip() for name in text.split(',') if name.strip()]


class ProjectTab(QWidget):
    """Project & Data tab: load example projects, manage metrics, view logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Example project row
        example_row = QHBoxLayout()
        example_row.addWidget(QLabel("Example Project:"))
        self.ui_example_project = QComboBox()
        self.ui_example_project.addItems(["— Select —"])
        self.btn_load_example = QPushButton("📂 Load Example Project")
        self.btn_load_example.setEnabled(False)
        example_row.addWidget(self.ui_example_project)
        example_row.addWidget(self.btn_load_example)
        example_row.addStretch()
        layout.addLayout(example_row)

        # Metrics buttons
        btn_layout = QHBoxLayout()
        self.btn_load_metrics = QPushButton("📂 Load Metrics")
        self.btn_save_metrics = QPushButton("💾 Save Metrics")
        btn_layout.addWidget(self.btn_load_metrics)
        btn_layout.addWidget(self.btn_save_metrics)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress bar (for plot rendering)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_output)


class VisualizationTab(QWidget):
    """Visualization tab with plot controls and embedded matplotlib canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _create_axis_controls(self, metrics, layout, x_label="X Range", y_label="Y Range"):
        controls = {}
        for m in metrics:
            grp = QGroupBox(m)
            grp_layout = QFormLayout()

            row = QHBoxLayout()
            cb_x = QCheckBox("Manual X")
            cb_y = QCheckBox("Manual Y")
            row.addWidget(cb_x)
            row.addWidget(cb_y)
            row.addStretch()
            grp_layout.addRow("Enable", row)

            x_layout = QHBoxLayout()
            x_layout.addWidget(QLabel("Min:"))
            sp_x_min = QDoubleSpinBox()
            sp_x_min.setRange(-1000, 1000)
            sp_x_min.setValue(0)
            sp_x_min.setEnabled(False)
            cb_x.toggled.connect(sp_x_min.setEnabled)
            x_layout.addWidget(sp_x_min)
            x_layout.addWidget(QLabel("Max:"))
            sp_x_max = QDoubleSpinBox()
            sp_x_max.setRange(-1000, 1000)
            sp_x_max.setValue(1)
            sp_x_max.setEnabled(False)
            cb_x.toggled.connect(sp_x_max.setEnabled)
            x_layout.addWidget(sp_x_max)
            grp_layout.addRow(x_label, x_layout)

            y_layout = QHBoxLayout()
            y_layout.addWidget(QLabel("Min:"))
            sp_y_min = QDoubleSpinBox()
            sp_y_min.setRange(-1000, 1000)
            sp_y_min.setValue(0)
            sp_y_min.setEnabled(False)
            cb_y.toggled.connect(sp_y_min.setEnabled)
            y_layout.addWidget(sp_y_min)
            y_layout.addWidget(QLabel("Max:"))
            sp_y_max = QDoubleSpinBox()
            sp_y_max.setRange(-1000, 1000)
            sp_y_max.setValue(1)
            sp_y_max.setEnabled(False)
            cb_y.toggled.connect(sp_y_max.setEnabled)
            y_layout.addWidget(sp_y_max)
            grp_layout.addRow(y_label, y_layout)

            grp.setLayout(grp_layout)
            layout.addWidget(grp)
            controls[m] = {
                'x_cb': cb_x, 'x_min': sp_x_min, 'x_max': sp_x_max,
                'y_cb': cb_y, 'y_min': sp_y_min, 'y_max': sp_y_max
            }
        return controls

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Viz Type:"))
        self.ui_viz_type = QComboBox()
        self.ui_viz_type.addItems(["SNR Trends", "FBID Trends", "Spectrogram Overlay (Together)",
                                   "Single Prediction", "Preprocessing", "Metrics Mosaic", "Metrics Mosaic v2", "Metric Violin", "Fragmentation Verification"])
        ctrl_layout.addWidget(self.ui_viz_type)

        ctrl_layout.addWidget(QLabel("Metric:"))
        self.ui_metric_selector = QComboBox()
        self.ui_metric_selector.addItems(["Coverage", "False Positives", "Frequency Difference", "Fragmentation"])
        self.ui_metric_selector.setVisible(False)
        ctrl_layout.addWidget(self.ui_metric_selector)

        ctrl_layout.addWidget(QLabel("Target WAV:"))
        self.ui_target_wav = QLineEdit()
        self.ui_target_wav.setPlaceholderText("Auto-selects first discovered file")
        ctrl_layout.addWidget(self.ui_target_wav)

        self.btn_browse_wav = QPushButton("Browse")
        ctrl_layout.addWidget(self.btn_browse_wav)

        self.btn_render_plot = QPushButton("📊 Render Plot")
        self.btn_save_plot = QPushButton("💾 Save Plot")
        ctrl_layout.addWidget(self.btn_render_plot)
        ctrl_layout.addWidget(self.btn_save_plot)
        layout.addLayout(ctrl_layout)

        # Collapsible Controls Panel
        self.controls_toggle = QToolButton()
        self.controls_toggle.setText("▼ Plot Controls")
        self.controls_toggle.setCheckable(True)
        layout.addWidget(self.controls_toggle)

        self.controls_frame = QFrame()
        self.controls_frame.setFrameShape(QFrame.Shape.Box)
        self.controls_frame.setVisible(False)
        controls_layout = QVBoxLayout(self.controls_frame)

        # Plot modifiers group
        self.mod_grp = QGroupBox("Plot Modifiers")
        mod_layout = QFormLayout()
        self.ui_plot_algo = QComboBox()
        self.ui_plot_algo.addItems(["Silbido Profundo", "SAM", "SMC-PHD", "CREPE"])
        mod_layout.addRow("Algorithm", self.ui_plot_algo)
        self.ui_show_spectrogram = QCheckBox("Show Spectrogram")
        mod_layout.addRow("Show Spectrogram", self.ui_show_spectrogram)
        self.mod_grp.setLayout(mod_layout)
        controls_layout.addWidget(self.mod_grp)

        # SNR Plot Range group
        self.snr_range_grp = QGroupBox("Spectrogram SNR Range")
        snr_range_layout = QFormLayout()

        self.ui_snr_plot_min = QDoubleSpinBox()
        self.ui_snr_plot_min.setRange(-50.0, 50.0)
        self.ui_snr_plot_min.setDecimals(1)
        self.ui_snr_plot_min.setSingleStep(1.0)
        self.ui_snr_plot_min.setValue(0.0)
        self.ui_snr_plot_min_btn = QCheckBox("Enable Min")
        self.ui_snr_plot_min_btn.toggled.connect(self.ui_snr_plot_min.setEnabled)

        self.ui_snr_plot_max = QDoubleSpinBox()
        self.ui_snr_plot_max.setRange(-50.0, 50.0)
        self.ui_snr_plot_max.setDecimals(1)
        self.ui_snr_plot_max.setSingleStep(1.0)
        self.ui_snr_plot_max.setValue(20.0)
        self.ui_snr_plot_max_btn = QCheckBox("Enable Max")
        self.ui_snr_plot_max_btn.toggled.connect(self.ui_snr_plot_max.setEnabled)

        min_row = QHBoxLayout()
        min_row.addWidget(self.ui_snr_plot_min_btn)
        min_row.addWidget(self.ui_snr_plot_min)
        snr_range_layout.addRow("Min SNR", min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(self.ui_snr_plot_max_btn)
        max_row.addWidget(self.ui_snr_plot_max)
        snr_range_layout.addRow("Max SNR", max_row)

        self.snr_range_grp.setLayout(snr_range_layout)
        controls_layout.addWidget(self.snr_range_grp)

        # Axis limits group (initially hidden)
        self.limits_grp = QGroupBox("Axis Limits")
        self.limits_layout = QVBoxLayout()

        self.metric_controls = QGroupBox("Metric Axes")
        self.metric_layout = QVBoxLayout()
        self.metric_axis_controls = self._create_axis_controls(["Coverage", "Fragmentation", "FreqDiff", "FalsePos"],
                                                               self.metric_layout)
        self.metric_controls.setLayout(self.metric_layout)

        self.spectrogram_controls = QGroupBox("Spectrogram Axes")
        self.spectrogram_layout = QVBoxLayout()
        self.spectrogram_axis_controls = self._create_axis_controls(["Spectrogram"], self.spectrogram_layout,
                                                                    x_label="Time", y_label="Freq")
        self.spectrogram_controls.setLayout(self.spectrogram_layout)

        self.limits_layout.addWidget(self.metric_controls)
        self.limits_layout.addWidget(self.spectrogram_controls)
        self.limits_grp.setLayout(self.limits_layout)
        self.limits_grp.setVisible(False)
        controls_layout.addWidget(self.limits_grp)

        self._current_axis_mode = "metrics"
        self.spectrogram_controls.setVisible(False)

        self.canvas = PlotCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas)

        # Connect toggle AFTER frame is fully initialized to prevent AttributeError
        self.controls_toggle.setChecked(False)
        self.controls_toggle.toggled.connect(self._toggle_controls)

    def _toggle_controls(self, checked: bool):
        self.controls_frame.setVisible(checked)
        self.controls_toggle.setText("▲ Plot Controls" if checked else "▼ Plot Controls")

    def set_algorithm_options(self, names: list):
        """Populate the algorithm dropdown from a list of names."""
        current = self.ui_plot_algo.currentText()
        self.ui_plot_algo.clear()
        self.ui_plot_algo.addItems(names)
        # Restore previous selection if still valid
        if current in names:
            self.ui_plot_algo.setCurrentText(current)

    def show_limits(self, visible: bool):
        self.limits_grp.setVisible(visible)

    def set_algo_selector_visible(self, visible: bool):
        self.mod_grp.setVisible(visible)

    def set_axis_mode(self, mode: str):
        self._current_axis_mode = mode
        if mode == "metrics":
            self.metric_controls.setVisible(True)
            self.spectrogram_controls.setVisible(False)
        else:
            self.metric_controls.setVisible(False)
            self.spectrogram_controls.setVisible(True)

    def get_limits(self) -> dict:
        controls = self.metric_axis_controls if self._current_axis_mode == "metrics" else self.spectrogram_axis_controls
        limits = {}
        for m, ctrls in controls.items():
            if ctrls['x_cb'].isChecked() or ctrls['y_cb'].isChecked():
                limits[m] = {
                    'x': (ctrls['x_min'].value(), ctrls['x_max'].value()) if ctrls['x_cb'].isChecked() else None,
                    'y': (ctrls['y_min'].value(), ctrls['y_max'].value()) if ctrls['y_cb'].isChecked() else None
                }
        return limits

    def get_snr_plot_range(self) -> tuple:
        min_val = self.ui_snr_plot_min.value() if self.ui_snr_plot_min_btn.isChecked() else None
        max_val = self.ui_snr_plot_max.value() if self.ui_snr_plot_max_btn.isChecked() else None
        return min_val, max_val

    def update_limits_from_plot(self, limits_dict: dict):
        controls = self.metric_axis_controls if self._current_axis_mode == "metrics" else self.spectrogram_axis_controls
        for m, ctrls in controls.items():
            if m in limits_dict:
                if 'x' in limits_dict[m]:
                    ctrls['x_min'].setValue(limits_dict[m]['x'][0])
                    ctrls['x_max'].setValue(limits_dict[m]['x'][1])
                if 'y' in limits_dict[m]:
                    ctrls['y_min'].setValue(limits_dict[m]['y'][0])
                    ctrls['y_max'].setValue(limits_dict[m]['y'][1])


class DtwFbidTab(QWidget):
    """Tab for DTW FBID Accuracy Pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl_layout = QFormLayout()

        self.ui_n_ground = QSpinBox()
        self.ui_n_ground.setValue(10)
        self.ui_n_ground.setRange(1, 100)

        self.ui_n_comp = QSpinBox()
        self.ui_n_comp.setValue(10)
        self.ui_n_comp.setRange(1, 100)

        ctrl_layout.addRow("N Ground Truth:", self.ui_n_ground)
        ctrl_layout.addRow("N Comparisons:", self.ui_n_comp)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run DTW FBID Accuracy")
        self.btn_save_plot = QPushButton("💾 Save Plot")
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_save_plot)

        layout.addLayout(ctrl_layout)
        layout.addLayout(btn_layout)

        # Plot Canvas
        self.canvas = PlotCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas)

        # Plot Type Selector
        plot_type_layout = QHBoxLayout()
        self.ui_plot_type = QComboBox()
        self.ui_plot_type.addItems(["Violin Plot", "Box Plot"])
        self.ui_plot_type.currentTextChanged.connect(self._on_plot_type_changed)
        plot_type_layout.addWidget(QLabel("Display:"))
        plot_type_layout.addWidget(self.ui_plot_type)
        layout.addLayout(plot_type_layout)

        self.current_fig_violin = None
        self.current_fig_box = None

    def _on_plot_type_changed(self, text):
        if text == "Violin Plot" and self.current_fig_violin:
            self.canvas.set_figure(self.current_fig_violin)
        elif text == "Box Plot" and self.current_fig_box:
            self.canvas.set_figure(self.current_fig_box)

    def update_plots(self, fig_violin, fig_box):
        self.current_fig_violin = fig_violin
        self.current_fig_box = fig_box
        # Default to violin
        self.ui_plot_type.setCurrentText("Violin Plot")
        self._on_plot_type_changed("Violin Plot")

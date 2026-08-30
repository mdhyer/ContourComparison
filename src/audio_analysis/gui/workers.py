from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QThread, pyqtSignal
import matplotlib
import traceback

try:
    matplotlib.use('QtAgg')
except RuntimeError:
    pass

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from audio_analysis.config import PipelineConfig
from audio_analysis.pipeline import execute_precompute, execute_evaluate, _parse_snr_float
from audio_analysis.utils.contour_utils import load_ground_truth, load_precomputed, fragment_contours, \
    identify_harmonics, _resolve_fbid_snr
from audio_analysis.evaluation.comparison import compare_contours
from audio_analysis.evaluation.metrics import _select_metrics, aggregate_metrics
from audio_analysis.plotting.plotting import (
    plot_results, fbid_plot, visualize_together, visualize_prediction,
    plot_preprocessing, plot_metrics_mosaic, plot_metrics_mosaic_v2, plot_fbid_trends,
    plot_fragmentation_verification
)
from audio_analysis.evaluation.dtw_fbid_accuracy import run_fbid_accuracy_pipeline, plot_violin, plot_boxplot


class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: PipelineConfig, wav_files: List[Path], algorithms: List[str], command: str,
                 overwrite: bool = True):
        super().__init__()
        self.config = config
        self.wav_files = wav_files
        self.algorithms = algorithms
        self.command = command
        self.overwrite = overwrite
        self.is_running = True

    def run(self):
        try:
            self.log_signal.emit(f"▶ Starting {self.command} for {len(self.wav_files)} files...")
            
            def _progress(current: int, total: int, msg: str):
                pct = int((current / total) * 100) if total > 0 else 0
                self.progress_signal.emit(pct)
                self.log_signal.emit(f"[{pct}%] {msg}")
                print(f"[{pct}%] {msg}")  # Keep console output for CLI/debugging

            if self.command == "precompute":
                execute_precompute(self.config, self.wav_files, self.algorithms, overwrite=self.overwrite, progress_callback=_progress)
            elif self.command == "evaluate":
                metrics = execute_evaluate(self.config, self.wav_files, self.algorithms, progress_callback=_progress)
                self.finished_signal.emit(metrics)
            elif self.command == "plot":
                self.log_signal.emit("ℹ️ Plotting is handled via the Visualization tab.")
            self.log_signal.emit(f"✅ {self.command} completed successfully.")
            self.progress_signal.emit(100)
        except Exception as e:
            self.error_signal.emit(f"❌ Error: {str(e)}")
            self.progress_signal.emit(0)


class PlotWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    figure_signal = pyqtSignal(object)
    limits_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: PipelineConfig, wav_files: List[Path], aggregated_metrics: Dict,
                 algo: str, plot_type: str, target_wav: Optional[str] = None, selected_algos: List[str] = None,
                 limits: Optional[Dict] = None, snr_plot_min: Optional[float] = None, snr_plot_max: Optional[float] = None,
                 show_spectrogram: bool = False):
        super().__init__()
        self.config = config
        self.wav_files = wav_files
        self.aggregated_metrics = aggregated_metrics
        self.algo = algo
        self.plot_type = plot_type
        self.target_wav = target_wav
        self.selected_algos = selected_algos or []
        self.limits = limits or {}
        self.snr_plot_min = snr_plot_min
        self.snr_plot_max = snr_plot_max
        self.show_spectrogram = show_spectrogram

    def run(self):
        try:
            self.log_signal.emit(f"📊 Computing {self.plot_type} plot...")
            fig = None
            if self.plot_type == "SNR Trends":
                fig = self._compute_snr_figure()
            elif self.plot_type == "FBID Trends":
                fig = self._compute_fbid_figure()
            elif self.plot_type == "Spectrogram Overlay (Together)":
                fig = self._compute_together_figure()
            elif self.plot_type == "Single Prediction":
                fig = self._compute_prediction_figure()
            elif self.plot_type == "Preprocessing":
                fig = self._compute_preprocessing_figure()
            elif self.plot_type == "Metrics Mosaic":
                fig = self._compute_mosaic_figure()
            elif self.plot_type == "Metrics Mosaic v2":
                fig = self._compute_mosaic_v2_figure()
            elif self.plot_type == "Fragmentation Verification":
                fig = self._compute_fragmentation_verification_figure()
            elif self.plot_type in ["Coverage", "False Positives", "Frequency Difference", "Fragmentation"]:
                fig = self._compute_metric_violin_figure()

            if fig:
                self.figure_signal.emit(fig)
                self.log_signal.emit("✅ Plot ready.")
            self.progress_signal.emit(100)
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(f"❌ Plot computation error: {str(e)}")
            self.progress_signal.emit(0)

    def _compute_snr_figure(self) -> Figure:
        algos = self.selected_algos if self.selected_algos else [self.algo]
        fig = None
        axs = None
        colors = plt.cm.viridis(np.linspace(0., 1., len(algos)))

        snr_wav_map = {}

        # FIX: Fallback discovery if wav_files list is empty
        if not self.wav_files:
            from audio_analysis.pipeline import _discover_wav_files
            self.wav_files = _discover_wav_files(self.config.paths.data_root, None, None, self.config.paths.data_layout)

        def _get_wav_base(p: Path) -> str:
            name = p.stem
            if '_snr_' in name:
                name = name.split('_snr_')[0]
            elif '_CLEAN_' in name:
                name = name.split('_CLEAN_')[0]
            return name

        if self.target_wav:
            target_path = Path(self.target_wav)
            try:
                target_fbid, target_snr = _resolve_fbid_snr(target_path)
                target_base = _get_wav_base(target_path)
                snr_wav_map[target_snr] = self.target_wav
                
                # First pass: try to match exact FBID and base name
                for wav_path in self.wav_files:
                    try:
                        p = Path(wav_path)
                        fbid, snr_level = _resolve_fbid_snr(p)
                        base = _get_wav_base(p)
                        if fbid == target_fbid and base == target_base and snr_level not in snr_wav_map:
                            snr_wav_map[snr_level] = str(p)
                    except Exception:
                        continue
                        
                # Second pass: fill remaining SNR levels with any available wav
                for wav_path in self.wav_files:
                    try:
                        p = Path(wav_path)
                        _, snr_level = _resolve_fbid_snr(p)
                        if snr_level not in snr_wav_map:
                            snr_wav_map[snr_level] = str(p)
                    except Exception:
                        continue
            except Exception:
                pass

        if not snr_wav_map:
            for wav_path in self.wav_files:
                try:
                    p = Path(wav_path)
                    _, snr_level = _resolve_fbid_snr(p)
                    if snr_level not in snr_wav_map:
                        snr_wav_map[snr_level] = str(p)
                except Exception:
                    continue

        # Resolve the exact mode key from the GUI configuration
        mode_key = f"{self.config.metrics.coverage_mode.value}_{self.config.metrics.freq_diff_mode.value}_{self.config.metrics.frag_mode.value}_{self.config.metrics.recall_mode.value}"

        for i, algo in enumerate(algos):
            algo_data = self.aggregated_metrics.get(algo, {})
            per_snr = algo_data.get('per_snr', algo_data) if isinstance(algo_data, dict) else {}
            if not per_snr: continue

            sorted_snrs = sorted(per_snr.keys(), key=lambda x: _parse_snr_float(x))
            
            # Access nested metrics using the resolved mode_key
            cov = np.array([per_snr[snr].get(mode_key, {}).get("coverage", np.nan) for snr in sorted_snrs])
            fp = np.array([per_snr[snr].get(mode_key, {}).get("false_pos", np.nan) for snr in sorted_snrs])
            fd = np.array([per_snr[snr].get(mode_key, {}).get("freq_diff", np.nan) for snr in sorted_snrs])
            frag = np.array([per_snr[snr].get(mode_key, {}).get("fragmentation", np.nan) for snr in sorted_snrs])
            noise = np.array([_parse_snr_float(snr) for snr in sorted_snrs])

            def _safe_unc(key):
                vals = [per_snr[snr].get(mode_key, {}).get(key, [np.nan, np.nan]) for snr in sorted_snrs]
                arr = np.array(vals)
                return arr if arr.shape == (len(sorted_snrs), 2) else None

            cov_unc = _safe_unc("coverage_5_95")
            fp_unc = _safe_unc("false_pos_5_95")
            fd_unc = _safe_unc("freq_diff_5_95")
            frag_unc = _safe_unc("fragmentation_5_95")

            fig, axs = plot_results(
                noise_floats=noise, coverage=cov, false_pos=fp, freq_diff=fd, frag=frag,
                coverage_unc=cov_unc, false_pos_unc=fp_unc, freq_diff_unc=fd_unc, frag_unc=frag_unc,
                algorithm=algo, layout="v2", fig=fig, axs=axs, setup_axes=(i == 0),
                color=colors[i], show_legend=(i == len(algos) - 1),
                metrics_config=self.config.metrics,
                snr_wav_map=snr_wav_map if i == 0 else None,
                snr_plot_min=self.snr_plot_min,
                snr_plot_max=self.snr_plot_max,
                limits=self.limits
            )
        if fig is None: raise ValueError("No valid metrics found for selected algorithms.")

        self.limits_signal.emit(self._extract_limits(axs, plot_type="snr"))
        return fig

    def _compute_fbid_figure(self) -> Figure:
        fig, axs = plot_fbid_trends(self.aggregated_metrics, self.selected_algos or [self.algo], self.limits, metrics_config=self.config.metrics)
        if fig is None: raise ValueError("No FBID data available for selected algorithms.")
        self.limits_signal.emit(self._extract_limits(axs, plot_type="fbid"))
        return fig

    def _extract_limits(self, axs, plot_type: str) -> dict:
        limits = {}
        if plot_type == "snr":
            key_map = {'E': 'Coverage', 'F': 'Fragmentation', 'H': 'FreqDiff'}
        else:
            key_map = {'A': 'Coverage', 'B': 'Fragmentation', 'C': 'FreqDiff'}

        for key, name in key_map.items():
            if key in axs:
                ax = axs[key]
                limits[name] = {'x': ax.get_xlim(), 'y': ax.get_ylim()}
        return limits

    def _compute_together_figure(self) -> Figure:
        if not self.target_wav: raise ValueError("Target WAV required for 'together' plot.")
        return visualize_together(
            self.target_wav, str(self.config.paths.params_dir), self.selected_algos,
            wav_src=str(self.config.paths.data_root), use_precomputed=True,
            load_src=str(self.config.paths.precompute_dir), pipeline_config=self.config,
            limits=self.limits
        )

    def _compute_prediction_figure(self) -> Figure:
        if not self.target_wav: raise ValueError("Target WAV required for 'prediction' plot.")
        return visualize_prediction(
            self.target_wav, self.algo, str(self.config.paths.data_root),
            str(self.config.paths.params_dir), pipeline_config=self.config,
            limits=self.limits
        )

    def _compute_preprocessing_figure(self) -> Figure:
        if not self.target_wav: raise ValueError("Target WAV required for 'preprocessing' plot.")
        return plot_preprocessing(
            self.target_wav,
            use_precomputed=True, pipeline_config=self.config,
            limits=self.limits
        )

    def _compute_mosaic_figure(self) -> Figure:
        if not self.target_wav: raise ValueError("Target WAV required for 'mosaic' plot.")
        return plot_metrics_mosaic(
            self.target_wav, str(self.config.paths.params_dir),
            str(self.config.paths.data_root), str(self.config.paths.precompute_dir),
            algorithm=self.algo, pipeline_config=self.config,
            limits=self.limits
        )

    def _compute_mosaic_v2_figure(self) -> Figure:
        if not self.target_wav: raise ValueError("Target WAV required for 'mosaic v2' plot.")
        return plot_metrics_mosaic_v2(
            self.target_wav, str(self.config.paths.params_dir),
            algorithm=self.algo, precompute_dir=str(self.config.paths.precompute_dir),
            pipeline_config=self.config,
            limits=self.limits
        )

    def _compute_fragmentation_verification_figure(self) -> Figure:
        if not self.target_wav:
            raise ValueError("Target WAV required for fragmentation verification plot.")
            
        target_path = Path(self.target_wav)
        # Pass Path objects directly to avoid 'str' / 'str' errors in utility functions
        ground, discont = load_ground_truth(wav_path=target_path, params_dir=self.config.paths.params_dir)
        if ground is None:
            raise ValueError(f"No ground truth found for {self.target_wav}")
            
        pred = load_precomputed(wav=str(target_path), ALGORITHM=self.algo, top_dir=self.config.paths.precompute_dir)
        if pred is None:
            raise ValueError(f"No precomputed contours found for {self.algo} on {self.target_wav}")
            
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
            
        # Apply post-processing to match evaluation pipeline
        if self.config.algorithm.split_contours:
            pred = fragment_contours(pred, dur=self.config.algorithm.contour_dur_threshold)
        if self.config.algorithm.remove_harmonics:
            pred = identify_harmonics(pred, tolerance=self.config.algorithm.harmonic_tolerance, freqdiff=self.config.algorithm.harmonic_freqdiff)
            
        return plot_fragmentation_verification(
            wav=self.target_wav,
            ground_contours=ground,
            pred_contours=pred,
            metrics_config=self.config.metrics,
            discont=discont,
            show_spectrogram=self.show_spectrogram,
            limits=self.limits
        )

    def _compute_metric_violin_figure(self) -> Figure:
        from audio_analysis.plotting.plotting import plot_metric_violin
        return plot_metric_violin(
            self.aggregated_metrics, 
            self.selected_algos or [self.algo], 
            self.plot_type, 
            self.config.metrics
        )


class DtwFbidWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict) # Contains 'metrics', 'fig_violin', 'fig_box'
    error_signal = pyqtSignal(str)

    def __init__(self, config: PipelineConfig, algorithms: List[str], n_ground: int, n_comparisons: int):
        super().__init__()
        self.config = config
        self.algorithms = algorithms
        self.n_ground = n_ground
        self.n_comparisons = n_comparisons

    def run(self):
        try:
            self.log_signal.emit("▶ Starting DTW FBID Accuracy pipeline...")
            
            # Define progress callback to update GUI
            def _progress(current: int, total: int, msg: str):
                pct = int((current / total) * 100) if total > 0 else 0
                self.progress_signal.emit(pct)
                self.log_signal.emit(f"[{pct}%] {msg}")

            # Run pipeline
            data = run_fbid_accuracy_pipeline(
                config=self.config,
                algorithms=self.algorithms,
                n_ground=self.n_ground,
                n_comparisons=self.n_comparisons,
                progress_callback=_progress
            )
            
            self.progress_signal.emit(90) # Pipeline done, starting plots
            self.log_signal.emit("✅ Pipeline complete. Generating plots...")
            
            # Generate plots
            fig_violin = plot_violin(data['acc_plot_data'], data['algorithms'])
            fig_box = plot_boxplot(data['acc_plot_data'], data['algorithms'])
            
            self.finished_signal.emit({
                'metrics': data['metrics'],
                'fig_violin': fig_violin,
                'fig_box': fig_box
            })
            self.progress_signal.emit(100)
            
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(f"❌ Error: {str(e)}")

"""
Consolidated Plotting Utilities
Merges functionality from plot_all_results.py, plot_all_results_v0.py, plot_fbid.py,
plotting.py, plot_metrics.py, plot_metrics_mosaic.py, plot_metrics_mosaic_v2.py,
and plot_preprocessing.py into a single, modular package.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple, Dict, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft, filtfilt, butter

from ..utils.contour_utils import run_ground, load_precomputed, fragment_contours, identify_harmonics
from ..contour_extraction.crepe import CrepePredictor
from ..config import CoverageMode, FreqDiffMode, FragMode, MetricsConfig, PipelineConfig
from ..evaluation.comparison import _split_ground_truth

logger = logging.getLogger(__name__)


def _resolve_metric_labels(metrics_config: Optional[MetricsConfig]) -> Dict[str, str]:
    """Centralized resolution of metric labels based on configuration modes."""
    if metrics_config:
        cov_mode, frag_mode, fd_mode = metrics_config.coverage_mode, metrics_config.frag_mode, metrics_config.freq_diff_mode
    else:
        cov_mode, frag_mode, fd_mode = CoverageMode.TOTAL, FragMode.TOTAL, FreqDiffMode.TOTAL

    cov_label = (
        "FFC Contour Coverage" if cov_mode == CoverageMode.FFC else
        "Total Contour Coverage" if cov_mode == CoverageMode.TOTAL else
        "Per Loop FFC Coverage" if cov_mode == CoverageMode.FFC_PER_LOOP else
        "Per Loop Total Coverage" if cov_mode == CoverageMode.TOTAL_PER_LOOP else
        "Per Contour Coverage"
    )

    frag_label = (
        "Total Fragmentation" if frag_mode == FragMode.TOTAL else
        "FFC Fragmentation" if frag_mode == FragMode.FFC else
        "Per Loop Fragmentation" if frag_mode == FragMode.TOTAL_PER_LOOP else
        "Per Loop FFC Fragmentation" if frag_mode == FragMode.FFC_PER_LOOP else
        "Fragmentation"
    )

    fd_label = (
        "FFC Frequency Error (%)" if fd_mode == FreqDiffMode.FFC else
        "Frequency Error (%)"
    )

    return {
        "coverage": cov_label,
        "fragmentation": frag_label,
        "freq_diff": fd_label,
        "false_pos": "False Positives"
    }


def _parse_snr_float(snr_str: str) -> float:
    if str(snr_str).upper() == "CLEAN":
        return 0.0
    match = re.search(r'-?\d+\.?\d*', str(snr_str))
    return float(match.group()) if match else 0.0


def _ensure_contour_list(contours) -> List[np.ndarray]:
    """Ensure contours is a list of valid 2D numpy arrays (N, 2)."""
    if contours is None:
        return []
    if isinstance(contours, np.ndarray):
        if len(contours) == 1 and contours[0].ndim == 2:
            return list(contours)
        if contours.ndim == 2 and contours.shape[0] > 0:
            return [contours]
        elif contours.ndim == 1 and len(contours) >= 1:
            res = []
            for c in contours:
                arr = np.asarray(c)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    res.append(arr)
            return res
        return []
    if isinstance(contours, (list, tuple)):
        res = []
        for c in contours:
            arr = np.asarray(c)
            if arr.ndim == 2 and arr.shape[0] > 0:
                res.append(arr)
        return res
    return []


def _plot_spectrogram(ax: plt.Axes, wav_path: str, snr_key: str, share_axes: Optional[plt.Axes] = None, show_title: bool = True) -> None:
    """Helper to plot spectrogram on a given axis."""
    try:
        audio, sr = sf.read(wav_path)
        f, t, Zxx = stft(audio, fs=sr, nperseg=1024, noverlap=round(1024 * .9), padded=False, scaling='psd')
        spec = np.abs(Zxx)
        spec = np.log(spec + 1e-6)
        ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec))
        ax.set_aspect(0.05)
        ax.set_ylim([0, 24])
        if share_axes:
            ax.sharey(share_axes)
            ax.sharex(share_axes)
        ax.set_xlabel('Time (s)')
        if show_title:
            snr_val = _parse_snr_float(snr_key)
            ax.set_title(f'SNR: {snr_val:.0f} dB')
    except Exception:
        pass


def _apply_limits_to_axes(axs: Dict[str, plt.Axes], limits: Optional[Dict]) -> None:
    """Apply axis limits to all axes in the figure if provided."""
    if not limits:
        return
    # Extract first available limit to apply globally to main axes
    first_limit = next(iter(limits.values()), None)
    if first_limit:
        for ax in axs.values():
            if first_limit.get('x'):
                ax.set_xlim(first_limit['x'])
            if first_limit.get('y'):
                ax.set_ylim(first_limit['y'])


def plot_results(
        noise_floats: np.ndarray,
        coverage: np.ndarray,
        false_pos: np.ndarray,
        freq_diff: np.ndarray,
        frag: np.ndarray,
        algorithm: str,
        color: Optional[str] = None,
        layout: str = "v1",
        fig: Optional[plt.Figure] = None,
        axs: Optional[Dict[str, plt.Axes]] = None,
        setup_axes: bool = True,
        show_legend: bool = True,
        coverage_unc: Optional[np.ndarray] = None,
        false_pos_unc: Optional[np.ndarray] = None,
        freq_diff_unc: Optional[np.ndarray] = None,
        frag_unc: Optional[np.ndarray] = None,
        snr_wav_map: Optional[Dict[str, str]] = None,
        metrics_config: Optional[MetricsConfig] = None,
        limits: Optional[Dict] = None,
        snr_plot_min: Optional[float] = None,
        snr_plot_max: Optional[float] = None,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    LABELS = {
        'Silbido Profundo': 'Silbido Profundo', 'SAM': 'SAM-whistle',
        'CREPE': 'CREPE-tt', 'SMC-PHD': 'SMC-PHD'
    }

    noise_floats = np.asarray(noise_floats, dtype=float)
    sort_idx = np.argsort(noise_floats)
    noise_floats = noise_floats[sort_idx]
    coverage = np.asarray(coverage)[sort_idx]
    false_pos = np.asarray(false_pos)[sort_idx]
    freq_diff = np.asarray(freq_diff)[sort_idx]
    frag = np.asarray(frag)[sort_idx]

    if coverage_unc is not None: coverage_unc = np.asarray(coverage_unc)[sort_idx]
    if false_pos_unc is not None: false_pos_unc = np.asarray(false_pos_unc)[sort_idx]
    if freq_diff_unc is not None: freq_diff_unc = np.asarray(freq_diff_unc)[sort_idx]
    if frag_unc is not None: frag_unc = np.asarray(frag_unc)[sort_idx]

    x_margin = (np.max(noise_floats) - np.min(noise_floats)) * 0.15 if len(noise_floats) > 1 else 2.0
    xlim_auto = [np.min(noise_floats) - x_margin, np.max(noise_floats) + x_margin]
    if xlim_auto[0] == xlim_auto[1]:
        xlim_auto = [xlim_auto[0] - 2.0, xlim_auto[1] + 2.0]

    if layout == "v1":
        mosaic = [['A', 'B', 'C', 'D'], ['E', 'E', 'F', 'F'], ['E', 'E', 'F', 'F'], ['G', 'G', 'H', 'H'], ['G', 'G', 'H', 'H']]
        figsize = (8, 10)
    else:
        mosaic = [['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
                  ['E', 'E', 'E', 'E', 'F', 'F', 'F', 'F', 'H', 'H', 'H', 'H'],
                  ['E', 'E', 'E', 'E', 'F', 'F', 'F', 'F', 'H', 'H', 'H', 'H']]
        figsize = (12, 6)

    if fig is None:
        fig, axs = plt.subplot_mosaic(mosaic, figsize=figsize, constrained_layout=True)

    # Explicit axis mapping per layout
    ax_cov = axs['E']
    ax_frag = axs['F']
    if layout == "v1":
        ax_fd = axs['G']
        ax_fp = axs['H']
    else:  # v2
        ax_fd = axs['H']
        ax_fp = None

    # Plot Coverage
    line_cov = ax_cov.plot(noise_floats, coverage, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
    cov_color = line_cov[0].get_color()
    if coverage_unc is not None:
        ax_cov.fill_between(noise_floats, coverage_unc[:, 0], coverage_unc[:, 1], alpha=0.3, color=cov_color)
    # if show_legend: ax_cov.legend()

    # Plot Fragmentation
    line_frag = ax_frag.plot(noise_floats, frag, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
    frag_color = line_frag[0].get_color()
    if frag_unc is not None:
        ax_frag.fill_between(noise_floats, frag_unc[:, 0], frag_unc[:, 1], alpha=0.3, color=frag_color)
    if show_legend: ax_frag.legend(loc='upper left' if layout != "v1" else None)

    # Plot Frequency Difference
    line_fd = ax_fd.plot(noise_floats, freq_diff, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
    fd_color = line_fd[0].get_color()
    if freq_diff_unc is not None:
        ax_fd.fill_between(noise_floats, freq_diff_unc[:, 0], freq_diff_unc[:, 1], alpha=0.3, color=fd_color)
    # if show_legend: ax_fd.legend()

    # Plot False Positives (v1 only)
    if ax_fp is not None:
        line_fp = ax_fp.plot(noise_floats, false_pos, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
        fp_color = line_fp[0].get_color()
        if false_pos_unc is not None:
            ax_fp.fill_between(noise_floats, false_pos_unc[:, 0], false_pos_unc[:, 1], alpha=0.3, color=fp_color)
        if show_legend: ax_fp.legend()

    if setup_axes:
        # Resolve dynamic labels based on metric modes
        labels = _resolve_metric_labels(metrics_config)
        ax_cov.set_ylabel(labels["coverage"])
        ax_frag.set_ylabel(labels["fragmentation"])
        ax_fd.set_ylabel(labels["freq_diff"])
        if ax_fp is not None:
            ax_fp.set_ylabel(labels["false_pos"])

        # Compute SNR keys for spectrogram insets
        snr_keys = []
        if snr_wav_map:
            all_snr_keys = sorted(snr_wav_map.keys(), key=lambda x: _parse_snr_float(x))
            if snr_plot_min is not None or snr_plot_max is not None:
                filtered_keys = [k for k in all_snr_keys if (snr_plot_min is None or _parse_snr_float(k) >= snr_plot_min) and (snr_plot_max is None or _parse_snr_float(k) <= snr_plot_max)]
                if len(filtered_keys) >= 4:
                    indices = np.linspace(0, len(filtered_keys) - 1, 4, dtype=int)
                    snr_keys = [filtered_keys[i] for i in indices]
                else:
                    snr_keys = filtered_keys[:4]
            else:
                snr_keys = all_snr_keys[:4]

        if layout == "v1":
            ax_cov.set_title('Contour Coverage')
            ax_frag.set_title('Fragmentation')
            ax_fd.set_title('Frequency Difference')
            ax_fp.set_title('False Positives')

            if snr_keys:
                for idx, snr_key in enumerate(snr_keys):
                    ax_key = ['A', 'B', 'C', 'D'][idx]
                    ax = axs[ax_key]
                    wav_path = snr_wav_map[snr_key]
                    share_ax = axs['A'] if idx > 0 else None
                    _plot_spectrogram(ax, wav_path, snr_key, share_ax)
                axs['A'].set_ylabel('Frequency (kHz)')
        else:
            if snr_keys:
                for idx, snr_key in enumerate(snr_keys):
                    ax_key = ['A', 'B', 'C', 'D'][idx]
                    ax = axs[ax_key]
                    wav_path = snr_wav_map[snr_key]
                    share_ax = axs['A'] if idx > 0 else None
                    _plot_spectrogram(ax, wav_path, snr_key, share_ax)
                axs['A'].set_ylabel('Frequency (kHz)')

            # Consistent auto-scaling & margins for v2
            ax_frag.set_ylim(bottom=0); ax_frag.margins(y=0.1)
            ax_fd.set_ylim(bottom=0); ax_fd.margins(y=0.1)

            ax_cov.text(-0.2, 1.05, '(a)', transform=ax_cov.transAxes, va='bottom', fontweight='bold', fontsize=14)
            ax_frag.text(-0.2, 1.05, '(b)', transform=ax_frag.transAxes, va='bottom', fontweight='bold', fontsize=14)
            ax_fd.text(-0.2, 1.05, '(c)', transform=ax_fd.transAxes, va='bottom', fontweight='bold', fontsize=14)

        ax_cov.set_xlabel('SNR'); ax_frag.set_xlabel('SNR'); ax_fd.set_xlabel('SNR')
        if ax_fp is not None: ax_fp.set_xlabel('SNR')

        ax_cov.set_ylim([0, 1]); ax_cov.set_xlim(xlim_auto)
        ax_frag.set_xlim(xlim_auto); ax_fd.set_xlim(xlim_auto)
        if ax_fp is not None: ax_fp.set_xlim(xlim_auto)

        if layout == "v1":
            ax_fd.set_ylim(bottom=0); ax_fd.margins(y=0.1); ax_fd.set_xlim(xlim_auto)
            ax_frag.set_ylim(bottom=0); ax_frag.margins(y=0.1)
            ax_fp.set_ylim(bottom=0); ax_fp.margins(y=0.1)

        # Apply manual limits per-axis if provided
        if limits:
            if 'Coverage' in limits:
                lim = limits['Coverage']
                if lim.get('x'): ax_cov.set_xlim(lim['x'])
                if lim.get('y'): ax_cov.set_ylim(lim['y'])
            if 'Fragmentation' in limits:
                lim = limits['Fragmentation']
                if lim.get('x'): ax_frag.set_xlim(lim['x'])
                if lim.get('y'): ax_frag.set_ylim(lim['y'])
            if 'FreqDiff' in limits:
                lim = limits['FreqDiff']
                if lim.get('x'): ax_fd.set_xlim(lim['x'])
                if lim.get('y'): ax_fd.set_ylim(lim['y'])
            if 'FalsePos' in limits and ax_fp:
                lim = limits['FalsePos']
                if lim.get('x'): ax_fp.set_xlim(lim['x'])
                if lim.get('y'): ax_fp.set_ylim(lim['y'])

    return fig, axs


def fbid_plot(
        FBs: np.ndarray,
        coverage_plot: np.ndarray,
        frag_plot: Optional[np.ndarray] = None,
        freqdiff_plot: Optional[np.ndarray] = None,
        algorithm: str = "CREPE",
        audiofolder: str = "",
        color: Optional[str] = None,
        layout: str = "v1",
        fig: Optional[plt.Figure] = None,
        axs: Optional[Dict[str, plt.Axes]] = None,
        setup_axes: bool = True,
        show_legend: bool = True,
        coverage_unc: Optional[np.ndarray] = None,
        frag_unc: Optional[np.ndarray] = None,
        freqdiff_unc: Optional[np.ndarray] = None,
        metrics_config: Optional[MetricsConfig] = None,
        limits: Optional[Dict] = None,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    LABELS = {
        'Silbido Profundo': 'Silbido Profundo', 'SAM': 'SAM-whistle', 'SAM-Raw': 'SAM-whistle',
        'CREPE-2': 'CREPE Consp.', 'SMC-PHD': 'SMC-PHD', 'CREPE': 'CREPE-tt', 'CREPE-PAM': 'CREPE-PAM'
    }

    cov_sort = np.argsort(coverage_plot)[::-1]
    frag_sort = np.argsort(frag_plot) if frag_plot is not None else cov_sort
    freqdiff_sort = np.argsort(freqdiff_plot) if freqdiff_plot is not None else cov_sort
    FBs_sorted = FBs[cov_sort]
    coverage_plot_sorted = coverage_plot[cov_sort]
    frag_plot_sorted = frag_plot[frag_sort] if frag_plot is not None else None
    freqdiff_plot_sorted = freqdiff_plot[freqdiff_sort] if freqdiff_plot is not None else None

    cov_unc_sorted = coverage_unc[cov_sort] if coverage_unc is not None else None
    frag_unc_sorted = frag_unc[frag_sort] if frag_unc is not None else None
    fd_unc_sorted = freqdiff_unc[freqdiff_sort] if freqdiff_unc is not None else None

    if layout == "v1":
        mosaic = [['AA', 'AA', 'AA', 'AA'], ['AA', 'AA', 'AA', 'AA'], ['AA', 'AA', 'AA', 'AA'],
                  ['A0', 'B0', 'C0', 'D0'], ['A1', 'B1', 'C1', 'D1'], ['A2', 'B2', 'C2', 'D2']]
        figsize = (6, 10)
    else:
        mosaic = [['A', 'B', 'C']]
        figsize = (12, 4)

    if fig is None:
        fig, axs = plt.subplot_mosaic(mosaic, figsize=figsize, constrained_layout=True)

    ax_coverage = axs['AA'] if layout == "v1" else axs['A']
    line_cov = ax_coverage.plot(range(len(coverage_plot_sorted)), coverage_plot_sorted, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
    cov_color = line_cov[0].get_color()
    if cov_unc_sorted is not None:
        ax_coverage.fill_between(range(len(coverage_plot_sorted)), cov_unc_sorted[:, 0], cov_unc_sorted[:, 1], alpha=0.3, color=cov_color)
    elif layout == "v1":
        ax_coverage.fill_between(range(len(coverage_plot_sorted)), coverage_plot_sorted - 0.05, coverage_plot_sorted + 0.05, alpha=0.3, color=cov_color)
    if show_legend: ax_coverage.legend()

    if setup_axes:
        labels = _resolve_metric_labels(metrics_config)
        ax_coverage.set_xlabel('FBID')
        ax_coverage.set_ylabel(labels["coverage"] if layout != "v1" else 'Coverage')
        ax_coverage.set_xticks([])
        ax_coverage.set_ylim([0, 1])
        ax_coverage.set_xlim([0, len(FBs_sorted) - 1])
        if layout == "v1": ax_coverage.set_title('Contour Coverage')

    if layout != "v1" and frag_plot_sorted is not None and freqdiff_plot_sorted is not None:
        ax_frag = axs['B']
        line_frag = ax_frag.plot(range(len(frag_plot_sorted)), frag_plot_sorted, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
        frag_color = line_frag[0].get_color()
        if frag_unc_sorted is not None:
            ax_frag.fill_between(range(len(frag_plot_sorted)), frag_unc_sorted[:, 0], frag_unc_sorted[:, 1], alpha=0.3, color=frag_color)

        ax_freqdiff = axs['C']
        line_fd = ax_freqdiff.plot(range(len(freqdiff_plot_sorted)), freqdiff_plot_sorted, label=LABELS.get(algorithm, algorithm), color=color, linewidth=2)
        fd_color = line_fd[0].get_color()
        if fd_unc_sorted is not None:
            ax_freqdiff.fill_between(range(len(freqdiff_plot_sorted)), fd_unc_sorted[:, 0], fd_unc_sorted[:, 1], alpha=0.3, color=fd_color)

        if setup_axes:
            ax_frag.set_xlabel('FBID')
            ax_frag.set_ylabel(labels["fragmentation"])
            ax_frag.set_xticks([])
            ax_frag.set_xlim([0, len(FBs_sorted) - 1])
            max_frag = np.nanmax(frag_plot_sorted)
            ax_frag.set_ylim(0, max_frag * 1.15 if max_frag > 0 else 1.0)

            ax_freqdiff.set_xlabel('FBID')
            ax_freqdiff.set_ylabel(labels["freq_diff"])
            ax_freqdiff.set_xticks([])
            ax_freqdiff.set_xlim([0, len(FBs_sorted) - 1])
            max_fd = np.nanmax(freqdiff_plot_sorted)
            ax_freqdiff.set_ylim(0, max_fd * 1.15 if max_fd > 0 else 1.0)

            axs['A'].text(-0.2, 1.05, '(a)', transform=axs['A'].transAxes, va='bottom', fontweight='bold', fontsize=14)
            axs['B'].text(-0.2, 1.05, '(b)', transform=axs['B'].transAxes, va='bottom', fontweight='bold', fontsize=14)
            axs['C'].text(-0.2, 1.05, '(c)', transform=axs['C'].transAxes, va='bottom', fontweight='bold', fontsize=14)

            # Apply manual limits per-axis if provided
            if limits:
                if 'Coverage' in limits:
                    lim = limits['Coverage']
                    if lim.get('x'): ax_coverage.set_xlim(lim['x'])
                    if lim.get('y'): ax_coverage.set_ylim(lim['y'])
                if 'Fragmentation' in limits:
                    lim = limits['Fragmentation']
                    if lim.get('x'): ax_frag.set_xlim(lim['x'])
                    if lim.get('y'): ax_frag.set_ylim(lim['y'])
                if 'FreqDiff' in limits:
                    lim = limits['FreqDiff']
                    if lim.get('x'): ax_freqdiff.set_xlim(lim['x'])
                    if lim.get('y'): ax_freqdiff.set_ylim(lim['y'])

    return fig, axs


def plot_fbid_trends(
    aggregated_metrics: Dict[str, Any],
    algorithms: List[str],
    limits: Optional[Dict] = None,
    metrics_config: Optional[MetricsConfig] = None
) -> Tuple[Optional[plt.Figure], Optional[Dict[str, plt.Axes]]]:
    """Shared function to plot unified FBID trends for multiple algorithms."""
    fig = None
    axs = None
    colors = plt.cm.viridis(np.linspace(0., 1., len(algorithms)))

    # Resolve mode key
    if metrics_config:
        mode_key = f"{metrics_config.coverage_mode.value}_{metrics_config.freq_diff_mode.value}_{metrics_config.frag_mode.value}_{metrics_config.recall_mode.value}"
    else:
        mode_key = "total_total_total_total"

    for i, algo in enumerate(algorithms):
        algo_data = aggregated_metrics.get(algo, {})
        per_fbid = algo_data.get('per_fbid', {})

        if not per_fbid:
            continue

        valid_fbids = list(per_fbid.keys())
        cov = np.array([per_fbid[fb].get(mode_key, {}).get('coverage', np.nan) for fb in valid_fbids])
        frag = np.array([per_fbid[fb].get(mode_key, {}).get('fragmentation', np.nan) for fb in valid_fbids])
        fd = np.array([per_fbid[fb].get(mode_key, {}).get('freq_diff', np.nan) for fb in valid_fbids])

        def _get_unc(key):
            vals = []
            for fb in valid_fbids:
                v = per_fbid[fb].get(mode_key, {}).get(key, [np.nan, np.nan])
                # Guard against inconsistent list lengths/types that cause "inhomogeneous" numpy errors
                if not isinstance(v, (list, tuple, np.ndarray)) or len(v) != 2:
                    v = [np.nan, np.nan]
                vals.append(v)
            try:
                arr = np.array(vals)
                return arr if arr.shape == (len(valid_fbids), 2) else None
            except ValueError:
                return None

        fig, axs = fbid_plot(
            FBs=np.array(valid_fbids), coverage_plot=cov, frag_plot=frag, freqdiff_plot=fd,
            coverage_unc=_get_unc("coverage_5_95"), frag_unc=_get_unc("fragmentation_5_95"),
            freqdiff_unc=_get_unc("freq_diff_5_95"),
            algorithm=algo, layout="v2", fig=fig, axs=axs, setup_axes=(i == 0),
            color=colors[i], show_legend=(i == len(algorithms) - 1),
            metrics_config=metrics_config,
            limits=limits
        )
    return fig, axs


def _run_algorithm_directly(wav: str, algorithm: str, cfg: Optional[PipelineConfig] = None) -> List[np.ndarray]:
    from ..contour_extraction.silbido import run_silbido, run_sam, run_smcphd
    pipeline_cfg = cfg or PipelineConfig()
    algo = algorithm.strip()
    if algo == "CREPE":
        predictor = CrepePredictor(model_dir=pipeline_cfg.paths.model_dir)
        return predictor.predict_crepe(wav, apply_mask=True)
    elif algo == "Silbido Profundo":
        return run_silbido(wav, pipeline_cfg)
    elif algo == "SAM":
        return run_sam(wav, pipeline_cfg)
    elif algo == "SMC-PHD":
        return run_smcphd(wav, pipeline_cfg)
    else:
        raise ValueError(f"Direct execution not configured for algorithm: {algo}")


def visualize_together(
    wav: str,
    params_dir: str,
    algorithms: Optional[List[str]] = None,
    wav_src: Optional[str] = None,
    use_precomputed: bool = False,
    load_src: Optional[str] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    metrics_config: Optional[MetricsConfig] = None,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    cfg = pipeline_config or PipelineConfig()
    if algorithms is None:
        algorithms = ['Silbido Profundo', 'SAM', 'SMC-PHD', 'CREPE']

    LABELS = {
        'Silbido Profundo': 'Silbido Profundo',
        'SAM': 'SAM-whistle',
        'SAM-Raw': 'SAM-Raw',
        'CREPE-2': 'CREPE Consp.',
        'CREPE': 'CREPE-tt',
        'SMC-PHD': 'SMC-PHD'
    }

    fig, axs = plt.subplot_mosaic(
        [['AA','AA'], ['B','C'], ['D','E']],
        figsize=(8, 6),
        constrained_layout=True
    )

    if wav_src is None:
        wav_src = load_src or params_dir

    audio, sr = sf.read(wav)
    nfft = 512
    f, t, Zxx = stft(audio, fs=sr, nperseg=nfft, noverlap=round(nfft * .9), padded=False, scaling='psd')
    spec = np.abs(Zxx)
    spec = np.log(spec + 1e-6)

    ground_contour, _ = run_ground(wav, params_dir=params_dir)
    ground_contour = _ensure_contour_list(ground_contour)

    ax = axs['AA']
    ax.pcolormesh(t, f/1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    for ground in ground_contour:
        ax.plot(ground[:,0], ground[:,1]/1000, color='w', alpha=1, linestyle=':', linewidth=2)

    ax.set_ylim([0, 20])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Ground Truth')

    for ALGORITHM, axi in zip(algorithms, ['B','C','D','E']):
        ax = axs[axi]
        ax.sharey(axs['AA'])

        if use_precomputed:
            pred_contour = load_precomputed(wav, ALGORITHM=ALGORITHM, top_dir=cfg.paths.precompute_dir)
        else:
            pred_contour = _run_algorithm_directly(wav, ALGORITHM, cfg=cfg)

        if cfg.algorithm.split_contours:
            pred_contour = fragment_contours(pred_contour, dur=cfg.algorithm.contour_dur_threshold)
        if cfg.algorithm.remove_harmonics:
            pred_contour = identify_harmonics(pred_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
            
        pred_contour = _ensure_contour_list(pred_contour)

        ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')

        colors = plt.cm.magma(np.linspace(0.5, .9, len(pred_contour)))
        for pred, c in zip(pred_contour, colors):
            ax.scatter(pred[:, 0], pred[:, 1] / 1000, color=c, zorder=10, alpha=1, s=2)

        ax.set_title(LABELS.get(ALGORITHM, ALGORITHM))

    axs['D'].set_xlabel('Time (s)')
    axs['D'].set_ylabel('Frequency (kHz)')
    axs['E'].set_xlabel('Time (s)')
    axs['B'].set_ylabel('Frequency (kHz)')

    # Apply limits to all axes if provided
    _apply_limits_to_axes(axs, limits)

    return fig


def visualize_prediction(
    wav: str,
    algorithm: str,
    wav_src: str,
    params_dir: str,
    pipeline_config: Optional[PipelineConfig] = None,
    metrics_config: Optional[MetricsConfig] = None,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    cfg = pipeline_config or PipelineConfig()
    audio, sr = sf.read(wav)
    nfft = 512
    f, t, Zxx = stft(audio, fs=sr, nperseg=nfft, noverlap=round(nfft * .9), padded=False, scaling='psd')
    spec = np.abs(Zxx)
    spec = np.log(spec + 1e-6)

    ground_contour, _ = run_ground(wav, params_dir=params_dir)
    ground_contour = _ensure_contour_list(ground_contour)
    
    pred_contour = load_precomputed(wav, ALGORITHM=algorithm, top_dir=cfg.paths.precompute_dir)
    if cfg.algorithm.split_contours:
        pred_contour = fragment_contours(pred_contour, dur=cfg.algorithm.contour_dur_threshold)
    if cfg.algorithm.remove_harmonics:
        pred_contour = identify_harmonics(pred_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
    pred_contour = _ensure_contour_list(pred_contour)

    fig, ax = plt.subplots(figsize=(4, 2), constrained_layout=True)
    ax.pcolormesh(t, f/1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    for ground in ground_contour:
        ax.plot(ground[:,0], ground[:,1]/1000, color='w', alpha=1, linestyle=':', linewidth=2)

    colors = plt.cm.magma(np.linspace(0.4, .8, len(pred_contour)))
    for pred, c in zip(pred_contour, colors):
        ax.scatter(pred[:,0], pred[:,1]/1000, color=c, zorder=10, alpha=1, s=1)

    ax.set_ylim([0, 20])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(algorithm)

    # Apply limits if provided
    if limits:
        first_limit = next(iter(limits.values()), None)
        if first_limit:
            if first_limit.get('x'): ax.set_xlim(first_limit['x'])
            if first_limit.get('y'): ax.set_ylim(first_limit['y'])

    return fig


def plot_metrics_mosaic(
    wav: str,
    params_dir: str,
    wav_src: str,
    load_src: str,
    algorithm: str = 'Silbido Profundo',
    pipeline_config: Optional[PipelineConfig] = None,
    metrics_config: Optional[MetricsConfig] = None,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    cfg = pipeline_config or PipelineConfig()
    audio, sr = sf.read(wav)
    b, a = butter(4, 2000, btype='highpass', fs=sr)
    audio = filtfilt(b, a, audio)

    nfft = 512
    f, t, Zxx = stft(audio, fs=sr, nperseg=nfft, noverlap=round(nfft * .9), padded=False, scaling='psd')
    spec = np.abs(Zxx)
    spec = np.log(spec + 1e-6)

    ground_contour, _ = run_ground(wav, params_dir=params_dir)
    ground_contour = _ensure_contour_list(ground_contour)
    
    pred_contour = load_precomputed(wav, ALGORITHM=algorithm, top_dir=cfg.paths.precompute_dir)
    if cfg.algorithm.split_contours:
        pred_contour = fragment_contours(pred_contour, dur=cfg.algorithm.contour_dur_threshold)
    if cfg.algorithm.remove_harmonics:
        pred_contour = identify_harmonics(pred_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
    pred_contour = _ensure_contour_list(pred_contour)

    fig, axs = plt.subplot_mosaic(
        [['AA','AA', 'F'], ['AA', 'AA', 'H'], ['B', 'C', 'G'], ['D', 'E', 'I']],
        figsize=(8, 8),
        constrained_layout=True
    )

    FRAG_AXS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    ax = axs['AA']
    ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    for ground in ground_contour:
        ax.plot(ground[:,0], 1.25 * ground[:,1]/1000, color='red', alpha=.5, linestyle='-', linewidth=2)
        ax.plot(ground[:, 0], .75 * ground[:, 1] / 1000, color='red', alpha=.5, linestyle='-', linewidth=2)
        ax.fill_between(ground[:, 0], .75 * ground[:, 1] / 1000, 1.25 * ground[:, 1] / 1000, color='red', alpha=.1)

    cids = np.flip(np.linspace(0.3, .9, len(pred_contour)))
    colors = plt.cm.magma(cids)
    for pred, c in zip(pred_contour, colors):
        ax.scatter(pred[:, 0], pred[:, 1] / 1000, color=c, zorder=10, alpha=1, s=1)

    ax.set_ylim([0, 20])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Ground Truth')

    for axi in FRAG_AXS:
        ax = axs[axi]
        ax.sharey(axs['AA'])
        ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')

        for ground in ground_contour:
            ax.plot(ground[:, 0], 1.25 * ground[:, 1] / 1000, color='red', alpha=.5, linestyle='-', linewidth=2)
            ax.plot(ground[:, 0], .75 * ground[:, 1] / 1000, color='red', alpha=.5, linestyle='-', linewidth=2)
            ax.fill_between(ground[:, 0], .75 * ground[:, 1] / 1000, 1.25 * ground[:, 1] / 1000, color='red', alpha=.1)

        for pred, c in zip(pred_contour, colors):
            ax.scatter(pred[:, 0], pred[:, 1] / 1000, color=c, zorder=10, alpha=1, s=1)

        ax.set_ylim([0, 20])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')

    axs['AA'].text(-0.2, 1.1, '(a)', transform=axs['AA'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['B'].text(-0.2, 1.1, '(b)', transform=axs['B'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['C'].text(-0.2, 1.1, '(c)', transform=axs['C'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['D'].text(-0.2, 1.1, '(d)', transform=axs['D'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['E'].text(-0.2, 1.1, '(e)', transform=axs['E'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['F'].text(-0.2, 1.1, '(f)', transform=axs['F'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['G'].text(-0.2, 1.1, '(g)', transform=axs['G'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['H'].text(-0.2, 1.1, '(h)', transform=axs['H'].transAxes, va='bottom', fontweight='bold', fontsize=14)
    axs['I'].text(-0.2, 1.1, '(i)', transform=axs['I'].transAxes, va='bottom', fontweight='bold', fontsize=14)

    axs['AA'].set_title('Raw Contours')
    axs['B'].set_title('Harmonic Removal')

    # Apply limits to all axes if provided
    _apply_limits_to_axes(axs, limits)

    return fig


def plot_preprocessing(
    wav: str,
    use_precomputed: bool = True,
    pipeline_config: Optional[PipelineConfig] = None,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    cfg = pipeline_config or PipelineConfig()
    audio, sr = sf.read(wav)

    # Load precomputed contours or fall back to direct execution
    if use_precomputed:
        try:
            silb_contour = load_precomputed(wav, ALGORITHM="Silbido Profundo", top_dir=cfg.paths.precompute_dir)
        except FileNotFoundError:
            from ..contour_extraction.silbido import run_silbido
            silb_contour = run_silbido(wav, cfg)

        try:
            crepe_contour = load_precomputed(wav, ALGORITHM="CREPE", top_dir=cfg.paths.precompute_dir)
        except FileNotFoundError:
            crepe_predictor = CrepePredictor(model_dir=cfg.paths.model_dir)
            crepe_contour = crepe_predictor.predict_crepe(wav)
    else:
        from ..contour_extraction.silbido import run_silbido
        crepe_predictor = CrepePredictor(model_dir=cfg.paths.model_dir)
        crepe_contour = crepe_predictor.predict_crepe(wav)
        silb_contour = run_silbido(wav, cfg)

    silb_contour = _ensure_contour_list(silb_contour)
    crepe_contour = _ensure_contour_list(crepe_contour)

    f, t, Zxx = stft(audio, fs=sr, nperseg=1024, noverlap=round(1024*.9), padded=False, scaling='psd')
    spec = np.abs(Zxx)
    spec = np.log(spec + 1e-6)

    fig, axs = plt.subplot_mosaic(
        [['AA','AA','.','BA','BA','.','CA','CA'], ['AA','AA','.','BB','BB','.','CB','CB']],
        figsize=(10, 6),
        constrained_layout=True
    )

    aspect = .075
    axs['AA'].sharex(axs['BB'])
    axs['AA'].sharey(axs['BB'])
    axs['BA'].sharex(axs['BB'])
    axs['BA'].sharey(axs['BB'])
    axs['CA'].sharex(axs['BB'])
    axs['CA'].sharey(axs['BB'])
    axs['CB'].sharex(axs['BB'])
    axs['CB'].sharey(axs['BB'])
    axs['BA'].set_aspect(aspect)
    axs['BB'].set_aspect(aspect)
    axs['CB'].set_aspect(aspect)
    axs['CA'].set_aspect(aspect)
    axs['BB'].set_ylim([0, 30])

    axs['AA'].pcolormesh(t, f/1000, spec, vmin=np.min(spec), vmax=np.max(spec))
    axs['AA'].set_aspect(aspect)
    axs['AA'].set_xlabel('Time (s)')
    axs['AA'].set_ylabel('Frequency (kHz)')

    axs['BA'].set_xlabel('Time (s)')
    axs['BA'].set_ylabel('Frequency (kHz)')
    axs['BB'].set_xlabel('Time (s)')
    axs['BB'].set_ylabel('Frequency (kHz)')
    axs['CA'].set_xlabel('Time (s)')
    axs['CA'].set_ylabel('Frequency (kHz)')
    axs['CB'].set_xlabel('Time (s)')
    axs['CB'].set_ylabel('Frequency (kHz)')

    axs['BA'].set_title('Tonal Estimation')
    axs['BB'].set_title('FFC Estimation')
    axs['CA'].set_title('Tonal Estimation')
    axs['CB'].set_title('FFC Estimation')

    _p = 2
    # Plot Raw Contours
    for contour in silb_contour:
        axs['BA'].scatter(contour[:,0], contour[:,1]/1000, s=_p)
    for contour in crepe_contour:
        axs['BB'].scatter(contour[:,0], contour[:,1]/1000, s=_p)

    # Apply Post-Processing
    if cfg.algorithm.split_contours:
        silb_contour = fragment_contours(silb_contour, dur=cfg.algorithm.contour_dur_threshold)
        crepe_contour = fragment_contours(crepe_contour, dur=cfg.algorithm.contour_dur_threshold)
    if cfg.algorithm.remove_harmonics:
        silb_contour = identify_harmonics(silb_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
        crepe_contour = identify_harmonics(crepe_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
        
    silb_contour = _ensure_contour_list(silb_contour)
    crepe_contour = _ensure_contour_list(crepe_contour)

    # Plot Processed Contours
    for contour in silb_contour:
        axs['CA'].scatter(contour[:,0], contour[:,1]/1000, s=_p)
    for contour in crepe_contour:
        axs['CB'].scatter(contour[:,0], contour[:,1]/1000, s=_p)

    # Apply limits to all axes if provided
    _apply_limits_to_axes(axs, limits)

    return fig


def plot_metrics_mosaic_v2(
    wav: str,
    params_dir: str,
    algorithm: str = 'Silbido Profundo',
    precompute_dir: Optional[str] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    metrics_config: Optional[MetricsConfig] = None,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    cfg = pipeline_config or PipelineConfig()
    audio, sr = sf.read(wav)
    b, a = butter(4, 2000, btype='highpass', fs=sr)
    audio = filtfilt(b, a, audio)

    nfft = 1024
    f, t, Zxx = stft(audio, fs=sr, nperseg=nfft, noverlap=round(nfft * .9), padded=False, scaling='psd')
    spec = np.abs(Zxx)
    spec = np.log(spec + 1e-6)

    ground_contour, _ = run_ground(wav, params_dir=params_dir)
    ground_contour = _ensure_contour_list(ground_contour)
    
    ground_dict = {}
    for ground in ground_contour:
        for i in range(ground.shape[0]):
            gx, gf = ground[i, 0], ground[i, 1]
            ground_dict[np.round(gx, decimals=3)] = gf

    pred_contour = load_precomputed(wav, ALGORITHM=algorithm, top_dir=precompute_dir or str(cfg.paths.precompute_dir))
    pred_contour = _ensure_contour_list(pred_contour)
    pred_contour = fragment_contours(pred_contour, dur=cfg.algorithm.contour_dur_threshold)
    pred_contour = _ensure_contour_list(pred_contour)

    fig, axs = plt.subplot_mosaic(
        [['A', 'B'], ['C', 'D']], figsize=(10, 8), constrained_layout=True
    )
    fig.subplots_adjust(hspace=.5, wspace=.3)

    FRAG_AXS = ['C', 'D']
    total_cov = {}
    total_freq_diff = []

    # Subplot A: Raw Contours
    ax = axs['A']
    ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    cids = np.flip(np.linspace(0.3, .9, len(pred_contour)))
    np.random.shuffle(cids)
    colors = plt.cm.magma(cids)
    for pred, c in zip(pred_contour, colors):
        ax.plot(pred[:, 0], pred[:, 1] / 1000, color='white', zorder=10, alpha=1, linewidth=2, linestyle='-')
    ax.set_ylim([2, 24]); ax.set_xlim([.18, .9])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Raw Contours')

    # Subplot B: Harmonic Removal
    pred_contour_harm = identify_harmonics(pred_contour, tolerance=cfg.algorithm.harmonic_tolerance, freqdiff=cfg.algorithm.harmonic_freqdiff)
    pred_contour_harm = _ensure_contour_list(pred_contour_harm)
    
    ax = axs['B']
    ax.pcolormesh(t, f / 1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    cids = np.flip(np.linspace(0.6, .9, len(pred_contour_harm)))
    np.random.shuffle(cids)
    colors = plt.cm.magma(cids)
    for pred, c in zip(pred_contour_harm, colors):
        ax.plot(pred[:, 0], pred[:, 1] / 1000, color='white', zorder=10, alpha=1, linewidth=2, linestyle='-')
    ax.set_ylim([2, 24]); ax.set_xlim([.18, .9])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (kHz)')
    ax.set_title('Harmonic Removal')

    # Subplots C, D: Fragments
    for axi, c, pred, fragi in zip(FRAG_AXS, colors, pred_contour_harm, range(len(FRAG_AXS))):
        ax = axs[axi]
        pred = np.array(pred)
        ax.plot(pred[:, 0], pred[:, 1] / 1000, color=c, zorder=10, alpha=1, linewidth=3, linestyle='-')

        pred_cov, fdiff, fill_low, fill_high = [], [], [], []
        for i in range(pred.shape[0]):
            px, pf = pred[i, 0], pred[i, 1]
            if np.round(px, decimals=3) in ground_dict:
                pred_cov.append(px)
                fill_low.append(pf/1000)
                fill_high.append(ground_dict[np.round(px, decimals=3)]/1000)
                fd = np.abs(pf - ground_dict[np.round(px, decimals=3)]) / ground_dict[np.round(px, decimals=3)]
                fdiff.append(fd)
                total_freq_diff.append(fd)
                if fd <= .25: total_cov[np.round(px, decimals=3)] = 1

        ax.fill_between(pred_cov, fill_low, fill_high, color='#a1a1a1')
        ax.text(0.05, 0.95, f'Fragment {fragi+1}:\nRel. Freq. Diff: {np.mean(fdiff):.3f}\nCoverage: {len(pred_cov)/len(ground_dict):.2f}',
                transform=ax.transAxes, va='top', ha='left')

        # FIXED: Axes-relative coordinates (y=1.1 is ~10% above the plot top)
        ax.plot(pred_cov, np.ones(len(pred_cov)) * 1.1, transform=ax.get_xaxis_transform(), color=c, zorder=10, alpha=1., clip_on=False, linewidth=4)

        for ground in ground_contour:
            ax.plot(ground[:, 0], ground[:, 1] / 1000, color='k', alpha=1, linestyle='-', linewidth=1)

            # Ground truth markers now use axes-relative coordinates to stay above the plot
            ax.plot(ground[:, 0], np.ones(len(ground[:,0])) * 1.1, transform=ax.get_xaxis_transform(), color='k', alpha=1, linestyle='-', linewidth=1.5, clip_on=False)
            ax.plot([ground[0,0], ground[0,0]], [1.02, 1.18], transform=ax.get_xaxis_transform(), color='k', alpha=1, linestyle='-', linewidth=1.5, clip_on=False)
            ax.plot([ground[-1, 0], ground[-1, 0]], [1.02, 1.18], transform=ax.get_xaxis_transform(), color='k', alpha=1, linestyle='-', linewidth=1.5, clip_on=False)
            ax.fill_between(ground[:, 0], .75 * ground[:, 1] / 1000, 1.25 * ground[:, 1] / 1000, color='red', alpha=.1)

        ax.set_ylim([2, 14]); ax.set_xlim([.18, .9])
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (kHz)')

    axs['B'].text(0.05, 0.95, f'Rel. Freq. Diff: {np.mean(total_freq_diff):.3f}\nFragmentation: {len(pred_contour_harm)}\nTotal Coverage: {len(total_cov)/len(ground_dict):.2f}',
                  transform=axs['B'].transAxes, va='top', ha='left', color='white', fontweight='bold')

    for i, ax_key in enumerate(['A', 'B', 'C', 'D']):
        axs[ax_key].text(-0.2, 1.1, f'({chr(97+i)})', transform=axs[ax_key].transAxes, va='bottom', fontweight='bold', fontsize=14)

    # Apply limits to all axes if provided
    _apply_limits_to_axes(axs, limits)

    return fig


def plot_metric_violin(
    aggregated_metrics: Dict[str, Any],
    algorithms: List[str],
    metric_name: str,
    metrics_config: Optional[MetricsConfig] = None
) -> plt.Figure:
    """Generates a legacy-styled violin plot comparing algorithms across a selected metric."""
    mode_key = f"{metrics_config.coverage_mode.value}_{metrics_config.freq_diff_mode.value}_{metrics_config.frag_mode.value}_{metrics_config.recall_mode.value}" if metrics_config else "total_total_total_total"
    
    key_map = {
        "Coverage": "coverage",
        "False Positives": "false_pos",
        "Frequency Difference": "freq_diff",
        "Fragmentation": "fragmentation"
    }
    data_key = key_map.get(metric_name, "coverage")
    
    plot_data = []
    valid_algos = []
    for algo in algorithms:
        per_fbid = aggregated_metrics.get(algo, {}).get('per_fbid', {})
        if not per_fbid: continue
        
        vals = []
        for fb_m in per_fbid.values():
            v = fb_m.get(mode_key, {}).get(data_key, np.nan)
            if not np.isnan(v): vals.append(v)
        if vals:
            plot_data.append(vals)
            valid_algos.append(algo)
            
    if not plot_data:
        raise ValueError(f"No valid data found for {metric_name}.")
        
    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(plot_data, np.arange(len(valid_algos)), showmedians=True, showextrema=True)
    
    LABELS = {'Silbido Profundo': 'Silbido Profundo', 'SAM': 'SAM-whistle', 'CREPE': 'CREPE-tt', 'SMC-PHD': 'SMC-PHD'}
    ax.set_xticks(np.arange(len(valid_algos)), labels=[LABELS.get(a, a) for a in valid_algos])
    ax.set_xlabel('Algorithm')
    
    labels = _resolve_metric_labels(metrics_config)
    ax.set_ylabel(labels.get(data_key, metric_name))
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    colors = plt.cm.viridis(np.linspace(0., 1., len(parts['bodies'])))
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)
        
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmedians'].set_linestyle(':')
    if 'cbars' in parts: parts['cbars'].set_visible(False)
    if 'cmins' in parts:
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(1.5)
        parts['cmins'].set_linestyle('-')
    if 'cmaxes' in parts:
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(1.5)
        parts['cmaxes'].set_linestyle('-')
        
    fig.tight_layout()
    return fig


def plot_fragmentation_verification(
    wav: str,
    ground_contours: List[np.ndarray],
    pred_contours: List[np.ndarray],
    metrics_config: Optional[MetricsConfig] = None,
    ffc_threshold: Optional[float] = None,
    discont: Optional[List] = None,
    show_spectrogram: Optional[bool] = False,
    limits: Optional[Dict] = None,
) -> plt.Figure:
    """Visualizes and verifies fragmentation logic for a single file."""
    threshold = ffc_threshold if ffc_threshold is not None else (metrics_config.freq_diff_threshold if metrics_config else 0.25)
    
    audio, sr = sf.read(wav)
    f, t, Zxx = stft(audio, fs=sr, nperseg=1024, noverlap=921, scaling='psd')
    spec = np.log(np.abs(Zxx) + 1e-6)

    fig, axs = plt.subplot_mosaic(
        [['A', 'B'], ['A', 'B']],
        figsize=(10, 6),
        constrained_layout=True,
        height_ratios=[3, 1]
    )

    # Ensure valid contour lists
    ground_contours = _ensure_contour_list(ground_contours)
    pred_contours = _ensure_contour_list(pred_contours)

    # Incorporate proper ground truth segmentation based on comparison.py
    segmented_ground = []
    for gt in ground_contours:
        segmented_ground.extend(_split_ground_truth(gt, discont=discont, gap_threshold=0.1))
    ground_contours = segmented_ground

    # 1. Compute fragmentation metrics explicitly for auditability
    n_gt_loops = len(ground_contours)
    total_frag = len(pred_contours)
    
    ffc_count = 0
    ffc_segments = []
    fp_segments = []
    
    for pred in pred_contours:
        is_ffc = False
        for gt in ground_contours:
            for i in range(pred.shape[0]):
                # Explicitly convert to float to prevent ambiguous truth value errors 
                # if array indexing returns numpy scalars or unexpected shapes
                px, pf = float(pred[i, 0]), float(pred[i, 1])
                for j in range(gt.shape[0]):
                    gx, gf = float(gt[j, 0]), float(gt[j, 1])
                    if gf == 0: continue
                    if abs(px - gx) < 0.05 and abs(pf -  gf) / gf < threshold:
                        is_ffc = True
                        break
                if is_ffc: break
            if is_ffc: break
            
        if is_ffc:
            ffc_count += 1
            ffc_segments.append(pred)
        else:
            fp_segments.append(pred)

    per_loop = total_frag / n_gt_loops if n_gt_loops > 0 else 0
    per_loop_ffc = ffc_count / n_gt_loops if n_gt_loops > 0 else 0

    # 2. Plot Spectrogram + Contours
    ax_spec = axs['A']
    if show_spectrogram:
        ax_spec.pcolormesh(t, f/1000, spec, vmin=np.min(spec), vmax=np.max(spec), cmap='viridis')
    
    # Ground Truth
    for idx, gt in enumerate(ground_contours):
        ax_spec.plot(gt[:,0], gt[:,1]/1000, color='grey', linewidth=3, linestyle='--', label='Ground Truth' if idx == 0 else "")
        
    # Predicted FFC segments
    colors = plt.cm.viridis(np.linspace(0., 1., len(ffc_segments)))
    for idx, seg in enumerate(ffc_segments):
        ax_spec.plot(seg[:,0], seg[:,1]/1000, color=colors[idx], linewidth=3, alpha=0.8, label='FFC Fragment' if idx == 0 else "")
        
    # Predicted FP segments
    colors = plt.cm.viridis(np.linspace(0., 1., len(fp_segments)))
    for idx, seg in enumerate(fp_segments):
        ax_spec.plot(seg[:,0], seg[:,1]/1000, color=colors[idx], linestyle=':', linewidth=3, alpha=0.8, label='FP Fragment' if idx == 0 else "")

    # Plot discontinuities as vertical lines
    if discont:
        for i, d in enumerate(discont):
            ax_spec.axvline(x=d, color='black', linestyle=':', alpha=0.8, linewidth=1.5, label='Discontinuity' if i == 0 else "")

    ax_spec.set_ylim([0, 24])
    ax_spec.set_xlabel('Time (s)')
    ax_spec.set_ylabel('Frequency (kHz)')
    ax_spec.set_title('Fragmentation Breakdown')
    ax_spec.legend(loc='upper right')

    _apply_limits_to_axes({'A': ax_spec}, limits)

    # Display summary statistics
    n_discont = len(discont) if discont else 0
    stats_text = (
        f"Pred Contours: {len(pred_contours)}\n"
        f"GT Contours: {len(ground_contours)}\n"
        f"FFC Contours: {ffc_count}\n"
        f"Discontinuities: {n_discont}"
    )
    ax_spec.text(0.02, 0.98, stats_text, transform=ax_spec.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 3. Summary Bar Chart
    ax_bar = axs['B']
    modes = ['Total', 'FFC', 'Per Loop', 'Per Loop FFC']
    values = [total_frag, ffc_count, per_loop, per_loop_ffc]
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    
    bars = ax_bar.bar(modes, values, color=colors, edgecolor='black')
    ax_bar.set_ylabel('Count / Avg per GT Loop')
    ax_bar.set_title('Computed Fragmentation Metrics')
    ax_bar.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    return fig

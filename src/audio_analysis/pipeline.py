"""
Orchestration module for the Audio Analysis Pipeline.
Handles batch execution of precomputation, evaluation, and plotting.
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import PipelineConfig, DataLayout
from .utils.contour_utils import load_ground_truth, load_precomputed, save_precomputed, fragment_contours, identify_harmonics, _resolve_fbid_snr
from .contour_extraction.silbido import run_silbido, run_sam, run_smcphd
from .evaluation.comparison import compare_contours, RawComparisonResult
from .evaluation.metrics import _select_metrics, aggregate_metrics, compute_all_mode_metrics
from .plotting.plotting import plot_results, fbid_plot

try:
    from .contour_extraction.crepe import CrepePredictor
except ImportError:
    CrepePredictor = None

logger = logging.getLogger(__name__)

def _extract_ind(wav_path: Path) -> Optional[str]:
    """Extract IND identifier from WAV filename (e.g., 'IND123_001.wav' -> '123')."""
    name = wav_path.name
    if "IND" not in name:
        return None
    return name.split("IND")[-1].split(".wav")[0].split("_")[0]

def _parse_snr_float(snr_str: str) -> float:
    """Convert SNR string to float for plotting. Handles CLEAN, SNR_20, 20dB, etc."""
    if str(snr_str).upper() == "CLEAN":
        return 0.0
    
    # Extract the first numeric value found in the string
    match = re.search(r'-?\d+\.?\d*', str(snr_str))
    if match:
        return float(match.group())
    
    return 0.0

def save_metrics(metrics: Dict, output_path: Path, config: Optional[PipelineConfig] = None) -> None:
    """Save aggregated metrics to a JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Wrap metrics and optionally embed the full run configuration
    payload = {"metrics": metrics}
    if config is not None:
        payload["config"] = config.model_dump(mode='json')
        
    with open(output_path, 'w') as f:
        json.dump(convert(payload), f, indent=4)
    logger.info(f"Metrics saved to {output_path}")

def _apply_postprocessing(contours: List[np.ndarray], config: PipelineConfig) -> List[np.ndarray]:
    """Apply contour splitting and harmonic removal exactly once."""
    if config.algorithm.split_contours:
        contours = fragment_contours(contours, dur=config.algorithm.contour_dur_threshold)
    if config.algorithm.remove_harmonics:
        contours = identify_harmonics(contours, tolerance=config.algorithm.harmonic_tolerance, freqdiff=config.algorithm.harmonic_freqdiff)
    return contours

def _discover_wav_files(data_root: Path, fbids: Optional[List[str]], snr_levels: Optional[List[str]], layout: DataLayout) -> List[Path]:
    """Discover WAV files based on FBID and SNR level filters."""
    wav_files = []
    if not data_root.exists():
        logger.warning(f"Data root directory not found: {data_root}")
        return wav_files

    if layout == DataLayout.NESTED_NOISE:
        snr_dir_base = data_root / "NoiseLevels"
        if not snr_dir_base.exists():
            logger.warning(f"NoiseLevels directory not found: {snr_dir_base}")
            return wav_files

        snr_targets = snr_levels or [d.name for d in snr_dir_base.iterdir() if d.is_dir()]
        for snr in snr_targets:
            snr_path = snr_dir_base / snr
            if not snr_path.exists():
                logger.warning(f"SNR directory not found: {snr_path}")
                continue
            label_targets = fbids or [d.name for d in snr_path.iterdir() if d.is_dir()]
            for label in label_targets:
                label_dir = snr_path / label
                if label_dir.exists():
                    wav_files.extend(sorted(label_dir.glob("*.wav")))
    else: # FLAT
        wav_files = sorted(data_root.rglob("*.wav"))
        if fbids:
            wav_files = [w for w in wav_files if w.parent.name in fbids]

    return sorted(set(wav_files))

def _build_param_lookup(params_dir: Path) -> Dict[str, Path]:
    """Scan params directory and build a lookup table mapping base names to file paths."""
    lookup = {}
    if not params_dir.exists():
        return lookup
    for p in params_dir.rglob("*_params.*"):
        if p.suffix in ('.mat', '.npy'):
            base = p.stem.replace("_params", "")
            lookup[base.lower()] = p
    return lookup

def _run_algorithm(wav_path: Path, algorithm: str, config: PipelineConfig, crepe_predictor: Optional["CrepePredictor"] = None) -> List[np.ndarray]:
    """Route to the correct contour extraction function."""
    if algorithm == "CREPE":
        if crepe_predictor is None:
            if CrepePredictor is None:
                raise ImportError("CREPE requires PyTorch. Install with: pip install audio-analysis[models]")
            crepe_predictor = CrepePredictor()
        return crepe_predictor.predict_crepe(str(wav_path))
    elif algorithm == "Silbido Profundo":
        return run_silbido(str(wav_path), config)
    elif algorithm == "SAM":
        return run_sam(str(wav_path), config)
    elif algorithm == "SMC-PHD":
        return run_smcphd(str(wav_path), config)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def execute_precompute(config: PipelineConfig, wav_files: List[Path], algorithms: List[str], overwrite: bool = True, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None:
    """Run contour extraction and save precomputed results."""
    logger.info(f"Starting precomputation for {len(wav_files)} files across {len(algorithms)} algorithms.")

    crepe_predictor = None
    if "CREPE" in algorithms:
        if CrepePredictor is None:
            raise ImportError("CREPE requires PyTorch. Install with: pip install audio-analysis[models]")
        crepe_predictor = CrepePredictor(model_dir=config.paths.model_dir)

    total_tasks = len(wav_files) * len(algorithms)
    current_task = 0
    use_tqdm = progress_callback is None

    for algo in (tqdm(algorithms, desc="Algorithms", unit="algo") if use_tqdm else algorithms):
        for wav_path in (tqdm(wav_files, desc=f"Processing {algo}", unit="file", leave=False) if use_tqdm else wav_files):
            try:
                label, intermediate = _resolve_fbid_snr(wav_path)
                save_file = config.paths.precompute_dir / algo / intermediate / label / f"{wav_path.stem}_contour.npy"

                if not overwrite and save_file.exists():
                    continue

                logger.debug(f"Running {algo} on {wav_path.name}")
                contours = _run_algorithm(wav_path, algo, config, crepe_predictor)

                save_precomputed(
                    wav=str(wav_path),
                    contours=contours,
                    algorithm=algo,
                    src=str(wav_path.parent),
                    top_dir=str(config.paths.precompute_dir)
                )
                current_task += 1
                if progress_callback:
                    progress_callback(current_task, total_tasks, f"Precomputing {algo}: {wav_path.name}")
            except Exception:
                logger.exception(f"Failed to precompute {algo} for {wav_path.name}")

def compute_metrics_from_precomputed(
    config: PipelineConfig,
    wav_files: List[Path],
    algorithms: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """Compute metrics from already precomputed contours.
    Decoupled from pipeline orchestration for fast threshold/mode iteration.
    """
    logger.info("Starting metric computation from precomputed contours.")
    raw_results_snr: Dict[str, Dict[str, List[RawComparisonResult]]] = {algo: {} for algo in algorithms}
    raw_results_fbid: Dict[str, Dict[str, List[RawComparisonResult]]] = {algo: {} for algo in algorithms}

    total_tasks = len(wav_files) * len(algorithms)
    current_task = 0
    use_tqdm = progress_callback is None

    for wav_path in (tqdm(wav_files, desc="Computing metrics", unit="file") if use_tqdm else wav_files):
        try:
            label, snr_level = _resolve_fbid_snr(wav_path)
            ground, discont = load_ground_truth(wav_path=wav_path, params_dir=config.paths.params_dir)
            if ground is None:
                logger.warning(f"No ground truth found for {wav_path.name}, skipping.")
                continue

            for algo in (tqdm(algorithms, desc="Algorithms", unit="algo", leave=False) if use_tqdm else algorithms):
                try:
                    pred_contours = load_precomputed(
                        wav=str(wav_path),
                        ALGORITHM=algo,
                        top_dir=config.paths.precompute_dir
                    )
                    if pred_contours is None:
                        logger.warning(f"No precomputed contours for {algo} on {wav_path.name}, skipping.")
                        continue

                    if isinstance(pred_contours, np.ndarray):
                        pred_contours = pred_contours.tolist()

                    pred_contours = _apply_postprocessing(pred_contours, config)

                    result = compare_contours(
                        ground=ground,
                        mat_contour=pred_contours,
                        discont=discont,
                        ffc_threshold=config.metrics.freq_diff_threshold
                    )
                    current_task += 1
                    if progress_callback:
                        progress_callback(current_task, total_tasks, f"Evaluating {algo}: {wav_path.name}")

                    if snr_level not in raw_results_snr[algo]:
                        raw_results_snr[algo][snr_level] = []
                    raw_results_snr[algo][snr_level].append(result)

                    if label not in raw_results_fbid[algo]:
                        raw_results_fbid[algo][label] = []
                    raw_results_fbid[algo][label].append(result)
                except Exception as e:
                    logger.warning(f"Skipping {algo} for {wav_path.name} due to error: {e}")
                    continue
        except Exception:
            logger.exception(f"Metric computation failed for {wav_path.name}")

    aggregated = {}
    for algo in algorithms:
        algo_data = {'global': {}, 'per_snr': {}, 'per_fbid': {}}

        for snr, results in raw_results_snr[algo].items():
            if results:
                algo_data['per_snr'][snr] = compute_all_mode_metrics(results, config.metrics)

        for fbid, results in raw_results_fbid[algo].items():
            if results:
                algo_data['per_fbid'][fbid] = compute_all_mode_metrics(results, config.metrics)

        all_results = []
        for results in raw_results_snr[algo].values():
            all_results.extend(results)
            
        if all_results:
            algo_data['global'] = compute_all_mode_metrics(all_results, config.metrics)
            
        aggregated[algo] = algo_data

    return aggregated

def execute_evaluate(config: PipelineConfig, wav_files: List[Path], algorithms: List[str], progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    """Run contour comparison, aggregate metrics, and save results."""
    # Apply optional SNR > 20dB filter
    if config.filter_snr_20db:
        filter_path = config.paths.slsnr_20_filter_path
        if filter_path and filter_path.exists():
            with open(filter_path, 'r') as f:
                snr_filter = json.load(f)
            snr_filter_sets = {fb: set(inds) for fb, inds in snr_filter.items()}
            
            filtered_wavs = []
            for wp in wav_files:
                fb, _ = _resolve_fbid_snr(wp)
                ind = _extract_ind(wp)
                if ind and fb in snr_filter_sets and ind in snr_filter_sets[fb]:
                    filtered_wavs.append(wp)
                else:
                    logger.debug(f"Filtered out {wp.name} (FBID: {fb}, IND: {ind}) due to SNR < 20dB filter.")
            
            logger.info(f"Applied SNR > 20dB filter: {len(wav_files)} -> {len(filtered_wavs)} files.")
            wav_files = filtered_wavs
        else:
            logger.warning(f"SNR filter requested but {filter_path} not found. Proceeding without filter.")

    aggregated = compute_metrics_from_precomputed(config, wav_files, algorithms, progress_callback=progress_callback)
    save_metrics(aggregated, config.paths.output_dir / "evaluation_metrics.json", config=config)
    logger.info("Evaluation complete. All metric modes aggregated.")
    return aggregated

def execute_plot(config: PipelineConfig, aggregated_metrics: Dict, algorithms: List[str], 
                 wav_files: Optional[List[Path]] = None, target_wav: Optional[str] = None,
                 snr_plot_min: Optional[float] = None, snr_plot_max: Optional[float] = None) -> None:
    """Generate unified SNR trends visualization from aggregated metrics."""
    logger.info("Starting plot generation.")
    output_dir = config.paths.output_dir

    # Build snr_wav_map for spectrogram insets
    snr_wav_map = {}
    if target_wav and wav_files:
        def _get_wav_base(p: Path) -> str:
            name = p.stem
            if '_snr_' in name:
                name = name.split('_snr_')[0]
            elif '_CLEAN_' in name:
                name = name.split('_CLEAN_')[0]
            return name

        try:
            target_path = Path(target_wav)
            target_fbid, target_snr = _resolve_fbid_snr(target_path)
            target_base = _get_wav_base(target_path)
            snr_wav_map[target_snr] = target_wav
            for wav_path in wav_files:
                try:
                    fbid, snr_level = _resolve_fbid_snr(wav_path)
                    base = _get_wav_base(wav_path)
                    if fbid == target_fbid and base == target_base and snr_level not in snr_wav_map:
                        snr_wav_map[snr_level] = str(wav_path)
                except Exception:
                    continue
        except Exception:
            pass

        if not snr_wav_map:
            for wav_path in wav_files:
                try:
                    _, snr_level = _resolve_fbid_snr(wav_path)
                    if snr_level not in snr_wav_map:
                        snr_wav_map[snr_level] = str(wav_path)
                except Exception:
                    continue

    # Resolve the exact mode key from configuration
    mode_key = f"{config.metrics.coverage_mode.value}_{config.metrics.freq_diff_mode.value}_{config.metrics.frag_mode.value}_{config.metrics.recall_mode.value}"

    fig = None
    axs = None
    colors = plt.cm.viridis(np.linspace(0., 1., len(algorithms)))

    for i, algo in enumerate(algorithms):
        if algo not in aggregated_metrics:
            continue
        algo_data = aggregated_metrics[algo]
        per_snr = algo_data.get('per_snr', {})
        if not per_snr:
            logger.warning(f"No per-SNR data for {algo}, skipping.")
            continue

        sorted_snrs = sorted(per_snr.keys(), key=lambda x: _parse_snr_float(x))
        
        # Access nested metrics using the resolved mode_key
        cov_arr = np.array([per_snr[snr].get(mode_key, {}).get("coverage", np.nan) for snr in sorted_snrs])
        fp_arr = np.array([per_snr[snr].get(mode_key, {}).get("false_pos", np.nan) for snr in sorted_snrs])
        fd_arr = np.array([per_snr[snr].get(mode_key, {}).get("freq_diff", np.nan) for snr in sorted_snrs])
        frag_arr = np.array([per_snr[snr].get(mode_key, {}).get("fragmentation", np.nan) for snr in sorted_snrs])
        noise_floats = np.array([_parse_snr_float(snr) for snr in sorted_snrs])

        def _get_unc(key):
            vals = [per_snr[snr].get(mode_key, {}).get(key, [np.nan, np.nan]) for snr in sorted_snrs]
            arr = np.array(vals)
            return arr if arr.shape == (len(sorted_snrs), 2) else None

        fig, axs = plot_results(
            noise_floats=noise_floats,
            coverage=cov_arr,
            false_pos=fp_arr,
            freq_diff=fd_arr,
            frag=frag_arr,
            coverage_unc=_get_unc("coverage_5_95"),
            false_pos_unc=_get_unc("false_pos_5_95"),
            freq_diff_unc=_get_unc("freq_diff_5_95"),
            frag_unc=_get_unc("fragmentation_5_95"),
            algorithm=algo,
            layout="v2",
            fig=fig,
            axs=axs,
            setup_axes=(i == 0),
            color=colors[i],
            show_legend=(i == len(algorithms) - 1),
            metrics_config=config.metrics,
            snr_wav_map=snr_wav_map if i == 0 else None,
            snr_plot_min=snr_plot_min,
            snr_plot_max=snr_plot_max
        )

    if fig:
        plot_path = output_dir / "results_snr_trends.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved unified SNR trends plot to {plot_path}")
    else:
        logger.warning("No valid data to plot SNR trends.")


def plot_metrics_per_fbid(config, wav_files, algorithms, subsample=None):
    """Compute and plot metrics per FBID using fbid_plot."""
    if subsample:
        wav_files = sorted(wav_files)[::subsample]

    fb_raw = {algo: {} for algo in algorithms}

    for wav_path in wav_files:
        fbid = wav_path.parent.name
        ground, discont = load_ground_truth(wav_path, config.paths.params_dir)
        if ground is None:
            continue

        for algo in algorithms:
            if fbid not in fb_raw[algo]:
                fb_raw[algo][fbid] = []
            try:
                pred = load_precomputed(str(wav_path), algo, top_dir=str(config.paths.precompute_dir))
                if pred is None:
                    continue

                if isinstance(pred, np.ndarray):
                    pred = pred.tolist()

                pred = _apply_postprocessing(pred, config)

                res = compare_contours(ground, pred, discont=discont, ffc_threshold=config.metrics.freq_diff_threshold)
                fb_raw[algo][fbid].append(res)
            except Exception:
                logger.exception(f"Skipping {wav_path.name} for {algo}")

    output_dir = Path(config.paths.output_dir)

    all_algo_metrics = {}
    for algo in algorithms:
        algo_metrics = {}
        for fb, results in fb_raw[algo].items():
            if results:
                agg = aggregate_metrics([_select_metrics(r, config.metrics) for r in results])
                algo_metrics[fb] = agg
        all_algo_metrics[algo] = algo_metrics

    if not all_algo_metrics:
        return

    first_algo = algorithms[0]
    first_algo_data = all_algo_metrics.get(first_algo, {})
    if not first_algo_data:
        return

    # Use all FBIDs from the first algorithm as the base list
    base_fbids = list(first_algo_data.keys())

    for algo in algorithms:
        algo_data = all_algo_metrics.get(algo, {})
        if not algo_data:
            continue

        valid_fbids = [fb for fb in base_fbids if fb in algo_data]
        if not valid_fbids:
            continue

        cov_arr = [algo_data[fb]['coverage'] for fb in valid_fbids]
        frag_arr = [algo_data[fb]['fragmentation'] for fb in valid_fbids]
        fd_arr = [algo_data[fb]['freq_diff'] for fb in valid_fbids]

        fig, _ = fbid_plot(
            FBs=np.array(valid_fbids),
            coverage_plot=np.array(cov_arr),
            frag_plot=np.array(frag_arr),
            freqdiff_plot=np.array(fd_arr),
            algorithm=algo,
            layout="v2",
            metrics_config=config.metrics
        )
        out_path = output_dir / f"fbid_plot_{algo.replace(' ', '_')}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[OK] Saved FBID plot for {algo} to {out_path}")

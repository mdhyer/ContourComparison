"""
FBID Accuracy Pipeline (DTW Nearest Neighbor)
Replicates logic from legacy_run_dtw_test_NN.py
"""
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dtw import dtw

from audio_analysis.utils.contour_utils import (
    load_precomputed, 
    run_ground, 
    fragment_contours, 
    identify_harmonics
)

from audio_analysis.config import PipelineConfig

# DTW Arguments matching legacy script
DTW_ARGS = dict(
    keep_internals=True,
    window_type='sakoechiba',
    window_args={'window_size': 10}
)

LABELS = {
    'Silbido Profundo': 'Silbido Profundo',
    'SAM': 'SAM-whistle',
    'CREPE': 'CREPE-tt',
    'SMC-PHD': 'SMC-PHD',
    'Ground': 'Ground'
}

def run_fbid_accuracy_pipeline(
    config: PipelineConfig,
    algorithms: List[str],
    n_ground: int = 10,
    n_comparisons: int = 10,
    seed: int = 100,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Runs the FBID accuracy pipeline.
    Returns metrics and plot figures.
    """
    random.seed(seed)

    data_root = config.paths.data_root
    precompute_dir = config.paths.precompute_dir
    params_dir = config.paths.params_dir

    # 1. Discover FBIDs and WAVs
    fb_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    fbids = sorted([d.name for d in fb_dirs])
    
    ground_wavs = {}
    comp_wavs = {}
    
    for fb in fbids:
        wav_path = data_root / fb
        wavs = sorted(glob.glob(str(wav_path / "*.wav")))
        random.seed(100)
        random.shuffle(wavs)
        ground_wavs[fb] = wavs[:n_ground]
        comp_wavs[fb] = wavs[n_ground:n_ground + n_comparisons]

    results = {}
    acc_plot_data = {} # For plotting: algo -> list of per-fbid accuracies
    
    # Calculate total tasks for progress tracking
    # Total = Algorithms * FBIDs * Comparisons per FBID
    total_tasks = len(algorithms) * len(fbids) * n_comparisons
    current_task = 0
    
    # 2. Run NN Test
    for algo in algorithms:
        results[algo] = {}
        acc_plot_data[algo] = []
        
        for fb in fbids:
            comp_wavs_list = comp_wavs[fb]
            results[algo][fb] = []
            
            for comp_wav in comp_wavs_list:
                # Load Comparison Contour
                if algo == 'Ground':
                    contour, _ = run_ground(comp_wav, params_dir=str(params_dir))
                else:
                    # Fixed: use 'top_dir' instead of 'src' to match contour_utils.py signature
                    contour = load_precomputed(comp_wav, ALGORITHM=algo, top_dir=str(precompute_dir))

                # Post-process
                contour = fragment_contours(contour)
                if algo != 'Ground':
                    contour = identify_harmonics(contour)

                if len(contour) == 0:
                    results[algo][fb].append([0, 0, 0])
                    current_task += 1
                    if progress_callback:
                        progress_callback(current_task, total_tasks, f"Skipping empty contour: {Path(comp_wav).name}")
                    continue

                dtw_scores = []

                # Compare against ALL ground truth whistles from ALL FBIDs
                for fb_ground in fbids:
                    ground_wavs_list = ground_wavs[fb_ground]
                    for ground_wav in ground_wavs_list:
                        # Load Ground Contour
                        contour_comp, _ = run_ground(ground_wav, params_dir=str(params_dir))

                        dtw_distances = []

                        # Pairwise DTW
                        for cont in contour:
                            for cont_comp in contour_comp:
                                # Safely convert to 2D float64 arrays
                                try:
                                    cont_arr = np.asarray(cont, dtype=np.float64)
                                    cont_comp_arr = np.asarray(cont_comp, dtype=np.float64)
                                except (ValueError, TypeError):
                                    continue
                                    
                                if cont_arr.ndim != 2 or cont_comp_arr.ndim != 2:
                                    continue

                                # Legacy Interpolation Logic
                                # Explicitly flatten to 1D to prevent "object too deep" errors
                                xp1 = cont_comp_arr[:,0].ravel()
                                fp1 = cont_comp_arr[:,1].ravel()
                                xp2 = cont_arr[:,0].ravel()
                                fp2 = cont_arr[:,1].ravel()
                                
                                # Skip invalid contours (too short or non-increasing time axis)
                                if len(xp1) < 2 or len(xp2) < 2 or xp1[-1] <= xp1[0] or xp2[-1] <= xp2[0]:
                                    continue
                                    
                                x1 = np.linspace(xp1[0], xp1[-1], 100)
                                x2 = np.linspace(xp2[0], xp2[-1], 100)

                                try:
                                    c1 = np.interp(x1, xp1, fp1)
                                    c2 = np.interp(x2, xp2, fp2)
                                except ValueError:
                                    continue

                                # Normalize by median
                                c1 = (c1 - np.median(c1))
                                c2 = (c2 - np.median(c2))

                                alignment = dtw(c1.astype(np.float32), c2.astype(np.float32), **DTW_ARGS)
                                dtw_distances.append(alignment.distance)

                        if len(dtw_distances) > 0:
                            this_dist = np.nanmin(dtw_distances)
                            dtw_scores.append([fb_ground, this_dist])

                # Determine Top-1, Top-5, Top-20
                if not dtw_scores:
                    results[algo][fb].append([0, 0, 0])
                else:
                    sorter = np.argsort([d for i, d in dtw_scores])
                    dtw_sort = np.array(dtw_scores, dtype=object)[sorter, :]

                    top1 = 1 if fb == dtw_sort[0, 0] else 0
                    top5 = 1 if fb in dtw_sort[:5, 0] else 0
                    top20 = 1 if fb in dtw_sort[:20, 0] else 0

                    results[algo][fb].append([top1, top5, top20])

                # Update Progress
                current_task += 1
                if progress_callback:
                    progress_callback(current_task, total_tasks, f"Processed {algo} | {fb} | {Path(comp_wav).name}")

    # 3. Aggregate Metrics
    metrics = {}
    for algo in algorithms:
        top1_list = []
        top5_list = []
        top20_list = []
        fb_metrics = {}

        for fb in fbids:
            fb_results = results[algo][fb]
            if not fb_results:  # FIX: Safely skip FBIDs with no comparison WAVs
                continue
                
            res = np.transpose(fb_results)
            top1_list.extend(res[0, :])
            top5_list.extend(res[1, :])
            top20_list.extend(res[2, :])

            # Per-FBID Accuracy (Top-1)
            fb_count = len(res[0, :])
            fb_sum = np.sum(res[0, :])
            acc = fb_sum / fb_count if fb_count > 0 else 0
            fb_metrics[fb] = acc
            acc_plot_data[algo].append(acc)

        metrics[algo] = {
            'top1': np.sum(top1_list) / len(top1_list) if top1_list else 0,
            'top5': np.sum(top5_list) / len(top5_list) if top5_list else 0,
            'top20': np.sum(top20_list) / len(top20_list) if top20_list else 0,
            'fb_metrics': fb_metrics
        }

    return {
        'results': results,
        'metrics': metrics,
        'acc_plot_data': acc_plot_data,
        'fbids': fbids,
        'algorithms': algorithms
    }

def plot_violin(data: Dict[str, List[float]], algorithms: List[str]) -> plt.Figure:
    """Replicates legacy violin plot."""
    fig, ax = plt.subplots(figsize=(4, 4.5))

    # Prepare data
    plot_data = [data[algo] for algo in algorithms]

    parts = ax.violinplot(plot_data, np.arange(len(algorithms)), showmedians=True, showextrema=True)

    ax.set_xticks(np.arange(len(algorithms)), labels=[LABELS.get(a, a) for a in algorithms], rotation=45)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('FBID Accuracy')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legacy color logic: Ground is black, others are viridis
    n_algos = len(algorithms)
    if n_algos > 1:
        colors = plt.cm.viridis(np.linspace(0., 1., n_algos - 1))
        colors = np.append(np.array([[0, 0, 0, 0.3]]), colors, axis=0)
    else:
        colors = np.array([[0, 0, 0, 0.3]])

    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)

    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmedians'].set_linestyle(':')
    if 'cbars' in parts:
        parts['cbars'].set_visible(False)
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

def plot_boxplot(data: Dict[str, List[float]], algorithms: List[str]) -> plt.Figure:
    """Replicates legacy box plot."""
    fig, ax = plt.subplots(figsize=(4, 4.5))

    plot_data = [data[algo] for algo in algorithms]

    # Matplotlib 3.9+ removed the `labels` kwarg; set ticks/labels explicitly
    box = ax.boxplot(plot_data, notch=True, showfliers=False,
                     positions=np.arange(len(algorithms)), patch_artist=True)

    # Legacy color logic: Ground is black, others are viridis
    n_algos = len(algorithms)
    if n_algos > 1:
        colors = plt.cm.viridis(np.linspace(0., 1., n_algos - 1))
        colors = np.append(np.array([[0, 0, 0, 0.3]]), colors, axis=0)
    else:
        colors = np.array([[0, 0, 0, 0.3]])

    for ai, (algo, patch, bmed) in enumerate(zip(algorithms, box['boxes'], box['medians'])):
        jittered = np.random.uniform(low=-.1, high=.1, size=len(data[algo]))
        jittered2 = np.random.uniform(low=-.025, high=.025, size=len(data[algo]))

        ax.scatter(jittered + ai, np.asarray(data[algo]) + jittered2, color='grey', s=1, zorder=10, alpha=.5)
        patch.set_facecolor(colors[ai])
        patch.set_alpha(.75)
        plt.setp(bmed, color='k')

    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_xticklabels([LABELS.get(a, a) for a in algorithms], rotation=45)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('FBID Accuracy')

    fig.tight_layout()
    return fig

"""
Command-line interface for the Audio Analysis Pipeline.
"""
from __future__ import annotations
import argparse
import logging
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .config import PipelineConfig, PathsConfig, CoverageMode, FreqDiffMode, FragMode, RecallMode, DataLayout
from .pipeline import _discover_wav_files, execute_precompute, execute_evaluate, execute_plot, plot_metrics_per_fbid, save_metrics
from .evaluation.dtw_fbid_accuracy import run_fbid_accuracy_pipeline, plot_violin, plot_boxplot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def _parse_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument("--project-root", type=str, help="Root directory of the project (defaults to current directory)")
    parser.add_argument("--data-root", type=str, help="Root directory for audio data")
    parser.add_argument("--output-dir", type=str, help="Directory for saving results")
    parser.add_argument("--params-dir", type=str, help="Directory for parameter files")
    parser.add_argument("--precompute-dir", type=str, help="Directory for precomputed contours")
    parser.add_argument("--model-dir", type=str, help="Directory for models")
    parser.add_argument("--fbids", nargs="+", type=str, help="List of FBIDs to process (default: all found)")
    parser.add_argument("--snr-levels", nargs="+", type=str, help="List of SNR levels to process (default: all found)")
    parser.add_argument("--algorithms", nargs="+", type=str, default=["SAM", "Silbido Profundo", "SMC-PHD", "CREPE"],
                        help="Algorithms to run (default: Silbido Profundo SAM SMC-PHD CREPE)")
    parser.add_argument("--sample-rate", type=int, help="Target sample rate for processing")
    parser.add_argument("--filter-band", type=int, nargs=2, help="Filter band (min, max)")
    parser.add_argument("--data-layout", type=str, choices=[l.value for l in DataLayout],
                        default="flat", help="Directory structure: 'flat' or 'nested'")

    # Metrics configuration arguments
    parser.add_argument("--coverage-mode", type=str, choices=[m.value for m in CoverageMode],
                        help="Coverage metric mode (total, ffc, per_loop)")
    parser.add_argument("--freq-diff-mode", type=str, choices=[m.value for m in FreqDiffMode],
                        help="Frequency difference metric mode (total, ffc)")
    parser.add_argument("--frag-mode", type=str, choices=[m.value for m in FragMode],
                        help="Fragmentation metric mode (total, ffc, per_loop)")
    parser.add_argument("--recall-mode", type=str, choices=[m.value for m in RecallMode],
                        help="Recall metric mode (total, ffc)")
    parser.add_argument("--freq-diff-threshold", type=float, help="Threshold for FFC calculations (default: 0.25)")
    parser.add_argument("--subsample", type=int, default=None, help="Process 1 out of every N files (e.g., 20)")

    # Contour processing controls
    parser.add_argument("--harmonic-tolerance", type=float, help="Tolerance ratio for harmonic removal (default: 0.5)")
    parser.add_argument("--harmonic-freqdiff", type=float, help="Relative frequency diff threshold for harmonics (default: 0.2)")
    parser.add_argument("--contour-dur-threshold", type=float, help="Time gap threshold for contour splitting (default: 0.05)")
    parser.add_argument("--no-harmonic-removal", action="store_true", help="Disable harmonic removal")
    parser.add_argument("--no-contour-splitting", action="store_true", help="Disable contour splitting")

    # Algorithm paths/names
    parser.add_argument("--matlab-silbido-path", type=str, help="Path to MATLAB Silbido script")
    parser.add_argument("--matlab-smcphd-path", type=str, help="Path to MATLAB SMC-PHD script")
    parser.add_argument("--crepe-model-name", type=str, help="Name of CREPE model file")
    parser.add_argument("--crepe-model-dir", type=str, help="Directory containing CREPE model")

    # Runtime flags
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing precomputed contours")
    parser.add_argument("--create-dirs", action="store_true", help="Automatically create missing data/output directories")

    # SNR Filtering
    parser.add_argument("--filter-snr-20db", action="store_true", help="Filter evaluation to only include files with native SNR > 20dB")
    parser.add_argument("--slsnr-filter-path", type=str, help="Path to JSON file containing INDs for SNR > 20dB filter")

    # Config serialization
    parser.add_argument("--export-config", type=str, help="Export current configuration to a JSON file")
    parser.add_argument("--load-config", type=str, help="Load configuration from a JSON file")

    # SNR Plot Range
    parser.add_argument("--snr-plot-min", type=float, help="Minimum SNR for spectrogram insets")
    parser.add_argument("--snr-plot-max", type=float, help="Maximum SNR for spectrogram insets")

    # Metrics I/O
    parser.add_argument("--load-metrics", type=str, help="Path to precomputed metrics JSON (skips evaluation & WAV discovery)")
    parser.add_argument("--save-metrics", nargs="?", const="default", default=None, type=str,
                        help="Save aggregated metrics to JSON. Defaults to output_dir/evaluation_metrics.json if no path given.")

def main():
    parser = argparse.ArgumentParser(
        description="Audio Analysis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audio-analysis precompute --data-root ./data --algorithms "Silbido Profundo" SAM SMC-PHD CREPE
  audio-analysis evaluate --data-root ./data --output-dir ./results --fbids FBID_001 FBID_002 --snr-levels CLEAN SNR_20
  audio-analysis plot --plot-snr-trends --plot-fbids
  audio-analysis plot --plot-mosaic-v2 --target-wav ./data/sample.wav
  audio-analysis plot --visualize-together --target-wav ./data/sample.wav
  audio-analysis plot --plot-metric-violin "False Positives"
  audio-analysis plot --load-metrics ./results/evaluation_metrics.json --plot-snr-trends
        """
    )

    # Add common arguments to the main parser so they can be used without a subcommand
    _parse_common_args(parser)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Precompute command
    precompute_parser = subparsers.add_parser("precompute", help="Precompute contours for specified algorithms")
    _parse_common_args(precompute_parser)

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Run contour comparison and metric aggregation")
    _parse_common_args(evaluate_parser)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate visualization plots")
    _parse_common_args(plot_parser)
    plot_parser.add_argument("--plot-snr-trends", action="store_true", help="Generate SNR trends plot")
    plot_parser.add_argument("--plot-fbids", action="store_true", help="Generate per-FBID metric plots")
    plot_parser.add_argument("--target-wav", type=str, help="Target WAV file for specific plots (e.g., mosaic v2)")
    plot_parser.add_argument("--plot-mosaic-v2", action="store_true", help="Generate Metrics Mosaic v2 plot")
    plot_parser.add_argument("--visualize-together", action="store_true", help="Generate Spectrogram Overlay (Together) plot")
    plot_parser.add_argument("--plot-metric-violin", nargs="?", const="Coverage", default=None,
                             choices=["Coverage", "False Positives", "Frequency Difference", "Fragmentation"],
                             help="Generate legacy-styled metric violin plot. Optionally specify metric (default: Coverage)")
    plot_parser.add_argument("--plot-preprocessing", action="store_true", help="Generate preprocessing plot (Silbido Profundo vs CREPE).")

    # DTW FBID Accuracy command
    dtw_fbid_parser = subparsers.add_parser("dtw-fbid", help="Run FBID Accuracy (DTW NN) pipeline")
    _parse_common_args(dtw_fbid_parser)
    dtw_fbid_parser.add_argument("--n-ground", type=int, default=10, help="Number of ground truth whistles per FBID")
    dtw_fbid_parser.add_argument("--n-comparisons", type=int, default=10, help="Number of comparison whistles per FBID")
    dtw_fbid_parser.add_argument("--plot-type", type=str, choices=["violin", "box", "both"], default="both")

    args = parser.parse_args()

    # Load base config or create default
    if args.load_config:
        from .utils.config_io import load_config
        config = load_config(Path(args.load_config))
    else:
        paths_kwargs = {}
        if args.project_root:
            paths_kwargs["project_root"] = Path(args.project_root)
        if args.data_root:
            paths_kwargs["data_root"] = Path(args.data_root)
        if args.output_dir:
            paths_kwargs["output_dir"] = Path(args.output_dir)
        if args.params_dir:
            paths_kwargs["params_dir"] = Path(args.params_dir)
        if args.precompute_dir:
            paths_kwargs["precompute_dir"] = Path(args.precompute_dir)
        if args.model_dir:
            paths_kwargs["model_dir"] = Path(args.model_dir)
        if args.data_layout:
            paths_kwargs["data_layout"] = DataLayout(args.data_layout)
        config = PipelineConfig(paths=PathsConfig(**paths_kwargs))

    # Apply CLI overrides (only if explicitly provided)
    if args.project_root: config.paths.project_root = Path(args.project_root)
    if args.data_root: config.paths.data_root = Path(args.data_root)
    if args.output_dir: config.paths.output_dir = Path(args.output_dir)
    if args.params_dir: config.paths.params_dir = Path(args.params_dir)
    if args.precompute_dir: config.paths.precompute_dir = Path(args.precompute_dir)
    if args.model_dir: config.paths.model_dir = Path(args.model_dir)
    if args.data_layout: config.paths.data_layout = DataLayout(args.data_layout)
    if args.create_dirs: config.paths.create_dirs = True

    if args.sample_rate:
        config.audio.sample_rate = args.sample_rate
    if args.filter_band:
        config.audio.filter_band = tuple(args.filter_band)

    # Override metrics config
    if args.coverage_mode:
        config.metrics.coverage_mode = CoverageMode(args.coverage_mode)
    if args.freq_diff_mode:
        config.metrics.freq_diff_mode = FreqDiffMode(args.freq_diff_mode)
    if args.frag_mode:
        config.metrics.frag_mode = FragMode(args.frag_mode)
    if args.recall_mode:
        config.metrics.recall_mode = RecallMode(args.recall_mode)
    if args.freq_diff_threshold is not None:
        config.metrics.freq_diff_threshold = args.freq_diff_threshold

    # Contour processing overrides
    if args.harmonic_tolerance is not None:
        config.algorithm.harmonic_tolerance = args.harmonic_tolerance
    if args.harmonic_freqdiff is not None:
        config.algorithm.harmonic_freqdiff = args.harmonic_freqdiff
    if args.contour_dur_threshold is not None:
        config.algorithm.contour_dur_threshold = args.contour_dur_threshold
    if args.no_harmonic_removal:
        config.algorithm.remove_harmonics = False
    if args.no_contour_splitting:
        config.algorithm.split_contours = False

    # Algorithm paths/names overrides
    if args.matlab_silbido_path:
        config.algorithm.matlab_silbido_path = args.matlab_silbido_path
    if args.matlab_smcphd_path:
        config.algorithm.matlab_smcphd_path = args.matlab_smcphd_path
    if args.crepe_model_name:
        config.algorithm.crepe_model_name = args.crepe_model_name
    if args.crepe_model_dir:
        config.algorithm.crepe_model_dir = Path(args.crepe_model_dir)

    # SNR Filtering overrides
    if args.filter_snr_20db:
        config.filter_snr_20db = True
    if args.slsnr_filter_path:
        config.paths.slsnr_20_filter_path = Path(args.slsnr_filter_path)

    print("Pipeline initialized with configuration:")
    print(f"  Project Root: {config.paths.project_root}")
    print(f"  Data Root: {config.paths.data_root}")
    print(f"  Output Dir: {config.paths.output_dir}")
    print(f"  Precompute Dir: {config.paths.precompute_dir}")
    print(f"  Params Dir: {config.paths.params_dir}")
    print(f"  Model Dir: {config.paths.model_dir}")
    print(f"  Create Dirs: {config.paths.create_dirs}")
    print(f"  Sample Rate: {config.audio.sample_rate}")
    print(f"  Filter Band: {config.audio.filter_band}")
    print(f"  FBIDs: {args.fbids or 'All'}")
    print(f"  SNR Levels: {args.snr_levels or 'All'}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Data Layout: {config.paths.data_layout.value}")
    print(f"  Coverage Mode: {config.metrics.coverage_mode.value}")
    print(f"  Freq Diff Mode: {config.metrics.freq_diff_mode.value}")
    print(f"  Frag Mode: {config.metrics.frag_mode.value}")
    print(f"  Recall Mode: {config.metrics.recall_mode.value}")
    print(f"  Freq Diff Threshold: {config.metrics.freq_diff_threshold}")
    print(f"  Remove Harmonics: {config.algorithm.remove_harmonics}")
    print(f"  Harmonic Tolerance: {config.algorithm.harmonic_tolerance}")
    print(f"  Harmonic FreqDiff: {config.algorithm.harmonic_freqdiff}")
    print(f"  Split Contours: {config.algorithm.split_contours}")
    print(f"  Contour Dur Threshold: {config.algorithm.contour_dur_threshold}")
    print(f"  Filter SNR > 20dB: {config.filter_snr_20db}")
    print(f"  SLSNR Filter Path: {config.paths.slsnr_20_filter_path}")
    if args.subsample:
        print(f"  Subsample: 1/{args.subsample}")

    # Export if requested
    if args.export_config:
        from .utils.config_io import save_config
        save_config(config, Path(args.export_config))
        logger.info(f"✅ Configuration exported to {args.export_config}")
        if not args.command:
            return

    aggregated = None
    wav_files = []

    # 1. Handle Precomputed Metrics Loading
    if args.load_metrics:
        try:
            with open(args.load_metrics, 'r') as f:
                data = json.load(f)
            aggregated = data.get("metrics", data)
            logger.info(f"✅ Loaded precomputed metrics from {args.load_metrics}")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return

    # 2. Handle WAV Discovery (Only if we aren't just loading metrics for plotting)
    if not args.load_metrics or args.command == "precompute":
        wav_files = _discover_wav_files(
            config.paths.data_root,
            args.fbids,
            args.snr_levels,
            config.paths.data_layout
        )

        # Apply subsampling if requested
        if args.subsample:
            wav_files = sorted(wav_files)[::args.subsample]

        if not wav_files and args.command in ["precompute", "evaluate"]:
            logger.warning("No WAV files found matching the provided FBIDs/SNR levels. Exiting.")
            return

    # 3. Execute Commands
    if args.command == "precompute":
        execute_precompute(config, wav_files, args.algorithms, overwrite=args.overwrite)

    elif args.command == "evaluate":
        if not args.load_metrics:
            aggregated = execute_evaluate(config, wav_files, args.algorithms)
        else:
            logger.info("Skipping evaluation (--load-metrics provided).")

    elif args.command == "plot":
        plot_snr = args.plot_snr_trends
        plot_fbid = args.plot_fbids
        plot_mosaic_v2 = args.plot_mosaic_v2
        plot_visualize_together = args.visualize_together
        plot_metric_violin = args.plot_metric_violin
        plot_preprocessing = args.plot_preprocessing

        # Fallback to SNR trends if no plot flags are provided
        if not (plot_snr or plot_fbid or plot_mosaic_v2 or plot_visualize_together or plot_metric_violin or plot_preprocessing):
            logger.warning("No plot type specified. Defaulting to SNR trends. Use --plot-snr-trends, --plot-fbids, --plot-mosaic-v2, --visualize-together, or --plot-preprocessing.")
            plot_snr = True

        # Only run expensive evaluation if metric-based plots are requested and metrics aren't already loaded
        needs_evaluation = (plot_snr or plot_fbid or plot_metric_violin) and aggregated is None
        if needs_evaluation:
            aggregated = execute_evaluate(config, wav_files, args.algorithms)

        if aggregated:
            if plot_snr:
                execute_plot(config, aggregated, args.algorithms, wav_files=wav_files, target_wav=args.target_wav,
                             snr_plot_min=args.snr_plot_min, snr_plot_max=args.snr_plot_max)

            # Unified FBID Plotting Logic (Identical to GUI)
            if plot_fbid:
                from audio_analysis.plotting.plotting import plot_fbid_trends

                fig, axs = plot_fbid_trends(aggregated, args.algorithms, limits=None, metrics_config=config.metrics)

                if fig:
                    out_path = config.paths.output_dir / "fbid_trends.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_path, dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"✅ Saved Unified FBID Trends plot to {out_path}")
                else:
                    logger.warning("Could not generate FBID plot (no valid data).")

            # Metric Violin Plotting Logic
            if plot_metric_violin:
                from audio_analysis.plotting.plotting import plot_metric_violin as violin_plot
                fig = violin_plot(aggregated, args.algorithms, plot_metric_violin, config.metrics)
                out_path = config.paths.output_dir / "metric_violin.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=600, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved Metric Violin plot to {out_path}")

        # Mosaic v2 runs independently without requiring evaluation
        if plot_mosaic_v2:
            target_wav = args.target_wav or (str(wav_files[0]) if wav_files else None)
            if not target_wav:
                logger.error("No target WAV specified or found for mosaic v2 plot.")
                return

            from audio_analysis.plotting.plotting import plot_metrics_mosaic_v2
            fig = plot_metrics_mosaic_v2(
                wav=target_wav,
                params_dir=str(config.paths.params_dir),
                algorithm=args.algorithms[0] if args.algorithms else "Silbido Profundo",
                precompute_dir=str(config.paths.precompute_dir),
                pipeline_config=config,
                metrics_config=config.metrics
            )

            out_path = config.paths.output_dir / "metrics_mosaic_v2.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"✅ Saved Metrics Mosaic v2 plot to {out_path}")

        # Visualize Together runs independently without requiring evaluation
        if plot_visualize_together:
            target_wav = args.target_wav or (str(wav_files[0]) if wav_files else None)
            if not target_wav:
                logger.error("No target WAV specified or found for visualize-together plot.")
                return

            from audio_analysis.plotting.plotting import visualize_together
            fig = visualize_together(
                wav=target_wav,
                params_dir=str(config.paths.params_dir),
                algorithms=args.algorithms,
                wav_src=str(config.paths.data_root),
                use_precomputed=True,
                load_src=str(config.paths.precompute_dir),
                pipeline_config=config,
                metrics_config=config.metrics
            )

            out_path = config.paths.output_dir / "visualize_together.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"✅ Saved Visualize Together plot to {out_path}")

        # Preprocessing plot runs independently without requiring evaluation
        if plot_preprocessing:
            target_wav = args.target_wav or (str(wav_files[0]) if wav_files else None)
            if not target_wav:
                logger.error("No target WAV specified or found for preprocessing plot.")
                return

            from audio_analysis.plotting.plotting import plot_preprocessing
            fig = plot_preprocessing(
                wav=target_wav,
                use_precomputed=True,
                pipeline_config=config,
                limits=None
            )

            out_path = config.paths.output_dir / "preprocessing_plot.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"✅ Saved Preprocessing plot to {out_path}")

    elif args.command == "dtw-fbid":

        logger.info(f"Starting DTW FBID Accuracy pipeline...")
        logger.info(f"Algorithms: {args.algorithms}")

        # Always include Ground truth for DTW baseline, placed first
        dtw_algorithms = list(args.algorithms)
        if "Ground" not in dtw_algorithms:
            dtw_algorithms.insert(0, "Ground")

        # Run Pipeline
        pipeline_data = run_fbid_accuracy_pipeline(
            config=config,
            algorithms=dtw_algorithms,
            n_ground=args.n_ground,
            n_comparisons=args.n_comparisons
        )

        # Print Metrics
        for algo, m in pipeline_data['metrics'].items():
            logger.info(f"--- {algo} ---")
            logger.info(f"Top-1: {m['top1']:.2%}, Top-5: {m['top5']:.2%}, Top-20: {m['top20']:.2%}")

        # Plotting
        out_dir = config.paths.output_dir if config.paths.output_dir else Path(args.output_dir) if args.output_dir else Path(".")

        if args.plot_type in ["violin", "both"]:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_violin(pipeline_data['acc_plot_data'], pipeline_data['algorithms'])
            fig.savefig(out_dir / "fbid_accuracy_violin.png", dpi=600)
            plt.close(fig)
            logger.info("Saved violin plot.")

        if args.plot_type in ["box", "both"]:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_boxplot(pipeline_data['acc_plot_data'], pipeline_data['algorithms'])
            fig.savefig(out_dir / "fbid_accuracy_boxplot.png", dpi=600)
            plt.close(fig)
            logger.info("Saved box plot.")

    else:
        logger.info("No command specified. Exiting after configuration print.")

    # 4. Explicit Save if requested
    if args.save_metrics is not None and aggregated is not None:
        if args.save_metrics == "default":
            save_path = config.paths.output_dir / "evaluation_metrics.json"
        else:
            save_path = Path(args.save_metrics)
        save_metrics(aggregated, save_path, config=config)
        logger.info(f"✅ Metrics explicitly saved to {save_path}")

if __name__ == "__main__":
    main()

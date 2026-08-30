"""
Precompute contours for all algorithms and save to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import PipelineConfig
from ..utils.contour_utils import save_precomputed
from .crepe import CrepePredictor
from .silbido import run_silbido, run_sam, run_smcphd

logger = logging.getLogger(__name__)


def run_precompute(
    wav_dir: str,
    algorithms: Optional[List[str]] = None,
    force: bool = False,
    pipeline_config: Optional[PipelineConfig] = None,
) -> None:
    """
    Precompute contours for all algorithms and save to disk.
    """
    cfg = pipeline_config or PipelineConfig()
    
    if algorithms is None:
        algorithms = ['Silbido Profundo', 'SAM', 'SMC-PHD', 'CREPE']

    wav_dir = Path(wav_dir)
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    wav_files = sorted(list(wav_dir.glob("*.wav")))
    if not wav_files:
        logger.warning(f"No .wav files found in {wav_dir}")
        return

    logger.info(f"Found {len(wav_files)} WAV files in {wav_dir}")

    crepe_predictor = None
    if "CREPE" in algorithms:
        try:
            crepe_predictor = CrepePredictor(model_dir=cfg.paths.model_dir)
        except Exception as e:
            logger.error(f"Failed to initialize CREPE predictor: {e}")
            raise

    for wav_file in wav_files:
        logger.info(f"Processing {wav_file.name}")
        try:
            for algo in algorithms:
                if algo == "CREPE":
                    if crepe_predictor is None:
                        logger.warning("CREPE predictor not initialized. Skipping.")
                        continue
                    contours = crepe_predictor.predict_crepe(str(wav_file), apply_mask=True)
                elif algo == "Silbido Profundo":
                    contours = run_silbido(str(wav_file), cfg)
                elif algo == "SAM":
                    contours = run_sam(str(wav_file), cfg)
                elif algo == "SMC-PHD":
                    contours = run_smcphd(str(wav_file), cfg)
                else:
                    logger.warning(f"Unknown algorithm: {algo}")
                    continue

                save_precomputed(
                    wav=str(wav_file),
                    contours=contours,
                    algorithm=algo,
                    src=wav_file.parent,
                    top_dir=str(cfg.paths.precompute_dir),
                )
                logger.info(f"Saved {algo} contours for {wav_file.name}")
        except Exception as e:
            logger.error(f"Failed to process {wav_file.name} with {algo}: {e}")
            continue

    logger.info("Precomputation complete.")

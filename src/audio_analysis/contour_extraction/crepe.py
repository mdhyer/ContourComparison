"""
CREPE pitch estimation wrapper.
"""
from __future__ import annotations

import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    import torch
    import torchcrepe
except ImportError:
    torch = None
    torchcrepe = None

try:
    import librosa
except ImportError:
    librosa = None

from ..config import PipelineConfig

class CrepePredictor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_dir: Optional[Path] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        batch_size: int = 2048,
        device: str = "cuda",
        downsample: int = 20,
        step: int = 2,
        threshold: float = 0.1,
        decoder: str = "argmax",
        interpolate: float = 0.005,
    ):
        """
        Initialize CREPE predictor.
        """
        if torch is None:
            raise ImportError(
                "CREPE requires PyTorch. Install with: pip install audio-analysis[models]"
            )

        # Fallback to CPU if CUDA is not available or not compiled
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available. Falling back to CPU for CREPE inference.")
            device = "cpu"
            
        self.device = device
        self.batch_size = batch_size
        self.tcrepe_model = torchcrepe.Crepe("full").eval().to(device)
        
        cfg = pipeline_config or PipelineConfig()
        resolved_model_name = model_name or cfg.algorithm.crepe_model_name
        resolved_model_dir = model_dir or cfg.algorithm.crepe_model_dir or cfg.paths.model_dir
        model_path = resolved_model_dir / resolved_model_name
        if not model_path.exists():
            raise FileNotFoundError(f"CREPE model not found at: {model_path}")
            
        self.tcrepe_model.load_state_dict(torch.load(str(model_path), map_location=device))
        self.step = step
        self.downsample = downsample
        self.decoder = torchcrepe.decode.__dict__[decoder]
        self.threshold = threshold
        self.interpolate = interpolate

    def predict_crepe(self, wavfile: str, apply_mask: bool = True) -> List[np.ndarray]:
        """
        Predict pitch contour from a WAV file using CREPE.
        Returns a list containing a single contour array of shape (N, 2) [time, f0].
        
        NOTE: Uses a sample-rate scaling trick to extend CREPE's ~2kHz range to ~40kHz.
        Audio is loaded at 16kHz * downsample, but preprocess is told it's 16kHz.
        This compresses frame duration by 20x, stretching the frequency axis accordingly.
        Output frequencies are then scaled by `downsample` to restore true Hz values.
        """
        if librosa is None:
            raise ImportError(
                "CREPE requires librosa. Install with: pip install audio-analysis[models]"
            )

        sig, _ = librosa.load(wavfile, sr=torchcrepe.SAMPLE_RATE * self.downsample)

        generator = torchcrepe.core.preprocess(
            torch.tensor(sig).unsqueeze(0),
            torchcrepe.SAMPLE_RATE,
            hop_length=int(self.step / self.downsample * torchcrepe.SAMPLE_RATE),
            batch_size=self.batch_size,
            device=self.device,
        )

        with torch.inference_mode():
            preds = torch.vstack([self.tcrepe_model(frames).cpu() for frames in generator]).T.unsqueeze(0)
            f0 = (torchcrepe.core.postprocess(preds, decoder=self.decoder) * self.downsample).squeeze()

        confidence = preds.max(axis=1)[0].squeeze()
        time = np.linspace(0, len(sig) / (self.downsample * torchcrepe.SAMPLE_RATE), len(f0))

        if apply_mask:
            mask = confidence > self.threshold
            time = time[mask]
            f0 = f0[mask]
            confidence = confidence[mask]

        # Optional interpolation (commented out in legacy script)
        # t0 = np.round(time[0] * (1/self.interpolate)) * self.interpolate
        # t1 = np.round(time[-1] * (1 / self.interpolate)) * self.interpolate
        # ix = np.arange(t0, t1, self.interpolate)
        # f0 = np.interp(ix, time, f0)
        # time = ix

        f0 = f0.detach().cpu().numpy()
        return [np.stack([time, f0], axis=1)]

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from enum import Enum

# Resolve package-relative paths safely without hardcoding absolute filesystem paths
_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_SLSNR_PATH = _PACKAGE_DIR / "utils" / "SLSNR_20.json"

class CoverageMode(str, Enum):
    TOTAL = "total"
    FFC = "ffc"
    TOTAL_PER_LOOP = "total_per_loop"
    FFC_PER_LOOP = "ffc_per_loop"

class FreqDiffMode(str, Enum):
    TOTAL = "total"
    FFC = "ffc"

class FragMode(str, Enum):
    TOTAL = "total"
    FFC = "ffc"
    TOTAL_PER_LOOP = "total_per_loop"
    FFC_PER_LOOP = "ffc_per_loop"

class RecallMode(str, Enum):
    TOTAL = "total"
    FFC = "ffc"

class DataLayout(str, Enum):
    FLAT = "flat"
    NESTED_NOISE = "nested"

class MetricsConfig(BaseModel):
    coverage_mode: CoverageMode = CoverageMode.TOTAL
    freq_diff_mode: FreqDiffMode = FreqDiffMode.TOTAL
    frag_mode: FragMode = FragMode.TOTAL
    recall_mode: RecallMode = RecallMode.TOTAL
    freq_diff_threshold: float = 0.25

class PathsConfig(BaseModel):
    """Configuration for data and output directories.
    
    Data/intermediate paths (precompute, noise_levels, params) 
    resolve relative to data_root. Output and model paths resolve relative to project_root.
    """
    project_root: Path = Path(".")
    data_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    precompute_dir: Optional[Path] = None
    noise_levels_dir: Optional[Path] = None
    params_dir: Optional[Path] = None
    model_dir: Optional[Path] = None
    data_layout: DataLayout = DataLayout.FLAT
    slsnr_20_filter_path: Optional[Path] = None
    create_dirs: bool = False

    @model_validator(mode="after")
    def resolve_paths(self) -> "PathsConfig":
        # Default data_root to project_root/data if not specified
        if self.data_root is None:
            self.data_root = self.project_root / "data"
            
        # Output and model directories remain independent of data_root
        if self.output_dir is None:
            self.output_dir = self.project_root / "results"
        if self.model_dir is None:
            self.model_dir = self.project_root / "models"
        if self.params_dir is None:
            self.params_dir = self.project_root / "Params"
        if self.precompute_dir is None:
            self.precompute_dir = self.project_root / "PrecomputeContours"
        if self.noise_levels_dir is None:
            self.noise_levels_dir = self.data_root / "NoiseLevels"
        if self.slsnr_20_filter_path is None:
            # Only set default if the file exists in the package
            if _DEFAULT_SLSNR_PATH.exists():
                self.slsnr_20_filter_path = _DEFAULT_SLSNR_PATH
            else:
                self.slsnr_20_filter_path = None
            
        return self

    def model_post_init(self, __context) -> None:
        # Directory creation is handled lazily at write sites (pipeline.py, cli.py).
        # This method is intentionally a no-op to avoid creating unused directories.
        pass

class AudioConfig(BaseModel):
    sample_rate: int = 48000
    filter_band: Tuple[int, int] = (2000, 22000)

class AlgorithmConfig(BaseModel):
    contour_dur_threshold: float = 0.025
    harmonic_tolerance: float = 0.7
    harmonic_freqdiff: float = 0.2
    remove_harmonics: bool = True
    split_contours: bool = True
    matlab_silbido_path: Optional[str] = None
    matlab_smcphd_path: Optional[str] = None
    
    # Silbido Profundo Parameters
    silbido_threshold1: float = 0.005
    silbido_threshold2: float = 0.4
    silbido_method: str = "DeepWhistle"
    
    crepe_model_name: str = "model_only-0_bottlenose_dolphins.pth"
    crepe_model_dir: Optional[Path] = None

class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    filter_snr_20db: bool = False

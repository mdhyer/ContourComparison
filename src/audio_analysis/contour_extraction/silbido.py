"""
MATLAB-based contour extraction wrappers for Silbido Profundo and SMC-PHD.
Handles lazy engine initialization, path resolution, data type conversion,
and robust error handling with graceful fallbacks.
"""
from __future__ import annotations
import logging
import io
from pathlib import Path
from typing import List, Any, Optional
import numpy as np

# Guard against missing matlab.engine package
try:
    import matlab
    import matlab.engine
    _MATLAB_ENGINE_AVAILABLE = True
except ImportError:
    matlab = None
    _MATLAB_ENGINE_AVAILABLE = False

from ..config import PipelineConfig

logger = logging.getLogger(__name__)

_matlab_eng = None
_matlab_initialized = False

def _is_matlab_available() -> bool:
    """Check if MATLAB engine can be started."""
    if not _MATLAB_ENGINE_AVAILABLE:
        logger.warning("matlab.engine package not installed. MATLAB algorithms will be skipped.")
        return False
    try:
        matlab.engine.find_matlab()
        return True
    except Exception:
        logger.warning("MATLAB runtime not found in system PATH. MATLAB algorithms will be skipped.")
        return False

def _get_matlab_engine(config: Optional[PipelineConfig] = None) -> Optional[Any]:
    """Initialize, validate, and return a shared MATLAB engine instance."""
    global _matlab_eng, _matlab_initialized
    
    # Reuse existing engine if healthy
    if _matlab_eng is not None:
        try:
            _matlab_eng.eval("1", nargout=0)
            return _matlab_eng
        except Exception:
            logger.warning("MATLAB engine crashed or disconnected. Attempting to restart...")
            _matlab_eng = None
            _matlab_initialized = False

    if not _is_matlab_available():
        return None

    try:
        _matlab_eng = matlab.engine.start_matlab("-batch")
        
        # Resolve paths from config or fallback to defaults
        cfg = config or PipelineConfig()
        silbido_base = Path(cfg.algorithm.matlab_silbido_path) if cfg.algorithm.matlab_silbido_path else Path.home() / "Documents" / "SilbidoProfundo"
        smcphd_base = Path(cfg.algorithm.matlab_smcphd_path) if cfg.algorithm.matlab_smcphd_path else Path.home() / "Documents" / "SMC-PHD"
        
        # Validate paths before adding to MATLAB
        if not silbido_base.exists():
            logger.warning(f"Silbido base path not found: {silbido_base}. MATLAB initialization may fail.")
        if not smcphd_base.exists():
            logger.warning(f"SMC-PHD base path not found: {smcphd_base}. MATLAB initialization may fail.")
            
        # Legacy initialization sequence wrapped in try/except for robustness
        try:
            _matlab_eng.addpath(str(silbido_base / "silbido-release3.0"), nargout=0)
            _matlab_eng.silbido_init(nargout=0)
            _matlab_eng.addpath(str(silbido_base), nargout=0)
            
            smcfold = _matlab_eng.genpath(str(smcphd_base))
            _matlab_eng.addpath(smcfold, nargout=0)
        except matlab.engine.MatlabExecutionError as e:
            logger.error(f"MATLAB script initialization failed: {e}")
            _matlab_eng = None
            return None
            
        _matlab_initialized = True
        logger.info("MATLAB engine initialized successfully.")
        return _matlab_eng
    except Exception as e:
        logger.error(f"Failed to initialize MATLAB engine: {e}")
        _matlab_eng = None
        return None

def _convert_matlab_to_numpy(data: Any) -> Any:
    """Recursively convert MATLAB types (matlab.double, cell arrays) to native NumPy/Python types."""
    if matlab is not None and hasattr(matlab, 'double'):
        try:
            if isinstance(data, matlab.double):
                return np.array(data)
        except (TypeError, AttributeError):
            pass

    if hasattr(data, '__iter__') and not isinstance(data, (str, bytes, np.ndarray)):
        return [_convert_matlab_to_numpy(item) for item in data]
    return data

def run_silbido(wav_path: str, config: PipelineConfig) -> List[np.ndarray]:
    """Run Silbido Profundo and return contours as a list of NumPy arrays."""
    eng = _get_matlab_engine(config)
    if eng is None:
        logger.warning(f"MATLAB engine unavailable. Skipping Silbido Profundo for {wav_path}")
        return []
        
    try:
        result = eng.silbido_process_file_allcontours(
            str(wav_path), 
            config.algorithm.silbido_threshold1, 
            config.algorithm.silbido_threshold2, 
            config.algorithm.silbido_method, 
            nargout=1, 
            stdout=io.StringIO()
        )
        return _convert_matlab_to_numpy(result)
    except matlab.engine.MatlabExecutionError as e:
        logger.error(f"MATLAB execution error for Silbido Profundo ({wav_path}): {e}")
        return []
    except Exception as e:
        logger.error(f"Silbido Profundo failed for {wav_path}: {e}")
        return []

def run_smcphd(wav_path: str, config: PipelineConfig) -> List[np.ndarray]:
    """Run SMC-PHD and return contours as a list of NumPy arrays."""
    eng = _get_matlab_engine(config)
    if eng is None:
        logger.warning(f"MATLAB engine unavailable. Skipping SMC-PHD for {wav_path}")
        return []
        
    try:
        result = eng.extractSMCPHD(str(wav_path), nargout=1, stdout=io.StringIO())
        return _convert_matlab_to_numpy(result)
    except matlab.engine.MatlabExecutionError as e:
        logger.error(f"MATLAB execution error for SMC-PHD ({wav_path}): {e}")
        return []
    except Exception as e:
        logger.error(f"SMC-PHD failed for {wav_path}: {e}")
        return []

def run_sam(wav_path: str, config: PipelineConfig) -> List[np.ndarray]:
    """Placeholder for SAM algorithm (currently bypassed in legacy code)."""
    logger.warning("SAM algorithm is currently bypassed.")
    return []

import pytest
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_analysis.contour_extraction import silbido
from src.audio_analysis.config import PipelineConfig, AlgorithmConfig

# Check if MATLAB is available
try:
    import matlab.engine
    _MATLAB_AVAILABLE = True
except ImportError:
    _MATLAB_AVAILABLE = False

@pytest.mark.skipif(not _MATLAB_AVAILABLE, reason="MATLAB engine not installed")
class TestSilbidoRealParity:
    @pytest.fixture
    def test_wav_path(self):
        """
        Points to a real WAV file for inference.
        Override via environment variable: SILBIDO_TEST_WAV=/path/to/test.wav
        """
        wav_path = os.environ.get("SILBIDO_TEST_WAV", "tests/data/sample.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
        return wav_path

    @pytest.fixture
    def reference_contour_path(self):
        """
        Points to a real precomputed contour file for comparison.
        Override via environment variable: SILBIDO_TEST_REF=/path/to/ref.npy
        """
        ref_path = os.environ.get("SILBIDO_TEST_REF", "tests/data/reference_silbido_output.npy")
        if not os.path.exists(ref_path):
            pytest.skip(f"Reference contour file not found: {ref_path}")
        return ref_path

    def test_real_matlab_execution_and_parity(self, test_wav_path, reference_contour_path):
        """
        Runs actual MATLAB Silbido Profundo inference and compares against precomputed contours.
        This test requires a valid MATLAB installation and test data files.
        """
        config = PipelineConfig(
            algorithm=AlgorithmConfig(
                silbido_threshold1=0.005,
                silbido_threshold2=0.4,
                silbido_method="DeepWhistle"
            )
        )

        # Run actual inference
        result = silbido.run_silbido(test_wav_path, config)
        
        # Load reference
        reference = np.load(reference_contour_path, allow_pickle=True)
        
        # Handle cases where reference might be a single array or list of arrays
        if isinstance(reference, list):
            reference = reference[0] if len(reference) > 0 else np.array([])
            
        assert len(result) > 0, "Silbido returned no contours for test WAV."
        
        # Compare first contour
        np.testing.assert_array_almost_equal(
            result[0], reference, 
            decimal=5, 
            err_msg="Real MATLAB output does not match precomputed reference contour."
        )

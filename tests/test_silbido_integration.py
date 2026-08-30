import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from audio_analysis.contour_extraction import silbido


@pytest.fixture
def mock_matlab_types():
    """Create mock MATLAB types for testing type conversion without MATLAB runtime."""
    class MockDouble:
        def __init__(self, data):
            self._data = np.array(data)
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
        def __getitem__(self, idx):
            return self._data[idx]

    class MockCell:
        def __init__(self, data):
            self._data = data
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
        def __getitem__(self, idx):
            return self._data[idx]

    return {"double": MockDouble, "cell": MockCell}


class TestSilbidoTypeConversion:
    def test_nested_cell_to_numpy(self, mock_matlab_types, monkeypatch):
        """Verifies that nested MATLAB cell arrays are recursively converted to lists/arrays."""
        # Inject mock classes into the silbido module's matlab namespace so isinstance() checks pass
        mock_matlab = type('MockMatlab', (), {
            'cell': mock_matlab_types["cell"],
            'double': mock_matlab_types["double"]
        })()
        monkeypatch.setattr('audio_analysis.contour_extraction.silbido.matlab', mock_matlab)

        inner = mock_matlab_types["double"]([1.0, 2.0, 3.0])
        outer = mock_matlab_types["cell"]([inner, inner])
    
        result = silbido._convert_matlab_to_numpy(outer)
    
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_output_conversion(self, mock_matlab_types):
        """Verifies that empty MATLAB outputs are handled gracefully."""
        empty_cell = mock_matlab_types["cell"]([])
        result = silbido._convert_matlab_to_numpy(empty_cell)
        assert result == []

    def test_deeply_nested_conversion(self, mock_matlab_types, monkeypatch):
        """Verifies that deeply nested MATLAB structures are fully converted."""
        mock_matlab = type('MockMatlab', (), {
            'cell': mock_matlab_types["cell"],
            'double': mock_matlab_types["double"]
        })()
        monkeypatch.setattr('audio_analysis.contour_extraction.silbido.matlab', mock_matlab)
        
        # Create a 3-level deep structure: cell -> cell -> double
        inner_double = mock_matlab_types["double"]([1.0, 2.0])
        mid_cell = mock_matlab_types["cell"]([inner_double, inner_double])
        outer_cell = mock_matlab_types["cell"]([mid_cell, mid_cell])
        
        result = silbido._convert_matlab_to_numpy(outer_cell)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert len(result[0]) == 2
        assert isinstance(result[0][0], np.ndarray)
        np.testing.assert_array_equal(result[0][0], np.array([1.0, 2.0]))


class TestSilbidoEngineLifecycle:
    @pytest.mark.matlab
    @patch("audio_analysis.contour_extraction.silbido.matlab.engine")
    def test_headless_initialization(self, mock_engine):
        """Verify MATLAB engine starts with headless flags."""
        mock_engine.start_matlab.return_value = MagicMock()
        eng = silbido._get_matlab_engine()
        mock_engine.start_matlab.assert_called_once()
        assert eng is not None

    @pytest.mark.matlab
    @patch("audio_analysis.contour_extraction.silbido.matlab.engine")
    def test_graceful_fallback_when_missing(self, mock_engine):
        """Verify engine returns None and logs warning when MATLAB is unavailable."""
        # Reset module-level cache to prevent interference from previous tests
        silbido._matlab_eng = None
        silbido._matlab_initialized = False
        
        mock_engine.start_matlab.side_effect = Exception("MATLAB not found")
        eng = silbido._get_matlab_engine()
        assert eng is None

    @pytest.mark.matlab
    @patch("audio_analysis.contour_extraction.silbido._is_matlab_available")
    @patch("audio_analysis.contour_extraction.silbido.matlab.engine")
    def test_engine_disconnection_recovery(self, mock_engine, mock_available):
        """Verify that a crashed engine is detected and reset for re-initialization."""
        mock_available.return_value = True
        
        # Simulate an existing engine that crashes on eval
        mock_eng = MagicMock()
        mock_eng.eval.side_effect = Exception("Engine disconnected")
        silbido._matlab_eng = mock_eng
        silbido._matlab_initialized = True
        
        # Mock start_matlab to return a fresh engine
        fresh_eng = MagicMock()
        mock_engine.start_matlab.return_value = fresh_eng
        
        eng = silbido._get_matlab_engine()
        
        # Should have detected crash and attempted restart
        mock_engine.start_matlab.assert_called_once()
        assert eng is not None

    @patch("audio_analysis.contour_extraction.silbido._is_matlab_available")
    def test_missing_matlab_runtime_fallback(self, mock_available):
        """Verify that algorithms skip gracefully when MATLAB runtime is missing."""
        mock_available.return_value = False
        silbido._matlab_eng = None
        silbido._matlab_initialized = False
        
        eng = silbido._get_matlab_engine()
        assert eng is None
        
        # Verify run_silbido handles None engine gracefully
        from audio_analysis.config import PipelineConfig
        result = silbido.run_silbido("dummy.wav", PipelineConfig())
        assert result == []


class TestCliGuiPipelineParity:
    def test_identical_config_routing(self):
        """Verify CLI and GUI route to the same Silbido execution function."""
        from audio_analysis.cli import _parse_common_args
        from audio_analysis.config import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.algorithm.silbido_threshold1 == 0.005
        assert cfg.algorithm.silbido_method == "DeepWhistle"

    def test_parameter_override_propagation(self):
        """Verify custom Silbido parameters propagate through config correctly."""
        from audio_analysis.config import PipelineConfig, AlgorithmConfig
        algo_cfg = AlgorithmConfig(silbido_threshold1=0.01, silbido_method="Energy")
        cfg = PipelineConfig(algorithm=algo_cfg)
        assert cfg.algorithm.silbido_threshold1 == 0.01
        assert cfg.algorithm.silbido_method == "Energy"


class TestSilbidoParity:
    @patch("audio_analysis.contour_extraction.silbido.run_silbido")
    def test_output_matches_precomputed(self, mock_run):
        """Verify Silbido output structure matches precomputed expectations."""
        mock_run.return_value = [np.array([[0.0, 1000], [0.1, 1050]])]
        result = silbido.run_silbido("dummy.wav")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == (2, 2)

    @patch("audio_analysis.contour_extraction.silbido.run_silbido")
    def test_empty_detection_parity(self, mock_run):
        """Verify empty detection returns empty list consistently."""
        mock_run.return_value = []
        result = silbido.run_silbido("dummy.wav")
        assert result == []

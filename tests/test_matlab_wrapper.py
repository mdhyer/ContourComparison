"""
Unit tests for MATLAB engine wrapper, type conversion, and Silbido configuration.
Runs entirely with mocks; requires no MATLAB installation.
"""
import pytest
import numpy as np
import io
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, ANY

from audio_analysis.config import PipelineConfig, AlgorithmConfig
from audio_analysis.contour_extraction import silbido


@pytest.fixture(autouse=True)
def reset_matlab_state():
    """Reset module-level engine state before/after each test."""
    silbido._matlab_eng = None
    silbido._matlab_initialized = False
    yield
    silbido._matlab_eng = None
    silbido._matlab_initialized = False


@pytest.fixture
def mock_config():
    """PipelineConfig with custom Silbido parameters for testing."""
    return PipelineConfig(algorithm=AlgorithmConfig(
        matlab_silbido_path="/fake/silbido",
        matlab_smcphd_path="/fake/smcphd",
        silbido_threshold1=0.01,
        silbido_threshold2=0.5,
        silbido_method="Energy"
    ))


class TestConvertMatlabToNumpy:
    """Tests for _convert_matlab_to_numpy() type conversion logic."""

    def test_matlab_double_conversion(self, mock_config):
        class MockMatlabDouble:
            pass
        mock_data = MockMatlabDouble()
        with patch.object(silbido, '_MATLAB_ENGINE_AVAILABLE', True), \
             patch.object(silbido, 'matlab', MagicMock(double=MockMatlabDouble)):
            result = silbido._convert_matlab_to_numpy(mock_data)
            assert isinstance(result, np.ndarray)

    def test_nested_iterable_conversion(self, mock_config):
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = silbido._convert_matlab_to_numpy(data)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_string_passthrough(self, mock_config):
        assert silbido._convert_matlab_to_numpy("test_string") == "test_string"

    @patch.object(silbido, '_MATLAB_ENGINE_AVAILABLE', False)
    def test_graceful_fallback_no_matlab(self, mock_config):
        data = [1, 2, 3]
        result = silbido._convert_matlab_to_numpy(data)
        assert result == [1, 2, 3]


class TestGetMatlabEngine:
    """Tests for lazy initialization, health checks, and path validation."""

    @pytest.mark.matlab
    @patch('audio_analysis.contour_extraction.silbido.matlab.engine')
    def test_lazy_initialization(self, mock_engine, mock_config):
        mock_engine.find_matlab.return_value = True
        mock_eng_instance = MagicMock()
        mock_engine.start_matlab.return_value = mock_eng_instance

        with patch.object(Path, 'exists', return_value=True):
            eng = silbido._get_matlab_engine(mock_config)
            assert eng is mock_eng_instance
            mock_engine.start_matlab.assert_called_once()
            mock_eng_instance.addpath.assert_called()
            mock_eng_instance.silbido_init.assert_called_once()

    @pytest.mark.matlab
    @patch('audio_analysis.contour_extraction.silbido.matlab.engine')
    def test_engine_health_check_restart(self, mock_engine, mock_config):
        mock_engine.find_matlab.return_value = True
        mock_eng1 = MagicMock()
        mock_eng1.eval.side_effect = Exception("Crashed")
        mock_eng2 = MagicMock()
        mock_engine.start_matlab.return_value = mock_eng2

        # Pre-set module state to trigger the health check path
        silbido._matlab_eng = mock_eng1

        with patch.object(Path, 'exists', return_value=True):
            eng = silbido._get_matlab_engine(mock_config)
            assert eng is mock_eng2
            assert mock_engine.start_matlab.call_count == 1

    @pytest.mark.matlab
    @patch('audio_analysis.contour_extraction.silbido.matlab.engine')
    def test_missing_matlab_runtime(self, mock_engine, mock_config):
        mock_engine.find_matlab.side_effect = Exception("Not found")
        eng = silbido._get_matlab_engine(mock_config)
        assert eng is None

    @pytest.mark.matlab
    @patch('audio_analysis.contour_extraction.silbido.matlab.engine')
    def test_path_validation_warnings(self, mock_engine, mock_config, caplog):
        caplog.set_level(logging.WARNING)
        mock_engine.find_matlab.return_value = True
        mock_eng = MagicMock()
        mock_engine.start_matlab.return_value = mock_eng

        with patch.object(Path, 'exists', return_value=False):
            _ = silbido._get_matlab_engine(mock_config)
            assert "Silbido base path not found" in caplog.text
            assert "SMC-PHD base path not found" in caplog.text


class TestRunSilbido:
    """Tests for run_silbido() parameter routing and error handling."""

    @patch('audio_analysis.contour_extraction.silbido._get_matlab_engine')
    def test_parameter_passing(self, mock_get_eng, mock_config):
        mock_eng = MagicMock()
        mock_get_eng.return_value = mock_eng
        mock_eng.silbido_process_file_allcontours.return_value = [[1, 2], [3, 4]]

        with patch.object(silbido, '_convert_matlab_to_numpy', side_effect=lambda x: x):
            result = silbido.run_silbido("/fake.wav", mock_config)
            mock_eng.silbido_process_file_allcontours.assert_called_once_with(
                "/fake.wav", 0.01, 0.5, "Energy", nargout=1, stdout=ANY
            )
            assert result == [[1, 2], [3, 4]]

    @pytest.mark.matlab
    @patch('audio_analysis.contour_extraction.silbido._get_matlab_engine')
    def test_matlab_execution_error_fallback(self, mock_get_eng, mock_config, caplog):
        caplog.set_level(logging.ERROR)
        mock_eng = MagicMock()
        mock_get_eng.return_value = mock_eng
        
        MockExecError = type("MatlabExecutionError", (Exception,), {})
        mock_eng.silbido_process_file_allcontours.side_effect = MockExecError("MATLAB Error")

        with patch.object(silbido.matlab.engine, 'MatlabExecutionError', MockExecError):
            result = silbido.run_silbido("/fake.wav", mock_config)
            assert result == []
            assert "MATLAB execution error" in caplog.text

    @patch('audio_analysis.contour_extraction.silbido._get_matlab_engine')
    def test_engine_unavailable_fallback(self, mock_get_eng, mock_config, caplog):
        caplog.set_level(logging.WARNING)
        mock_get_eng.return_value = None
        result = silbido.run_silbido("/fake.wav", mock_config)
        assert result == []
        assert "MATLAB engine unavailable" in caplog.text


class TestSilbidoConfig:
    """Tests for AlgorithmConfig Silbido fields."""

    def test_default_values(self):
        cfg = AlgorithmConfig()
        assert cfg.silbido_threshold1 == 0.005
        assert cfg.silbido_threshold2 == 0.4
        assert cfg.silbido_method == "DeepWhistle"

    def test_custom_values(self):
        cfg = AlgorithmConfig(
            silbido_threshold1=0.01,
            silbido_threshold2=0.6,
            silbido_method="Energy"
        )
        assert cfg.silbido_threshold1 == 0.01
        assert cfg.silbido_threshold2 == 0.6
        assert cfg.silbido_method == "Energy"

    def test_config_serialization_roundtrip(self):
        cfg = AlgorithmConfig(silbido_threshold1=0.02, silbido_method="SpectralFlux")
        json_str = cfg.model_dump_json()
        restored = AlgorithmConfig.model_validate_json(json_str)
        assert restored.silbido_threshold1 == 0.02
        assert restored.silbido_method == "SpectralFlux"

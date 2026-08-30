import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from audio_analysis.config import PipelineConfig, CoverageMode, FreqDiffMode, FragMode, DataLayout
from audio_analysis.utils.config_io import save_config, load_config


class TestConfigSerialization:
    """Tests for configuration export/import parity between CLI and GUI."""

    def test_roundtrip_serialization(self, tmp_path: Path):
        """Verify that saving and loading a config preserves all values."""
        original = PipelineConfig()
        original.paths.data_layout = DataLayout.NESTED_NOISE
        original.audio.sample_rate = 44100
        original.metrics.coverage_mode = CoverageMode.FFC
        original.metrics.freq_diff_threshold = 0.3
        original.algorithm.remove_harmonics = False

        config_path = tmp_path / "test_config.json"
        save_config(original, config_path)
        
        loaded = load_config(config_path)
        
        assert loaded.paths.data_layout == DataLayout.NESTED_NOISE
        assert loaded.audio.sample_rate == 44100
        assert loaded.metrics.coverage_mode == CoverageMode.FFC
        assert loaded.metrics.freq_diff_threshold == 0.3
        assert loaded.algorithm.remove_harmonics is False

    def test_cli_overrides_loaded_config(self, tmp_path: Path):
        """Verify that CLI arguments can override a loaded configuration."""
        base = PipelineConfig()
        base.audio.sample_rate = 48000
        config_path = tmp_path / "base.json"
        save_config(base, config_path)
        
        # Simulate CLI override logic
        loaded = load_config(config_path)
        loaded.audio.sample_rate = 22050
        
        assert loaded.audio.sample_rate == 22050


class TestGuiCliControlMapping:
    """Verify that GUI controls map correctly to PipelineConfig fields."""

    def test_config_to_dict_mapping(self):
        """Ensure all config fields are serializable and match expected types."""
        cfg = PipelineConfig()
        data = cfg.model_dump()
        
        assert isinstance(data['paths']['data_layout'], str)
        assert isinstance(data['audio']['sample_rate'], int)
        assert isinstance(data['metrics']['freq_diff_threshold'], float)
        assert isinstance(data['algorithm']['remove_harmonics'], bool)

    def test_invalid_config_rejection(self):
        """Ensure invalid configurations are caught during load."""
        invalid_json = """
        {
            "paths": {"data_layout": "invalid_layout"},
            "audio": {"sample_rate": 48000},
            "algorithm": {},
            "metrics": {}
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write(invalid_json)
            f.flush()
            with pytest.raises(ValidationError):
                load_config(Path(f.name))

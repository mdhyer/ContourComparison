"""
Parity tests ensuring CLI and GUI produce identical configurations and execution arguments.
Run with: pytest tests/test_cli_gui_parity.py
Run with existing config: pytest tests/test_cli_gui_parity.py --config-path ./path/to/config.json
"""
import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from audio_analysis.config import PipelineConfig
from audio_analysis.utils.config_io import save_config, load_config
from audio_analysis.cli import main as cli_main
from audio_analysis.gui.main_window import MainWindow
from audio_analysis.gui.workers import PipelineWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for all GUI tests."""
    if not QApplication.instance():
        return QApplication([])
    return QApplication.instance()

@pytest.fixture
def test_config(request):
    """Load user-provided config or generate a temporary default."""
    config_path = request.config.getoption("--config-path")
    if config_path:
        return load_config(Path(config_path))
    
    # Fallback: generate default config
    return PipelineConfig()

@pytest.fixture
def temp_config_file(test_config):
    """Serialize test_config to a temporary JSON file for CLI/GUI loading."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        save_config(test_config, Path(f.name))
        return Path(f.name)


class TestConfigurationParity:
    """Verify that CLI and GUI serialize/deserialize configurations identically."""

    def test_cli_export_import_roundtrip(self, temp_config_file):
        """CLI --export-config and --load-config must preserve all fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            out_path = Path(out_f.name)
            
        with patch("sys.argv", ["cli", "--load-config", str(temp_config_file), "--export-config", str(out_path)]), \
             patch("audio_analysis.cli._discover_wav_files", return_value=[]):
            cli_main()
            
        loaded = load_config(out_path)
        assert loaded.model_dump() == load_config(temp_config_file).model_dump()

    def test_gui_export_import_roundtrip(self, qapp, temp_config_file):
        """GUI Save/Load Config buttons must preserve all fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
            out_path = Path(out_f.name)
            
        win = MainWindow()
        win.config = load_config(temp_config_file)
        win._load_defaults()
        
        with patch("audio_analysis.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(out_path), "")):
            win._save_config()
            
        loaded = load_config(out_path)
        assert loaded.model_dump() == load_config(temp_config_file).model_dump()
        win.close()

    def test_cli_gui_config_equivalence(self, temp_config_file, qapp):
        """CLI and GUI must produce bitwise-identical JSON from the same source."""
        # CLI export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cli_f:
            cli_path = Path(cli_f.name)
        with patch("sys.argv", ["cli", "--load-config", str(temp_config_file), "--export-config", str(cli_path)]), \
             patch("audio_analysis.cli._discover_wav_files", return_value=[]):
            cli_main()
            
        # GUI export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as gui_f:
            gui_path = Path(gui_f.name)
        win = MainWindow()
        win.config = load_config(temp_config_file)
        win._load_defaults()
        with patch("audio_analysis.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(gui_path), "")):
            win._save_config()
        win.close()
            
        with open(cli_path) as f1, open(gui_path) as f2:
            assert json.load(f1) == json.load(f2), "CLI and GUI exported different JSON structures"


class TestExecutionParity:
    """Verify that execution workers receive identical configuration objects."""

    def test_worker_receives_validated_config(self, test_config):
        """PipelineWorker must accept and forward the exact PipelineConfig instance."""
        wav_files = [Path("/fake/audio.wav")]
        algos = ["CREPE", "Silbido Profundo"]
        
        worker = PipelineWorker(test_config, wav_files, algos, "evaluate", overwrite=False)
        
        # Assert worker holds the exact same config reference/data
        assert worker.config.model_dump() == test_config.model_dump()
        assert worker.wav_files == wav_files
        assert worker.algorithms == algos
        assert worker.command == "evaluate"
        assert worker.overwrite is False

    def test_cli_passes_config_to_pipeline(self, temp_config_file):
        """CLI must pass the loaded PipelineConfig to execute_evaluate/precompute."""
        with patch("audio_analysis.cli.execute_evaluate") as mock_eval, \
             patch("audio_analysis.cli._discover_wav_files", return_value=[Path("/fake.wav")]), \
             patch("sys.argv", ["cli", "evaluate", "--load-config", str(temp_config_file)]):
            cli_main()
            
        mock_eval.assert_called_once()
        passed_config = mock_eval.call_args[0][0]
        assert isinstance(passed_config, PipelineConfig)
        assert passed_config.model_dump() == load_config(temp_config_file).model_dump()

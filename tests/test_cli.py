import sys
import pytest
from unittest.mock import patch
from audio_analysis.cli import main

class TestCLI:
    def test_cli_invalid_metric_mode(self, capsys):
        """Verify CLI raises SystemExit for invalid metric mode."""
        with pytest.raises(SystemExit):
            sys.argv = ['audio-analysis', '--coverage-mode', 'invalid']
            main()
        captured = capsys.readouterr()
        assert 'invalid' in captured.err.lower() or 'invalid choice' in captured.err.lower()

    def test_cli_default_args(self, capsys):
        """Verify CLI initializes correctly with default arguments."""
        sys.argv = ['audio-analysis']
        main()
        captured = capsys.readouterr()
        assert 'Pipeline initialized' in captured.out
        assert 'Coverage Mode: total' in captured.out

    def test_cli_custom_paths(self, capsys):
        """Verify CLI accepts custom data and output paths."""
        sys.argv = ['audio-analysis', '--data-root', './custom_data', '--output-dir', './custom_out']
        main()
        captured = capsys.readouterr()
        # Path normalization strips the leading './', so we check for the base name
        assert 'custom_data' in captured.out
        assert 'custom_out' in captured.out

    @patch("audio_analysis.cli.execute_precompute")
    @patch("audio_analysis.cli._discover_wav_files")
    def test_cli_precompute_command(self, mock_discover, mock_precompute, capsys):
        """Verify CLI routes to execute_precompute correctly."""
        mock_discover.return_value = ["dummy.wav"]
        sys.argv = ['audio-analysis', 'precompute', '--data-root', './data', '--algorithms', 'CREPE']
        main()
        mock_precompute.assert_called_once()
        call_args = mock_precompute.call_args
        # Verify wav_files and algorithms are passed correctly
        assert call_args[0][1] == mock_discover.return_value
        assert call_args[0][2] == ['CREPE']

    @patch("audio_analysis.cli.execute_evaluate")
    @patch("audio_analysis.cli._discover_wav_files")
    def test_cli_evaluate_command(self, mock_discover, mock_evaluate, capsys):
        """Verify CLI routes to execute_evaluate correctly."""
        mock_discover.return_value = ["dummy.wav"]
        mock_evaluate.return_value = {}
        sys.argv = ['audio-analysis', 'evaluate', '--data-root', './data']
        main()
        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        assert call_args[0][1] == mock_discover.return_value

    @patch("audio_analysis.cli.execute_plot")
    @patch("audio_analysis.cli.execute_evaluate")
    @patch("audio_analysis.cli._discover_wav_files")
    def test_cli_plot_command(self, mock_discover, mock_evaluate, mock_plot, capsys):
        """Verify CLI routes to execute_plot correctly."""
        mock_discover.return_value = ["dummy.wav"]
        mock_evaluate.return_value = {"Silbido Profundo": {
            "global": {},
            "per_snr": {},
            "per_fbid": {}
        }}
        sys.argv = ['audio-analysis', 'plot', '--data-root', './data']
        main()
        mock_evaluate.assert_called_once()
        mock_plot.assert_called_once()
        # Verify execute_plot receives the aggregated metrics from execute_evaluate
        plot_call_args = mock_plot.call_args
        assert plot_call_args[0][1] == mock_evaluate.return_value

    @patch("audio_analysis.cli._discover_wav_files")
    def test_cli_no_wav_files_exits_gracefully(self, mock_discover, caplog):
        """Verify CLI exits gracefully when no WAV files are found."""
        mock_discover.return_value = []
        sys.argv = ['audio-analysis', 'evaluate', '--data-root', './empty_data']
        main()
        # pytest captures logging output in caplog, not capsys
        assert 'No WAV files found' in caplog.text

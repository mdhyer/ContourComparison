import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from audio_analysis.utils.contour_utils import (
    get_data_paths,
    fragment_contours,
    identify_harmonics,
    _find_matching_param,
    load_ground_truth,
    save_precomputed,
    load_precomputed,
    run_ground
)


class TestContourUtilsCoverage:
    """Tests specifically designed to increase coverage for contour_utils.py"""

    def test_get_data_paths_defaults(self, tmp_path):
        """Cover line 38: get_data_paths return statement with defaults."""
        paths = get_data_paths()
        assert "precompute" in paths
        assert "noise_levels" in paths
        assert "params" in paths

    def test_fragment_contours_empty_contour(self):
        """Cover line 54: fragment_contours handling empty contour arrays."""
        empty_contour = np.array([]).reshape(0, 2)
        valid_contour = np.array([[0.0, 1000.0], [0.01, 1000.0]])
        contours = [empty_contour, valid_contour]

        result = fragment_contours(contours)
        assert len(result) == 1
        assert np.array_equal(result[0], valid_contour)

    def test_identify_harmonics_single_contour(self):
        """Cover line 97: identify_harmonics early return for < 2 contours."""
        contour = np.array([[0.0, 1000.0], [0.1, 1000.0]])
        result = identify_harmonics([contour])
        assert len(result) == 1

    def test_identify_harmonics_remove_second(self):
        """Cover line 110: identify_harmonics removing the second contour (j)."""
        c_i = np.array([[0.0, 1000.0], [0.1, 1000.0]])
        c_j = np.array([[0.0, 2000.0], [0.1, 2000.0]])

        result = identify_harmonics([c_i, c_j])
        assert len(result) == 1
        assert np.array_equal(result[0], c_i)

    def test_find_matching_param_partial_multiple(self, tmp_path):
        """Cover lines 167-169: _find_matching_param with multiple partial matches."""
        p1 = tmp_path / "ind_params.mat"
        p2 = tmp_path / "ind123_params.mat"
        p1.touch()
        p2.touch()

        lookup = {
            "ind": p1,
            "ind123": p2
        }

        result = _find_matching_param("IND123_001", lookup)
        assert result == p2

    def test_load_ground_truth_npy(self, tmp_path):
        """Cover lines 174-187: load_ground_truth with .npy file."""
        label_dir = tmp_path / "FBID_001"
        label_dir.mkdir()

        gt_data = np.array([[0.0, 1000.0], [0.1, 1000.0]])
        np.save(label_dir / "test_file_params.npy", gt_data)

        wav_path = label_dir / "test_file.wav"
        wav_path.touch()

        contour, discont = load_ground_truth(wav_path, tmp_path)
        assert contour is not None
        assert np.array_equal(contour, gt_data)
        assert discont is None

    def test_save_and_load_precomputed(self, tmp_path):
        """Cover lines 205, 207, 210: save_precomputed directory creation and saving."""
        wav_path = tmp_path / "FBID_001" / "CLEAN" / "test_file.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.touch()

        contours = [np.array([[0.0, 1000.0]])]

        save_precomputed(
            wav=str(wav_path),
            contours=contours,
            algorithm="TEST_ALGO",
            src=str(wav_path.parent),
            top_dir=str(tmp_path / "Precompute")
        )

        saved_file = tmp_path / "Precompute" / "TEST_ALGO" / "FBID_001" / "CLEAN" / "test_file_contour.npy"
        assert saved_file.exists()

    def test_load_precomputed_fallback(self, tmp_path):
        """Cover lines 222-230: load_precomputed fallback glob search."""
        algo_dir = tmp_path / "TEST_ALGO" / "FBID_001" / "CLEAN"
        algo_dir.mkdir(parents=True)

        contours = [np.array([[0.0, 1000.0]])]
        np.save(algo_dir / "IND123_001_contour.npy", np.array(contours, dtype=object))

        wav_path = tmp_path / "FBID_001" / "CLEAN" / "IND123.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.touch()

        result = load_precomputed(
            wav=str(wav_path),
            ALGORITHM="TEST_ALGO",
            top_dir=str(tmp_path)
        )

        assert result is not None

    def test_run_ground_npy(self, tmp_path):
        """Cover lines 239-259: run_ground with .npy file."""
        label_dir = tmp_path / "FBID_001"
        label_dir.mkdir()

        gt_data = np.array([[0.0, 1000.0], [0.1, 1000.0]])
        np.save(label_dir / "test_file_params.npy", gt_data)

        wav_path = label_dir / "test_file.wav"
        wav_path.touch()

        contours, discont = run_ground(str(wav_path), str(tmp_path))
        assert len(contours) == 1
        assert np.array_equal(contours[0], gt_data)
        assert discont == []

    @patch('audio_analysis.utils.contour_utils.loadmat')
    def test_run_ground_mat_with_discont(self, mock_loadmat, tmp_path):
        """Cover lines 267-309: run_ground with .mat file and discontinuities."""
        label_dir = tmp_path / "FBID_001"
        label_dir.mkdir()

        # Create a dummy .mat file so param_path.exists() returns True
        param_file = label_dir / "test_file_params.mat"
        param_file.touch()

        contour_data = np.array([
            [0.0, 1000.0], [0.1, 1000.0],
            [0.5, 1000.0], [0.6, 1000.0]
        ])
        discont_data = np.array([[0.1, 0.5]])

        # Mock loadmat to return 2D object arrays mimicking MATLAB struct fields
        mock_loadmat.return_value = {
            'W': {
                'contour': np.array([[contour_data]], dtype=object),
                'discont': np.array([[discont_data]], dtype=object)
            }
        }

        wav_path = label_dir / "test_file.wav"
        wav_path.touch()

        contours, discont = run_ground(str(wav_path), str(tmp_path))

        assert len(contours) == 2
        assert len(discont) == 1
